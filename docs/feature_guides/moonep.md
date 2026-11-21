# MoonEP

MoonEP 是 DeepEP token dispatch/combine 路径上的专家副本扩展。它在 EP 组内按路由计数为各
rank 规划远程专家副本（虚拟专家空间为 2E），把副本权重预取到消费 rank，使部分 token 在本地
副本上完成专家计算；副本产生的权重梯度在反向归并回 home rank。该路径用于专家路由倾斜明显、
跨 rank token 流量集中在少数专家时的 EP 训练。

MoonEP 不改变 router、SwiGLU 和专家 GEMM 的数学定义，只改变 token 与专家权重的放置和传输
方式；专家计算仍由 PyTorch 侧的 grouped experts 完成。它依赖 DeepEP 的 `ElasticBuffer` 和
`deepep::dispatch/combine`，当前不脱离 DeepEP 单独使用。

## 背景与挑战

EP 把每个 token 按路由结果发送到持有目标专家的 rank。路由倾斜时，持有热门专家的 rank 会同时
成为 token 接收、专家计算和通信的热点，EP 组的步进时间由最慢的 rank 决定。仅靠 token
all-to-all 无法消除这种结构性不均衡。

一种处理方式是把热门专家复制到消费 rank：如果某个 rank 本来就要接收大量发往同一远程专家的
token，就让该 rank 持有一份该专家的副本，token 直接落到本地虚拟专家上，减少跨 rank token
流量。这个方案引入三类额外工作：

- 需要根据全局每专家 token 计数决定「每个 rank 复制哪些远程专家」，并让所有 rank 得到一致的
  虚拟专家 id 映射；
- 副本权重需要从 home rank 传输到消费 rank，且不能与 token 通信争用同一块 buffer/heap；
- 副本行产生的权重梯度需要归并回 home rank，否则专家权重更新会丢梯度。

MoonEP 把这三部分封装为 `moonep_plan`、`moonep_prefetch_weight`、`moonep_reduce_grad` 三个
算子，围绕现有 DeepEP 传输完成规划、权重预取和梯度归并。token dispatch/combine 本身仍复用
`ElasticBuffer`，不另建一套 token 通信。

## 特性概述

- **虚拟专家规划**：`moonep_plan` 输入每专家 token 计数和 top-k 专家 id，输出虚拟专家 id
  （扩展空间 2E）、接收区累计偏移 `cu_seqlens` 和副本清单 `experts_to_copy`。
- **权重预取**：`moonep_prefetch_weight` 把本 rank 的 gate/up/down 权重和副本行写入私有
  symmetric heap 的预取槽，返回 `2 × 本地专家数` 行的 BF16 快照；快照是独立存储，后续层
  可以复用同一个通信槽。
- **副本梯度归并**：预取算子的反向调用 `moonep_reduce_grad`，把副本行的梯度跨 rank 加回
  home 行，返回原始本地专家形状的权重梯度。
- **复用 DeepEP 传输**：dispatch 使用虚拟专家 id 和 `num_experts=2E` 进入现有
  `deepep::dispatch`；combine 直接继承 `AscDeepEPTokenDispatcher` 的实现。
- **score absorption 兼容**：保留 DeepEP 的 `absorb_router_scores` 开关，pre-W2 和 post-W2
  两条路径都可用。

## 启用方式

MoonEP 由模型配置和 override 两处共同选择：

| 配置 | 作用 |
| --- | --- |
| `moe_comm_backend="deepep"` | 让模型工厂生成 `DeepEPTokenDispatcher.Config`；`asc_moonep` 只匹配这个目标 |
| `asc_moonep` override | 把 DeepEP dispatcher 换成 `AscMoonEPTokenDispatcher` |
| `expert_hidden_dim` | 专家 FFN 宽度，用于计算私有 heap 和权重视图；模型工厂的 `make_routed_experts_config()` 会自动填入 `inner_experts.hidden_dim`，手工构造配置时必须提供 |
| `absorb_router_scores` | 沿用 DeepEP 语义；MoonEP 不修改该开关 |
| `hidden_dim`、`num_max_tokens_per_rank`、`top_k` | 与 DeepEP 相同，由配置阶段的 `update_from_config()` 填写或校验 |

override 名称：

```text
torchtitan_npu.override.common.token_dispatcher.asc_moonep
```

MoonEP override 只负责替换 dispatcher；fused SwiGLU 由单独的 override 选择，脚本同时启用
两者，例如 `torchtitan_npu.override.common.swiglu_group.asc` 和
`torchtitan_npu.override.common.swiglu_group.asc_shared_experts`。

A5 MoonEP 入口在默认的 `AC_MODE=selective` 下会一并配置 SelectiveAC 的 save-op 编辑，
避免反向重放规划算子：

```text
torchtitan_npu.override.common.activation_checkpoint.update_save_ops={"edits":"+torch.ops.torchtitan_npu.moonep_plan.default"}
```

启动脚本：

```bash
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_moonep.sh
```

该脚本默认：

- `CONFIG=deepseek_v4_flash_43layers_16experts_deepep`；
- `CLI_OVERRIDES` 包含 `asc_moonep`、fused RoPE/SwiGLU/compressor override；
- `AC_MODE=selective`，并自动追加上面的 `moonep_plan` save-op 编辑。

`USE_GOLDEN=1` 时脚本回到非 deepep recipe，且不启用上述 override，MoonEP 不生效。

MoonEP 要求 `expert_parallel_degree > 1`，不满足时 `init_buffer()` 会抛出 `ValueError`。
专家权重传输只使用 BF16。量化 dispatcher（带
`mxfp8_quant_params`）会被 `asc_moonep` 拒绝。

## 架构

```text
model config (moe_comm_backend="deepep")
  → DeepEPTokenDispatcher.Config（expert_hidden_dim 由 make_routed_experts_config 填入）
  → asc_moonep override
  → AscMoonEPTokenDispatcher
      ├─ init_buffer()
      │    ├─ super().init_buffer() → ElasticBuffer（与 DeepEP 共用，见 deepep.md）
      │    └─ 私有 HCCL group（group_desc="asc_moonep_weights"）
      │         + MoonEP symmetric heap / context（按 group、device、专家数、宽度缓存）
      ├─ dispatch()
      │    ├─ moonep.plan(...) → virtual_ids / cu_seqlens / experts_to_copy
      │    ├─ prefetch_expert_weights(...) → staged_weights（2×本地专家行）
      │    └─ super().dispatch_experts(..., num_experts=2E)
      ├─ routed experts（functional_call 使用 staged_weights）
      └─ combine()（继承 AscDeepEPTokenDispatcher）
```

`AscMoonEPTokenDispatcher` 继承 `AscDeepEPTokenDispatcher`，因此 token 传输、score
absorption、combine 和 DeepEP 反向路径全部复用；MoonEP 只在 dispatch 前增加规划与权重预取。

## 组件清单

| 模块 | 职责 |
| --- | --- |
| `override/common/token_dispatcher.py` | `AscMoonEPTokenDispatcher`、`MoonEPDispatchState`、`asc_moonep` override、weight context 缓存 |
| `ops/ascendc/moonep.py` | `torchtitan_npu::moonep_plan`、`moonep_prefetch_weight`、`moonep_reduce_grad`、`MoonEPContextHandle`、fake/meta 与反向 |
| `patches/torchtitan/models/common/moe.py` | 给 dispatcher 传专家权重；有 `staged_weights` 时用 `functional_call` 替换 W1/W3/W2；填充 `expert_hidden_dim` |
| `patches/torchtitan/models/common/token_dispatcher.py` | DeepEP dispatcher 配置增加 `expert_hidden_dim` 字段 |
| `override/common/activation_checkpoint.py` | `update_save_ops`，用于把 `moonep_plan` 加入 SelectiveAC 保存集合 |
| `examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_moonep.sh` | A5 MoonEP 入口 |
| `tests/unit_tests/ops/ascendc/test_moonep.py`、`tests/smoke_tests/ops/test_moonep.py`、`tests/integration_tests/moonep_config.py` | 契约、2 卡传输和真实模型验证入口 |

## 数据路径

### 1. plan

`moonep.plan(counts, ids, num_ranks, context)` 在私有 context 上执行跨 rank 的每专家计数交换
和贪心副本分配，返回：

- `virtual_ids`：与输入 top-k id 同 shape 的 int32 张量，值落在扩展空间 `[0, 2E)`；
- `cu_seqlens`：虚拟专家升序的接收区累计偏移，长度为 `2 × 本地专家数`；
- `experts_to_copy`：`[num_ranks, num_local_experts]` 的副本清单，`-1` 表示空槽。

算子内部把后端返回的视图 `reshape_as`/`clone` 成独立张量，避免后续调用覆盖；`MoonEPDispatchState`
保存这三个结果以及本次分发状态。

### 2. prefetch_weight

`prefetch_expert_weights` 把 `w1_EFD`、`w3_EFD`、`w2_EDF` 和 `experts_to_copy` 交给
`moonep_prefetch_weight`。算子在 context 的 3 个权重组和 1 个可复用通信槽上：

- 由 home rank 把副本行写入目标 rank 的预取槽；
- 返回 3 个 `2 × num_local_experts` 行的 BF16 快照，前半是本地行，后半是副本行；
- 在算子边界内完成 `clone`，使快照不随通信槽被下一层覆盖。

快照通过 `functional_call` 替换 routed experts 的 `w1_EFD`/`w3_EFD`/`w2_EDF`，使 grouped
experts 按虚拟专家 id 和 `2 × 本地专家数` 的权重布局计算。

### 3. dispatch 与专家计算

`AscMoonEPTokenDispatcher.dispatch` 使用 `virtual_ids` 调用
`super().dispatch_experts(..., num_experts=2 * self.num_experts)`。DeepEP 按每个 rank
`2E / EP` 个本地虚拟专家返回接收 token、每虚拟专家计数、`recv_scores` 和 handle；
`MoonEPDispatchState` 在此基础上保留规划结果和 staged weights。

专家计算仍走现有 routed experts：`routed_scores_R` 在 `absorb_router_scores=True` 时于
SwiGLU 之后、W2 之前乘 score，否则由 combine 前加权。plan 返回的 `cu_seqlens` 与 dispatch
返回计数的累计和一致；当前 `_ClampGroupedExperts` 从每专家计数计算 grouped GEMM offsets。

### 4. 激活检查点

`moonep_prefetch_weight` 的返回值是独立快照，因此反向可以在任意时刻重建；`moonep_plan`
包含跨 rank 的计数交换，重放会增加一次规划通信。A5 脚本在 SelectiveAC 下把
`torch.ops.torchtitan_npu.moonep_plan.default` 加入保存集合，缓存规划结果、让预取权重在
重算时重建。

## 通信域与容量

MoonEP 使用两块互相独立的资源：

**token 传输**沿用 DeepEP 的 `ElasticBuffer`。`AscMoonEPTokenDispatcher` 继承
`get_buffer_and_group()`，和普通 DeepEP 一样按
`(deepep_group, torch.npu.current_device(), hidden_dim, num_max_tokens_per_rank, top_k)`
缓存；参数相同时与 `AscDeepEPTokenDispatcher` 共用同一个 token buffer，不额外申请。详见
[DeepEP](deepep.md#通信域与容量)。

**权重传输**使用独立 HCCL group：

```text
group_desc = "asc_moonep_weights"
ranks      = deepep_group 的 ranks
```

其 `hccl_buffer_size` 由 MoonEP 私有 heap 需求换算，而不是 `HCCL_BUFFSIZE`：

```text
width = hidden_dim * expert_hidden_dim
need  = HEAP_BASE + reduce_region_end(
    num_ranks, num_local_experts, width,
    num_groups=3, slots=1, grad_bytes_per_elem=4,
)
```

`3` 个权重组对应 gate/up/down，`1` 个可复用通信槽承载副本推送，FP32 梯度槽用于副本梯度归并。
算出的字节数折算成 `hccl_buffer_size`，创建 context 后调用 `check_heap_capacity()`，容量不足
会在初始化阶段报错。

context 由 `AscMoonEPTokenDispatcher._contexts` 按
`(deepep_group, torch.npu.current_device(), num_experts, hidden_dim * expert_hidden_dim)`
缓存；同一进程内参数相同的 dispatcher 共享一个 context，不同专家数或宽度各自创建。权重传输
故意不复用 `ElasticBuffer` 的 communicator 和 heap，避免 token 通信与权重预取争用同一份
对称内存。

## 支持组合

| 组合 | 状态 | 说明 |
| --- | --- | --- |
| DP shard + EP（EP>1） | 支持 | DeepSeek V4/V4.1可使能asc_moonep  |
| AC none / full / selective | 支持 | 集成套件三种 policy；selective 由脚本缓存 plan |
| fused SwiGLU | 支持 | 与 MoonEP 是独立的 override，脚本同时启用 |
| torch.compile | 支持 | ops 提供 fake/meta |
| PP / TP | 不支持 | 当前没有 PP/TP 组合的验证路径 |
| 量化权重 | 不支持 | `asc_moonep` 检测到 `mxfp8_quant_params` 时直接报错 |
| A3 | 不支持 | 仅Ascend 950支持 |

## 与其他特性的关系

| 特性 | 关系 |
| --- | --- |
| DeepEP | MoonEP 复用 `ElasticBuffer`、`deepep::dispatch/combine` 和反向 handle；见 [DeepEP](deepep.md) |
| EP 通信域分离 | token 组仍由 `get_deepep_group()` 创建；权重组另建 `asc_moonep_weights`；见 [EP 通信域分离](ep_process_group.md) |
| fused SwiGLU | 独立 override，需要和 `asc_moonep` 同时选择；见 [融合算子](fused_ops.md) |
| score absorption | 继承 DeepEP 的 `absorb_router_scores`，两种模式均可运行 |
| 激活检查点 | SelectiveAC 可缓存 `moonep_plan`；预取快照支持重算 |
| 低精度训练 | 不支持；量化 dispatch/combine 会被拒绝 |

## 限制

- 需要 Ascend 950；缺少
  `moonep_context`/`moonep_plan`/`prefetch_weight`/`reduce_grad` 时在初始化或算子调用处导入失败。
- 要求 `moe_comm_backend="deepep"` 且 `expert_parallel_degree > 1`。不支持 PP、TP。
- 只支持 BF16 专家权重传输；当前使能量化 会被 `asc_moonep` 拒绝。
- 权重传输只有一个可复用通信槽，下一层的预取会覆盖该槽。
- 私有 heap 容量按 3 个权重组、1 个通信槽和 FP32 梯度槽计算；容量不足在初始化阶段报错，
  `HCCL_BUFFSIZE`/`hccl_buffer_size` 不是这条路径的直接配置入口。
- `moonep_plan` 的重放会增加跨 rank 规划通信；SelectiveAC 下建议按脚本把它加入保存集合。
