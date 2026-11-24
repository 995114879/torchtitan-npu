# DeepSeek-V4 ACLGraph 的分段入图与捕获边界

ACLGraph 采用分段入图策略：在 GraphTrainer 的前反向 joint FX 图中，将形状稳定的连续设备计算捕获为多个区域并重放，以减少逐算子下发开销。MoE 的路由专家路径继续按真实路由量运行，不为捕获补齐专家容量。数据加载、模型预处理和优化器更新仍由原训练流程执行。

## 启用方式与限制

使用普通 GraphTrainer recipe，例如 `graph_trainer_deepseek_v4_flash_43layers_16experts`，通过 `compile.pass_pipeline` 选择。DSV4 recipe 默认使用 `mutation-functionalization`：它保留 GraphTrainer 默认图变换，并修复重计算时的原地操作。ACLGraph 只叠加在这条普通管线上：

| 场景 | pipeline | EP overlap |
| --- | --- | --- |
| 原始 GraphTrainer | `mutation-functionalization` | 关闭 |
| 分段入图 | `mutation-functionalization+aclgraph` | 关闭 |

ACLGraph pipeline 要求 `compile.mode=aot_fx_trace`、`compile.inductor_compilation=regional`、关闭 EP overlap、PP=1、CP=1。不提供与 Auto Overlap 的组合。启用 PP 时，初始化会在队列切换和模型准备之前报错，需设置 PP=1 或关闭 ACLGraph。MTP 会在修改模型前报错，需设置 `num_mtp_layers=0` 或关闭 ACLGraph。TP 及量化组合应在目标设备单独验证。

`compile.backend` 与 ACLGraph pass 是两个层次的设置。四卡集成用例使用 `aot_eager`；该设置仍经过 AOT joint graph、区域编译与图捕获，不等于跳过 GraphTrainer。

在 ACLGraph pipeline 中添加 `--compile.disable-passes aclgraph_pass`，会跳过模型 metadata 准备、队列切换和附加捕获 pass，返回原 pipeline。CLI 禁用列表会覆盖 recipe 默认列表，hook 会补回 NPU 所需的 `cudagraph_pass` 禁用项。该对照测量的是完整适配的效果。

## 哪些部分留在图外

[aclgraph.py](../../torchtitan_npu/extensions/torchtitan/experiments/graph_trainer/aclgraph.py) 的 `_eager_boundary` 检查当前节点及其直接输入的 FX metadata：

| 情况 | 原因 | 处理 |
| --- | --- | --- |
| `module_fqn` 包含完整路径分量 `routed_experts` | 路由量决定通信 splits、重排输出行数和专家计算尺寸 | 整个路由专家子树保持 eager，包括重计算与反向 |
| `aten._local_scalar_dense` | 把设备或 CPU Tensor 内容读成 Python 标量，需要执行当前批次的读取 | 保留运行时读取 |
| 自身或直接输入含 CPU Tensor | 包括 D2H、CPU splits 整理及其他 host 计算 | 保持 eager |
| Tensor 的 shape/stride 含 unbacked 符号，或输入/输出是含 unbacked 符号的标量 | 尺寸或标量依赖本批实际数据，捕获时的具体值不能代表后续步骤 | 连同消费这些动态值的节点保持 eager |

unbacked 符号指由数据内容产生的符号，例如真实接收 token 数。它不同于仅用符号表达但布局已知的普通输入尺寸。这里只排除包含数据相关符号的部分，不把所有符号尺寸一概排除。

路由专家边界覆盖：

```text
dispatch：统计/交换路由量、构造 splits、token 重排、All-to-All
    → inner_experts：随接收行数变化的专家计算
    → combine：恢复专家顺序、反向 All-to-All、还原 token 布局
```

仅排除 CPU 的 `item()` 不够：后面的 All-to-All 和 GEMM 仍可能用变化的尺寸，因此显式排除整个 `routed_experts` 子树。动态形状检查则补充保护模块标记不完整的节点。

Attention、共享专家、路由器中固定布局的设备计算，以及其他稳定的前反向工作，可按最终图顺序捕获。实际捕获范围取决于 FX metadata 与调度结果，不能据模块名称承诺整层都进入同一张图。

## MoE 保留原来的动态布局

[模型准备](../../torchtitan_npu/models/deepseek_v4/aclgraph.py) 不替换 `token_dispatcher`。显式选择 Ascend dispatcher 时，仍使用 [AscAllToAllTokenDispatcher](../../torchtitan_npu/override/common/token_dispatcher.py)；其他配置继续使用原 dispatcher。

每个 rank 每步按真实路由数组织紧凑数据和通信 splits，专家输入行数可增可减。图外执行会重新计算这些值，不要求各 rank 容量相同，也不要求各 step 容量相同。每条通信边的发送量和接收量仍须匹配，这是 All-to-All 自身的要求。

模型准备保留原 dispatcher 的 `absorb_router_scores` 行为、mesh 和参数对象，不引入专家容量桶或专用重排算子。

这样减少了 padding、无效专家计算与固定布局维护量，也放弃了路由专家区域的捕获收益。剩余捕获能否抵消区域调用、输入复制和输出保护开销，需要同卡性能对照。

## Attention 为什么仍需要固定 metadata

EP 的动态容量与 Attention 的文档 metadata 是两个独立问题。Attention 主输入可以始终为 `[B,S,D]`，但 packed batch 每步包含的文档数不同，`cu_seqlens` 和 `seqused` 的长度仍可能变化。继续捕获 Attention 时，需要解决这部分变化。

[prepare_aclgraph_trainer](../../torchtitan_npu/extensions/torchtitan/experiments/graph_trainer/aclgraph.py) 按实际训练与 dataloader 配置计算：

```python
document_capacity = local_batch_size * (seq_len // document_alignment)
prepare(document_capacity=document_capacity)
```

每段至少占一个对齐单元；B=1、S=4096、对齐粒度为 128 时，最多预留 32 段。对齐粒度必须为正并整除序列长度；未知 loader 按每 token 一段给出上界。

[aclgraph_metadata.py](../../torchtitan_npu/models/deepseek_v4/aclgraph_metadata.py) 在文档边界数组左侧补零：额外槽位表示零长度文档，不增加真实 token。压缩率对应的 query、SWA、compressed KV 边界保持同一文档槽位关系。实际文档数超过上界时明确报错。

[模型准备](../../torchtitan_npu/models/deepseek_v4/aclgraph.py) 仅修改当前模型实例的方法：

1. `_fixed_attention_metadata` 按 `attention_metadata_key` 为各类 Attention geometry 构造固定长度 metadata。
2. `reuse_metadata_buffers` 将每步新值复制进 `_static_metadata_buffers`，保持设备地址稳定。
3. [Ascend Attention backend](../../torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py) 按这些边界生成自身需要的 CANN records。

metadata 中的 Tensor 是位置、文档边界和 backend 辅助记录；输入 token、activation、参数不是通过这些数组承载。普通 token 数据仍按原训练输入传入。

## 分区 pass 做什么

`construct_aclgraph_passes` 保留普通 pipeline 的 pass 顺序，去掉 CUDA 捕获与逐节点 kernel 注解，随后追加 loss 累积缓冲复用和 ACLGraph 分区。

`reinplace_loss_accumulation_pass` 只恢复 loss 中独占的二维 FP32 梯度累积缓冲，将满足别名安全条件的 `add` 改回 `add_`。这处理的是单个 microbatch 内分块 LM head 的反向累加，与 GBS 是否触发跨 microbatch 梯度累积不同。

`aclgraph_pass` 按最终 FX 顺序，把连续 eager 节点与连续可捕获节点分别组成子图。边界由计算类型和存储所有权共同决定：

1. `_eager_boundary` 保留路由专家、设备标量读取和动态尺寸计算。
2. 与 caller 输入共享存储的返回值，以及空存储等无法安全重建的输出别名，按 `eager_alias_storages` 保留原地址。
3. `_partition_storage_safe_regions` 追踪动态 eager 来源。对这些存储的写入、跨分区或返回 step 外的 view，以及含动态 Tensor 的容器解包留在 eager；普通只读计算仍可捕获。新增边界后重新检查，直到没有新的不安全节点。

图内生成、只在图内消费的 view 可以继续捕获。动态 eager 结果的 view 若进入稳定输入副本并逃逸，则无法同时保留来源别名和跨步结果，因此需保留原始 eager 运算。

`split_module(..., keep_original_order=True)` 保持原有执行顺序。eager 子图沿用其 `forward`，其余子图安装 `ACLGraphWrapper`。摘要保存在 `gm.meta["aclgraph_partitions"]`，使用 `eager` 和 `capture` 字段。

## 运行时为什么仍需要管理地址和存活期

[aclgraph_runtime.py](../../torchtitan_npu/extensions/torchtitan/experiments/graph_trainer/aclgraph_runtime.py) 继续负责稳定区域的捕获与重放：

- 首次调用执行 warmup；第二次调用捕获并重放；后续检查输入、更新缓冲并 replay。
- 参数、状态和地址稳定的被写入输入保留原地址；写入每步新分配的 eager 结果时，相关写操作留在 eager。可复制的 root 动态输入写入共享持久缓冲，允许每步数据与来源地址变化。
- root 输入若共享同一存储和布局，按整个别名组决定是否 staging；组内任一输入标记为 producer 持有的静态缓冲时，全组透传。运行前检查实际别名和 Tensor 元数据。
- 图外专家计算返回固定形状的 token 输出时，后续捕获区域将新内容复制进自身稳定输入；不能把 eager producer 的地址当作捕获所有权。
- 图内 producer 的中间存储可供后续区域借用；weak Tensor handle 与 FX 最后使用管理减少长期持有 activation 和未分片权重。
- 返回 joint 图外的梯度按需要 clone，避免下一步覆盖 `param.grad`；与静态状态共享存储的输出保留别名。

正常路径在 joint step 出口保护返回值。区域内和区域间保留捕获存储的原始别名，在最后一次写入后，按设备和底层分配地址分组，每组复制一次原始字节，再重建返回 view 的 shape、stride、dtype 和 offset。同一分配的多个返回值继续共享复制后的存储，下一步 replay 不会覆盖已返回的梯度。

首次捕获返回拥有存储的 Tensor，FX 负责将其保留到最后一次使用或 step 出口复制完成；后续 replay 使用非拥有 handle。缺少输出 alias metadata 时，退回区域级复制，并限制后续借用。`ACLGraphWrapper` 的 `clone_output_indices` 接收扁平输出叶子索引，默认 `None` 保持独立 wrapper 的保护范围。

输入 staging 分两类：FX 已知只读取连续逻辑区域时，仅复制 Tensor 的逻辑元素，允许相同前缀来自不同长度的底层存储；非零 offset、非连续布局、conj/neg view 或涉及 `storage_offset` 的访问则保留完整字节存储。完整存储复制不按 dtype 截断尾部字节。独立 wrapper 缺少 FX 访问分析，默认采用完整存储策略。

guard 检查 shape、stride、dtype、device、Tensor 类型与 dispatch keys、offset、静态地址及 Python 标量；完整存储策略还检查底层字节数。所有输入检查通过后才更新稳定缓冲，避免后面的非法参数导致前面的输入或共享槽位先被修改。C++ guard 拒绝时始终报错，即使 Python 诊断未找到更具体原因，也不会放行。稳定区域的布局变化不会自动重捕获。路由专家的动态尺寸已经留在图外，不再依靠容量上限容纳它们。

`_CaptureCoordinator` 在 warmup/capture 时核对跨 rank 区域顺序并协调进入捕获，稳定 replay 不额外执行 Gloo barrier。捕获内对设备输出调用 `wait_tensor`，接入待完成的 HCCL stream。关闭 trainer 时先释放图和协调组。

初始化还会将不支持捕获的任务队列模式 2 切换到 1。该模式为进程级设置，关闭时不会恢复；性能对照必须记录这项差异。

## 验证与判读

[CPU 回归](../../tests/unit_tests/extensions/torchtitan/experiments/graph_trainer/test_aclgraph.py) 直接执行分区后的 joint 图，让路由行数按 2、4、1、0、3 变化，以独立公式比较 loss 与输入梯度，并确认稳定区域各捕获一次。模型准备测试检查 dispatcher 对象未被替换，同时验证固定文档 metadata 的计算语义与地址复用。

[单卡 NPU 冒烟](../../tests/smoke_tests/extensions/torchtitan/experiments/graph_trainer/test_aclgraph_smoke.py) 验证 16K 文档帧的 Compressor 重放和捕获存储所有权。[四卡集成用例](../../tests/integration_tests/deepseek_v4.py) 的 `dsv4_aclgraph_ep4_fsdp4` 使用真实路由、EP4/FSDP4、S640、普通管线与 ACLGraph，关闭 EP overlap，检查四步训练完成。

### A3 8 卡 100 步精度对比

2026-11-13 使用 [训练指标比较工具](../../scripts/numerical_alignment/loss_compare.py)，在同一组 A3 设备上依次运行修改前后两个提交，从 TensorBoard 读取未截断的逐步 `loss` 和 `grad_norm`，对齐 step 1～100。**本次是不开 ACLGraph 与开启 ACLGraph 的对比；数值近似检查通过，逐位一致检查未通过，严格确定性验证未完成。**

#### 对照配置

| 项目 | 基线组 | ACLGraph 组 |
| --- | --- | --- |
| 实测提交 | `6aa5a7f56554e09922c4796398c71a769c200569` | `9be63bdf1c44b1955846dc06be3afc58ed2a2cc9` |
| `compile.pass_pipeline` | `mutation-functionalization` | `mutation-functionalization+aclgraph` |
| ACLGraph | 关闭 | 开启 |
| 任务队列模式 | 2 | 初始化时由 2 切换为 1 |
| `debug.deterministic` / `debug.deterministic_warn_only` | 均为 `false` | 均为 `false` |
| EP overlap / profiler | 均关闭 | 均关闭 |

除 pipeline 和各自输出目录外，两侧打印的配置一致；配置核对仅忽略初始化函数文本表示中的进程内存地址。该对照包含固定 Attention metadata、任务队列切换和图捕获等完整适配，不只比较 `replay()` 调用。

- **设备与环境：**8 卡 A3 `Ascend910_9362`，同一组逻辑设备 0～7；CANN 9.2.0，Python 3.12.10，torch `2.14.0.dev20260811+cpu`，torch_npu `2.14.0+git2d91fb5`；上游 torchtitan 提交 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`。
- **模型与并行：**`graph_trainer_deepseek_v4_flash_43layers_16experts`，43 层、16 专家，共 24,578,655,831 参数，MTP=0；EP8/FSDP8，DP replicate、TP、CP、PP 均为 1，SPMD backend 为 `partial_dtensor`。
- **批量与精度：**MBS=1、GBS=8、S4096，梯度累积 1 次，各运行 100 步；FP32 主参数、BF16 混合精度计算、FP32 梯度归约，量化关闭，`training.extension.allow_hf32=true`。
- **编译路径：**`aot_fx_trace`、`compile.backend=inductor`、`inductor_compilation=regional`、`memory_policy=dsv4-mhc-moe-save`；保留融合 Attention 和真实 MoE 路由。
- **共同起点与数据：**加载同一份完整 step-0 DCP 模型权重，`seed=42`，相同 tokenizer、`c4_test` 数据和 128 token 文档对齐；相同 DistMuon/AdamW 参数分组及学习率调度器，优化器和数据游标均从新训练开始。

#### 指标与判定

数值近似阈值沿用 [精度对齐开发指南](../developer_guides/numerical_alignment.md)：`MEAN_AE <= 0.02`、`MAX_RE <= 0.005`、`MEAN_RE <= 0.005`。相对误差分母为 `max(abs(baseline), 1e-4)`；下表的相对误差以比例表示，`0.005` 对应 `0.5%`。

| 指标 | MEAN_AE | MAX_RE | MEAN_RE | 数值近似判定 |
| --- | ---: | ---: | ---: | --- |
| `loss` | `1.59034729e-4` | `5.33851308e-5` | `1.55679142e-5` | PASS |
| `grad_norm` | `2.16837883e-3` | `2.06360508e-3` | `3.50064229e-4` | PASS |

两组均完成 100 步，训练退出码和比较工具退出码均为 0，指标没有缺步、NaN 或 Inf。ACLGraph 组 rank0 捕获 173 个区域，每个区域只捕获一次，后续持续 replay；基线没有 ACLGraph 捕获记录。

按零容差逐步检查，`loss` 仅第 1 步完全相同，最大绝对误差为 `5.50270081e-4`（step 44）；`grad_norm` 的 100 步均存在差异，最大绝对误差为 `1.20439529e-2`（step 72）。
