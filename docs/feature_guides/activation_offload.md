# FullAC 激活卸载

激活卸载为 eager 执行的 FullAC 训练提供逐层边界卸载：前向结束后把满足条件的层边界激活通过 NovaSwap 异步发送到 CPU，反向重计算需要该激活时再异步搬回 NPU。该能力默认关闭，当前实现面向 Ascend NPU 的 eager 训练路径。

## 适用范围

启用激活卸载时必须同时满足以下条件：

- 使用 FullAC（CLI 子命令 `activation-checkpoint:full`）；
- 使用 eager 执行，不能启用 `compile`；
- 不使用 pipeline parallelism；
- 运行环境提供 `torch_npu`、CANN 和可用的 Ascend NPU。

未启用 `--training.enable-activation-offload` 时，训练仍走原有路径，不安装激活卸载 hook。

| 能力 | 激活方式 | 数据约束 | 验证范围 | 状态 |
| --- | --- | --- | --- | --- |
| FullAC 层边界激活 D2H/H2D | `--training.enable-activation-offload` | NPU、需梯度、连续且覆盖完整 storage 的 Tensor；默认至少 1 MiB | `tests/unit_tests/extensions/cpu_offload/test_activation_offload.py` 的 CPU 逻辑单测；NPU 传输需在目标环境验证 | 有限支持 |

## 启用方式

以 DeepSeek-V4 的 FullAC 配置为例：

```bash
torchrun ... \
  --training.enable-activation-offload \
  --training.activation-offload-min-bytes=1048576 \
  activation-checkpoint:full
```

`activation-offload-min-bytes` 用于过滤小 Tensor；调大该值可以缩小卸载范围。配置字段定义在 `torchtitan_npu/config/configs.py`，Trainer 接入和约束检查位于 `torchtitan_npu/extensions/torchtitan/trainer.py`。

## 工作方式

`ActivationBoundaryOffload` 在已有 FullAC `CheckpointWrapper` 上安装 hook，不改变模型层级。每个 wrapper 的前向执行期间，saved-tensor hook 收集候选 Tensor；前向返回后选择最大的安全候选，并提交一次 NovaSwap `D2H`。复制使用 NovaSwap 的 pinned memory、copy stream、事件和存储回收机制。

反向重计算由 autograd 触发 `unpack`。`unpack` 对当前 token 提交 `H2D`，同时尝试预取前一层已经卸载的 token；在读取前提交 `WAIT_DEVICE`，随后从 NovaSwap 取回 Tensor。预取失败时退回当前 token 的按需 H2D。激活被消费后释放其 swap 注册，避免把所有重载 Tensor 保留到整个 backward 结束。

## 限制

- 当前策略每个 FullAC 层最多选择一个最大的安全候选；它不是通用的“保存了什么就全部卸载”。参数、非连续或部分 storage view、内部 overlap、无梯度和低于阈值的 Tensor 会被跳过。
- 只覆盖 FullAC 的 eager 路径，不覆盖 SelectiveAC、pipeline parallelism 或 compile 图路径。
- 未针对 MoE 专家、路由和 combine 做独立选择策略；它们只有在其 Tensor 满足通用候选条件并被 FullAC wrapper 保存时才可能参与卸载。
- 对无法由 PyTorch pytree 展开的输出容器采取保守策略，保持候选驻留在 NPU，避免错误地释放 autograd 可见别名。
- 本次 CPU 单测验证调度、预取、生命周期、FSDP 输入 view 和边界过滤；没有把 CPU 单测结果当作 NPU 传输或性能结论。

## 代码与验证

- 实现：`torchtitan_npu/extensions/cpu_offload/activation_offload.py`
- Trainer 接入：`torchtitan_npu/extensions/torchtitan/trainer.py`
- 配置：`torchtitan_npu/config/configs.py`
- CPU 单测：`tests/unit_tests/extensions/cpu_offload/test_activation_offload.py`

在仓库根目录运行：

```bash
python -m pytest tests/unit_tests/extensions/cpu_offload/test_activation_offload.py
```
