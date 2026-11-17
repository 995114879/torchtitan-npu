# FSDP 通信缓冲的计算流内存复用

该功能使 FSDP 通信缓冲复用计算流已释放的缓存，减少不同流各自保留大块空闲内存造成的 Reserved 增长。AllGather（AG，收集完整参数）使用原生异步解分片；ReduceScatter（RS，归约并分片梯度）在计算流申请输出缓冲。两个开关独立控制，默认关闭。

## 启用方式

在所有 worker 启动、首次导入 `torchtitan_npu` 之前设置：

```bash
export TORCHTITAN_NPU_FSDP_AG_ALLOC_ON_COMPUTE_STREAM=1
export TORCHTITAN_NPU_FSDP_RS_ALLOC_ON_COMPUTE_STREAM=1
export MULTI_STREAM_MEMORY_REUSE=2
```

仅值 `1` 表示开启。环境变量在首次导入时读取，AG 策略作用于之后创建的 FSDP 参数组；运行中修改配置后需重启训练进程。

`MULTI_STREAM_MEMORY_REUSE` 控制 torch-npu 后端的跨流内存保护方式，与 `MULTI_STREAM_MEMORY_RESERVE` 是不同配置。本功能不会覆盖后端环境变量；上述配置为 Ascend 910（A3）已验证组合。

## AG：原生异步解分片

AG 开关为每个 FSDP 参数组启用原生 `FSDPParamGroup.unshard_async_op`，无需重编译。当前固定依赖 `torch==2.14.0` 通过内部方法 `FSDPModule._set_unshard_async_op` 提供相同模式；上游在 [PyTorch #202343](https://github.com/pytorch/pytorch/pull/202343) 中将其公开为 `FSDPModule.set_unshard_in_current_stream`，并由 [TorchTitan #5627](https://github.com/pytorch/torchtitan/pull/5627) 集成。

缓冲申请、参数类型转换和输入拷贝在调用时的当前流执行，通常是模型计算流。HCCL 异步通信等待输入就绪，消费权重时由原生 FSDP 等待通信结果；输出拷贝、缓冲持有和未消费预取清理也由原生实现管理。

需要注意以下两条执行边界：

- 开关不自动创建前向预取链。基础 DeepSeek-V4/V4.1 的 EP>1 路径已有显式预取；EP=1 或自定义模型需通过 `set_modules_to_forward_prefetch` 配置下一层。第 N 层计算前先准备并提交第 N+1 层的 AG，通信才能与第 N 层计算重叠；没有预取时通信会逐层暴露为等待。
- 类型转换和输入拷贝会占用计算流，减少与计算重叠的机会；在嵌套 FSDP 且单次反向前经历多次前向并结合 activation checkpointing 重算的路径中，原生 `torch==2.14.0` 会回退到默认 `async_op=False` 解分片。

## RS：计算流申请，保留归约流

RS 开关将原生归约输出缓冲的申请切换到计算流，通信继续使用原生归约流。使用前在通信流上同步等待计算流完成分配，并通过 `record_stream` 登记 `reduce_scatter_stream` 以及实际参与消费的 `all_reduce_stream`（如 HSDP 同步归约或使用 `all_reduce_hook` 时）。

分片组为空或大小为 1、没有后续 AllReduce 且梯度除数为 1 时，直接走原生 copy 分支，省去 `input / 1` 的临时张量。输出仍独立分配，多卡归约和其他缩放因子保留原生参数。

## 验证与限制

在 Ascend 910（A3）、`torch 2.14.0`、`CANN 9.2.0` 与 `MULTI_STREAM_MEMORY_REUSE=2` 环境下，已完成真实 NPU/HCCL 合成校验（单 rank 分片、FSDP、HSDP 与 FP32/BF16）、DeepSeek-V4 确定性训练对照与整网显存观测；CPU 单元测试仅验证开关接入与流依赖契约，不代替设备内存复用验证。

自定义通信 allocator、对称内存、CPU offload、GraphTrainer/capture、Ascend 950（A5），以及量化 AG、MTP、多模态、PP 和 LoRA 组合未包含在本次验证范围内。Reserved 是分配器保留的内存池，Allocated 是张量申请统计，二者均不是整卡占用；显存与步时表现取决于模型负载、分配历史与执行时序，需结合 profiler 或带历史的 memory snapshot 测量。
