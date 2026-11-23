# Checkpoint 使用指南

本文说明如何使用 TorchTitan 的 checkpoint 机制保存、加载
`torchtitan-npu` 训练状态，以及如何加载或保存 Hugging Face safetensors 权重。

当前实现基于上游 `torchtitan.components.checkpointer.CheckpointManager`
（torchtitan v0.3.0 起 `torchtitan.components.checkpoint` 仅为兼容再导出）：

- DCP（Distributed Checkpoint）用于保存和加载分布式训练状态。
- Hugging Face safetensors 只用于模型权重的加载或保存，不能加载优化器和训练步数。

除下文「导出 vLLM Ascend 量化权重」外，本文不覆盖其他权重转换工具。需要转换 Hugging Face 权重时，
使用上游仓库提供的转换工具。

## 当前配置入口

训练配置由以下注册表提供：

| 模块 | 配置入口 | checkpoint 默认值 |
| --- | --- | --- |
| `torchtitan_npu.models.deepseek_v4.config_registry` | `deepseek_v4_debugmodel`、`deepseek_v4_flash`、`deepseek_v4_pro` | `enable=False`，`interval=100` |
| `torchtitan_npu.models.deepseek_v3_2.config_registry` | `deepseek_v3_2_debugmodel` | `enable=False`，`interval=10`，`last_save_model_only=False` |

`enable=False` 时既不会保存 checkpoint，也不会执行 checkpoint 加载。训练命令需要显式
传入 `--checkpoint.enable`，或在 Python 配置中设置 `checkpoint.enable = True`。

## checkpoint 格式与目录

### DCP

DCP 是 PyTorch Distributed Checkpoint 格式，可以保存模型、优化器、学习率调度器、
dataloader 和训练状态，适合断点续训以及改变并行切分后的加载。

启用 `checkpoint.folder=checkpoint` 时，目录结构通常为：

```text
<dump_folder>/checkpoint/step-<step>/
├── .metadata
└── *.distcp
```

其中 `<dump_folder>` 来自 `--dump-folder`，`step-<step>` 是训练步数。DCP 加载会检查
`.metadata`，并按 `--checkpoint.load-step` 选择指定步数；默认值 `-1` 表示选择最新步数。

### Hugging Face safetensors

Hugging Face 权重通常包含 `model.safetensors` 或
`model.safetensors.index.json`，也应准备对应的 tokenizer 和 `config.json` 等配置文件。
该格式只表示模型权重，不包含优化器、学习率调度器和训练步数。

## 常用配置项

配置类型来自上游 `torchtitan.components.checkpointer.base`，Python 配置使用下划线命名，命令行
使用 tyro 的连字符命名：

| 配置项 | 作用 |
| --- | --- |
| `enable` | 启用 checkpoint 保存和加载。默认 `False`。 |
| `folder` | checkpoint 子目录名，默认 `checkpoint`；最终路径为 `{dump_folder}/{folder}`。 |
| `interval` | 保存 DCP checkpoint 的步数间隔。 |
| `load_step` | 要加载的步数；`-1` 表示最新 checkpoint。 |
| `initial_load_path` | 当前输出目录没有 checkpoint 时使用的初始 checkpoint 路径，必须是绝对路径或远程 URI。 |
| `initial_load_model_only` | 初始加载是否只加载模型权重，默认 `True`。设为 `False` 才会尝试加载完整训练状态。 |
| `initial_load_in_hf` | 将初始路径按 Hugging Face safetensors 读取。HF 加载只能加载模型权重。 |
| `initial_load_in_hf_quantized` | 从 HF 量化权重加载；使用前必须启用 `initial_load_in_hf`。支持的格式与适用范围见下文「加载量化 HF 权重」。 |
| `last_save_model_only` | 最后一步是否只保存模型权重，默认 `True`。设为 `False` 才保存完整训练状态。 |
| `last_save_in_hf` | 最后一步是否以 Hugging Face safetensors 格式保存。必须同时使用模型权重保存模式。 |
| `export_dtype` | 保存时模型权重导出的 dtype，可用 `float16`、`bfloat16`、`float32`。 |
| `async_mode` | 保存方式：`disabled`、`async` 或 `async_with_pinned_mem`。 |
| `keep_latest_k` | 保留最近的 checkpoint 数量；`0` 表示全部保留，不能设置为 `1`。 |
| `exclude_from_loading` | 从 DCP 加载时排除状态，例如 `optimizer,lr_scheduler,dataloader`。 |
| `enable_first_step_checkpoint` | 是否在第一个训练 step 后立即保存一次 checkpoint。 |
| `create_seed_checkpoint` | 不应用并行切分，创建可供后续任务重分片加载的 seed checkpoint。 |
| `load_only` | 只加载、不保存 checkpoint，适合验证或调试。 |
命令行布尔字段使用反向选项关闭，例如 `--checkpoint.no-enable`；不要写成
`--checkpoint.enable false`。列表字段使用英文逗号分隔，不要用空格拆成多个 token。

## 保存 DCP checkpoint

以下命令使用当前脚本默认的 `deepseek_v3_debugmodel` 配置。运行前准备 Ascend/CANN 环境：

```bash
NGPU=1 \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dump-folder ./outputs/dsv3_checkpoint \
  --checkpoint.enable \
  --checkpoint.folder checkpoint \
  --checkpoint.interval 100 \
  --training.steps 200
```

训练过程中会生成：

```text
./outputs/dsv3_checkpoint/checkpoint/step-100/
./outputs/dsv3_checkpoint/checkpoint/step-200/
```

也可以在配置注册表中设置：

```python
from torchtitan.components.checkpointer import CheckpointManager

checkpoint = CheckpointManager.Config(
    enable=True,
    folder="checkpoint",
    interval=100,
    keep_latest_k=5,
    async_mode="disabled",
)
```

## 加载 DCP checkpoint

### 自动加载最新 checkpoint

以下示例使用与前述保存示例相同的 DeepSeek V3 debug 模型和 tokenizer。加载时模型配置必须与 checkpoint 匹配。使用相同的 `--dump-folder` 和 `--checkpoint.folder` 重新启动，并启用 checkpoint：

```bash
NGPU=1 \
MODULE=torchtitan.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dump-folder ./outputs/dsv3_checkpoint \
  --checkpoint.enable \
  --checkpoint.folder checkpoint
```

当输出目录中存在可用的 `step-*` checkpoint 时，框架会默认加载最新一步。

### 指定加载步数

```bash
MODULE=torchtitan.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dump-folder ./outputs/dsv3_checkpoint \
  --checkpoint.enable \
  --checkpoint.load-step 100
```

### 从其他目录初始化

使用新的输出目录，并通过 `--checkpoint.initial-load-path` 指向旧 checkpoint 的完整
step 目录。该参数必须使用绝对路径或远程 URI；相对路径会触发上游的 `ValueError`：

```bash
NGPU=1 \
MODULE=torchtitan.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dump-folder ./outputs/dsv3_new_job \
  --checkpoint.enable \
  --checkpoint.initial-load-path /absolute/path/to/dsv3_checkpoint/checkpoint/step-100
```

默认只加载模型权重。若要加载优化器、学习率调度器和训练状态，应显式关闭
`initial_load_model_only`：

```bash
MODULE=torchtitan.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dump-folder ./outputs/dsv3_new_job \
  --checkpoint.enable \
  --checkpoint.initial-load-path /absolute/path/to/dsv3_checkpoint/checkpoint/step-100 \
  --checkpoint.no-initial-load-model-only
```

如果 `{dump_folder}/{checkpoint.folder}` 已经存在可用 checkpoint，框架会优先从该目录加载，并忽略 `initial_load_path`。从新权重启动实验时，应使用新的 `dump-folder` 或
清理旧的 checkpoint 目录。

如只需要模型和部分训练状态，可以排除不需要的键：

```bash
MODULE=torchtitan.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dump-folder ./outputs/dsv3_model_only \
  --checkpoint.enable \
  --checkpoint.initial-load-path /absolute/path/to/dsv3_checkpoint/checkpoint/step-100 \
  --checkpoint.exclude-from-loading optimizer,lr_scheduler,dataloader
```

## 加载 Hugging Face 权重

从 HF safetensors 初始化时，需要启用 checkpoint、设置
`--checkpoint.initial-load-in-hf`，并确保 `--checkpoint.initial-load-model-only` 保持
为 `True`。以下使用 `deepseek_v4_debugmodel`，初始权重和 tokenizer 必须与该模型配置匹配；完整规格的 DeepSeek-V4 权重需要改用相应的模型配置和并行环境：

```bash
NGPU=1 \
MODULE=torchtitan_npu.models.deepseek_v4 CONFIG=deepseek_v4_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path /path/to/dsv4_tokenizer \
  --dump-folder ./outputs/dsv4_from_hf \
  --checkpoint.enable \
  --checkpoint.initial-load-in-hf \
  --checkpoint.initial-load-path /absolute/path/to/matching-dsv4-debug-hf
```

如果不传 `--checkpoint.initial-load-path`，checkpoint 管理器会尝试使用模型配置中的
`hf_assets_path`。`initial_load_path` 优先级高于 `hf_assets_path`。HF 权重不能加载
优化器或训练步数。

### 加载量化 HF 权重

`initial_load_in_hf_quantized` 启用后，由 state-dict adapter 按模型和目录内容选择 reader，
无需额外的格式开关：

- `deepseek_v4`：自动探测两种 MX 量化格式。
  - 固定格式 MX：`.weight`（mxfp8 为 `float8_e4m3fn`，mxfp4 为打包字节，safetensors 标签为
    `uint8` 或 `int8` 均可）与 e8m0 `.scale` 成对存放的目录。`.scale` 按行分块（每个 32 列块一个
    系数）或按 DeepSeek 官方 attn/shared_experts 的 32×32 二维分块存放均可，读取时在 CPU 侧展开
    为行式后交给反量化 kernel。
  - torchao 协议：`torchtitan_npu/tools/checkpoint_conversion/export_quantized_hf.py` 导出的目录，
    量化元数据写在 safetensors 的 `__metadata__` 中。探测时优先于固定格式 MX。
- `deepseek_v4_1`：官方 MXFP8 整包目录由组合 reader 按 FQN 路由，与训练目标是否配置 Engram 表
  无关（未配置时官方 Engram 张量保留在元数据中但不被读取）——Engram
  row/block MX 交给 Engram reader，非 Engram 的固定格式 MX 对交给 MX reader（`.scale` 可以与
  权重不在同一 shard，单个张量也可以跨多个 shard；官方包的 attn/shared_experts 为 32×32 分块
  e8m0 scale，routed experts 为 int8 标签的打包 MXFP4，均在支持范围内），普通 bf16/f32 张量走上游
  HF 读取路径。
  训练目标未引用的额外张量（如 MTP、vision）保持不动，不被读取；不受支持的量化格式在被读取时
  明确报错，不会静默加载错误权重。

其他模型的 adapter 不会选择这些 MX reader，对这类目录启用该标志只会报出与量化无关的错误；
此类模型请先使用反量化后的 bf16 权重。

## 保存 Hugging Face 权重

训练最后一步可以直接保存 HF safetensors。该模式只保存模型权重，不能作为完整训练
断点使用。以下示例使用 DeepSeek-V4 debug 配置及与其匹配的 tokenizer：

```bash
NGPU=1 \
MODULE=torchtitan_npu.models.deepseek_v4 CONFIG=deepseek_v4_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path /path/to/dsv4_tokenizer \
  --dump-folder ./outputs/dsv4_hf \
  --checkpoint.enable \
  --checkpoint.last-save-in-hf \
  --checkpoint.export-dtype bfloat16 \
  --training.steps 100
```

`last_save_in_hf` 需要模型提供 state-dict adapter，并且必须保持
`last_save_model_only=True`。`deepseek_v3_2` 和 `deepseek_v4` 模型均提供
对应的 state-dict adapter；具体可用性仍取决于模型配置和权重格式。

## 导出 vLLM Ascend 量化权重

`torchtitan_npu/tools/checkpoint_conversion/export_quantized_vllm.py` 是离线导出工具：读取训练
checkpoint 的 master 权重，按量化计划重新量化，写出 vLLM Ascend 可直接加载的 Hugging Face
safetensors。量化权重是普通的 `weight` 与 `weight_scale` 张量，逐参数类型写在
`quant_model_description.json`（ModelSlim 格式），serving 时对应 `--quantization ascend`。

输入可以是 DCP 或 HF safetensors：DCP 覆盖 `deepseek_v4`、`deepseek_v4_1`、`qwen3_5`；HF 只支持
Qwen3.5，DeepSeek 的 HF 发布件本身是 fp8/fp4 量化格式，必须从 DCP 导出。工具为单进程程序：
DCP 读取与命名转换在 CPU 上完成，量化 kernel 在 `--quant_device` 指定的 NPU 上执行，运行前需要
`torch_npu`、CANN 和可用的 Ascend NPU 环境。

### 快速开始

`--hf_assets_path` 指向 HF 发布件目录（`config.json`、tokenizer，以及 DCP 导出时用于补齐冻结
视觉塔的 `*.safetensors`；所选模型配置不含视觉模块、但 serving 需要视觉塔时加 `--needs_vision`），
非权重文件会复制到输出目录。DeepSeek 的 `config.json` 复制后会把结构字段（层数、专家数、
hidden/intermediate/vocab、rope 等）按 `--model_flavor` 的模型配置改写，因此导出裁剪层数或专家数
的 flavor 时不会残留发布件的完整规格；模型配置未声明的字段保持 assets 原值。视觉开关另行按导出
的张量集对齐：DeepSeek 的 `vision_n_layers` 在导出没有视觉塔时置为 0，有视觉塔但配置为 0 时保留
原值并告警；Qwen3.5 声明 `vision_config` 而导出没有视觉塔时直接报错。Qwen3.5 的配置项在嵌套的
`text_config` 中，暂不改写，保持 assets 原样。运行后先核对日志中的 `plan: ...`（显式 `--recipe`
时为 `recipe=... plan: ...`，回放 DCP 配置时为 `DCP replay plan: ...`）与 `Quantized N/M tensors`。

DeepSeek-V4.1，DCP 输入，回放 checkpoint 内嵌的量化配置：

```bash
python -m torchtitan_npu.tools.checkpoint_conversion.export_quantized_vllm \
  /absolute/path/to/dump/checkpoint/step-100 \
  ./outputs/vllm_quantized \
  --hf_assets_path /path/to/DeepSeek-V4.1 \
  --model_name deepseek_v4_1 \
  --model_flavor deepseek_v4_1_flash
```

Qwen3.5，HF safetensors 输入，必须显式指定 recipe：

```bash
python -m torchtitan_npu.tools.checkpoint_conversion.export_quantized_vllm \
  /path/to/Qwen3.5-27B \
  ./outputs/qwen35_quantized \
  --hf_assets_path /path/to/Qwen3.5-27B \
  --model_name qwen3_5 \
  --model_flavor 27B \
  --recipe W4A8_MXFP
```

### 量化层与 recipe

量化范围由训练 recipe 的 filter 决定（训练侧的层选择规则见
[低精度训练](../feature_guides/low_precision_training.md)），不按「所有 Linear」推断。导出侧在
`vllm_export_specs.py` 的 `QUANTIZATION_EXPORT_RULES` 中把训练 FQN 后缀换成 HF 名字后缀，只有命中
的张量在描述文件中是非 `FLOAT` 条目，清单以该表为准（V4.1 Engram 表是例外，见
[输出与校验](#输出与校验)）：

- `deepseek_v4`：`DEEPSEEK_V4_DENSE_WEIGHT_SUFFIXES`；routed expert 由 `ROUTED_WEIGHT_PATTERN` 匹配。
- `deepseek_v4_1`：`DEEPSEEK_V41_DENSE_WEIGHT_SUFFIXES`，在 V4 的清单上增加 `.engram.wkv`。
- `qwen3_5`：`QWEN35_DENSE_WEIGHT_SUFFIXES`，无 routed expert。

其余权重（embedding 与输出层、norm、MoE router、vision 塔、MTP，以及 Qwen3.5 的 `conv1d` 与 MoE
专家）不量化，在描述文件中记为 `FLOAT`。

量化类型由 recipe 的权重量化配置推导（`vllm_export_contract.py` 的 `quant_description_type`）：

| recipe | 量化类型 |
| --- | --- |
| `all_mxfp8`、`mix`、`all_block_fp8` | `W8A8_MXFP8` |
| `mix`、`all_block_fp8` + `--enable_mxfp4_qat` | routed expert 为 `W4A8_MXFP`，dense 仍为 `W8A8_MXFP8` |
| `W4A8_MXFP`（Qwen3.5，内置 MXFP4） | `W4A8_MXFP` |
| `W8A8_MXFP8`（Qwen3.5） | `W8A8_MXFP8` |
| `all_hif8` | 不支持，构建计划阶段报错 |

表中的类型是 `quant_model_description.json` 里每个权重的条目；顶层 `model_quant_type` 只有在所有
命中的权重都是 FP4 时才是 `W4A8_MXFP`，混合导出（`mix`、`all_block_fp8` + `--enable_mxfp4_qat`）
顶层仍记 `W8A8_MXFP8`。`--enable_mxfp4_qat` 只作用于 DeepSeek `mix`/`all_block_fp8` 的 routed
expert，`all_mxfp8` 与 Qwen3.5 的 recipe 不读取该开关（Qwen3.5 `W4A8_MXFP` 固有启用）。

省略 `--recipe` 时工具回放 DCP 中 `torchao_npu` wrapper 记录的逐参数配置：Qwen3.5 的
`in_proj_q/k/v` 融合成 `in_proj_qkv` 前会校验三份配置齐全且一致；训练未包成 Linear 的
serving-only 权重（V4.1 `.engram.wkv`）借用 dense 配置；任何量化目标缺配置都会报错，而不是导出
与训练不一致的权重。回放模式下传入 `--enable_mxfp4_qat` 或 `--dst_type_max` 会报错，需改用
`--recipe`。

### 输出与校验

输出目录包含模型 shard（`model.safetensors` 或 `model-*-of-*.safetensors`）、
`quant_model_description.json` 和从 `--hf_assets_path` 复制的配置文件；多 shard 或含 Engram 表时
还会生成 `model.safetensors.index.json`。输出目录不能残留上一次导出的权重或索引文件，否则直接
报错。导出只包含模型权重与量化描述，不含优化器和训练状态。未量化权重在描述文件中记为 `FLOAT`；
DCP 输入时其 dtype 按 HF 发布件的原生 dtype 还原，读不到发布件的 tensor header 或 dtype 不兼容时
保留 DCP dtype 并告警。V4.1 Engram 表按 vLLM Ascend 的逐行 MXFP8 布局导出：
`layers.<n>.engram.embed.weight` 为 `float8_e4m3fn`，同前缀的 `.scale` 为 `float8_e8m0fnu`，
每行每 32 个元素一个 scale；这两类张量不在 `quant_model_description.json` 里逐项记录，只写一条
`layers.<n>.engram.embed.quant_type: "MXFP8"` 元数据，vLLM Ascend 的 Engram loader 直接读
safetensors dtype 与 index。

写盘后工具自检 dtype、scale 形状、描述条目、shard 张量集合与 `config.json`；不满足 vLLM Ascend
加载约定时报错并删除本次写出的权重文件。复制过来的 `quantization_config` 保持 assets 原样：其中
声明的 `quant_method`（如 fp8）不会改写成导出格式，自检只记录警告，serving 时必须用
`--quantization ascend` 覆盖。`--keep_scale_tail_dim` 保留 kernel 的
`(..., ceil(blocks/2), 2)` scale 布局，只能用于检查，不能直接 serving。其余参数见 `--help`。

## 创建 seed checkpoint

seed checkpoint 用于先创建未应用并行切分的模型状态，再由多卡任务通过 DCP 重分片加载。
以下使用 DeepSeek-V4 debug 配置与匹配的 tokenizer；创建时使用单卡，并将各并行度设为 `1`：

```bash
NGPU=1 \
MODULE=torchtitan_npu.models.deepseek_v4 CONFIG=deepseek_v4_debugmodel \
bash scripts/run_train.sh \
  --hf-assets-path /path/to/dsv4_tokenizer \
  --dump-folder ./outputs/dsv4_seed \
  --checkpoint.enable \
  --checkpoint.create-seed-checkpoint \
  --parallelism.data-parallel-replicate-degree 1 \
  --parallelism.data-parallel-shard-degree 1 \
  --parallelism.tensor-parallel-degree 1 \
  --parallelism.pipeline-parallel-degree 1 \
  --parallelism.context-parallel-degree 1 \
  --parallelism.expert-parallel-degree 1
```

生成的 `step-0` 目录可以作为后续任务的 `checkpoint.initial_load_path`。

## 常见限制与验证范围

- 需要可用的 `torch`、`torch_npu`、TorchTitan、CANN 和 Ascend NPU 环境；本文命令未在当前
  Windows 环境执行。
- DCP 适用于完整训练状态加载；HF safetensors 只适用于模型权重。
- 文档中的模型入口、配置字段和命令来自源码与脚本的静态核对；多卡训练、权重格式兼容性
  和 NPU 性能未在本次文档修改中重新实测。

## 相关文档

- [快速上手](../user-guides/quickstart.md)
- [安装指南](../user-guides/installation.md)
- [训练启动脚本](../../scripts/run_train.sh)
- [DeepSeek-V4 checkpoint 配置](../../torchtitan_npu/models/deepseek_v4/config_registry.py)
