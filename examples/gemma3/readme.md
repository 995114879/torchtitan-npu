# Gemma3 4B GSM8K SFT

本指南提供 Gemma3 4B 权重在 Ascend NPU 上进行 GSM8K SFT 的启动流程。训练入口复用 TorchTitan 的单轮 `ChatDataLoader`，直接加载 `openai/gsm8k` 的 `main/train` split。

## 前置条件

- 已准备好与本仓库 `requirements.txt` 匹配的 TorchTitan、`torch_npu` 和 Python 环境。
- 已加载 CANN 环境并有可用的 NPU；本指南不安装驱动、CANN、TorchTitan 或 `torch_npu`。
- 已安装 `modelscope` CLI。
- 使用 `google/gemma-3-4b-pt` 模型资产；上游 TorchTitan 0.3.0 tokenizer 会读取 PT 资产目录中的 `chat_template.jinja`。

## 1. 准备模型资产

在仓库根目录执行以下命令，将 PT 权重放在仓库内的 `.gemma3_gsm8k`，再从 IT 资产下载官方 `chat_template.json`，提取其中的 `chat_template` 字段写入 PT 资产目录的 `chat_template.jinja`。训练始终使用 PT 权重；IT 目录只作为模板文件的来源。

```bash
MODEL_DIR="$PWD/.gemma3_gsm8k/models/gemma-3-4b-pt"
mkdir -p "$MODEL_DIR"
# Download the PT weights used for training.
modelscope download google/gemma-3-4b-pt \
  --repo-type model \
  --revision "${GEMMA3_MODEL_REVISION:-master}" \
  --local_dir "$MODEL_DIR"

TEMPLATE_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMPLATE_DIR"' EXIT
# Download only the IT chat-template source; IT weights are not used.
modelscope download google/gemma-3-4b-it \
  --repo-type model \
  --revision "${GEMMA3_TEMPLATE_REVISION:-master}" \
  --include "chat_template.json" \
  --local_dir "$TEMPLATE_DIR"
python -c 'import json, pathlib, sys; source, target = map(pathlib.Path, sys.argv[1:]); target.write_text(json.loads(source.read_text(encoding="utf-8"))["chat_template"], encoding="utf-8")' \
  "$TEMPLATE_DIR/chat_template.json" "$MODEL_DIR/chat_template.jinja"
```

## 2. 启动 4B SFT

GSM8K 不随 torchtitan 或本仓库提供。recipe 复用上游 `datasets.load_dataset("openai/gsm8k", name="main", split="train")`：先访问 Hub，失败时尝试本地缓存；没有网络且没有缓存时训练无法开始。官方 Hub 不可访问时，直接给启动命令加兼容镜像 endpoint：

准备完成后，在同一仓库根目录执行：

```bash
HF_ENDPOINT=https://hf-mirror.com \
OUTPUT_DIR="$PWD/.gemma3_gsm8k/runs/gsm8k-s4096-st300" \
bash examples/gemma3/gemma3_gsm8k_4b_pt_fsdp2.sh \
  --metrics.enable-tensorboard
```

`hf-mirror.com` 是第三方服务；网络正常时删除 `HF_ENDPOINT` 即可。当前 recipe 未配置 `Validator.Config`，不会自动执行 validation。

模型、并行策略和 HF checkpoint 初始化由 `sft_gemma3_4b_fsdp2` recipe 管理。launcher 设置资产路径、常改的 `STEPS`、`WARMUP_STEPS`、`SEQ_LEN`、`MBS`、`GBS`、`OUTPUT_DIR` 和 `CHECKPOINT_INTERVAL`。`WARMUP_STEPS` 默认按 `ceil(3% * STEPS)` 计算，也可以显式覆盖；这个训练策略属于示例入口，Gemma3 模型包只使用 TorchTitan 的通用 scheduler 配置。checkpoint 保存在 `OUTPUT_DIR/checkpoint`，stdout 日志默认写入 `logs/sft_gemma3_4b_fsdp2_<timestamp>.log`；启用 TensorBoard 后，原始指标位于 `OUTPUT_DIR/tb/` 下。

## 3. 绘制 loss 和 grad_norm

训练结束后，将 `<timestamp>` 替换为实际日志时间戳：

```bash
python .agents/skills/training-log-visualization/scripts/plot_training_logs.py \
  --log-a "logs/sft_gemma3_4b_fsdp2_<timestamp>.log" \
  --output "$PWD/.gemma3_gsm8k/runs/gsm8k-s4096-st300/loss_grad_norm.png" \
  --no-show
```

图中包含每步 `loss` 和 `grad_norm`。
