# Unit Test 架构与迁移方案

本文记录 CPU unit test 的目录、执行和生产代码归属。它是开发约定，不替代 `test-review` skill 中针对某个 PR 的审查报告。

## 目标

测试目录要能回答「被保护的对象是谁」，测试入口要能回答「这条测试是否会被执行」。目录归属和执行归属是两个维度，不能因为一个测试不计入产品 UT，就让它从 CI 中静默消失。

## 目录与统计边界

`tests/unit_tests/` 是 CPU 测试的统一执行根目录。生产语义测试按 `torchtitan_npu/` 的职责镜像放置：

- `tests/unit_tests/compile/`、`config/`、`models/`、`ops/` 和 `override/` 分别保护对应的生产模块；
- `tests/unit_tests/patches/torchtitan/` 只镜像有上游依据的临时 patch；`tests/unit_tests/patches/torch_npu/` 镜像当前有效的 NPU runtime compatibility patch；
- `tests/unit_tests/tooling/` 放仓库脚本和 skill 的自测，便于沿用现有 unit-test 入口执行，但这类测试不计入产品行为覆盖；
- 多进程 worker 与启动它的测试放在同一个语义目录；
- `tests/integration_tests/` 只放进入真实 `torchtitan.train` 的 NPU 模型 ST，不把 CPU UT、kernel pytest 或 runner 自测计入 ST。

本次目录调整将以下文件归入统一执行根，但保留其 tooling 属性：

```text
tests/unit_tests/tooling/profiler/test_profiler_tools.py
tests/unit_tests/tooling/training_log_visualization/test_training_log_visualization.py
```

它们不再散落在 `scripts/tests/` 或 `.agents/skills/.../tests/`，也不因此变成产品 UT。产品 UT 统计和报告必须按语义归属排除 `tooling/`。

## `patches/` 的职责和边界

`torchtitan_npu/patches/torchtitan/` 只保存当前固定上游版本尚未包含、且已有上游 PR 或 commit 依据的临时 patch。上游版本吸收对应改动后，应删除 patch、导入入口和专用测试。

`torchtitan_npu/patches/torch_npu/` 中的文件不是「上游 torchtitan 临时 patch」：它们改变 PyTorch、Inductor 或 `torch_npu` 的 NPU 运行时行为，例如 `determinism.py` 替换 `torch.use_deterministic_algorithms`，`device_copy.py` 修改 Inductor 的 `DeviceCopy` 和 `prims.device_put`。对应测试镜像到 `tests/unit_tests/patches/torch_npu/`；强制 eager 的 TorchTitan 兼容行为则归 `extensions/torchtitan/`。

## 执行归属

只要测试放在 `tests/unit_tests/`，现有 `.ci/unit_test.sh` 的 `pytest tests/unit_tests` 就能发现它；报告仍须单独标注 `tooling`，避免把 tooling 通过数写成产品覆盖。后续若保留其他目录外的测试，必须在对应 CI suite 显式列出执行路径，不能依赖开发者手工运行。

`experiments/torchao-npu` 保留独立测试目录，通过 `scripts/run_tests.sh cpu` 显式接入主仓 UT CI；其设备和集成测试使用手工入口。环境准备、依赖安装副作用、分层执行及报告判读见 [TorchAO-NPU 测试使用说明](../../experiments/torchao-npu/tests/README.md)。

## CPU 环境与可选依赖

CPU 单测在独立 Python 3.12 环境中执行。先根据 [安装指南](../user-guides/installation.md)、[运行依赖](../../requirements.txt)和 [开发依赖](../../requirements-dev.txt)准备配套运行时；TorchTitan 的固定版本由 `requirements.txt` 与 `.ci/setup_torchtitan.sh` 推导。以下命令创建环境并注册当前 checkout，`--no-deps` 不代替基础运行依赖的安装：

```bash
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
# 在该环境中准备上述配套运行时，再安装开发依赖。
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -v --tb=short tests/unit_tests
```

所有 CPU 测试命令都设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免 PyTorch 自动加载设备后端；CPU UT 不证明 NPU kernel、HCCL 或真实训练结果。

严格 CPU 验证应在没有可用加速器的环境中执行。`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 不会隐藏已透传的 NPU，也不会阻止显式 `torch_npu` 调用；主机暴露设备时，需使用不透传设备的独立执行环境或明确记录的 CPU 隔离措施，不能仅凭该变量声称未初始化设备。

基础 CPU 与含可选依赖的完整 CPU 使用同一测试入口，按实际依赖和跳过原因区分结果：

| 环境 | TorchFT 测试范围 | 结果判读 |
| --- | --- | --- |
| 基础 CPU，未安装 TorchFT | `patches/torchft` 下的正向用例跳过，故意阻止 TorchFT 导入的用例仍执行；TorchFT 扩展目录的测试不收集 | 分别报告缺依赖的 skip 与未收集范围，不能声称完成含 TorchFT 的完整验证 |
| 含 TorchFT 的完整 CPU | 安装 `torchft` extra 后，收集 TorchFT 扩展目录，执行 patch 用例及完整 CPU 套件 | 已安装依赖的导入异常、缺少传递依赖或行为失败均为失败，不能转换成 skip |

缺包时，[TorchFT 扩展目录的 `conftest.py`](../../tests/unit_tests/extensions/torchtitan/experiments/torchft/conftest.py)通过 `collect_ignore_glob = ["test_*.py"]` 不收集 `tests/unit_tests/extensions/torchtitan/experiments/torchft/` 下的测试。该范围与 [`patches/torchft/test_accelerator.py`](../../tests/unit_tests/patches/torchft/test_accelerator.py) 中单个正向用例的 skip 不同；仅从 pytest 的 skipped 数量无法看出这部分未收集的测试。

只有模块和 distribution metadata 均不存在时才按未安装处理。metadata 仍在但模块缺失属于损坏安装，正向测试继续失败，扩展目录也继续收集并暴露导入错误。

[`pyproject.toml`](../../pyproject.toml) 的 extra 固定 `torchft==0.2.0`。在已准备基础运行时的隔离环境中安装后，重新执行完整入口：

```bash
python -m pip install -e '.[torchft]'
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -v --tb=short tests/unit_tests
```

[`.ci/unit_test.sh`](../../.ci/unit_test.sh) 为 TorchFT 配置了预编译 wheel 索引，并固定已安装的 Torch 版本。使用该来源前须确认 HTTPS 证书验证通过，且 wheel 的 Python、平台和架构标签匹配当前环境；不能以关闭 TLS 验证代替检查，也不能仅凭 CI 配置认定当前平台已有可用 wheel。

没有匹配 wheel 时，`pip` 可能从源码构建。[TorchFT v0.2.0 源码安装说明](https://github.com/meta-pytorch/torchft/blob/v0.2.0/README.md#installing-from-source)要求 Rust 工具链、Protobuf 编译器 `protoc` 与开发库（Ubuntu 包名 `protobuf-compiler`、`libprotobuf-dev`），构建后端使用 `maturin`。Python 环境本身不会提供这些系统工具；`Could not find protoc` 表示源码构建前置条件未满足。应使用已具备前置条件的独立构建环境或已验证的匹配 wheel；无法满足时，将含可选依赖的完整验证记为 blocked，并记录安装日志，基础 CPU 结果单独报告。
