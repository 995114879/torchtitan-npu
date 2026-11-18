# TorchAO-NPU 测试使用说明

[统一入口](../scripts/run_tests.sh)负责安装实验包及指定依赖、校验环境、执行所选测试层，并将结果输出到终端和本地报告。以下命令从 torchtitan-npu 仓库根目录执行。

## 测试分层与 CI

| 模式 | 执行目录（相对于本目录） | 验证范围 | 执行方式 |
| --- | --- | --- | --- |
| `cpu` | `unit_tests/` | 配置、参数替换、wrapper 元数据、CPU/meta 张量、序列化及模拟算子的调用契约 | UT CI 和手工执行；不要求物理 NPU |
| `device` | `device_tests/` | 真实 NPU 算子、量化、wrapper、编译及设备迁移 | 手工执行，需要可用 NPU |
| `integration` | `integration_tests/` | MoE QAT 训练及 checkpoint 保存/恢复 | 手工执行，需要可用 NPU |
| `npu` | 依次执行 `device_tests/`、`integration_tests/` | 上述两层 NPU 验证 | 手工执行，需要可用 NPU |

当前只有 `cpu` 接入主仓 [UT CI](../../../.ci/unit_test.sh)。设备和集成测试没有接入主仓 Smoke/Integration CI。`_support/` 保存共享辅助代码，不作为独立测试层执行。CPU 模拟算子测试不能证明真实 NPU 算子的数值正确性。

## 环境准备

运行测试需要 Linux/Bash 环境和 Python ≥ 3.12，默认解释器为 `python3.12`。解释器中需预装 `pip`、`packaging`、`pytest`、`setuptools>=64`，以及与测试代码兼容的 PyTorch、torch-npu、CANN Python 包、torchtitan 和主仓依赖。安装方法见[软件安装指南](../../../docs/user-guides/installation.md)，版本声明见[主仓依赖](../../../requirements.txt)和[实验包配置](../pyproject.toml)；CI 使用其预置的软件栈。

`cpu` 不需要物理 NPU，但测试仍会导入 `torch_npu` 和 CANN Python 模块，因此仅安装 CPU PyTorch 不足以运行。NPU 模式还需要匹配的驱动、固件、CANN 算子包和可用设备；编译用例需要相应编译工具链，通信用例需要可用的 HCCL。具体设备、dtype 和算子支持范围由所选用例及运行时决定。

脚本不会安装或替换整个 PyTorch/CANN 软件栈。NPU 模式在 pytest 收集前检查设备可用性，并实际执行一次 NPU 张量分配、求和及同步；失败时直接退出。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `PYTHON_BIN` | `python3.12` | 所有安装、校验和 pytest 操作使用的 Python 可执行文件；可设为虚拟环境解释器的绝对路径，不包含额外命令参数 |
| `CANN_ENV_PATH` | 未设置，不额外 source 脚本 | 设置后执行 `source "$CANN_ENV_PATH"`；未设置时沿用调用者已准备的环境 |
| `TORCHAO_TEST_REPORT_DIR` | `<仓库根目录>/test_reports/torchao-npu` | 日志、XML 和状态文件的存放目录；相对路径按仓库根目录解析 |

脚本会切换到仓库根目录，建议环境变量中的文件路径使用绝对路径。空值按上述默认值处理。脚本设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，需要时显式导入 `torch_npu`。

## 依赖安装与副作用

每次正常运行均先执行以下准备，再启动 pytest：

1. 从实验包 `pyproject.toml` 读取 torchao 约束，执行 `pip install --no-deps`；当前为 `torchao==0.17.0`。
2. 仅在 `cpu` 模式执行 `pip install --no-deps spmd_types==0.2.3`，补齐当前 CPU CI 所需的 API 版本。
3. 使用 `pip install --no-deps --no-build-isolation -e` 安装当前实验包，校验依赖版本及 `torchao_npu` 的源码导入位置。

上述操作会修改 `PYTHON_BIN` 对应的 Python 环境，需要安装权限，并需要配置可用的 package index、离线源或缓存。`--no-deps` 避免安装过程自动升级整套运行时，也意味着其他依赖必须预先准备；`--no-build-isolation` 要求构建依赖已存在。仓内有源码不等于实验包已经安装。

**`--collect-only` 不是无副作用的预览。** 例如 `cpu --collect-only` 仍先安装依赖和实验包；`device --collect-only` 还会执行 NPU 检查。只有将 `--help` 或 `-h` 放在第一个参数位置，才会在安装、设备检查及报告目录创建之前直接返回帮助。

## 分层执行与 pytest 参数

```bash
# 查看脚本帮助，不安装依赖或生成报告
bash experiments/torchao-npu/scripts/run_tests.sh --help

# 不传参数时默认执行 cpu
bash experiments/torchao-npu/scripts/run_tests.sh cpu

# CANN 路径按实际环境设置；若已 source，可省略该变量
CANN_ENV_PATH=/usr/local/Ascend/cann/set_env.sh \
    bash experiments/torchao-npu/scripts/run_tests.sh device

bash experiments/torchao-npu/scripts/run_tests.sh integration
bash experiments/torchao-npu/scripts/run_tests.sh npu

# 模式后的参数原样传给 pytest
bash experiments/torchao-npu/scripts/run_tests.sh cpu -k param_swap -x
bash experiments/torchao-npu/scripts/run_tests.sh device -k test_to_mx_then_mm_forward
bash experiments/torchao-npu/scripts/run_tests.sh cpu --collect-only -q

# 指定解释器，并用独立报告目录保留本次运行结果
PYTHON_BIN=/opt/venvs/torchao/bin/python \
    TORCHAO_TEST_REPORT_DIR=/tmp/torchao-reports/run-001 \
    bash experiments/torchao-npu/scripts/run_tests.sh cpu
```

从实验包目录执行时，入口改为 `bash scripts/run_tests.sh <模式>`。脚本显式设置 pytest 配置、仓库根目录、导入路径和所选测试目录，避免主仓与实验包的 `tests` 模块重名。传 pytest 参数时必须先写模式；`cpu --help` 会先完成环境准备，再显示 pytest 帮助。

`npu` 为两层分别启动 pytest 进程，参数会传给两层；`-k` 若在其中一层筛掉全部测试，该层会以「未收集到测试」的非零状态结束。第一层 pytest 失败后仍会尝试第二层；环境准备失败则不会开始任何测试。直接执行 `pytest experiments/torchao-npu/tests` 会同时收集 NPU 层，不适合作为 CPU 入口。

## 报告位置与覆盖规则

默认报告目录为 `<仓库根目录>/test_reports/torchao-npu/`。脚本通过 `tee` 将输出同时写入终端和日志，CI 可直接查看作业日志。

| 文件 | 内容与生成时机 |
| --- | --- |
| `run-<模式>.log` | 依赖安装、环境校验及所选 pytest 测试层的完整输出 |
| `status-<模式>.txt` | 各测试层的退出码，以及脚本最终的 `exit=<退出码>` |
| `<测试目录>.log` | 对应 pytest 执行的输出，例如 `device_tests.log` |
| `<测试目录>.xml` | pytest 生成的 JUnit XML，例如 `unit_tests.xml`；提前中断时可能不存在 |

每次运行会覆盖当前模式的 `run-*.log` 和 `status-*.txt`，并在环境准备前删除所选测试层的旧 `.log` 和 `.xml`。未选中的模式和测试层文件保留。

`npu` 与单独的 `device`、`integration` 共用测试层报告文件名，后续运行会覆盖对应层的报告。并发执行或需要保留历史时，应为每次运行设置不同的 `TORCHAO_TEST_REPORT_DIR`。pytest 参数可以覆盖 JUnit XML 的输出位置；其他日志和状态文件名由脚本固定。

## 区分环境准备失败与测试失败

以 `cpu` 为例，正常完成时状态文件为：

```text
unit_tests=0
exit=0
```

状态文件只有非零 `exit=...`、没有测试层记录时，失败发生在 pytest 完成之前。应查看 `run-<模式>.log` 中的依赖安装、环境校验或 NPU 设备检查错误。

存在 `<测试目录>=<非零值>` 时，该测试层已经执行完成。应结合对应 `.log` 和 XML 判断是测试失败、收集错误、参数错误、未收集到测试，还是该层日志写入失败。该值是执行管道的退出码，不是失败用例数量。

脚本退出前会关闭完整日志的写端并等待 `tee` 刷盘，再写入最终 `exit` 状态。因此测试层均为 `0` 而最终 `exit` 非零时，应检查 `run-<模式>.log` 的写入错误。

报告目录创建或输出重定向失败时，状态文件可能不存在；进程被强制终止时，状态文件也可能不完整。日志中的 `SKIPPED` 或 `--collect-only` 成功不代表真实 NPU 计算已通过，应结合执行模式和 pytest 汇总判断。
