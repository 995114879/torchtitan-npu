# TorchFT 遗言保存

遗言机制在故障或提交拒绝后，让剩余健康 replica 停在完整更新边界，选出一个 replica 保存完整 checkpoint，减少训练进度损失。本功能只保存并结束本次训练，不启动、重新拉起或重新接入训练进程；writer 保存失败或超时后直接报失败，不切换备用 writer。

## 适用条件

- 使用仓库支持的 DeepSeek-V4 Flash TorchFT recipe，同步 FSDP + EP；不支持 TP、CP、PP、LocalSGD、DiLoCo。
- 至少两个完整 replica；一个 replica 的分片可以跨多台机器，rank 使用该 replica 的全局 rank。
- 开启 checkpoint 和每 replica 的 dataloader checkpoint，设置 `--fault-tolerance.no-enable-elastic`，且 `min_replica_size == group_size`。关闭遗言后，原生 TorchFT elastic 功能不受本功能控制。
- 所有 replica 使用同一 checkpoint 根目录，要求 POSIX 文件系统正确支持原子替换和 fsync。部署方应预留完整 checkpoint 及全部游标副本所需空间，不依赖自动清理。
- 依赖见[安装说明](../user-guides/installation.md)和 [requirements.txt](../../requirements.txt)。service 仅使用 Python 标准库，训练仍需要匹配的 TorchTitan、TorchFT、PyTorch 和 NPU 软件栈。

## 启动方式

部署方为整个作业启动一个 Lighthouse 和一个独立遗言 service，再在每个节点启动普通 torchrun。每个训练 rank 启动一个心跳控制线程，上报本 rank 的状态并接收停止请求。遗言 service 检测 rank 失联和全局保存超时。每个 replica 的 rank 0 另用后台线程持久化游标并发布标记。

旧 `last_words_job.py`、`last_words_external.py`、`last_words_nodes.py` 入口及 job/cluster spec 已移除。两个仓库 launcher 使用普通 torchrun，不再自动启动 service 或 Lighthouse。

同一作业的遗言 service 和全部 worker 应使用同一源码版本。正常收尾使用 RPC v4，部署时请一起升级并重新启动作业。已有 checkpoint 可按原格式加载。

以下命令从源码根目录执行，所有变量由部署方指定。`RUN_ID` 每次启动必须全新，所有 replica 共用；`SESSION_DIR` 必须是尚不存在的私有目录；`SHARDS_PER_REPLICA` 是一个 replica 跨全部节点的 rank 总数。

单机 service（单独终端）：

```bash
python torchtitan_npu/extensions/torchtitan/experiments/torchft/last_words/service.py \
  --bind="127.0.0.1:${SERVICE_PORT}" \
  --session-dir="${SESSION_DIR}" --run-id="${RUN_ID}" \
  --num-replicas="${NUM_REPLICAS}" --world-size="${SHARDS_PER_REPLICA}" \
  --heartbeat-seconds=60 --startup-seconds=900 --timeout-seconds=300
```

跨主机必须使用双向 TLS，证书 SAN 包含 `SERVICE_HOST`。使用以下 service 命令：

```bash
python torchtitan_npu/extensions/torchtitan/experiments/torchft/last_words/service.py \
  --bind="0.0.0.0:${SERVICE_PORT}" \
  --advertise-address="https://${SERVICE_HOST}:${SERVICE_PORT}" \
  --tls-cert="${SERVER_CERT}" --tls-key="${SERVER_KEY}" --tls-ca="${CLIENT_CA}" \
  --session-dir="${SESSION_DIR}" --run-id="${RUN_ID}" \
  --num-replicas="${NUM_REPLICAS}" --world-size="${SHARDS_PER_REPLICA}" \
  --heartbeat-seconds=60 --startup-seconds=900 --timeout-seconds=300
```

service 生成 `replica-0.json` 等私有凭据文件，只交给对应 replica 的节点，不应改写或放入公开日志。共享目录或受控文件分发均可。各节点设置 `LAST_WORDS_SESSION_FILE`；TLS worker 另设 `LAST_WORDS_SERVICE_CA_FILE`、`LAST_WORDS_SERVICE_CERT_FILE`、`LAST_WORDS_SERVICE_KEY_FILE`，不需要 SSH 隧道。

Lighthouse 由用户或外部平台独立启动，全部 replica 使用同一可达地址：

```bash
torchft_lighthouse --bind="${LIGHTHOUSE_BIND}:${LIGHTHOUSE_PORT}" --min_replicas="${NUM_REPLICAS}"
```

每个节点执行普通训练命令。`REPLICA_ID` 是 replica 编号；`NODE_RANK` 是 replica 内节点编号；`MASTER_ADDR` 指向该 replica 首节点，同机不同 replica 的端口不能冲突。`NNODES * NPUS_PER_NODE` 必须等于 `SHARDS_PER_REPLICA`。

```bash
export TORCHFT_LIGHTHOUSE="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"
export LAST_WORDS_SESSION_FILE="${SESSION_DIR}/replica-${REPLICA_ID}.json"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
torchrun --nnodes="${NNODES}" --node-rank="${NODE_RANK}" \
  --nproc-per-node="${NPUS_PER_NODE}" --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" --max-restarts=0 \
  -m torchtitan_npu.train \
  --module=torchtitan_npu.models.deepseek_v4 --config=deepseek_v4_flash_8p_torchft \
  --dump-folder="${DUMP_FOLDER}" --hf-assets-path="${HF_ASSETS}" \
  --dataloader.dataset-path="${DATASET}" \
  --parallelism.data-parallel-shard-degree="${SHARDS_PER_REPLICA}" \
  --parallelism.expert-parallel-degree="${EP_DEGREE}" \
  --training.global-batch-size="${GLOBAL_BATCH_SIZE}" \
  --fault-tolerance.group-size="${NUM_REPLICAS}" \
  --fault-tolerance.min-replica-size="${NUM_REPLICAS}" \
  --fault-tolerance.replica-id="${REPLICA_ID}" \
  --fault-tolerance.no-enable-elastic --fault-tolerance.process-group-timeout-ms=60000 \
  --fault-tolerance.last-words.enable --fault-tolerance.last-words.run-id="${RUN_ID}" \
  --fault-tolerance.last-words.heartbeat-timeout-seconds=60 \
  --fault-tolerance.last-words.timeout-seconds=300
```

两机各 16 卡、每机一个 replica：`NUM_REPLICAS=2`、`NNODES=1`、`NODE_RANK=0`、`SHARDS_PER_REPLICA=16`。四机各 16 卡、每 replica 两机：`NUM_REPLICAS=2`、`NNODES=2`、各 replica 内 `NODE_RANK=0/1`、`SHARDS_PER_REPLICA=32`。EP、batch、网络接口等仍须匹配实际拓扑。

## 保存流程

1. worker 上报训练阶段；异常、提交拒绝或心跳超时触发停止请求。部分更新、初始 healing 未完成等不安全 replica 不能保存。
2. 健康 replica 等待完整更新边界。优化器、pre/post-hook、调度器和设备同步全部完成后才记录完成版本；当前 TorchFT 遗言不支持 EMA 权重。replica 内各 rank 在 CPU 控制组上对停训请求投票。冻结后，每个 rank 根据 service 返回的分片状态核对所有 rank 的完成版本。
3. service 收集冻结状态和游标准备结果，在完整就绪候选中按完整更新计数、训练步、replica ID 选择唯一 writer，授权固定为 round 1。其他健康 replica 等待；故障 replica 不参与保存 collective，不拼接不同 replica 的模型分片。
4. writer 的 rank 0 先持久化本轮授权绑定，复制各 replica 的安全游标，并把准备结果广播给其他 writer rank。准备成功后，所有 writer rank 共同保存模型和训练状态；rank 0 同步文件与目录，发布 manifest 和 publication。各 writer rank 看到 publication 后向 service 确认；rank 0 从 service 快照确认全部 writer rank 已上报，再执行生产预检，检查模型、优化器、调度器、训练状态、RNG、游标 schema 和 DCP 分片范围。service 收到全 rank 确认及 rank 0 校验报告后才宣布成功，存活的健康 rank 等待该最终结果后结束本次训练。确认过程不再写入或轮询 `ack-<rank>.json`。
5. 授权前缺少分片的 replica 不能入选，其他就绪 replica 仍可获选。授权后若 writer 在 service 裁决前失联、未集齐发布确认、预检失败或超过保存期限，本轮失败。service 的裁决一旦发布就不能更改，迟到确认不会触发重新授权。收到成功裁决后，writer 的 rank 0 先持久化回执，所有 writer rank 再通过最终 barrier 并正常退出。写回执或 barrier 失败仍可能导致训练进程非零退出。远端进程收尾由用户或外部调度平台负责。

worker/service 的心跳和保存期限必须一致。`startup_seconds` 为尚未注册的 rank 留出初始化宽限；已有候选就绪时，service 最多等到故障发生后的一个心跳期限，再从就绪候选中选 writer。若仍有副本在准备且没有候选就绪，service 可继续等待，直至有候选就绪、所有副本失去资格或总保存期限到达。当仍有副本在准备时，若总保存期限不大于心跳期限，即使其他副本已就绪，也可能在选出 writer 前超时。设置总期限时应为 writer 选举后的写盘与预检留出时间。

每个 rank 的心跳线程报告 CPU 侧状态。service 检查心跳是否逾期，并从首次收到停训请求或检测到故障时开始计算保存期限。若 NPU 卡死但 CPU 仍发送心跳，或 service 本身失效，由外部调度平台处理作业收尾。

## 结果与检查点

service 私有目录保存 `grant.json` 和 `result.json`。结果 `success=true` 且 `replica_id` 非空表示紧急保存确认成功，`success=false` 表示失败。

达到训练步数或同步读完有限数据集时，各 rank 等待后台 checkpoint 写入结束。若已结束的异步写入仅因游标存储错误而失败，该 rank 上报停训，保留使用共同游标保存的机会；写入成功的 rank 报告准备退出。等同一 replica 的所有 rank 都准备好，service 通过心跳要求它们作最终确认。每个 rank 此时再次检查本地停训请求，有请求就上报停训，否则投正常退出票。投票后，各 rank 继续等待同一个 service 决定。service 若先收到停训请求，该副本转入故障保存流程；若先收齐全部正常退出票，则确认该副本结束。所有副本正常结束后，作业进入 `COMPLETED`，结果为 `success=true, round=0, replica_id=null`。

service 在返回裁决前同步结果文件和所在目录。若目录同步失败，即使 `result.json` 已经可见，也需等 service 重试成功再使用该文件。service 保留终态并继续运行，由外部停止；每次新作业启动新 service 和会话目录，不复用旧授权或 run ID。

```text
checkpoint/
  ft-replicat-<id>/<run_id>/step-<step>/  # 已发布游标及 RNG
  last_words/control/<run_id>/         # 可选的 worker 故障诊断，读写失败时记日志
  last_words/<run_id>/replica-<writer_id>/
    model/
    cursors/replica-<id>/
    coordinator-binding.json
    manifest.json
    publication.json
    service-result.json
```

游标沿用 TorchFT 每步异步上传，只在 future 成功并完成持久化后发布。已结束且仅包含存储错误的游标上传可以使用旧成功游标；未结束的 staging、通信错误、模型保存失败不能这样回退。只选当前启动分支及明确绑定的恢复 anchor；游标不能晚于模型，但允许落后，因此可能重读数据，不承诺逐样本恰好一次或恢复轨迹逐位一致。

不完整目录保留诊断价值，但不能凭 manifest 或 publication 可见认定 service 已确认成功。旧 checkpoint 中的 ack 文件不影响加载，也不作为加载条件。service 成功表示保存和预检已经完成，不保证所有进程都收到最终响应后外部平台才结束进程。分片范围预检识别缺失、截断和路径越界，不是内容 checksum。目录 fsync 失败后文件仍可能可见，不能据此声称断电持久性。

需要再训练时，外部先确认旧作业收尾，再独立启动新作业。保留现有 checkpoint 加载能力：可设置 `--fault-tolerance.last-words.resume-path` 指定已核验候选，未指定时选择兼容已发布候选或周期 checkpoint。加载不创建新进程；正步数加载要求完整状态及恢复链，旧或 schema 不匹配工件拒绝加载。writer 的 rank 0 收到与本次授权匹配的成功裁决后，将其原子写入候选目录的 `service-result.json`，同步文件及目录后，与同一 writer 的其他 rank 在已有控制进程组汇合，再结束保存。心跳先收到成功裁决时，训练线程复用该结果，仍执行回执和汇合步骤。自动选择要求裁决与授权的 run ID、service 实例、round 和 writer 身份一致，并检查 publication 和 DCP。外部平台在正常收尾前应等待该文件可见。若 writer 在 service 成功后未能写回裁决，可以将该次 service 私有目录中已核实的成功 `result.json` 复制为候选的 `service-result.json`。

自动续训合并同一 replica 各 rank 枚举到的周期目录和遗言 manifest。发现与本作业配置匹配的正步数遗言或周期 checkpoint 后，会优先保留已提交的训练进度。若回执、模型分片或游标恢复链校验失败，会继续查找有效的正步数恢复点。例如，周期 `step-8` 的游标均不可用时，可以回退到有效的 `step-4`；若只剩 `step-0` 或没有可用恢复点，启动会报出失败候选的路径和原因。只有初态的新作业可以正常启动，也可以显式指定 `--checkpoint.load-step 0` 加载初态。

恢复周期 checkpoint 时，同一副本先合并各 rank 列出的游标候选，再按恢复顺序逐个核对。若最新游标仅部分 rank 可见，但有各 rank 都能验证的较早游标，则保留原模型步数并加载该共同游标。核实原 service 成功后，缺回执的 v1 候选可通过 `--fault-tolerance.last-words.resume-path` 显式加载；v2 候选必须先补齐匹配的成功回执。外部候选的游标导入后作为不可变 anchor，供后续周期 checkpoint 引用。

训练最后一步默认只导出模型权重。需要从最后一步续训时，设置 `--checkpoint.no-last-save-model-only`，让最终保存包含完整训练状态和恢复绑定。

自动和显式加载都只接受遗言 manifest 版本 1 和 2。每个逻辑 replica 的游标必须对应自己的 `cursors/replica-<id>/`，各 shard 的 RNG 设备类型必须与本次训练一致。显式指定的候选校验失败时会报错，自动选择则继续检查其他候选。

新作业使用版本化的恢复签名，按 FSDP/EP 实际拓扑参数绑定并行布局；模型、数据、优化器和调度器配置仍参与匹配。在支持的 DeepSeek-V4 TorchFT 配置（CP=1）下，加载器也识别旧版压缩器含 `carry_exchange` 时生成的配置签名。它按本次作业的完整配置重建该签名，仍会校验并行拓扑、数据配置和模型状态。恢复成功后，新游标和后续 checkpoint 沿用已加载的签名，使同一条恢复链可以继续保存和加载。

新保存的游标与 checkpoint 还记录分词器资产的 SHA-256。相同资产可以迁移目录，改变生效的词表、分词规则或配置会使对应候选失去恢复资格。旧 checkpoint 缺少该摘要时仍按原规则校验，部署方需确认分词器与保存时一致。从旧 checkpoint 续训后，新产物会记录本次使用的资产摘要。

空间清理由部署方负责。不要删除仍被 `last_words_binding.json` 引用的游标、anchor 或候选。本功能不再提供自动清理、迁移或容量准入脚本。

## 验证入口

CPU 单测覆盖正常训练收尾、唯一 writer 授权、遗言 checkpoint 成功保存与续训加载。跨主机 HCCL 和 NPU 训练需在相应硬件上验证。

```bash
python -m pytest tests/unit_tests/extensions/torchtitan/experiments/torchft
```

四卡用例 `dsv4_torchft_last_words_emergency_save_2x2` 在安全边界注入进程退出，逐 rank 核验唯一 writer 完成最终 barrier，并等待其 torchrun 以退出码 0 结束。测试 runner 随后启动新作业，检查完整训练状态、各副本游标及 RNG 的恢复结果，并验证续训提交。`dsv4_torchft_last_words_numerical_2x2` 从统一初态比较基线、遗言关闭和遗言开启三组结果。当前版本的实测范围记录在 PR 中。

未配置独立基线时，suite 跳过精度用例，故障保存用例仍可运行。精度用例通过 `LAST_WORDS_BASELINE_REPO` 指定本次修复前的基线源码（`324ecaf27c70ed8c3cbe7944ed40ae584c5e74d3`）。支持该提交的干净 Git 工作树或完整源码归档；启动训练前会核验 `torchtitan_npu` 下全部 Python 文件的路径和内容。
