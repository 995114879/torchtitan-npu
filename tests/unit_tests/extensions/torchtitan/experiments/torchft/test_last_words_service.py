# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import json
import runpy
from contextlib import ExitStack

from tests.unit_tests.extensions.torchtitan.experiments.torchft.service_helpers import FEATURE, service_session

WorkerSessionClient = runpy.run_path(str(FEATURE / "client.py"))["WorkerSessionClient"]


def test_service_elects_most_advanced_replica_and_confirms_save(tmp_path, monkeypatch):
    with service_session(tmp_path, "save") as directory, ExitStack() as clients:
        workers = []
        for replica, step in enumerate((2, 3)):
            monkeypatch.setenv("LAST_WORDS_SESSION_FILE", str(directory / f"replica-{replica}.json"))
            state = {"step": step, "last_words_num_updates": step}
            status = dict(phase="frozen", state=state, world_size=2, heartbeat_timeout_seconds=30)
            ranks = []
            for rank in range(2):
                worker = WorkerSessionClient("save", replica, rank)
                clients.callback(worker.client.close)
                worker.set_ready(state, "model")
                worker.report(status, requested=True)
                ranks.append(worker)
            workers.append(ranks)

        expected_state = {"step": 3, "last_words_num_updates": 3}
        grant = {"round": 1, "replica_id": 1, "state": expected_state, "signature": "model"}
        assert all(worker.poll()["grant"] == grant for ranks in workers for worker in ranks)

        published = dict(
            phase="frozen", state=expected_state, world_size=2, heartbeat_timeout_seconds=30, published_round=1
        )
        writer = workers[1]
        for worker in writer:
            assert worker.report(published, requested=True)["result"] is None
        result = writer[0].report(dict(published, validated_round=1), requested=True)
        assert result["phase"] == "SUCCEEDED"
        expected = {"round": 1, "replica_id": 1, "success": True, "error": ""}
        assert result["result"] == expected
        assert workers[0][1].poll()["result"] == expected
        audit = json.loads((directory / "result.json").read_text())
        assert audit == dict(
            expected, version=result["version"], run_id="save", coordinator_instance=result["instance"]
        )
