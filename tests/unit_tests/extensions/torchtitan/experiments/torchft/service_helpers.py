# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import runpy
import threading
from contextlib import contextmanager
from pathlib import Path

# Start the standalone service without importing the optional training runtime.
FEATURE = Path(__file__).resolve().parents[6] / "torchtitan_npu/extensions/torchtitan/experiments/torchft/last_words"
SERVICE = runpy.run_path(str(FEATURE / "service.py"))


@contextmanager
def service_session(root, run_id, *, num_replicas=2, world_size=2):
    directory = root / f"session-{run_id}"
    with SERVICE["ServiceServer"](("127.0.0.1", 0)) as server:
        SERVICE["prepare_session"](
            server,
            directory,
            address=f"http://127.0.0.1:{server.server_port}",
            run_id=run_id,
            num_replicas=num_replicas,
            world_size=world_size,
            heartbeat_seconds=30,
            startup_seconds=60,
            timeout_seconds=100,
        )
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield directory
        finally:
            server.shutdown()
            thread.join()
