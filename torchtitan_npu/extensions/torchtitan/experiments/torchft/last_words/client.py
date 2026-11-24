# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""HTTP transport and session clients for the torchtitan-npu save service.

This module must remain standard-library-only: control imports cannot initialize
torch/NPU. It uses direct HTTP(S) RPC and does not own training processes.
"""

import http.client
import ipaddress
import json
import os
import ssl
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

RPC_PROTOCOL_VERSION = 4


def validate_service_address(address, *, allow_port_zero=False):
    """Parse a root service URL without opening sockets or loading credentials."""
    parsed = urlsplit(address)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or parsed.port is None
        or (parsed.port == 0 and not allow_port_zero)
        or any(character.isspace() for character in address)
    ):
        raise ValueError("Provide an HTTP(S) service root URL with an explicit port and no credentials")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if parsed.scheme == "http" and not loopback:
        raise ValueError("Non-loopback coordination requires HTTPS with mutual TLS")
    return parsed


class ServiceClient:
    """Bounded persistent HTTP transport; imports no tensor or native extension."""

    def __init__(self, address):
        parsed = validate_service_address(address)
        self.connection: http.client.HTTPConnection
        if parsed.scheme == "https":
            files = [os.environ.get(f"LAST_WORDS_SERVICE_{name}_FILE", "") for name in ("CA", "CERT", "KEY")]
            if not all(files):
                raise ValueError("HTTPS coordination requires LAST_WORDS_SERVICE_CA_FILE, CERT_FILE and KEY_FILE")
            context = ssl.create_default_context(cafile=files[0])
            context.load_cert_chain(certfile=files[1], keyfile=files[2])
            self.connection = http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=2, context=context)
        else:
            self.connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
        self.lock = threading.Lock()

    def call(self, document, deadline) -> str:
        with self.lock:
            delay = 0.1
            while (remaining := deadline - time.monotonic()) > 0:
                self.connection.timeout = min(2.0, remaining)
                if self.connection.sock is not None:
                    self.connection.sock.settimeout(self.connection.timeout)
                try:
                    self.connection.request(
                        "POST", "/coordinate", body=document.encode(), headers={"Content-Type": "application/json"}
                    )
                    response = self.connection.getresponse()
                    raw = response.read(8 * 1024 * 1024 + 1)
                    if len(raw) > 8 * 1024 * 1024:
                        raise RuntimeError("Coordinator response exceeds limit")
                    result = raw.decode()
                    if response.status == 503:
                        raise ConnectionError("Coordinator temporarily unavailable")
                    if response.status != 200:
                        raise RuntimeError(json.loads(result).get("error", "Coordinator request failed"))
                    return result
                except (OSError, http.client.HTTPException):
                    # The server may already have committed this report. Retry
                    # exactly the same sequence and payload, including on 503.
                    self.connection.close()
                    time.sleep(min(delay, max(0, deadline - time.monotonic())))
                    delay = min(1.0, 2 * delay)
                except BaseException:
                    self.connection.close()
                    raise
        raise TimeoutError("Coordinator transport deadline expired")

    def close(self):
        with self.lock:
            self.connection.close()


class WorkerSessionClient:
    def __init__(self, run_id, replica, rank):
        self.session = json.loads(Path(os.environ["LAST_WORDS_SESSION_FILE"]).read_text())
        session = self.session
        if (
            session.get("version") != RPC_PROTOCOL_VERSION
            or session.get("run_id") != run_id
            or session.get("replica") != replica
        ):
            raise ValueError("Session file does not match this run/replica/protocol")
        if not 0 <= rank < session["world_size"]:
            raise ValueError("Rank outside registered replica")
        self.client = ServiceClient(session["address"])
        self.identity = dict(
            version=RPC_PROTOCOL_VERSION,
            run_id=run_id,
            instance=session["instance"],
            token=session["token"],
            replica=replica,
            rank=rank,
        )
        self.lock = threading.Lock()
        self.seq = 0
        self.snapshot = None
        self.ready = None

    def report(self, status, requested, *, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + self.session["heartbeat_seconds"]
        with self.lock:
            self.seq += 1
            status = dict(status, requested=requested, ready=self.ready)
            document = dict(self.identity, seq=self.seq, op="shard_report", data=status)
            snapshot = json.loads(self.client.call(json.dumps(document), deadline))
            if any(snapshot.get(key) != self.identity[key] for key in ("version", "run_id", "instance")):
                raise RuntimeError("Coordinator response belongs to another save session")
            self.snapshot = snapshot
            return snapshot

    def get_cached_snapshot(self):
        # Replies replace the entire snapshot. A reader may still need the
        # preceding freeze snapshot while the next heartbeat is being retried.
        return self.snapshot

    def poll(self):
        deadline = time.monotonic() + self.session["heartbeat_seconds"]
        with self.lock:
            self.seq += 1
            document = json.dumps(dict(self.identity, seq=self.seq, op="poll", data=None))
            snapshot = json.loads(self.client.call(document, deadline))
            if any(snapshot.get(key) != self.identity[key] for key in ("version", "run_id", "instance")):
                raise RuntimeError("Coordinator poll belongs to another save session")
            self.snapshot = snapshot
            return snapshot

    def set_ready(self, state, signature):
        with self.lock:
            ready = dict(state=state, signature=signature)
            if self.ready is not None and self.ready != ready:
                raise RuntimeError("Cannot change frozen ready state")
            self.ready = ready
