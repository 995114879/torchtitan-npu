# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Emergency-save authority, independent of process launch, TorchFT and torch.

All shard aggregation and writer decisions are serialized by one lock. Server
incarnations are deliberately ephemeral: losing this process fails the attempt.
Writer ranks acknowledge publication before rank zero validates the checkpoint.
This service never launches, terminates or restarts training. A failed writer is never replaced.
"""

import argparse
import copy
import hmac
import ipaddress
import json
import logging
import math
import os
import secrets
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

if __package__:
    from .client import RPC_PROTOCOL_VERSION, validate_service_address
else:
    import runpy

    _client = runpy.run_path(str(Path(__file__).with_name("client.py")))
    validate_service_address = _client["validate_service_address"]
    RPC_PROTOCOL_VERSION = _client["RPC_PROTOCOL_VERSION"]

MAX_REQUEST_BYTES = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_uint64(value):
    return type(value) is int and 0 <= value < 2**64


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Session:
    def __init__(self, config, now):
        self.config = config
        require(is_uint64(config["world_size"]) and 1 <= config["world_size"] <= 64, "invalid shard count")
        tokens = config["replica_tokens"]
        require(isinstance(tokens, list) and 1 <= len(tokens) <= 64, "invalid replica count")
        require(
            all(isinstance(token, str) and len(token) >= 32 for token in tokens),
            "invalid replica token",
        )
        require(len(set(tokens)) == len(tokens), "duplicate replica token")
        for key in ("heartbeat_seconds", "startup_seconds", "timeout_seconds"):
            value = config[key]
            require(type(value) in (float, int) and math.isfinite(value) and value > 0, "invalid time budget")
        self.started = now
        self.fault_at = None
        self.reason = ""
        self.phase = "RUNNING"
        self.reports: dict[tuple[int, int], dict[str, Any]] = {}
        self.excluded = set()
        self.finished_replicas = set()
        self.grant = None
        self.round = 0
        self.result = None
        self.revision = 1

    def request_terminal_save(self, now, reason):
        if self.fault_at is None and self.result is None:
            self.fault_at, self.reason, self.phase = now, reason, "PREPARING"
            self.revision += 1

    def finish(self, error=""):
        if self.result is None:
            self.phase = "FAILED" if error else "SUCCEEDED"
            self.result = dict(
                success=not error,
                error=error,
                round=self.round,
                replica_id=self.grant["replica_id"] if self.grant else None,
            )
            self.revision += 1

    def shards(self, replica):
        return [self.reports.get((replica, rank), {}).get("data") for rank in range(self.config["world_size"])]

    def can_confirm_finish(self, replica):
        return all(shard and shard["phase"] in {"finishing", "finish_vote"} for shard in self.shards(replica))

    def ready(self, replica):
        shards = self.shards(replica)
        if any(not shard or shard["phase"] != "frozen" or shard.get("ready") is None for shard in shards):
            return None
        shards = cast("list[dict[str, Any]]", shards)
        ready = shards[0]["ready"]
        require(
            all(shard["ready"] == ready and shard["state"] == ready["state"] for shard in shards),
            "ready shards disagree",
        )
        state = ready["state"]
        require(
            bool(ready["signature"])
            and is_uint64(state.get("step"))
            and is_uint64(state.get("last_words_num_updates")),
            "invalid completed state",
        )
        return ready

    def tick(self, now):
        if self.result is not None:
            return
        if self.grant is not None:
            shards = self.shards(self.grant["replica_id"])
            if (
                all(shard and shard.get("published_round") == self.round for shard in shards)
                and shards[0] is not None
                and shards[0].get("validated_round") == self.round
            ):
                self.finish()
                return
        if len(self.finished_replicas) == len(self.config["replica_tokens"]):
            self.phase = "COMPLETED"
            self.result = dict(success=True, error="", round=0, replica_id=None)
            self.revision += 1
            return
        for replica in range(len(self.config["replica_tokens"])):
            if replica in self.excluded or replica in self.finished_replicas:
                continue
            for rank in range(self.config["world_size"]):
                report = self.reports.get((replica, rank))
                stale = (
                    now - report["at"] > self.config["heartbeat_seconds"]
                    if report
                    else now - self.started > self.config["startup_seconds"]
                )
                if stale:
                    self.excluded.add(replica)
                    self.request_terminal_save(now, "missing replica/shard heartbeat")
                    break
        if self.fault_at is None:
            return
        remaining = self.config["timeout_seconds"] - (now - self.fault_at)
        if remaining <= 0:
            self.finish("global save deadline expired")
            return
        if self.grant is not None:
            writer = self.grant["replica_id"]
            if writer in self.excluded:
                self.finish("selected writer failed or timed out; no writer replacement")
            return
        candidates, pending = [], False
        for replica in range(len(self.config["replica_tokens"])):
            if replica in self.excluded or replica in self.finished_replicas:
                continue
            ready = self.ready(replica)
            if ready is None:
                pending = True
            else:
                candidates.append((replica, ready))
        if pending and (not candidates or now - self.fault_at < self.config["heartbeat_seconds"]):
            return
        if not candidates:
            self.finish("no complete live replica ready")
            return
        signatures = {ready["signature"] for _, ready in candidates}
        if len(signatures) != 1:
            self.finish("replica signatures disagree")
            return
        writer, ready = max(
            candidates,
            key=lambda item: (item[1]["state"]["last_words_num_updates"], item[1]["state"]["step"], -item[0]),
        )
        self.round += 1
        self.grant = dict(round=self.round, replica_id=writer, **ready)
        self.phase = "SAVING"
        self.revision += 1

    def snapshot(self, instance, run_id, *, replicas=None):
        if replicas is None:
            replicas = range(len(self.config["replica_tokens"]))
        return copy.deepcopy(
            dict(
                version=RPC_PROTOCOL_VERSION,
                instance=instance,
                run_id=run_id,
                phase=self.phase,
                revision=self.revision,
                stop=self.fault_at is not None,
                reason=self.reason,
                grant=self.grant,
                result=self.result,
                reports={str(replica): dict(shards=self.shards(replica)) for replica in replicas},
            )
        )

    def report(self, command, now):
        rank, replica, data = command["rank"], command["replica"], command["data"]
        require(
            isinstance(data, dict)
            and data.get("world_size") == self.config["world_size"]
            and data.get("heartbeat_timeout_seconds") == self.config["heartbeat_seconds"],
            "shard topology/budget differs from registration",
        )
        require(
            data.get("phase")
            in {
                "initializing",
                "forward_backward",
                "updating",
                "completed",
                "frozen",
                "ineligible",
                "finishing",
                "finish_vote",
            },
            "invalid shard phase",
        )
        if data["phase"] == "finish_vote" and self.fault_at is None:
            require(self.can_confirm_finish(replica), "finish vote before every shard is finishing")
        for field in ("published_round", "validated_round"):
            if field in data:
                require(
                    type(data[field]) is int
                    and self.grant is not None
                    and replica == self.grant["replica_id"]
                    and data[field] == self.round
                    and data.get("ready") == {key: self.grant[key] for key in ("state", "signature")}
                    and data["phase"] == "frozen",
                    "invalid publication acknowledgement",
                )
                require(field != "validated_round" or rank == 0, "validation requires writer rank zero")
        require(type(data.get("requested")) is bool, "invalid stop request")
        ready = data.get("ready")
        if ready is not None:
            require(
                isinstance(ready, dict)
                and set(ready) == {"state", "signature"}
                and isinstance(ready["state"], dict)
                and isinstance(ready["signature"], str),
                "invalid ready",
            )
        key = (replica, rank)
        old = self.reports.get(key)
        if old:
            require(
                command["seq"] > old["seq"] or (command["seq"] == old["seq"] and data == old["data"]),
                "stale or regressed shard report",
            )
            require(
                (old["data"].get("ready") is None or old["data"]["ready"] == ready)
                and (old["data"]["phase"] != "ineligible" or data["phase"] == "ineligible"),
                "frozen state regressed",
            )
            if command["seq"] == old["seq"]:
                return
        self.reports[key] = dict(seq=command["seq"], data=data, at=now)
        if data["phase"] == "ineligible":
            self.excluded.add(replica)
            self.request_terminal_save(now, "ineligible replica")
        if data["requested"]:
            self.request_terminal_save(now, "worker requested stop")
        if (
            data["phase"] == "finish_vote"
            and self.fault_at is None
            and all(shard and shard["phase"] == "finish_vote" for shard in self.shards(replica))
        ):
            self.finished_replicas.add(replica)
        if data["phase"] == "frozen" and ready is not None:
            self.ready(replica)


class LastWordsServiceCoordinator:
    """One locally provisioned run; remote callers can only report or poll."""

    def __init__(self, clock=time.monotonic):
        self.instance = secrets.token_hex(16)
        self.clock, self.lock, self.sessions = clock, threading.Lock(), {}
        self.audit_directory = None
        self._audited = set()

    def register(self, run_id, config):
        require(
            isinstance(run_id, str)
            and 0 < len(run_id) <= 200
            and Path(run_id).name == run_id
            and run_id not in {".", ".."},
            "invalid run id",
        )
        with self.lock:
            require(not self.sessions, "service already has a run; use a fresh service for a new run")
            self.sessions[run_id] = Session(config, self.clock())

    def _audit(self, run_id, session):
        if self.audit_directory is None:
            return
        for name, value in (("grant", session.grant), ("result", session.result)):
            if value is None or name in self._audited:
                continue
            path = self.audit_directory / (name + ".json")
            temporary = path.with_suffix(".tmp")
            with temporary.open("w") as stream:
                json.dump(
                    dict(version=RPC_PROTOCOL_VERSION, run_id=run_id, coordinator_instance=self.instance, **value),
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            sync_directory(self.audit_directory)
            self._audited.add(name)

    def tick(self):
        with self.lock:
            for run_id, session in self.sessions.items():
                session.tick(self.clock())
                self._audit(run_id, session)

    def command(self, command):
        require(
            isinstance(command, dict)
            and set(command) <= {"version", "op", "run_id", "instance", "token", "seq", "replica", "rank", "data"},
            "invalid command fields",
        )
        require(command.get("version") == RPC_PROTOCOL_VERSION, "unsupported last words protocol")
        require(command.get("op") in {"shard_report", "poll"}, "unsupported save-only operation")
        require(command.get("instance") == self.instance, "Coordinator incarnation changed")
        require(isinstance(command.get("token"), str), "invalid actor")
        require(is_uint64(command.get("seq")) and command["seq"] > 0, "invalid sequence")
        replica, rank = command.get("replica"), command.get("rank")
        require(is_uint64(replica) and is_uint64(rank), "invalid replica/rank")
        with self.lock:
            run_id = command.get("run_id")
            require(isinstance(run_id, str) and run_id in self.sessions, "unknown save session")
            session = self.sessions[run_id]
            tokens = session.config["replica_tokens"]
            require(
                replica < len(tokens)
                and rank < session.config["world_size"]
                and hmac.compare_digest(command["token"], tokens[replica]),
                "invalid actor",
            )
            now = self.clock()
            session.tick(now)
            if command["op"] == "shard_report" and session.result is None and replica not in session.finished_replicas:
                try:
                    session.report(command, now)
                except (ValueError, KeyError, TypeError):
                    session.excluded.add(replica)
                    session.request_terminal_save(now, "invalid replica report")
                    if session.grant is not None and session.grant["replica_id"] == replica:
                        session.finish("invalid writer report")
                    session.tick(now)
                    self._audit(run_id, session)
                    raise
                session.tick(now)
            self._audit(run_id, session)
            replicas = None
            if command["op"] == "shard_report" and session.result is None and replica not in session.finished_replicas:
                # Only frozen workers inspect peers. Every caller receives the same final snapshot.
                replicas = (replica,) if command["data"]["phase"] == "frozen" else ()
            snapshot = session.snapshot(self.instance, run_id, replicas=replicas)
            if replica in session.finished_replicas:
                snapshot["replica_finished"] = True
            elif (
                session.fault_at is None and command["op"] == "shard_report" and command["data"]["phase"] == "finishing"
            ):
                snapshot["confirm_finish"] = session.can_confirm_finish(replica)
            return snapshot


def prepare_session(
    server,
    directory,
    *,
    address,
    run_id,
    num_replicas,
    world_size,
    heartbeat_seconds=60,
    startup_seconds=900,
    timeout_seconds=300,
):
    """Provision private per-replica credentials without starting training."""
    require(type(num_replicas) is int and 1 <= num_replicas <= 64, "invalid replica count")
    validate_service_address(address)
    config = dict(
        world_size=world_size,
        replica_tokens=[secrets.token_hex(32) for _ in range(num_replicas)],
        heartbeat_seconds=heartbeat_seconds,
        startup_seconds=startup_seconds,
        timeout_seconds=timeout_seconds,
    )
    directory = Path(directory)
    # mkdir(parents=True) can create several directory entries. Make the
    # session directory itself reachable after a crash before serving workers.
    new_directories = []
    parent = directory
    while not parent.exists():
        new_directories.append(parent)
        parent = parent.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    server.coordinator.register(run_id, config)
    server.coordinator.audit_directory = directory
    for replica, token in enumerate(config["replica_tokens"]):
        path = directory / f"replica-{replica}.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                dict(
                    version=RPC_PROTOCOL_VERSION,
                    address=address,
                    run_id=run_id,
                    instance=server.coordinator.instance,
                    replica=replica,
                    token=token,
                    num_replicas=num_replicas,
                    world_size=world_size,
                    heartbeat_seconds=heartbeat_seconds,
                    timeout_seconds=timeout_seconds,
                ),
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
    sync_directory(directory)
    for created in new_directories:
        sync_directory(created.parent)


class ServiceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, *, tls_cert=None, tls_key=None, tls_ca=None):
        try:
            loopback = ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            loopback = address[0] == "localhost"
        files = (tls_cert, tls_key, tls_ca)
        require(not any(files) or all(files), "Provide all TLS certificate, key and client CA files")
        require(loopback or all(files), "Non-loopback coordination requires mutual TLS")
        self.tls = None
        if all(files):
            assert tls_cert is not None and tls_key is not None and tls_ca is not None
            self.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.tls.load_cert_chain(tls_cert, tls_key)
            self.tls.load_verify_locations(cafile=tls_ca)
            self.tls.verify_mode = ssl.CERT_REQUIRED
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        self.coordinator = LastWordsServiceCoordinator()
        self.stopped = threading.Event()
        self.ticker = None
        super().__init__(address, Handler)
        self.ticker = threading.Thread(target=self._tick, daemon=True)
        self.ticker.start()

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(5)
        return request, address

    def process_request_thread(self, request, client_address):
        # Handshake in the request thread: a silent TCP peer must not block
        # heartbeats from every authenticated worker at the accept loop.
        try:
            if self.tls is not None:
                request = self.tls.wrap_socket(request, server_side=True)
            super().process_request_thread(request, client_address)
        except OSError:
            self.shutdown_request(request)

    def _tick(self):
        audit_failed = False
        while not self.stopped.wait(0.1):
            try:
                self.coordinator.tick()
                audit_failed = False
            except OSError:
                if not audit_failed:
                    logging.exception("Could not persist the last-words audit; retrying")
                audit_failed = True

    def server_close(self):
        self.stopped.set()
        if self.ticker is not None:
            self.ticker.join(timeout=2)
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_args, **_kwargs):
        pass  # Never log tokens, request bodies, or capabilities.

    def respond(self, code, data):
        raw = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def get_status(self):
        if self.path != "/status":
            self.respond(404, dict(error="unknown endpoint"))
            return
        server = cast("ServiceServer", self.server)
        self.respond(200, dict(version=RPC_PROTOCOL_VERSION, backend="service", instance=server.coordinator.instance))

    def coordinate(self):
        try:
            require(self.path == "/coordinate", "unknown endpoint")
            require(not self.headers.get("Transfer-Encoding"), "chunked requests unsupported")
            length = int(self.headers.get("Content-Length", "0"))
            require(0 < length <= MAX_REQUEST_BYTES, "report exceeds 1 MiB or missing body")
            raw = self.rfile.read(length)
            require(len(raw) == length, "incomplete request")
            command = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            result = cast("ServiceServer", self.server).coordinator.command(command)
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            self.close_connection = True
            self.respond(400, dict(error=str(error)))
            return
        except OSError:
            self.close_connection = True
            self.respond(503, dict(error="Coordinator temporarily unavailable"))
            return
        self.respond(200, result)

    # BaseHTTPRequestHandler dispatches by HTTP verb.
    do_GET = get_status  # noqa: N815
    do_POST = coordinate  # noqa: N815


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1:0")
    parser.add_argument("--advertise-address", help="Client-facing root URL; port 0 uses the actual bound port")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--tls-ca", help="CA allowed to authenticate coordinator clients")
    parser.add_argument(
        "--session-dir", type=Path, required=True, help="New private directory for this run's credentials and result"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-replicas", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True, help="Shard count per replica, across all its nodes")
    parser.add_argument("--heartbeat-seconds", type=float, default=60)
    parser.add_argument("--startup-seconds", type=float, default=900)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args()
    host, port = args.bind.rsplit(":", 1)
    host = host.strip("[]")
    scheme = "https" if args.tls_cert else "http"
    advertised = args.advertise_address
    if advertised is None:
        require(host in {"127.0.0.1", "localhost", "::1"}, "Network bind requires --advertise-address")
        advertised = f"{scheme}://{'[' + host + ']' if ':' in host else host}:0"
    endpoint = validate_service_address(advertised, allow_port_zero=True)
    require(endpoint.scheme == scheme, "Advertised scheme must match the server TLS configuration")
    require(endpoint.hostname not in {"0.0.0.0", "::"}, "Advertise a reachable hostname, not a wildcard")
    with ServiceServer((host, int(port)), tls_cert=args.tls_cert, tls_key=args.tls_key, tls_ca=args.tls_ca) as server:
        require(endpoint.port in {0, server.server_port}, "Advertised port must match the bound service port")
        hostname = f"[{endpoint.hostname}]" if ":" in endpoint.hostname else endpoint.hostname
        address = f"{scheme}://{hostname}:{server.server_port}"
        prepare_session(
            server,
            args.session_dir,
            address=address,
            run_id=args.run_id,
            num_replicas=args.num_replicas,
            world_size=args.world_size,
            heartbeat_seconds=args.heartbeat_seconds,
            startup_seconds=args.startup_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(dict(address=address, run_id=args.run_id, session_dir=str(args.session_dir))), flush=True)
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
