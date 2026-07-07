"""HTTP server for slm-router. See DESIGN.md sections 5 and 7.1."""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import statistics
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# The foundation modules use flat imports (`from model import Decision`), so the
# router package directory must be on sys.path before we import them.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import yaml  # noqa: E402

from heuristic_fallback import heuristic_decide  # noqa: E402
from model import Decision, Encoder, Head, head_pointer_read  # noqa: E402
from policy import decide as policy_decide  # noqa: E402


DECISION_CACHE_MAX = 10_000
LATENCY_WINDOW_MAX = 1_000


class RouterState:
    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path

        paths_cfg = config.get("paths", {})
        self.heads_dir = _resolve(_HERE, paths_cfg.get("heads_dir", "heads"))
        self.head_pointer_path = _resolve(_HERE, paths_cfg.get("head_pointer", "heads/HEAD"))
        self.feedback_path = _resolve(_HERE, paths_cfg.get("feedback_log", "feedback.jsonl"))

        self.encoder: Encoder | None = None
        self.encoder_error: str | None = None

        self._head: Head | None = None
        self._head_version: str | None = None
        self._head_pointer_mtime: float | None = None

        self._head_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._decisions_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

        self._decisions: OrderedDict[str, dict[str, Any]] = OrderedDict()

        self._latencies_ms: list[float] = []
        self._predictions_total = {"solo": 0, "mixture": 0, "unsure": 0}
        self._predictions_by_policy = {"learned": 0, "heuristic": 0, "explore": 0}
        self._feedback_count = self._count_existing_feedback()

        self._started_at = time.time()
        self._rng = random.Random()
        self._auth_token = os.environ.get("SLM_ROUTER_TOKEN") or None

        self._load_encoder()
        self._maybe_reload_head(force=True)

    def _load_encoder(self) -> None:
        try:
            model_name = (self.config.get("encoder") or {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.encoder = Encoder(model_name)
        except Exception as e:
            self.encoder = None
            self.encoder_error = f"{type(e).__name__}: {e}"
            _log({"event": "encoder_load_failed", "error": self.encoder_error})

    def _count_existing_feedback(self) -> int:
        if not self.feedback_path.exists():
            return 0
        n = 0
        try:
            with self.feedback_path.open("rb") as f:
                for _ in f:
                    n += 1
        except Exception:
            return 0
        return n

    def _maybe_reload_head(self, force: bool = False) -> None:
        pointer = self.head_pointer_path
        try:
            mtime = pointer.stat().st_mtime if pointer.exists() else None
        except OSError:
            mtime = None

        if not force and mtime == self._head_pointer_mtime:
            return

        with self._head_lock:
            if not force and mtime == self._head_pointer_mtime:
                return
            self._head_pointer_mtime = mtime

            version = head_pointer_read(pointer) if pointer.exists() else None
            if not version:
                if self._head is not None:
                    _log({"event": "head_unloaded", "reason": "pointer_missing"})
                self._head = None
                self._head_version = None
                return

            head_file = self.heads_dir / f"{version}.joblib"
            if not head_file.exists():
                _log({"event": "head_missing", "version": version, "path": str(head_file)})
                self._head = None
                self._head_version = None
                return

            try:
                self._head = Head.load(head_file)
                self._head_version = version
                _log({"event": "head_loaded", "version": version})
            except Exception as e:
                _log({
                    "event": "head_load_failed",
                    "version": version,
                    "error": f"{type(e).__name__}: {e}",
                })
                self._head = None
                self._head_version = None

    def force_reload(self) -> str:
        self._maybe_reload_head(force=True)
        return self._head_version or "heuristic"

    def route(self, prompt: str) -> Decision:
        self._maybe_reload_head()

        head = self._head
        head_version = self._head_version

        if head is None or self.encoder is None:
            decision = heuristic_decide(prompt)
        else:
            try:
                x = self.encoder.encode(prompt)
                pred_red, pred_qual, conf = head.predict(x)
                policy_cfg = self.config.get("policy", {}) or {}
                decision = policy_decide(
                    pred_red,
                    pred_qual,
                    conf,
                    head_version=head_version or "unknown",
                    policy_config=policy_cfg,
                    rng=self._rng,
                )
            except Exception as e:
                _log({"event": "predict_failed", "error": f"{type(e).__name__}: {e}"})
                decision = heuristic_decide(prompt)

        decision.decision_id = f"d_{secrets.token_hex(6)}"
        self._remember_decision(decision, prompt)
        return decision

    def _remember_decision(self, decision: Decision, prompt: str) -> None:
        entry = {
            "prompt": prompt,
            "decision": decision.to_dict(),
            "timestamp": time.time(),
        }
        with self._decisions_lock:
            self._decisions[decision.decision_id] = entry
            while len(self._decisions) > DECISION_CACHE_MAX:
                self._decisions.popitem(last=False)

    def record_metrics(self, decision: Decision, latency_ms: float) -> None:
        with self._metrics_lock:
            if decision.decision in self._predictions_total:
                self._predictions_total[decision.decision] += 1
            if decision.policy in self._predictions_by_policy:
                self._predictions_by_policy[decision.policy] += 1
            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > LATENCY_WINDOW_MAX:
                self._latencies_ms = self._latencies_ms[-LATENCY_WINDOW_MAX:]

    def append_feedback(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        with self._feedback_lock:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with self.feedback_path.open("a", encoding="utf-8") as f:
                f.write(line)
            self._feedback_count += 1

    def known_decision(self, decision_id: str) -> bool:
        with self._decisions_lock:
            return decision_id in self._decisions

    def snapshot_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            samples = list(self._latencies_ms)
            preds = dict(self._predictions_total)
            by_policy = dict(self._predictions_by_policy)

        p50 = p99 = 0.0
        if samples:
            if len(samples) >= 2:
                # quantiles(n=100) returns 99 cut points -> index 49 ~ p50, 98 ~ p99.
                qs = statistics.quantiles(samples, n=100, method="inclusive")
                p50 = qs[49]
                p99 = qs[98]
            else:
                p50 = p99 = samples[0]

        head_version: str | None
        if self._head_version:
            head_version = self._head_version
        elif self.encoder is not None:
            head_version = "heuristic"
        else:
            head_version = "heuristic"

        return {
            "predictions_total": preds,
            "predictions_by_policy": by_policy,
            "prediction_latency_ms_p50": round(p50, 3),
            "prediction_latency_ms_p99": round(p99, 3),
            "head_version": head_version,
            "feedback_records_total": self._feedback_count,
            "uptime_s": round(time.time() - self._started_at, 3),
        }


def _resolve(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def _log(payload: dict[str, Any]) -> None:
    payload.setdefault("ts", time.time())
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


class RouterHandler(BaseHTTPRequestHandler):
    state: RouterState = None  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # silence default access log
        return

    def _read_json_body(self) -> dict[str, Any] | None:
        length_hdr = self.headers.get("Content-Length")
        if not length_hdr:
            return {}
        try:
            length = int(length_hdr)
        except ValueError:
            return None
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        token = self.state._auth_token
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        return secrets.compare_digest(header[len(prefix):].strip(), token)

    def _reject_unauthorized(self) -> None:
        self._send_json(401, {"error": "unauthorized"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth_ok():
            self._reject_unauthorized()
            return
        path = self.path.split("?", 1)[0]
        try:
            if path == "/route":
                self._handle_route()
            elif path == "/feedback":
                self._handle_feedback()
            elif path == "/reload":
                self._handle_reload()
            else:
                self._send_json(404, {"error": f"unknown path: {path}"})
        except Exception as e:
            _log({"event": "handler_error", "path": path, "error": f"{type(e).__name__}: {e}"})
            try:
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def do_GET(self) -> None:  # noqa: N802
        if not self._auth_ok():
            self._reject_unauthorized()
            return
        path = self.path.split("?", 1)[0]
        try:
            if path == "/metrics":
                self._send_json(200, self.state.snapshot_metrics())
            elif path == "/health":
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": f"unknown path: {path}"})
        except Exception as e:
            _log({"event": "handler_error", "path": path, "error": f"{type(e).__name__}: {e}"})
            try:
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def _handle_route(self) -> None:
        body = self._read_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            self._send_json(400, {"error": "missing or empty 'prompt' field"})
            return

        t0 = time.perf_counter()
        decision = self.state.route(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.state.record_metrics(decision, latency_ms)

        _log({
            "event": "route",
            "decision_id": decision.decision_id,
            "path": "/route",
            "decision": decision.decision,
            "policy": decision.policy,
            "head_version": decision.head_version,
            "latency_ms": round(latency_ms, 3),
        })
        self._send_json(200, asdict(decision))

    def _handle_feedback(self) -> None:
        body = self._read_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        decision_id = body.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            self._send_json(400, {"error": "missing 'decision_id'"})
            return
        if "timestamp" not in body or not body.get("timestamp"):
            body["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # We accept feedback even for decision_ids we don't recognize (server may
        # have restarted); "known" only controls the idempotency log message.
        known = self.state.known_decision(decision_id)
        self.state.append_feedback(body)
        _log({
            "event": "feedback",
            "decision_id": decision_id,
            "path": "/feedback",
            "known": known,
        })
        self._send_json(200, {"accepted": True})

    def _handle_reload(self) -> None:
        version = self.state.force_reload()
        _log({"event": "reload", "path": "/reload", "head_version": version})
        self._send_json(200, {"reloaded": True, "head_version": version})


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} is not a mapping")
    return data


def build_server(config_path: Path, port_override: int | None) -> tuple[ThreadingHTTPServer, RouterState]:
    config = _load_config(config_path)
    server_cfg = config.get("server", {}) or {}
    host = server_cfg.get("host", "127.0.0.1")
    port = port_override if port_override is not None else int(server_cfg.get("port", 8092))

    state = RouterState(config, config_path)
    handler = type("BoundRouterHandler", (RouterHandler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="slm-router HTTP server")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=str(_HERE / "config.yaml"))
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        _log({"event": "config_missing", "path": str(config_path)})
        return 2

    httpd, state = build_server(config_path, args.port)
    host, port = httpd.server_address[0], httpd.server_address[1]
    _log({
        "event": "server_start",
        "host": host,
        "port": port,
        "config": str(config_path),
        "head_version": state._head_version,
        "encoder_loaded": state.encoder is not None,
        "auth_required": state._auth_token is not None,
    })
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log({"event": "server_stop", "reason": "keyboard_interrupt"})
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
