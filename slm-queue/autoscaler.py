"""Reactive autoscaler for the slm-queue worker pool.

Every ``tick_interval_s`` seconds it samples per-model queue depth from
the dispatcher and decides whether to add or remove a replica:

  - Scale up when depth stays above ``scale_up_queue_threshold`` for
    ``scale_up_after_ticks`` consecutive ticks.
  - Scale down when the queue has been empty for
    ``scale_down_after_idle_ticks`` consecutive ticks.

Guards:

  - Per-model ``replicas_min`` / ``replicas_max`` from slms.yaml.
  - Global ``max_total_replicas`` cap (one GPU serves all Ollama models;
    unbounded scale-up thrashes).
  - Per-model ``cooldown_s`` after each scale event to prevent flapping.

Design notes:

  - Add-replica is a plain thread spawn; the loaded Ollama weights are
    already resident, so no cold-start delay.
  - Remove-replica enqueues a sentinel that any worker picks up on its
    next dequeue. This is graceful — in-flight tasks always finish.
  - The autoscaler holds no persistent state; a queue restart resets to
    ``spec.replicas`` from the config, which matches k8s Deployment
    semantics where autoscaling is transient.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _ModelState:
    ticks_over_threshold: int = 0
    ticks_idle: int = 0
    last_scale_ts: float = 0.0
    last_scale_reason: str = ""
    last_scale_action: str = ""  # "up" | "down" | ""


@dataclass
class AutoScalerDecision:
    ts: float
    model: str
    action: str          # "up" | "down" | "noop"
    reason: str
    from_replicas: int
    to_replicas: int


class AutoScaler:
    """Timer-driven feedback loop wrapping a Dispatcher.

    The dispatcher must expose:
      - ``models``: list[str]
      - ``queues[model]``: queue.Queue
      - ``scale_bounds[model]``: (initial, min, max)
      - ``replica_count(model) -> int``
      - ``add_replica(model) -> int``
      - ``remove_replica(model) -> int``
    """

    def __init__(
        self,
        dispatcher: Any,
        config: dict[str, Any],
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.dispatcher = dispatcher
        self.tick_interval_s = float(config.get("tick_interval_s", 10))
        self.scale_up_queue_threshold = int(config.get("scale_up_queue_threshold", 3))
        self.scale_up_after_ticks = int(config.get("scale_up_after_ticks", 3))
        self.scale_down_after_idle_ticks = int(config.get("scale_down_after_idle_ticks", 6))
        self.cooldown_s = float(config.get("cooldown_s", 30))

        # 0 = derive at startup from sum of replicas_max.
        configured_total = int(config.get("max_total_replicas", 0))
        derived_total = sum(hi for _, _, hi in dispatcher.scale_bounds.values())
        self.max_total_replicas = configured_total if configured_total > 0 else derived_total

        self._state: dict[str, _ModelState] = {m: _ModelState() for m in dispatcher.models}
        self._history: deque[AutoScalerDecision] = deque(maxlen=200)
        self._clock = clock
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="autoscaler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                # Autoscaler must never crash the process; log and continue.
                print(f"autoscaler tick failed: {type(e).__name__}: {e}")
            self._stop.wait(self.tick_interval_s)

    # ---- core decision ---------------------------------------------------

    def tick(self) -> list[AutoScalerDecision]:
        """One observation + decision pass. Returns whichever decisions
        were emitted this tick (0-N; usually 0 or 1)."""
        now = self._clock()
        decisions: list[AutoScalerDecision] = []
        total = self.dispatcher.total_replicas()
        for model in self.dispatcher.models:
            d = self._decide(model, now, total)
            if d is None:
                continue
            decisions.append(d)
            self._history.append(d)
            # Update the total we're comparing against as we apply decisions,
            # so within one tick we never exceed the budget.
            if d.action == "up":
                total += 1
            elif d.action == "down":
                total -= 1
        return decisions

    def _decide(
        self, model: str, now: float, current_total: int
    ) -> AutoScalerDecision | None:
        st = self._state[model]
        _, lo, hi = self.dispatcher.scale_bounds[model]
        depth = self.dispatcher.queues[model].qsize()
        replicas = self.dispatcher.replica_count(model)

        # Update rolling counters.
        if depth >= self.scale_up_queue_threshold:
            st.ticks_over_threshold += 1
            st.ticks_idle = 0
        elif depth == 0:
            st.ticks_idle += 1
            st.ticks_over_threshold = 0
        else:
            st.ticks_over_threshold = 0
            st.ticks_idle = 0

        # Cooldown gate.
        in_cooldown = (now - st.last_scale_ts) < self.cooldown_s and st.last_scale_ts > 0

        # Scale up?
        if (
            st.ticks_over_threshold >= self.scale_up_after_ticks
            and replicas < hi
            and not in_cooldown
        ):
            if current_total >= self.max_total_replicas:
                # Budget-blocked; note it but don't move counters — we want
                # to fire immediately once budget frees up.
                return AutoScalerDecision(
                    ts=now, model=model, action="noop",
                    reason=f"budget_full (total={current_total} >= {self.max_total_replicas})",
                    from_replicas=replicas, to_replicas=replicas,
                )
            new_count = self.dispatcher.add_replica(model)
            st.last_scale_ts = now
            st.last_scale_reason = (
                f"depth>={self.scale_up_queue_threshold} for "
                f"{st.ticks_over_threshold} ticks"
            )
            st.last_scale_action = "up"
            st.ticks_over_threshold = 0
            return AutoScalerDecision(
                ts=now, model=model, action="up", reason=st.last_scale_reason,
                from_replicas=replicas, to_replicas=new_count,
            )

        # Scale down?
        if (
            st.ticks_idle >= self.scale_down_after_idle_ticks
            and replicas > lo
            and not in_cooldown
        ):
            self.dispatcher.remove_replica(model)
            st.last_scale_ts = now
            st.last_scale_reason = (
                f"idle for {st.ticks_idle} ticks"
            )
            st.last_scale_action = "down"
            st.ticks_idle = 0
            return AutoScalerDecision(
                ts=now, model=model, action="down", reason=st.last_scale_reason,
                from_replicas=replicas, to_replicas=max(0, replicas - 1),
            )

        return None

    # ---- inspection ------------------------------------------------------

    def state_snapshot(self) -> dict[str, Any]:
        """Serializable snapshot for /status."""
        per_model: dict[str, Any] = {}
        for model in self.dispatcher.models:
            _, lo, hi = self.dispatcher.scale_bounds[model]
            st = self._state[model]
            per_model[model] = {
                "replicas": self.dispatcher.replica_count(model),
                "min": lo,
                "max": hi,
                "ticks_over_threshold": st.ticks_over_threshold,
                "ticks_idle": st.ticks_idle,
                "last_scale_ts": st.last_scale_ts or None,
                "last_scale_action": st.last_scale_action or None,
                "last_scale_reason": st.last_scale_reason or None,
            }
        return {
            "per_model": per_model,
            "total_replicas": self.dispatcher.total_replicas(),
            "max_total_replicas": self.max_total_replicas,
            "config": {
                "tick_interval_s": self.tick_interval_s,
                "scale_up_queue_threshold": self.scale_up_queue_threshold,
                "scale_up_after_ticks": self.scale_up_after_ticks,
                "scale_down_after_idle_ticks": self.scale_down_after_idle_ticks,
                "cooldown_s": self.cooldown_s,
            },
            "recent_decisions": [
                {
                    "ts": d.ts, "model": d.model, "action": d.action,
                    "reason": d.reason, "from": d.from_replicas, "to": d.to_replicas,
                }
                for d in list(self._history)[-20:]
            ],
        }
