"""SQLite-backed call log and daily metrics rollup."""
from __future__ import annotations

import sqlite3
import time
from datetime import date
from pathlib import Path

from slowandsweet.paths import DB_PATH, ensure_state_dir

CURRENT_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS calls (
  call_id TEXT PRIMARY KEY,
  ts REAL NOT NULL,
  status TEXT NOT NULL,
  reason TEXT,
  frontier_tokens_in INTEGER,
  frontier_tokens_out INTEGER,
  slm_tokens_out INTEGER,
  wall_ms INTEGER,
  estimated_solo_tokens INTEGER,
  estimated_solo_wall_ms INTEGER
);

CREATE INDEX IF NOT EXISTS calls_ts_idx ON calls(ts);

CREATE TABLE IF NOT EXISTS metrics_daily (
  date TEXT PRIMARY KEY,
  delegated_calls INTEGER NOT NULL DEFAULT 0,
  abstained_calls INTEGER NOT NULL DEFAULT 0,
  failed_calls INTEGER NOT NULL DEFAULT 0,
  frontier_tokens_saved INTEGER NOT NULL DEFAULT 0,
  wall_ms_saved INTEGER NOT NULL DEFAULT 0
);
"""


def _connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    ensure_state_dir()
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path | None = None) -> None:
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA)
        cur = conn.execute("SELECT version FROM schema_version")
        row = cur.fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )
        conn.commit()
    finally:
        conn.close()


def schema_version(path: Path | None = None) -> int:
    conn = _connect(path)
    try:
        try:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["version"]) if row else 0
    finally:
        conn.close()


def _today_str() -> str:
    return date.today().isoformat()


def record_call(
    call_id: str,
    status: str,
    reason: str | None = None,
    frontier_tokens_in: int | None = None,
    frontier_tokens_out: int | None = None,
    slm_tokens_out: int | None = None,
    wall_ms: int | None = None,
    estimated_solo_tokens: int | None = None,
    estimated_solo_wall_ms: int | None = None,
    ts: float | None = None,
    path: Path | None = None,
) -> None:
    if status not in ("delegated", "abstained", "failed"):
        raise ValueError(f"invalid status: {status!r}")
    ts = ts if ts is not None else time.time()
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO calls "
            "(call_id, ts, status, reason, frontier_tokens_in, frontier_tokens_out, "
            " slm_tokens_out, wall_ms, estimated_solo_tokens, estimated_solo_wall_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                call_id, ts, status, reason,
                frontier_tokens_in, frontier_tokens_out,
                slm_tokens_out, wall_ms,
                estimated_solo_tokens, estimated_solo_wall_ms,
            ),
        )

        # Savings count only when the call actually delegated successfully.
        if status == "delegated":
            tokens_saved = max(
                0,
                (estimated_solo_tokens or 0)
                - (frontier_tokens_in or 0)
                - (frontier_tokens_out or 0),
            )
            wall_saved = max(0, (estimated_solo_wall_ms or 0) - (wall_ms or 0))
        else:
            tokens_saved = 0
            wall_saved = 0

        day = _today_str()
        conn.execute(
            "INSERT INTO metrics_daily "
            "(date, delegated_calls, abstained_calls, failed_calls, "
            " frontier_tokens_saved, wall_ms_saved) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET "
            "  delegated_calls = delegated_calls + excluded.delegated_calls, "
            "  abstained_calls = abstained_calls + excluded.abstained_calls, "
            "  failed_calls = failed_calls + excluded.failed_calls, "
            "  frontier_tokens_saved = frontier_tokens_saved + excluded.frontier_tokens_saved, "
            "  wall_ms_saved = wall_ms_saved + excluded.wall_ms_saved",
            (
                day,
                1 if status == "delegated" else 0,
                1 if status == "abstained" else 0,
                1 if status == "failed" else 0,
                tokens_saved,
                wall_saved,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def read_today(path: Path | None = None) -> dict:
    day = _today_str()
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT * FROM metrics_daily WHERE date = ?", (day,)
        ).fetchone()
        totals = dict(row) if row else {
            "date": day,
            "delegated_calls": 0,
            "abstained_calls": 0,
            "failed_calls": 0,
            "frontier_tokens_saved": 0,
            "wall_ms_saved": 0,
        }

        # Top abstain reasons among today's calls (status = 'abstained').
        day_start = time.mktime(date.today().timetuple())
        reasons = conn.execute(
            "SELECT reason, COUNT(*) AS count FROM calls "
            "WHERE status = 'abstained' AND ts >= ? AND reason IS NOT NULL "
            "GROUP BY reason ORDER BY count DESC LIMIT 3",
            (day_start,),
        ).fetchall()
        top_reasons = [{"reason": r["reason"], "count": int(r["count"])} for r in reasons]
        return {"totals": totals, "top_abstain_reasons": top_reasons}
    finally:
        conn.close()
