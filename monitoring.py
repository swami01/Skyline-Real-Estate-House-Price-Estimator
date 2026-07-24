"""
Minimal prediction monitoring, backed by SQLite (stdlib, no new
dependency). Every call to /predict gets logged: when it happened,
which model version served it, how long it took, and what it
predicted. /stats aggregates this into something you'd actually look
at to notice problems -- a sudden shift in average predicted price, a
latency spike, or a version that's serving way more traffic than
expected.

This is a deliberately small version of what a real monitoring stack
(Prometheus + Grafana, or a hosted APM tool) does: log every request,
then look for drift in the aggregates over time. The full version
would also compare live input distributions against the training
distribution to catch data drift -- noted as a limitation, not
implemented here, since it needs a stored reference distribution and
a drift test (e.g. population stability index) to do properly.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "models", "monitoring.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    predicted_price REAL NOT NULL,
    input_json TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(model_version: str, latency_ms: float, predicted_price: float, input_dict: dict):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, model_version, latency_ms, predicted_price, input_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                model_version,
                latency_ms,
                predicted_price,
                json.dumps(input_dict),
            ),
        )


def get_stats(limit_recent: int = 10) -> dict:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]

        if total == 0:
            return {"total_predictions": 0}

        agg = conn.execute(
            "SELECT AVG(latency_ms) AS avg_latency_ms, "
            "MIN(latency_ms) AS min_latency_ms, MAX(latency_ms) AS max_latency_ms, "
            "AVG(predicted_price) AS avg_predicted_price, "
            "MIN(predicted_price) AS min_predicted_price, MAX(predicted_price) AS max_predicted_price "
            "FROM predictions"
        ).fetchone()

        by_version = conn.execute(
            "SELECT model_version, COUNT(*) AS n, AVG(predicted_price) AS avg_price "
            "FROM predictions GROUP BY model_version ORDER BY n DESC"
        ).fetchall()

        recent = conn.execute(
            "SELECT timestamp, model_version, latency_ms, predicted_price "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit_recent,),
        ).fetchall()

        return {
            "total_predictions": total,
            "avg_latency_ms": round(agg["avg_latency_ms"], 2),
            "min_latency_ms": round(agg["min_latency_ms"], 2),
            "max_latency_ms": round(agg["max_latency_ms"], 2),
            "avg_predicted_price": round(agg["avg_predicted_price"], 2),
            "min_predicted_price": round(agg["min_predicted_price"], 2),
            "max_predicted_price": round(agg["max_predicted_price"], 2),
            "by_model_version": [dict(r) for r in by_version],
            "recent_predictions": [dict(r) for r in recent],
        }


class Timer:
    """Tiny context manager for measuring latency in milliseconds."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
