"""Durable allocation intents and pool policy for a single control plane."""

import hashlib
import json
import sqlite3
from dataclasses import astuple, dataclass
from pathlib import Path
from uuid import UUID

from tokenflow.provisioner.base import PoolSpec


@dataclass
class Allocation:
    key: str
    pool: str
    created: float
    last_seen: float
    phase: str = "pending"
    external_id: str | None = None
    base_url: str | None = None
    retiring: bool = False
    terminate_after: float = 0
    ever_ready: bool = False
    unready_since: float | None = None
    worker_id: str | None = None


class State:
    def __init__(self, path: Path, pools: list[PoolSpec]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS pools (
                    name TEXT PRIMARY KEY, identity TEXT NOT NULL,
                    minimum INTEGER, maximum INTEGER,
                    failures INTEGER NOT NULL DEFAULT 0,
                    retry_at REAL NOT NULL DEFAULT 0, error TEXT
                );
                CREATE TABLE IF NOT EXISTS allocations (
                    key TEXT PRIMARY KEY, pool TEXT NOT NULL, created REAL NOT NULL,
                    last_seen REAL NOT NULL, phase TEXT NOT NULL, external_id TEXT,
                    base_url TEXT, retiring INTEGER NOT NULL, terminate_after REAL NOT NULL,
                    ever_ready INTEGER NOT NULL, unready_since REAL, worker_id TEXT
                );
                CREATE INDEX IF NOT EXISTS allocations_pool ON allocations(pool);
            """)
            self._configure(pools)
        except BaseException:
            self.close()
            raise

    def _configure(self, pools: list[PoolSpec]) -> None:
        """Reject changes that would strand an allocation outside discovery."""
        identities = {}
        for pool in pools:
            identity = pool.model_dump(mode="json", include={
                "provider", "model", "checkpoint", "backend", "gpu_type", "gpu_count",
                "image", "port", "capacity", "options",
            })
            identities[pool.name] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()

        with self.db:
            for row in self.db.execute("SELECT name, identity FROM pools").fetchall():
                name = row["name"]
                if identities.get(name) == row["identity"]:
                    continue
                if self.allocations(name):
                    raise ValueError(f"Drain pool {name!r} before removing it or changing its deployment")
                self.db.execute("DELETE FROM pools WHERE name = ?", (name,))
            self.db.executemany(
                "INSERT OR IGNORE INTO pools(name, identity) VALUES (?, ?)", identities.items()
            )
            for pool in pools:
                row = self.policy(pool.name)
                if row["maximum"] is not None and row["maximum"] > pool.max_replicas:
                    raise ValueError(f"Lower the saved replica limits for {pool.name!r} before reducing its ceiling")

    def allocations(self, pool: str) -> list[Allocation]:
        return [Allocation(**dict(row)) for row in self.db.execute(
            "SELECT * FROM allocations WHERE pool = ? ORDER BY created, key", (pool,)
        )]

    def registrations(self) -> dict[str, UUID]:
        return {row["key"]: UUID(row["worker_id"]) for row in self.db.execute(
            "SELECT key, worker_id FROM allocations WHERE worker_id IS NOT NULL"
        )}

    def save(self, allocation: Allocation) -> None:
        with self.db:
            self.db.execute("""
                INSERT OR REPLACE INTO allocations
                (key, pool, created, last_seen, phase, external_id, base_url,
                 retiring, terminate_after, ever_ready, unready_since, worker_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, astuple(allocation))

    def remove(self, key: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM allocations WHERE key = ?", (key,))

    def policy(self, pool: str) -> sqlite3.Row:
        return self.db.execute("SELECT * FROM pools WHERE name = ?", (pool,)).fetchone()

    def limits(self, pool: str, minimum: int, maximum: int) -> None:
        with self.db:
            self.db.execute("UPDATE pools SET minimum = ?, maximum = ? WHERE name = ?",
                            (minimum, maximum, pool))

    def failure(self, pool: str, now: float, error: str) -> None:
        failures = self.policy(pool)["failures"] + 1
        delay = min(300, 5 * 2 ** min(failures - 1, 6))
        with self.db:
            self.db.execute("UPDATE pools SET failures = ?, retry_at = ?, error = ? WHERE name = ?",
                            (failures, now + delay, error, pool))

    def healthy(self, pool: str) -> None:
        with self.db:
            self.db.execute("UPDATE pools SET failures = 0, retry_at = 0, error = NULL WHERE name = ?",
                            (pool,))

    def close(self) -> None:
        self.db.close()
