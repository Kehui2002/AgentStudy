from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalStore:
    """SQLite metadata plus a content-addressed filesystem object store."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.database_path = state_dir / "metadata.sqlite3"
        self.objects_dir = state_dir / "objects" / "sha256"
        self.operator = os.environ.get("ORIGIN_FIT_OPERATOR") or getpass.getuser()
        state_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_snapshots (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    imported_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fit_specifications (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    dataset_snapshot_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    specification_json TEXT NOT NULL,
                    FOREIGN KEY (dataset_snapshot_id) REFERENCES dataset_snapshots(id)
                );
                CREATE TABLE IF NOT EXISTS approved_fit_recipes (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL UNIQUE,
                    version INTEGER NOT NULL UNIQUE,
                    fit_specification_id TEXT NOT NULL UNIQUE,
                    approved_by TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    recipe_json TEXT NOT NULL,
                    FOREIGN KEY (fit_specification_id) REFERENCES fit_specifications(id)
                );
                CREATE TABLE IF NOT EXISTS fit_results (
                    id TEXT PRIMARY KEY,
                    dataset_snapshot_id TEXT NOT NULL,
                    approved_fit_recipe_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY (dataset_snapshot_id) REFERENCES dataset_snapshots(id),
                    FOREIGN KEY (approved_fit_recipe_id) REFERENCES approved_fit_recipes(id)
                );
                CREATE TABLE IF NOT EXISTS fit_jobs (
                    id TEXT PRIMARY KEY,
                    dataset_snapshot_id TEXT NOT NULL,
                    approved_fit_recipe_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    fit_result_id TEXT,
                    FOREIGN KEY (dataset_snapshot_id) REFERENCES dataset_snapshots(id),
                    FOREIGN KEY (approved_fit_recipe_id) REFERENCES approved_fit_recipes(id)
                );
                CREATE TABLE IF NOT EXISTS accepted_fits (
                    id TEXT PRIMARY KEY,
                    fit_result_id TEXT NOT NULL UNIQUE,
                    accepted_by TEXT NOT NULL,
                    accepted_at TEXT NOT NULL,
                    accepted_fit_json TEXT NOT NULL,
                    FOREIGN KEY (fit_result_id) REFERENCES fit_results(id)
                );
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def put_object(self, digest: str, content: bytes) -> Path:
        path = self.objects_dir / digest
        try:
            with path.open("xb") as object_file:
                object_file.write(content)
            path.chmod(0o444)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError("content-addressed object collision")
        return path

    def audit(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        object_id: str,
        details: dict,
        *,
        actor: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (
                event_type, occurred_at, actor, object_id, details_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                utc_now(),
                actor or self.operator,
                object_id,
                json.dumps(details, sort_keys=True, separators=(",", ":")),
            ),
        )
