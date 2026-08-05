from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterator
import uuid

from pydantic import ValidationError

from origin_fit.contracts import (
    GraphProfileCapability,
    WorkerCapabilities,
    WorkerJob,
    WorkerSubmission,
)
from origin_fit.errors import OriginFitError
from origin_fit.execution import (
    OriginAdapter,
    execute_approved_fit,
    load_approved_fit_execution_request,
)
from origin_fit.storage import LocalStore, utc_now
from origin_fit.specifications import REQUIRED_OUTPUTS

from .bundle import build_result_bundle


class WorkerError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OriginWorker:
    """Durable single-Origin execution service behind the transport adapter."""

    def __init__(
        self,
        state_dir: Path,
        adapter: OriginAdapter,
        *,
        max_dataset_bytes: int = 100 * 1024 * 1024,
        max_rows: int = 1_000_000,
        max_y_series: int = 20,
        job_timeout: float = 30 * 60,
    ) -> None:
        self.state_dir = state_dir
        self.database_path = state_dir / "worker.sqlite3"
        self.jobs_dir = state_dir / "jobs"
        self.adapter = adapter
        self.max_dataset_bytes = max_dataset_bytes
        self.max_rows = max_rows
        self.max_y_series = max_y_series
        self.job_timeout = job_timeout
        self._queue_lock = asyncio.Lock()
        state_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS fit_jobs (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    submission_hash TEXT NOT NULL,
                    submission_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    workspace_ref TEXT NOT NULL,
                    bundle_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fit_job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    error_code TEXT,
                    FOREIGN KEY (fit_job_id) REFERENCES fit_jobs(id)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )
            interrupted = connection.execute(
                "SELECT id FROM fit_jobs WHERE status = 'running'"
            ).fetchall()
            for row in interrupted:
                finished_at = utc_now()
                connection.execute(
                    """
                    UPDATE fit_jobs
                    SET status = 'failed', finished_at = ?,
                        error_code = 'worker_restarted',
                        error_message = 'Worker restarted while the Fit Job was running.'
                    WHERE id = ?
                    """,
                    (finished_at, row["id"]),
                )
                self._transition(
                    connection, row["id"], "failed", "worker_restarted"
                )
                self._audit(
                    connection,
                    "fit_job.failed",
                    row["id"],
                    {"error_code": "worker_restarted"},
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

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(
            transport_schema_version="1.0",
            fit_specification_schema_versions=["1.0"],
            fit_result_schema_versions=["1.0"],
            manifest_schema_versions=["1.0"],
            models=["ExpDec2"],
            graph_profiles=[
                GraphProfileCapability(id="expdec2-standard", version="1.0")
            ],
            max_dataset_bytes=self.max_dataset_bytes,
            max_rows=self.max_rows,
            max_y_series=self.max_y_series,
        )

    def health(self) -> dict[str, str]:
        with self.connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return {"status": "ok", "transport_schema_version": "1.0"}

    def submit(self, submission: WorkerSubmission, idempotency_key: str) -> WorkerJob:
        if not idempotency_key or len(idempotency_key) > 200:
            raise WorkerError(
                "invalid_idempotency_key", "A bounded Idempotency-Key is required."
            )
        canonical = json.dumps(
            submission.model_dump(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        submission_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM fit_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["submission_hash"] != submission_hash:
                    raise WorkerError(
                        "idempotency_conflict",
                        "The Idempotency-Key was already used for different content.",
                    )
                return self._job(existing)

        dataset = self._validate_submission(submission)
        worker_job_id = f"fit-job:{uuid.uuid4()}"
        workspace_ref = f"jobs/{worker_job_id.removeprefix('fit-job:')}"
        workspace = self._workspace(workspace_ref)
        workspace.mkdir(parents=False, exist_ok=False)
        snapshot_path = workspace / "dataset-snapshot.csv"
        snapshot_path.write_bytes(dataset)
        snapshot_path.chmod(0o444)
        persisted_submission = submission.model_dump()
        persisted_submission.pop("dataset_base64")
        persisted_json = json.dumps(
            persisted_submission,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        submitted_at = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO fit_jobs (
                    id, idempotency_key, submission_hash, submission_json,
                    status, submitted_at, workspace_ref
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    worker_job_id,
                    idempotency_key,
                    submission_hash,
                    persisted_json,
                    submitted_at,
                    workspace_ref,
                ),
            )
            self._transition(connection, worker_job_id, "queued")
            self._audit(
                connection,
                "fit_job.submitted",
                worker_job_id,
                {
                    "dataset_snapshot_id": submission.dataset_snapshot_id,
                    "approved_fit_recipe_id": submission.approved_fit_recipe_id,
                    "submission_hash": submission_hash,
                },
            )
        return self.get_job(worker_job_id)

    def get_job(self, worker_job_id: str) -> WorkerJob:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM fit_jobs WHERE id = ?", (worker_job_id,)
            ).fetchone()
        if row is None:
            raise WorkerError("not_found", f"Fit Job '{worker_job_id}' not found.")
        return self._job(row)

    def cancel(self, worker_job_id: str) -> WorkerJob:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM fit_jobs WHERE id = ?", (worker_job_id,)
            ).fetchone()
            if row is None:
                raise WorkerError(
                    "not_found", f"Fit Job '{worker_job_id}' not found."
                )
            if row["status"] == "queued":
                finished_at = utc_now()
                connection.execute(
                    "UPDATE fit_jobs SET status = 'cancelled', finished_at = ? WHERE id = ?",
                    (finished_at, worker_job_id),
                )
                self._transition(connection, worker_job_id, "cancelled")
                self._audit(connection, "fit_job.cancelled", worker_job_id, {})
            elif row["status"] == "running":
                finished_at = utc_now()
                connection.execute(
                    """
                    UPDATE fit_jobs SET status = 'failed', finished_at = ?,
                        error_code = 'cancelled_during_execution',
                        error_message = 'Running Fit Job cancellation terminated execution.'
                    WHERE id = ?
                    """,
                    (finished_at, worker_job_id),
                )
                self._transition(
                    connection,
                    worker_job_id,
                    "failed",
                    "cancelled_during_execution",
                )
                self._audit(
                    connection,
                    "fit_job.cancelled",
                    worker_job_id,
                    {"while_running": True},
                )
                terminate = getattr(self.adapter, "terminate", None)
                if callable(terminate):
                    terminate()
        return self.get_job(worker_job_id)

    async def run_queued(self) -> None:
        async with self._queue_lock:
            while True:
                worker_job_id = self._start_next()
                if worker_job_id is None:
                    return
                await self._execute(worker_job_id)

    def get_bundle(self, worker_job_id: str) -> bytes:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status, workspace_ref, bundle_sha256 FROM fit_jobs WHERE id = ?",
                (worker_job_id,),
            ).fetchone()
        if row is None:
            raise WorkerError("not_found", f"Fit Job '{worker_job_id}' not found.")
        if row["status"] != "succeeded" or not row["bundle_sha256"]:
            raise WorkerError("bundle_unavailable", "Fit Result Bundle is not available.")
        path = self._workspace(row["workspace_ref"]) / "fit-result-bundle.zip"
        try:
            content = path.read_bytes()
        except OSError as error:
            raise WorkerError(
                "bundle_unavailable", "Fit Result Bundle is not available."
            ) from error
        if hashlib.sha256(content).hexdigest() != row["bundle_sha256"]:
            raise WorkerError(
                "bundle_integrity_error", "Fit Result Bundle failed Worker verification."
            )
        with self.connect() as connection:
            self._audit(
                connection,
                "fit_result_bundle.downloaded",
                worker_job_id,
                {"bundle_sha256": row["bundle_sha256"]},
            )
        return content

    def _validate_submission(self, submission: WorkerSubmission) -> bytes:
        try:
            dataset = base64.b64decode(submission.dataset_base64, validate=True)
        except ValueError as error:
            raise WorkerError(
                "invalid_dataset", "Dataset Snapshot content is not valid base64."
            ) from error
        if len(dataset) > self.max_dataset_bytes:
            raise WorkerError("dataset_too_large", "Dataset Snapshot exceeds Worker limits.")
        if hashlib.sha256(dataset).hexdigest() != submission.dataset_content_hash:
            raise WorkerError(
                "dataset_integrity_error",
                "Dataset Snapshot content does not match its content hash.",
            )
        if submission.dataset_snapshot_id != f"sha256:{submission.dataset_content_hash}":
            raise WorkerError(
                "dataset_integrity_error", "Dataset Snapshot identifier is invalid."
            )
        recipe = submission.approved_fit_recipe
        specification = recipe.get("fit_specification", {})
        if not isinstance(specification, dict):
            raise WorkerError(
                "incompatible_submission",
                "Submission is incompatible with Worker capabilities.",
            )
        graph_profile = specification.get("graph_profile", {})
        model = specification.get("model", {})
        recipe_version = recipe.get("version")
        if (
            recipe.get("schema_version") != "1.0"
            or isinstance(recipe_version, bool)
            or not isinstance(recipe_version, int)
            or recipe_version < 1
            or specification.get("schema_version") != "1.0"
            or not isinstance(model, dict)
            or model.get("name") != "ExpDec2"
            or graph_profile != {"id": "expdec2-standard", "version": "1.0"}
            or specification.get("output_requirements") != REQUIRED_OUTPUTS
            or specification.get("dataset_snapshot_id")
            != submission.dataset_snapshot_id
            or specification.get("dataset_content_hash")
            != submission.dataset_content_hash
        ):
            raise WorkerError(
                "incompatible_submission",
                "Submission is incompatible with Worker capabilities.",
            )
        canonical_recipe = json.dumps(
            recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        if hashlib.sha256(canonical_recipe).hexdigest() != submission.approved_fit_recipe_hash:
            raise WorkerError(
                "recipe_integrity_error",
                "Approved Fit Recipe does not match its content hash.",
            )
        if submission.approved_fit_recipe_id != (
            f"recipe:sha256:{submission.approved_fit_recipe_hash}"
        ):
            raise WorkerError(
                "recipe_integrity_error", "Approved Fit Recipe identifier is invalid."
            )
        canonical_specification = json.dumps(
            specification,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        specification_hash = hashlib.sha256(canonical_specification).hexdigest()
        if (
            recipe.get("fit_specification_hash") != specification_hash
            or recipe.get("fit_specification_id")
            != f"spec:sha256:{specification_hash}"
        ):
            raise WorkerError(
                "recipe_integrity_error",
                "Fit Specification does not match its approved hash.",
            )
        row_count = submission.dataset_summary.get("row_count")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
            or row_count > self.max_rows
        ):
            raise WorkerError("dataset_too_large", "Dataset Snapshot exceeds Worker limits.")
        y_columns = submission.dataset_metadata.get("y_columns")
        if (
            not isinstance(y_columns, list)
            or not y_columns
            or any(not isinstance(name, str) or not name for name in y_columns)
            or len(y_columns) > self.max_y_series
        ):
            raise WorkerError("dataset_too_large", "Dataset Snapshot exceeds Worker limits.")
        return dataset

    def _start_next(self) -> str | None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM fit_jobs WHERE status = 'queued' ORDER BY submitted_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            started_at = utc_now()
            connection.execute(
                "UPDATE fit_jobs SET status = 'running', started_at = ? WHERE id = ?",
                (started_at, row["id"]),
            )
            self._transition(connection, row["id"], "running")
            self._audit(connection, "fit_job.running", row["id"], {})
            return str(row["id"])

    async def _execute(self, worker_job_id: str) -> None:
        try:
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT submission_json, workspace_ref FROM fit_jobs WHERE id = ?",
                    (worker_job_id,),
                ).fetchone()
            assert row is not None
            workspace = self._workspace(row["workspace_ref"])
            persisted_submission = json.loads(row["submission_json"])
            persisted_submission["dataset_base64"] = base64.b64encode(
                (workspace / "dataset-snapshot.csv").read_bytes()
            ).decode("ascii")
            submission = WorkerSubmission.model_validate(
                persisted_submission, strict=True
            )
            execution_store = self._hydrate_workspace(workspace, submission)
            request, _, _ = load_approved_fit_execution_request(
                execution_store,
                submission.dataset_snapshot_id,
                submission.approved_fit_recipe_id,
            )
            result = await asyncio.wait_for(
                execute_approved_fit(
                    execution_store,
                    submission.dataset_snapshot_id,
                    submission.approved_fit_recipe_id,
                    self.adapter,
                    fit_job_id=worker_job_id,
                ),
                timeout=self.job_timeout,
            )
            adapter_name = getattr(self.adapter, "adapter_name", type(self.adapter).__name__)
            originpro_version = getattr(self.adapter, "originpro_version", "fake-2025")
            bundle, _ = build_result_bundle(
                worker_job_id=worker_job_id,
                request=request,
                result=result,
                dataset_snapshot_id=submission.dataset_snapshot_id,
                dataset_content_hash=submission.dataset_content_hash,
                approved_fit_recipe_id=submission.approved_fit_recipe_id,
                approved_fit_recipe_hash=submission.approved_fit_recipe_hash,
                approved_fit_recipe=submission.approved_fit_recipe,
                adapter_name=str(adapter_name),
                originpro_version=str(originpro_version),
            )
            bundle_hash = hashlib.sha256(bundle).hexdigest()
            bundle_path = workspace / "fit-result-bundle.zip"
            bundle_path.write_bytes(bundle)
            bundle_path.chmod(0o444)
            with self.connect() as connection:
                current = connection.execute(
                    "SELECT status FROM fit_jobs WHERE id = ?", (worker_job_id,)
                ).fetchone()
                if current is None or current["status"] != "running":
                    return
                finished_at = utc_now()
                connection.execute(
                    """
                    UPDATE fit_jobs SET status = 'succeeded', finished_at = ?,
                        bundle_sha256 = ? WHERE id = ?
                    """,
                    (finished_at, bundle_hash, worker_job_id),
                )
                self._transition(connection, worker_job_id, "succeeded")
                self._audit(
                    connection,
                    "fit_job.succeeded",
                    worker_job_id,
                    {"bundle_sha256": bundle_hash},
                )
        except asyncio.TimeoutError:
            terminate = getattr(self.adapter, "terminate", None)
            if callable(terminate):
                terminate()
            self._fail(
                worker_job_id,
                "worker_timeout",
                "Fit Job exceeded the Worker execution timeout.",
            )
        except (OriginFitError, WorkerError, ValidationError) as error:
            code = getattr(error, "code", "invalid_submission")
            message = getattr(error, "message", "Worker rejected invalid job content.")
            self._fail(worker_job_id, str(code), str(message))
        except Exception:
            self._fail(
                worker_job_id,
                "worker_execution_error",
                "Worker could not complete the Fit Job.",
            )

    def _hydrate_workspace(
        self, workspace: Path, submission: WorkerSubmission
    ) -> LocalStore:
        store = LocalStore(workspace / "execution")
        dataset = base64.b64decode(submission.dataset_base64, validate=True)
        store.put_object(submission.dataset_content_hash, dataset)
        recipe = submission.approved_fit_recipe
        specification = recipe["fit_specification"]
        now = utc_now()
        with store.connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_snapshots (
                    id, content_hash, imported_at, metadata_json, summary_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    submission.dataset_snapshot_id,
                    submission.dataset_content_hash,
                    now,
                    json.dumps(submission.dataset_metadata, sort_keys=True),
                    json.dumps(submission.dataset_summary, sort_keys=True),
                ),
            )
            connection.execute(
                """
                INSERT INTO fit_specifications (
                    id, content_hash, dataset_snapshot_id, created_at,
                    specification_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    recipe["fit_specification_id"],
                    recipe["fit_specification_hash"],
                    submission.dataset_snapshot_id,
                    now,
                    json.dumps(specification, sort_keys=True, separators=(",", ":")),
                ),
            )
            connection.execute(
                """
                INSERT INTO approved_fit_recipes (
                    id, content_hash, version, fit_specification_id,
                    approved_by, approved_at, recipe_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    submission.approved_fit_recipe_id,
                    submission.approved_fit_recipe_hash,
                    recipe["version"],
                    recipe["fit_specification_id"],
                    recipe["approved_by"],
                    recipe["approved_at"],
                    json.dumps(recipe, sort_keys=True, separators=(",", ":")),
                ),
            )
        return store

    def _fail(self, worker_job_id: str, code: str, message: str) -> None:
        with self.connect() as connection:
            current = connection.execute(
                "SELECT status FROM fit_jobs WHERE id = ?", (worker_job_id,)
            ).fetchone()
            if current is None or current["status"] != "running":
                return
            finished_at = utc_now()
            connection.execute(
                """
                UPDATE fit_jobs SET status = 'failed', finished_at = ?,
                    error_code = ?, error_message = ? WHERE id = ?
                """,
                (finished_at, code, message, worker_job_id),
            )
            self._transition(connection, worker_job_id, "failed", code)
            self._audit(
                connection,
                "fit_job.failed",
                worker_job_id,
                {"error_code": code},
            )

    def _workspace(self, reference: str) -> Path:
        candidate = (self.state_dir / reference).resolve()
        root = self.jobs_dir.resolve()
        if candidate.parent != root:
            raise WorkerError("invalid_workspace", "Fit Job workspace is invalid.")
        return candidate

    @staticmethod
    def _job(row: sqlite3.Row) -> WorkerJob:
        return WorkerJob(
            worker_job_id=row["id"],
            status=row["status"],
            submitted_at=row["submitted_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            bundle_sha256=row["bundle_sha256"],
        )

    @staticmethod
    def _transition(
        connection: sqlite3.Connection,
        worker_job_id: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO state_transitions (
                fit_job_id, status, occurred_at, error_code
            ) VALUES (?, ?, ?, ?)
            """,
            (worker_job_id, status, utc_now(), error_code),
        )

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        event_type: str,
        object_id: str,
        details: dict[str, object],
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (
                event_type, occurred_at, object_id, details_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                event_type,
                utc_now(),
                object_id,
                json.dumps(details, sort_keys=True, separators=(",", ":")),
            ),
        )
