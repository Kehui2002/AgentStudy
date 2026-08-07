from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
import hashlib
import io
import json
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse
import zipfile

from pydantic import ValidationError

from mini_agent import ToolError

from .contracts import (
    BUNDLE_ARTIFACTS,
    BUNDLE_FILES,
    FIT_RESULT_SCHEMA_VERSION,
    FIT_SPECIFICATION_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    TRANSPORT_SCHEMA_VERSION,
    FitResultManifest,
    StrictModel,
    WorkerCapabilities,
    WorkerJob,
    WorkerSubmission,
)
from .errors import OriginFitError
from .execution import FitResult
from .storage import LocalStore, utc_now


class WorkerTransport(Protocol):
    async def capabilities(self) -> WorkerCapabilities: ...

    async def health(self) -> dict[str, str]: ...

    async def submit(
        self, submission: WorkerSubmission, idempotency_key: str
    ) -> WorkerJob: ...

    async def status(self, worker_job_id: str) -> WorkerJob: ...

    async def cancel(self, worker_job_id: str) -> WorkerJob: ...

    async def download_bundle(self, worker_job_id: str) -> bytes: ...


class PendingFitJob(StrictModel):
    status: Literal["pending"]
    worker_job_id: str
    worker_status: Literal["queued", "running"]
    message: str


class ArchivedFitResult(StrictModel):
    status: Literal["succeeded"]
    worker_job_id: str
    fit_archive_id: str
    bundle_sha256: str
    fit_result: FitResult


RemoteFitOutcome = ArchivedFitResult | PendingFitJob


class InProcessWorkerTransport:
    """Transport substitute that exercises the same durable Worker service in tests."""

    def __init__(self, worker: object) -> None:
        self.worker = worker
        self._tasks: set[asyncio.Task[None]] = set()

    async def capabilities(self) -> WorkerCapabilities:
        return self.worker.capabilities()  # type: ignore[attr-defined,no-any-return]

    async def health(self) -> dict[str, str]:
        return self.worker.health()  # type: ignore[attr-defined,no-any-return]

    async def submit(
        self, submission: WorkerSubmission, idempotency_key: str
    ) -> WorkerJob:
        job = self.worker.submit(submission, idempotency_key)  # type: ignore[attr-defined]
        if job.status == "queued" and not self._tasks:
            task = asyncio.create_task(self.worker.run_queued())  # type: ignore[attr-defined]
            self._tasks.add(task)
            task.add_done_callback(self._task_finished)
        return job

    async def status(self, worker_job_id: str) -> WorkerJob:
        job = self.worker.get_job(worker_job_id)  # type: ignore[attr-defined]
        if job.status not in ("queued", "running") and self._tasks:
            await asyncio.gather(*tuple(self._tasks))
        return job  # type: ignore[no-any-return]

    async def cancel(self, worker_job_id: str) -> WorkerJob:
        return self.worker.cancel(worker_job_id)  # type: ignore[attr-defined,no-any-return]

    async def download_bundle(self, worker_job_id: str) -> bytes:
        return self.worker.get_bundle(worker_job_id)  # type: ignore[attr-defined,no-any-return]

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()


class HttpWorkerTransport:
    """HTTPS Worker transport with certificate verification delegated to httpx."""

    def __init__(self, client: object, token: str) -> None:
        if not token:
            raise ValueError("Worker token must not be empty")
        self.client = client
        self._headers = {"Authorization": f"Bearer {token}"}
        self._owns_client = False

    @classmethod
    def with_pinned_certificate(
        cls,
        base_url: str,
        *,
        token: str,
        pinned_certificate: Path,
        timeout: float = 30,
    ) -> HttpWorkerTransport:
        """Create a production transport that requires HTTPS and one pinned CA file."""
        if urlparse(base_url).scheme.lower() != "https":
            raise ValueError("Origin Worker base URL must use HTTPS")
        if not pinned_certificate.is_file():
            raise ValueError("Pinned Origin Worker certificate does not exist")
        import httpx

        transport = cls(
            httpx.AsyncClient(
                base_url=base_url,
                verify=str(pinned_certificate),
                timeout=timeout,
            ),
            token,
        )
        transport._owns_client = True
        return transport

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()  # type: ignore[attr-defined]

    async def capabilities(self) -> WorkerCapabilities:
        response = await self.client.get(  # type: ignore[attr-defined]
            "/v1/capabilities", headers=self._headers
        )
        self._raise_for_status(response)
        return WorkerCapabilities.model_validate(response.json(), strict=True)

    async def health(self) -> dict[str, str]:
        response = await self.client.get("/v1/health", headers=self._headers)  # type: ignore[attr-defined]
        self._raise_for_status(response)
        value = response.json()
        if not isinstance(value, dict):
            raise OriginFitError("invalid_worker_response", "Worker health response is invalid.")
        return value

    async def submit(
        self, submission: WorkerSubmission, idempotency_key: str
    ) -> WorkerJob:
        response = await self.client.post(  # type: ignore[attr-defined]
            "/v1/jobs",
            headers={**self._headers, "Idempotency-Key": idempotency_key},
            json=submission.model_dump(mode="json"),
        )
        self._raise_for_status(response)
        return WorkerJob.model_validate(response.json(), strict=True)

    async def status(self, worker_job_id: str) -> WorkerJob:
        response = await self.client.get(  # type: ignore[attr-defined]
            f"/v1/jobs/{worker_job_id}", headers=self._headers
        )
        self._raise_for_status(response)
        return WorkerJob.model_validate(response.json(), strict=True)

    async def cancel(self, worker_job_id: str) -> WorkerJob:
        response = await self.client.post(  # type: ignore[attr-defined]
            f"/v1/jobs/{worker_job_id}/cancel", headers=self._headers
        )
        self._raise_for_status(response)
        return WorkerJob.model_validate(response.json(), strict=True)

    async def download_bundle(self, worker_job_id: str) -> bytes:
        response = await self.client.get(  # type: ignore[attr-defined]
            f"/v1/jobs/{worker_job_id}/bundle", headers=self._headers
        )
        self._raise_for_status(response)
        return bytes(response.content)

    @staticmethod
    def _raise_for_status(response: object) -> None:
        if response.is_success:  # type: ignore[attr-defined]
            return
        try:
            detail = response.json().get("detail", {})  # type: ignore[attr-defined]
            code = detail.get("code", "worker_request_failed")
            message = detail.get("message", "Worker request failed.")
        except (AttributeError, ValueError):
            code = "worker_request_failed"
            message = "Worker request failed."
        raise OriginFitError(str(code), str(message))


class RemoteOriginExecutor:
    def __init__(self, transport: WorkerTransport) -> None:
        self.transport = transport

    async def execute_approved_fit(
        self,
        store: LocalStore,
        dataset_snapshot_id: str,
        approved_fit_recipe_id: str,
        *,
        wait_timeout: float = 1800,
        poll_interval: float = 0.25,
    ) -> RemoteFitOutcome:
        submission = self.prepare_submission(
            store, dataset_snapshot_id, approved_fit_recipe_id
        )
        capabilities = await self.transport.capabilities()
        self._negotiate(capabilities, submission)
        self._audit_negotiation(store, capabilities, submission)
        idempotency_key = "sha256:" + hashlib.sha256(
            f"{dataset_snapshot_id}\0{approved_fit_recipe_id}".encode("utf-8")
        ).hexdigest()
        job = await self.transport.submit(submission, idempotency_key)
        self._record_job(store, job, submission)
        deadline = asyncio.get_running_loop().time() + max(wait_timeout, 0)
        while job.status in ("queued", "running"):
            if asyncio.get_running_loop().time() >= deadline:
                self._record_job(store, job, submission)
                return PendingFitJob(
                    status="pending",
                    worker_job_id=job.worker_job_id,
                    worker_status=job.status,
                    message="Fit Job remains pending at the Worker; it has not failed.",
                )
            await asyncio.sleep(max(poll_interval, 0))
            job = await self.transport.status(job.worker_job_id)
            self._record_job(store, job, submission)
        if job.status != "succeeded":
            raise OriginFitError(
                job.error_code or "remote_fit_failed",
                job.error_message or f"Remote Fit Job ended with status '{job.status}'.",
            )
        bundle = await self.transport.download_bundle(job.worker_job_id)
        return self._verify_and_archive(store, submission, job, bundle)

    @staticmethod
    def prepare_submission(
        store: LocalStore,
        dataset_snapshot_id: str,
        approved_fit_recipe_id: str,
    ) -> WorkerSubmission:
        with store.connect() as connection:
            snapshot = connection.execute(
                """
                SELECT content_hash, metadata_json, summary_json
                FROM dataset_snapshots WHERE id = ?
                """,
                (dataset_snapshot_id,),
            ).fetchone()
            recipe = connection.execute(
                """
                SELECT content_hash, recipe_json FROM approved_fit_recipes
                WHERE id = ?
                """,
                (approved_fit_recipe_id,),
            ).fetchone()
        if snapshot is None:
            raise OriginFitError(
                "not_found", f"Dataset Snapshot '{dataset_snapshot_id}' not found."
            )
        if recipe is None:
            raise OriginFitError(
                "approval_required",
                f"Approved Fit Recipe '{approved_fit_recipe_id}' not found.",
            )
        path = store.objects_dir / snapshot["content_hash"]
        try:
            dataset = path.read_bytes()
        except OSError as error:
            raise OriginFitError(
                "dataset_integrity_error", "Dataset Snapshot content is unavailable."
            ) from error
        if hashlib.sha256(dataset).hexdigest() != snapshot["content_hash"]:
            raise OriginFitError(
                "dataset_integrity_error", "Dataset Snapshot failed local verification."
            )
        return WorkerSubmission(
            transport_schema_version="1.0",
            dataset_snapshot_id=dataset_snapshot_id,
            dataset_content_hash=snapshot["content_hash"],
            dataset_base64=base64.b64encode(dataset).decode("ascii"),
            dataset_metadata=json.loads(snapshot["metadata_json"]),
            dataset_summary=json.loads(snapshot["summary_json"]),
            approved_fit_recipe_id=approved_fit_recipe_id,
            approved_fit_recipe_hash=recipe["content_hash"],
            approved_fit_recipe=json.loads(recipe["recipe_json"]),
        )

    @staticmethod
    def _negotiate(
        capabilities: WorkerCapabilities, submission: WorkerSubmission
    ) -> None:
        recipe = submission.approved_fit_recipe
        specification = recipe["fit_specification"]
        profile = specification["graph_profile"]
        supported_profiles = {
            (item.id, item.version) for item in capabilities.graph_profiles
        }
        registered_templates = {
            (
                item.template_id,
                item.version,
                item.sha256,
                item.graph_profile.id,
                item.graph_profile.version,
            )
            for item in capabilities.graph_templates
        }
        template = specification.get("graph_template", {})
        template_reference = (
            template.get("template_id"),
            template.get("version"),
            template.get("sha256"),
            profile["id"],
            profile["version"],
        )
        if (
            capabilities.transport_schema_version.split(".", 1)[0]
            != TRANSPORT_SCHEMA_VERSION.split(".", 1)[0]
            or specification["schema_version"]
            not in capabilities.fit_specification_schema_versions
            or FIT_RESULT_SCHEMA_VERSION not in capabilities.fit_result_schema_versions
            or MANIFEST_SCHEMA_VERSION not in capabilities.manifest_schema_versions
            or specification["model"]["name"] not in capabilities.models
            or (profile["id"], profile["version"]) not in supported_profiles
            or template_reference not in registered_templates
        ):
            raise OriginFitError(
                "incompatible_worker",
                "Worker capabilities are incompatible with the Approved Fit Recipe.",
            )
        row_count = submission.dataset_summary.get("row_count")
        y_columns = submission.dataset_metadata.get("y_columns")
        try:
            dataset_size = len(
                base64.b64decode(submission.dataset_base64, validate=True)
            )
        except ValueError as error:
            raise OriginFitError(
                "dataset_integrity_error", "Dataset Snapshot upload is invalid."
            ) from error
        if (
            dataset_size > capabilities.max_dataset_bytes
            or (
                capabilities.max_submission_bytes is not None
                and len(submission.model_dump_json().encode("utf-8"))
                > capabilities.max_submission_bytes
            )
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count > capabilities.max_rows
            or not isinstance(y_columns, list)
            or len(y_columns) > capabilities.max_y_series
        ):
            raise OriginFitError(
                "worker_limit_exceeded",
                "Dataset Snapshot exceeds the negotiated Worker limits.",
            )

    @staticmethod
    def _record_job(
        store: LocalStore, job: WorkerJob, submission: WorkerSubmission
    ) -> None:
        now = utc_now()
        with store.connect() as connection:
            connection.execute(
                """
                INSERT INTO worker_job_mappings (
                    worker_job_id, dataset_snapshot_id, approved_fit_recipe_id,
                    status, submitted_at, updated_at, error_code, bundle_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_job_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    error_code = excluded.error_code,
                    bundle_hash = excluded.bundle_hash
                """,
                (
                    job.worker_job_id,
                    submission.dataset_snapshot_id,
                    submission.approved_fit_recipe_id,
                    job.status,
                    job.submitted_at,
                    now,
                    job.error_code,
                    job.bundle_sha256,
                ),
            )
            store.audit(
                connection,
                "worker_job.status_observed",
                job.worker_job_id,
                {"status": job.status, "error_code": job.error_code},
            )

    @staticmethod
    def _audit_negotiation(
        store: LocalStore,
        capabilities: WorkerCapabilities,
        submission: WorkerSubmission,
    ) -> None:
        specification = submission.approved_fit_recipe["fit_specification"]
        with store.connect() as connection:
            store.audit(
                connection,
                "worker.capabilities.negotiated",
                submission.approved_fit_recipe_id,
                {
                    "transport_schema_version": capabilities.transport_schema_version,
                    "fit_specification_schema_version": specification["schema_version"],
                    "fit_result_schema_version": FIT_RESULT_SCHEMA_VERSION,
                    "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                    "model": specification["model"]["name"],
                    "graph_profile": specification["graph_profile"],
                    "graph_template": specification.get("graph_template"),
                },
            )

    @staticmethod
    def _verify_and_archive(
        store: LocalStore,
        submission: WorkerSubmission,
        job: WorkerJob,
        bundle: bytes,
    ) -> ArchivedFitResult:
        bundle_hash = hashlib.sha256(bundle).hexdigest()
        if job.bundle_sha256 != bundle_hash:
            raise OriginFitError(
                "bundle_integrity_error", "Downloaded bundle hash does not match Worker metadata."
            )
        try:
            with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if (
                    len(names) != len(set(names))
                    or set(names) != BUNDLE_FILES
                    or any(info.is_dir() or Path(info.filename).name != info.filename for info in infos)
                ):
                    raise OriginFitError(
                        "bundle_integrity_error",
                        "Fit Result Bundle has missing, extra, duplicate, or unsafe entries.",
                    )
                contents = {name: archive.read(name) for name in names}
        except (OSError, zipfile.BadZipFile, KeyError) as error:
            raise OriginFitError(
                "bundle_integrity_error", "Fit Result Bundle is not a valid archive."
            ) from error
        try:
            manifest = FitResultManifest.model_validate_json(
                contents["manifest.json"], strict=True
            )
        except ValidationError as error:
            raise OriginFitError(
                "bundle_integrity_error", "Fit Result manifest is invalid."
            ) from error
        if set(manifest.files) != BUNDLE_ARTIFACTS:
            raise OriginFitError(
                "bundle_integrity_error", "Fit Result manifest file set is invalid."
            )
        for name, expected in manifest.files.items():
            if hashlib.sha256(contents[name]).hexdigest() != expected:
                raise OriginFitError(
                    "bundle_integrity_error", f"Fit Result artifact '{name}' failed verification."
                )
        recipe = submission.approved_fit_recipe
        specification = recipe["fit_specification"]
        if (
            manifest.worker_job_id != job.worker_job_id
            or manifest.dataset_snapshot.id != submission.dataset_snapshot_id
            or manifest.dataset_snapshot.sha256 != submission.dataset_content_hash
            or manifest.approved_fit_recipe.id != submission.approved_fit_recipe_id
            or manifest.approved_fit_recipe.sha256
            != submission.approved_fit_recipe_hash
            or manifest.fit_specification.id != recipe["fit_specification_id"]
            or manifest.fit_specification.sha256
            != recipe["fit_specification_hash"]
            or manifest.schemas.fit_specification
            != FIT_SPECIFICATION_SCHEMA_VERSION
            or manifest.schemas.fit_result != FIT_RESULT_SCHEMA_VERSION
            or manifest.schemas.manifest != MANIFEST_SCHEMA_VERSION
            or manifest.graph_profile.model_dump() != specification["graph_profile"]
            or manifest.graph_template.model_dump()
            != specification["graph_template"]
        ):
            raise OriginFitError(
                "bundle_integrity_error", "Fit Result manifest provenance is inconsistent."
            )
        try:
            result = FitResult.model_validate_json(contents["result.json"], strict=True)
        except ValidationError as error:
            raise OriginFitError(
                "bundle_integrity_error", "Fit Result document is invalid."
            ) from error
        if (
            result.fit_job_id != job.worker_job_id
            or result.dataset_snapshot_id != submission.dataset_snapshot_id
            or result.approved_fit_recipe_id != submission.approved_fit_recipe_id
        ):
            raise OriginFitError(
                "bundle_integrity_error", "Fit Result provenance is inconsistent."
            )
        archive_id = f"fit-archive:sha256:{bundle_hash}"
        store.put_object(bundle_hash, bundle)
        with store.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO fit_results (
                    id, dataset_snapshot_id, approved_fit_recipe_id,
                    created_at, result_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.fit_result_id,
                    result.dataset_snapshot_id,
                    result.approved_fit_recipe_id,
                    manifest.created_at,
                    result.model_dump_json(),
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO fit_archives (
                    id, dataset_snapshot_id, approved_fit_recipe_id,
                    worker_job_id, fit_result_id, bundle_hash,
                    archived_at, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    archive_id,
                    result.dataset_snapshot_id,
                    result.approved_fit_recipe_id,
                    job.worker_job_id,
                    result.fit_result_id,
                    bundle_hash,
                    utc_now(),
                    manifest.model_dump_json(),
                ),
            )
            connection.execute(
                """
                UPDATE worker_job_mappings
                SET status = 'succeeded', updated_at = ?, bundle_hash = ?,
                    fit_archive_id = ? WHERE worker_job_id = ?
                """,
                (utc_now(), bundle_hash, archive_id, job.worker_job_id),
            )
            store.audit(
                connection,
                "fit_result_bundle.hash_verified",
                job.worker_job_id,
                {"bundle_sha256": bundle_hash, "file_count": len(BUNDLE_FILES)},
            )
            store.audit(
                connection,
                "fit_archive.created",
                archive_id,
                {
                    "worker_job_id": job.worker_job_id,
                    "bundle_sha256": bundle_hash,
                },
            )
        return ArchivedFitResult(
            status="succeeded",
            worker_job_id=job.worker_job_id,
            fit_archive_id=archive_id,
            bundle_sha256=bundle_hash,
            fit_result=result,
        )


def make_remote_execute_approved_fit_tool(
    store: LocalStore, executor: RemoteOriginExecutor
) -> Callable[[str, str], Awaitable[RemoteFitOutcome]]:
    async def model_execute_approved_fit(
        dataset_snapshot_id: str,
        approved_fit_recipe_id: str,
    ) -> RemoteFitOutcome:
        """Execute and archive one stored Dataset Snapshot with one Approved Fit Recipe."""
        try:
            return await executor.execute_approved_fit(
                store, dataset_snapshot_id, approved_fit_recipe_id
            )
        except OriginFitError as error:
            raise ToolError(
                code=error.code,
                message=error.message,
                details={
                    "dataset_snapshot_id": dataset_snapshot_id,
                    "approved_fit_recipe_id": approved_fit_recipe_id,
                },
            ) from error

    model_execute_approved_fit.__name__ = "execute_approved_fit"
    return model_execute_approved_fit
