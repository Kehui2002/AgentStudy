from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from origin_fit.execution import DeterministicFakeOriginAdapter
from origin_fit.errors import OriginFitError
from origin_fit.remote import InProcessWorkerTransport, RemoteOriginExecutor
from origin_fit.storage import LocalStore
from origin_worker.api import create_app
from origin_worker.cli import main as worker_main
from origin_worker.service import OriginWorker, WorkerError
from tests.test_origin_remote import approved_fixture


AUTHORIZATION = {"Authorization": f"Bearer {'t' * 32}"}


def replace_submission_dataset(submission, dataset: bytes):  # type: ignore[no-untyped-def]
    content_hash = hashlib.sha256(dataset).hexdigest()
    snapshot_id = f"sha256:{content_hash}"
    recipe = deepcopy(submission.approved_fit_recipe)
    specification = recipe["fit_specification"]
    specification["dataset_snapshot_id"] = snapshot_id
    specification["dataset_content_hash"] = content_hash
    canonical_specification = json.dumps(
        specification,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    specification_hash = hashlib.sha256(canonical_specification).hexdigest()
    recipe["fit_specification_id"] = f"spec:sha256:{specification_hash}"
    recipe["fit_specification_hash"] = specification_hash
    canonical_recipe = json.dumps(
        recipe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    recipe_hash = hashlib.sha256(canonical_recipe).hexdigest()
    return submission.model_copy(
        update={
            "dataset_snapshot_id": snapshot_id,
            "dataset_content_hash": content_hash,
            "dataset_base64": base64.b64encode(dataset).decode("ascii"),
            "approved_fit_recipe_id": f"recipe:sha256:{recipe_hash}",
            "approved_fit_recipe_hash": recipe_hash,
            "approved_fit_recipe": recipe,
        }
    )


class OriginWorkerHardeningTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_upload_is_rejected_before_an_oversized_body_is_parsed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = OriginWorker(
                Path(temporary_directory) / "worker",
                DeterministicFakeOriginAdapter(),
            )
            app = create_app(
                worker,
                bearer_token="t" * 32,
                max_upload_bytes=64,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://origin-worker.test",
            ) as client:
                response = await client.post(
                    "/v1/jobs",
                    headers={
                        **AUTHORIZATION,
                        "Idempotency-Key": "oversized-upload",
                        "Content-Type": "application/json",
                    },
                    content=b"{" + (b"x" * 64) + b"}",
                )

            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json()["detail"]["code"], "upload_too_large")
            self.assertEqual(tuple(worker.jobs_dir.iterdir()), ())

    def test_worker_rejects_a_workspace_root_symlinked_outside_its_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state_dir = root / "worker"
            outside = root / "outside"
            state_dir.mkdir()
            outside.mkdir()
            (state_dir / "jobs").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(WorkerError) as raised:
                OriginWorker(state_dir, DeterministicFakeOriginAdapter())

            self.assertEqual(raised.exception.code, "invalid_workspace_root")

    def test_worker_counts_parsed_rows_instead_of_trusting_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            worker = OriginWorker(
                root / "worker",
                DeterministicFakeOriginAdapter(),
                max_rows=12,
            )
            submission = RemoteOriginExecutor(
                InProcessWorkerTransport(worker)
            ).prepare_submission(store, snapshot_id, recipe_id)
            dataset = base64.b64decode(submission.dataset_base64)
            extra_row = dataset.splitlines()[-1] + b"\n"
            tampered = replace_submission_dataset(submission, dataset + extra_row)

            with self.assertRaises(WorkerError) as raised:
                worker.submit(tampered, "parsed-row-limit")

            self.assertEqual(raised.exception.code, "dataset_too_large")
            self.assertEqual(tuple(worker.jobs_dir.iterdir()), ())

    async def test_linux_rejects_worker_limits_before_uploading_a_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            limited_worker = OriginWorker(
                root / "worker",
                DeterministicFakeOriginAdapter(),
                max_dataset_bytes=1,
                max_rows=1,
                max_y_series=1,
            )

            class UploadMustNotStart:
                async def capabilities(self):  # type: ignore[no-untyped-def]
                    return limited_worker.capabilities()

                async def submit(self, submission, idempotency_key):  # type: ignore[no-untyped-def]
                    raise AssertionError("oversized Dataset Snapshot was uploaded")

            executor = RemoteOriginExecutor(UploadMustNotStart())  # type: ignore[arg-type]

            with self.assertRaises(OriginFitError) as raised:
                await executor.execute_approved_fit(store, snapshot_id, recipe_id)

            self.assertEqual(raised.exception.code, "worker_limit_exceeded")

    async def test_concurrent_queue_drains_through_only_one_origin_execution(self) -> None:
        class MeasuringAdapter:
            def __init__(self) -> None:
                self.active = 0
                self.maximum_active = 0

            async def execute(self, request):  # type: ignore[no-untyped-def]
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                try:
                    await asyncio.sleep(0.01)
                    return ()
                finally:
                    self.active -= 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = MeasuringAdapter()
            worker = OriginWorker(root / "worker", adapter)
            submission = RemoteOriginExecutor(
                InProcessWorkerTransport(worker)
            ).prepare_submission(store, snapshot_id, recipe_id)
            jobs = [
                worker.submit(submission, "concurrent-a"),
                worker.submit(submission, "concurrent-b"),
            ]

            await asyncio.gather(worker.run_queued(), worker.run_queued())

            self.assertEqual(adapter.maximum_active, 1)
            self.assertEqual(
                [worker.get_job(job.worker_job_id).status for job in jobs],
                ["failed", "failed"],
            )

    async def test_failed_execution_is_terminated_before_the_next_job(self) -> None:
        class DirtyAfterCrashAdapter:
            def __init__(self) -> None:
                self.execution_count = 0
                self.terminated = 0
                self.delegate = DeterministicFakeOriginAdapter()

            async def execute(self, request):  # type: ignore[no-untyped-def]
                self.execution_count += 1
                if self.execution_count == 1:
                    raise RuntimeError("private Origin crash details")
                if self.terminated != 1:
                    raise RuntimeError("dirty Origin instance was reused")
                return await self.delegate.execute(request)

            def terminate(self) -> None:
                self.terminated += 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = DirtyAfterCrashAdapter()
            worker = OriginWorker(root / "worker", adapter)
            submission = RemoteOriginExecutor(
                InProcessWorkerTransport(worker)
            ).prepare_submission(store, snapshot_id, recipe_id)
            failed = worker.submit(submission, "crash-first")
            following = worker.submit(submission, "clean-second")

            await worker.run_queued()

            self.assertEqual(worker.get_job(failed.worker_job_id).status, "failed")
            self.assertEqual(worker.get_job(following.worker_job_id).status, "succeeded")
            self.assertEqual(adapter.terminated, 1)
            audit = json.dumps(worker.inspect_audit_events(), sort_keys=True)
            self.assertNotIn("private Origin crash details", audit)
            self.assertNotIn(
                b"private Origin crash details", worker.database_path.read_bytes()
            )

    def test_serve_preflight_rejects_an_unusable_work_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            certfile = root / "worker.crt"
            keyfile = root / "worker.key"
            state_file = root / "not-a-directory"
            certfile.write_text("test certificate", encoding="utf-8")
            keyfile.write_text("test key", encoding="utf-8")
            state_file.write_text("not a directory", encoding="utf-8")

            with patch.dict("os.environ", {"ORIGIN_WORKER_TOKEN": "t" * 32}):
                with self.assertRaises(SystemExit) as raised:
                    worker_main(
                        [
                            "serve",
                            "--state-dir",
                            str(state_file),
                            "--host",
                            "192.168.56.1",
                            "--host-only-network",
                            "192.168.56.0/24",
                            "--certfile",
                            str(certfile),
                            "--keyfile",
                            str(keyfile),
                            "--fake-origin",
                        ]
                    )

            self.assertIn("state directory", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
