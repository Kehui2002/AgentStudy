from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import httpx

from origin_fit.datasets import ImportSelection, import_dataset
from origin_fit.execution import (
    DeterministicFakeOriginAdapter,
    FakeOriginAdapter,
    OriginExecutionRequest,
    OriginSeriesResponse,
)
from origin_fit.contracts import WorkerCapabilities
from origin_fit.errors import OriginFitError
from origin_fit.remote import (
    ArchivedFitResult,
    HttpWorkerTransport,
    InProcessWorkerTransport,
    PendingFitJob,
    RemoteOriginExecutor,
)
from origin_fit.specifications import (
    approve_fit_specification,
    inspect_persisted_object,
    propose_fit_specification,
)
from origin_fit.storage import LocalStore
from origin_worker.service import OriginWorker
from origin_worker.service import WorkerError
from origin_worker.api import create_app
from origin_worker.cli import main as origin_worker_main


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_expdec2.csv"


def approved_fixture(store: LocalStore) -> tuple[str, str]:
    snapshot = import_dataset(
        store,
        FIXTURE,
        ImportSelection(
            x="time_s",
            ys=("decay_a", "decay_b", "decay_c"),
            uncertainties={
                "decay_a": "decay_a_error",
                "decay_b": "decay_b_error",
                "decay_c": "decay_c_error",
            },
            units={
                "time_s": "s",
                "decay_a": "dimensionless",
                "decay_b": "dimensionless",
                "decay_c": "dimensionless",
                "decay_a_error": "dimensionless",
                "decay_b_error": "dimensionless",
                "decay_c_error": "dimensionless",
            },
        ),
    )
    specification = propose_fit_specification(
        store,
        snapshot["dataset_snapshot_id"],
        experiment_id="synthetic-expdec2",
        fit_minimum=0,
        fit_maximum=11,
        weighting="instrument",
        initialization="origin-auto",
        graph_profile_id="expdec2-standard",
        graph_profile_version="1.0",
    )
    recipe = approve_fit_specification(
        store, specification["fit_specification_id"]
    )
    return snapshot["dataset_snapshot_id"], recipe["approved_fit_recipe_id"]


def successful_response(
    series_name: str,
    *,
    y0: float,
    a1: float,
    t1: float,
    a2: float,
    t2: float,
) -> OriginSeriesResponse:
    parameters = {"y0": y0, "A1": a1, "t1": t1, "A2": a2, "t2": t2}
    return OriginSeriesResponse(
        series_name=series_name,
        converged=True,
        raw_parameters=parameters,
        standard_errors={name: 0.01 for name in parameters},
        confidence_intervals={
            name: (value - 0.02, value + 0.02)
            for name, value in parameters.items()
        },
        covariance=[
            [1.0 if row == column else 0.0 for column in range(5)]
            for row in range(5)
        ],
        correlations={"A1:t1": 0.2},
        fit_statistics={"reduced_chi_square": 1.02},
        actual_initial_values=parameters,
    )


class RemoteOriginExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_and_archives_a_verified_fixed_result_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            worker = OriginWorker(
                root / "worker",
                FakeOriginAdapter(
                    [
                        successful_response(
                            "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                        ),
                        successful_response(
                            "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                        ),
                        successful_response(
                            "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                        ),
                    ]
                ),
            )
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))

            outcome = await executor.execute_approved_fit(
                store,
                snapshot_id,
                recipe_id,
                wait_timeout=2,
                poll_interval=0.01,
            )

            self.assertIsInstance(outcome, ArchivedFitResult)
            assert isinstance(outcome, ArchivedFitResult)
            self.assertEqual(outcome.status, "succeeded")
            self.assertEqual(outcome.fit_result.fit_job_id, outcome.worker_job_id)
            self.assertTrue(outcome.fit_archive_id.startswith("fit-archive:sha256:"))

            inspected_archive = inspect_persisted_object(
                store, outcome.fit_archive_id
            )
            assert inspected_archive is not None
            self.assertEqual(
                inspected_archive["worker_job_id"], outcome.worker_job_id
            )
            self.assertEqual(
                inspected_archive["fit_result_id"], outcome.fit_result.fit_result_id
            )

            with store.connect() as connection:
                archive = connection.execute(
                    "SELECT bundle_hash, manifest_json FROM fit_archives WHERE id = ?",
                    (outcome.fit_archive_id,),
                ).fetchone()
            assert archive is not None
            bundle_path = store.objects_dir / archive["bundle_hash"]
            with zipfile.ZipFile(bundle_path) as bundle:
                names = set(bundle.namelist())
                self.assertEqual(
                    names,
                    {
                        "result.json",
                        "fitted-data.csv",
                        "residuals.csv",
                        "exclusions.csv",
                        "combined.png",
                        "combined.pdf",
                        "project.opju",
                        "manifest.json",
                    },
                )
                manifest = json.loads(bundle.read("manifest.json"))
            self.assertEqual(manifest["schema_version"], "1.0")
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["dataset_snapshot"]["id"], snapshot_id)
            self.assertEqual(manifest["approved_fit_recipe"]["id"], recipe_id)
            self.assertEqual(
                set(manifest["files"]), names - {"manifest.json"}
            )
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            event_types = [
                event["event_type"] for event in audit["audit_events"]
            ]
            self.assertIn("fit_result_bundle.hash_verified", event_types)
            self.assertIn("fit_archive.created", event_types)
            self.assertIn("worker.capabilities.negotiated", event_types)
            worker_events = [
                event["event_type"] for event in worker.inspect_audit_events()
            ]
            self.assertIn("worker.capabilities.reported", worker_events)
            self.assertIn("fit_job.status_queried", worker_events)

    async def test_reuses_the_same_worker_job_for_an_idempotent_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = FakeOriginAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    successful_response(
                        "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                    ),
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )
            executor = RemoteOriginExecutor(
                InProcessWorkerTransport(OriginWorker(root / "worker", adapter))
            )

            first = await executor.execute_approved_fit(
                store, snapshot_id, recipe_id, wait_timeout=2, poll_interval=0.01
            )
            second = await executor.execute_approved_fit(
                store, snapshot_id, recipe_id, wait_timeout=2, poll_interval=0.01
            )

            assert isinstance(first, ArchivedFitResult)
            assert isinstance(second, ArchivedFitResult)
            self.assertEqual(second.worker_job_id, first.worker_job_id)
            self.assertEqual(second.fit_archive_id, first.fit_archive_id)
            self.assertEqual(len(adapter.requests), 1)

    async def test_wait_timeout_returns_pending_and_can_be_resumed(self) -> None:
        class SlowAdapter(FakeOriginAdapter):
            async def execute(self, request):  # type: ignore[no-untyped-def]
                await asyncio.sleep(0.05)
                return await super().execute(request)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = SlowAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    successful_response(
                        "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                    ),
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )
            executor = RemoteOriginExecutor(
                InProcessWorkerTransport(OriginWorker(root / "worker", adapter))
            )

            pending = await executor.execute_approved_fit(
                store, snapshot_id, recipe_id, wait_timeout=0
            )
            self.assertIsInstance(pending, PendingFitJob)
            assert isinstance(pending, PendingFitJob)
            self.assertIn(pending.worker_status, {"queued", "running"})

            completed = await executor.execute_approved_fit(
                store, snapshot_id, recipe_id, wait_timeout=2, poll_interval=0.01
            )
            self.assertIsInstance(completed, ArchivedFitResult)
            assert isinstance(completed, ArchivedFitResult)
            self.assertEqual(completed.worker_job_id, pending.worker_job_id)

    async def test_worker_restart_fails_running_job_without_replaying_it(self) -> None:
        class BlockingAdapter(FakeOriginAdapter):
            def __init__(self, responses: list[OriginSeriesResponse]) -> None:
                super().__init__(responses)
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def execute(
                self, request: OriginExecutionRequest
            ) -> tuple[OriginSeriesResponse, ...]:
                self.started.set()
                await self.release.wait()
                return await super().execute(request)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = BlockingAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    successful_response(
                        "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                    ),
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )
            worker_state = root / "worker"
            worker = OriginWorker(worker_state, adapter)
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))
            submission = executor.prepare_submission(store, snapshot_id, recipe_id)
            job = worker.submit(submission, "restart-test")
            running = asyncio.create_task(worker.run_queued())
            await adapter.started.wait()

            restarted = OriginWorker(worker_state, FakeOriginAdapter([]))
            recovered_job = restarted.get_job(job.worker_job_id)

            self.assertEqual(recovered_job.status, "failed")
            self.assertEqual(recovered_job.error_code, "worker_restarted")
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            self.assertEqual(len(adapter.requests), 0)

    async def test_worker_restart_preserves_queued_job_and_cancel_is_durable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = FakeOriginAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    successful_response(
                        "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                    ),
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )
            worker_state = root / "worker"
            first_process = OriginWorker(worker_state, adapter)
            executor = RemoteOriginExecutor(InProcessWorkerTransport(first_process))
            submission = executor.prepare_submission(store, snapshot_id, recipe_id)
            job = first_process.submit(submission, "queued-test")

            restarted = OriginWorker(worker_state, adapter)
            self.assertEqual(restarted.get_job(job.worker_job_id).status, "queued")
            cancelled = restarted.cancel(job.worker_job_id)
            self.assertEqual(cancelled.status, "cancelled")

            restarted_again = OriginWorker(worker_state, adapter)
            self.assertEqual(
                restarted_again.get_job(job.worker_job_id).status, "cancelled"
            )
            await restarted_again.run_queued()
            self.assertEqual(adapter.requests, [])

    async def test_api_startup_resumes_a_queued_job_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = FakeOriginAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    successful_response(
                        "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                    ),
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )
            worker_state = root / "worker"
            first_process = OriginWorker(worker_state, adapter)
            executor = RemoteOriginExecutor(InProcessWorkerTransport(first_process))
            submission = executor.prepare_submission(store, snapshot_id, recipe_id)
            submitted = first_process.submit(submission, "startup-resume-test")

            restarted = OriginWorker(worker_state, adapter)
            app = create_app(restarted, bearer_token="test-secret-token")
            async with app.router.lifespan_context(app):
                for _ in range(100):
                    status = restarted.get_job(submitted.worker_job_id)
                    if status.status == "succeeded":
                        break
                    await asyncio.sleep(0.01)

            self.assertEqual(status.status, "succeeded")

    async def test_rejects_incompatible_transport_major_before_submission(self) -> None:
        class IncompatibleTransport:
            async def capabilities(self) -> WorkerCapabilities:
                return WorkerCapabilities(
                    transport_schema_version="2.0",
                    fit_specification_schema_versions=["1.0"],
                    fit_result_schema_versions=["1.0"],
                    manifest_schema_versions=["1.0"],
                    models=["ExpDec2"],
                    graph_profiles=[
                        {"id": "expdec2-standard", "version": "1.0"}
                    ],
                    max_dataset_bytes=100 * 1024 * 1024,
                    max_rows=1_000_000,
                    max_y_series=20,
                )

            async def submit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("incompatible submission reached Worker")

        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            executor = RemoteOriginExecutor(IncompatibleTransport())  # type: ignore[arg-type]

            with self.assertRaises(OriginFitError) as raised:
                await executor.execute_approved_fit(store, snapshot_id, recipe_id)

            self.assertEqual(raised.exception.code, "incompatible_worker")

    async def test_rejects_missing_extra_and_tampered_bundle_artifacts(self) -> None:
        def rewrite_bundle(
            content: bytes, *, remove: str | None = None, extra: bool = False,
            tamper: str | None = None
        ) -> bytes:
            with zipfile.ZipFile(io.BytesIO(content)) as source:
                members = {
                    name: source.read(name)
                    for name in source.namelist()
                    if name != remove
                }
            if extra:
                members["unexpected.txt"] = b"unexpected"
            if tamper is not None:
                members[tamper] += b"tampered"
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
                for name, value in members.items():
                    target.writestr(name, value)
            return output.getvalue()

        transformations = {
            "missing": lambda content: rewrite_bundle(
                content, remove="residuals.csv"
            ),
            "extra": lambda content: rewrite_bundle(content, extra=True),
            "tampered": lambda content: rewrite_bundle(
                content, tamper="result.json"
            ),
        }

        class BundleTransformTransport:
            def __init__(self, delegate, transform):  # type: ignore[no-untyped-def]
                self.delegate = delegate
                self.transform = transform
                self.bundle: bytes | None = None

            async def capabilities(self):  # type: ignore[no-untyped-def]
                return await self.delegate.capabilities()

            async def submit(self, submission, idempotency_key):  # type: ignore[no-untyped-def]
                return await self.delegate.submit(submission, idempotency_key)

            async def status(self, worker_job_id):  # type: ignore[no-untyped-def]
                job = await self.delegate.status(worker_job_id)
                if job.status == "succeeded" and self.bundle is None:
                    original = await self.delegate.download_bundle(worker_job_id)
                    self.bundle = self.transform(original)
                if self.bundle is not None:
                    return job.model_copy(
                        update={
                            "bundle_sha256": hashlib.sha256(self.bundle).hexdigest()
                        }
                    )
                return job

            async def download_bundle(self, worker_job_id):  # type: ignore[no-untyped-def]
                assert self.bundle is not None
                return self.bundle

        for name, transform in transformations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                store = LocalStore(root / "linux")
                snapshot_id, recipe_id = approved_fixture(store)
                adapter = FakeOriginAdapter(
                    [
                        successful_response(
                            "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                        ),
                        successful_response(
                            "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                        ),
                        successful_response(
                            "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                        ),
                    ]
                )
                transport = BundleTransformTransport(
                    InProcessWorkerTransport(OriginWorker(root / "worker", adapter)),
                    transform,
                )
                executor = RemoteOriginExecutor(transport)  # type: ignore[arg-type]

                with self.assertRaises(OriginFitError) as raised:
                    await executor.execute_approved_fit(
                        store,
                        snapshot_id,
                        recipe_id,
                        wait_timeout=2,
                        poll_interval=0.01,
                    )

                self.assertEqual(raised.exception.code, "bundle_integrity_error")
                with store.connect() as connection:
                    count = connection.execute(
                        "SELECT COUNT(*) AS count FROM fit_archives"
                    ).fetchone()
                assert count is not None
                self.assertEqual(count["count"], 0)

    async def test_v1_http_interface_authenticates_and_runs_the_full_protocol(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            worker = OriginWorker(
                root / "worker",
                FakeOriginAdapter(
                    [
                        successful_response(
                            "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                        ),
                        successful_response(
                            "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                        ),
                        successful_response(
                            "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                        ),
                    ]
                ),
            )
            app = create_app(worker, bearer_token="test-secret-token")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://origin-worker.test",
            ) as client:
                unauthorized = await client.get("/v1/health")
                self.assertEqual(unauthorized.status_code, 401)
                self.assertNotIn("test-secret-token", unauthorized.text)

                transport = HttpWorkerTransport(client, "test-secret-token")
                health = await transport.health()
                self.assertEqual(health["status"], "ok")
                executor = RemoteOriginExecutor(transport)
                outcome = await executor.execute_approved_fit(
                    store,
                    snapshot_id,
                    recipe_id,
                    wait_timeout=2,
                    poll_interval=0.01,
                )

            self.assertIsInstance(outcome, ArchivedFitResult)

    async def test_worker_timeout_fails_once_and_terminates_its_adapter(self) -> None:
        class HangingAdapter:
            def __init__(self) -> None:
                self.terminated = 0

            async def execute(
                self, request: OriginExecutionRequest
            ) -> tuple[OriginSeriesResponse, ...]:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            def terminate(self) -> None:
                self.terminated += 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = HangingAdapter()
            worker = OriginWorker(root / "worker", adapter, job_timeout=0.01)
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))
            submission = executor.prepare_submission(store, snapshot_id, recipe_id)
            submitted = worker.submit(submission, "timeout-test")

            await worker.run_queued()
            timed_out = worker.get_job(submitted.worker_job_id)

            self.assertEqual(timed_out.status, "failed")
            self.assertEqual(timed_out.error_code, "worker_timeout")
            self.assertEqual(adapter.terminated, 1)
            await worker.run_queued()
            self.assertEqual(adapter.terminated, 1)

    async def test_running_cancel_terminates_execution_and_remains_failed(self) -> None:
        class CancellableAdapter:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.terminated = 0

            async def execute(
                self, request: OriginExecutionRequest
            ) -> tuple[OriginSeriesResponse, ...]:
                self.started.set()
                await self.release.wait()
                return ()

            def terminate(self) -> None:
                self.terminated += 1
                self.release.set()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = CancellableAdapter()
            worker = OriginWorker(root / "worker", adapter)
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))
            submission = executor.prepare_submission(store, snapshot_id, recipe_id)
            submitted = worker.submit(submission, "cancel-running-test")
            task = asyncio.create_task(worker.run_queued())
            await adapter.started.wait()

            cancelled = worker.cancel(submitted.worker_job_id)
            await task

            self.assertEqual(cancelled.status, "failed")
            self.assertEqual(cancelled.error_code, "cancelled_during_execution")
            self.assertEqual(adapter.terminated, 1)
            self.assertEqual(
                worker.get_job(submitted.worker_job_id).status, "failed"
            )

    async def test_unexpected_worker_failure_is_sanitized_and_not_persisted(
        self,
    ) -> None:
        secret = "secret-token /private/origin/stack.py:42"

        class CrashingAdapter:
            async def execute(
                self, request: OriginExecutionRequest
            ) -> tuple[OriginSeriesResponse, ...]:
                raise RuntimeError(secret)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            worker = OriginWorker(root / "worker", CrashingAdapter())
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))

            with self.assertRaises(OriginFitError) as raised:
                await executor.execute_approved_fit(
                    store,
                    snapshot_id,
                    recipe_id,
                    wait_timeout=2,
                    poll_interval=0.01,
                )

            self.assertEqual(raised.exception.code, "worker_execution_error")
            self.assertNotIn(secret, raised.exception.message)
            self.assertNotIn(secret.encode(), worker.database_path.read_bytes())

    async def test_expired_terminal_workspace_is_removed_but_metadata_remains(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            worker = OriginWorker(
                root / "worker",
                DeterministicFakeOriginAdapter(),
                workspace_retention_days=0,
            )
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))
            outcome = await executor.execute_approved_fit(
                store,
                snapshot_id,
                recipe_id,
                wait_timeout=2,
                poll_interval=0.01,
            )
            assert isinstance(outcome, ArchivedFitResult)

            removed = worker.cleanup_expired_workspaces()

            self.assertEqual(removed, [outcome.worker_job_id])
            self.assertEqual(worker.get_job(outcome.worker_job_id).status, "succeeded")
            with self.assertRaises(WorkerError) as raised:
                worker.get_bundle(outcome.worker_job_id)
            self.assertEqual(raised.exception.code, "bundle_unavailable")

    async def test_api_periodically_cleans_expired_terminal_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            worker = OriginWorker(
                root / "worker",
                DeterministicFakeOriginAdapter(),
                workspace_retention_days=0,
            )
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))
            submission = executor.prepare_submission(store, snapshot_id, recipe_id)
            app = create_app(
                worker,
                bearer_token="test-secret-token",
                cleanup_interval=0.01,
            )

            async with app.router.lifespan_context(app):
                submitted = worker.submit(submission, "periodic-cleanup-test")
                await worker.run_queued()
                for _ in range(100):
                    try:
                        worker.get_bundle(submitted.worker_job_id)
                    except WorkerError as error:
                        self.assertEqual(error.code, "bundle_unavailable")
                        break
                    await asyncio.sleep(0.01)
                else:
                    self.fail("expired workspace was not cleaned periodically")

    def test_worker_cli_rejects_an_address_outside_declared_host_only_network(
        self,
    ) -> None:
        with self.assertRaises(SystemExit) as raised:
            origin_worker_main(
                [
                    "serve",
                    "--state-dir",
                    "/tmp/origin-worker-test-state",
                    "--host",
                    "8.8.8.8",
                    "--host-only-network",
                    "192.168.56.0/24",
                    "--certfile",
                    "/tmp/missing.crt",
                    "--keyfile",
                    "/tmp/missing.key",
                    "--fake-origin",
                ]
            )

        self.assertIn("host-only", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
