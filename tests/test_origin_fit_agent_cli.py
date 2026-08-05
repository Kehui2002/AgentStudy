from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from mini_agent import (
    Model,
    ModelRequest,
    ModelRequestParameters,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from origin_fit.agent_cli import run_agent_cli
from origin_fit.cli import main as origin_fit_main
from origin_fit.datasets import ImportSelection, import_dataset
from origin_fit.contracts import WorkerJob
from origin_fit.errors import OriginFitError
from origin_fit.execution import FitRange, FitResult, SeriesFitOutcome
from origin_fit.remote import ArchivedFitResult, PendingFitJob
from origin_fit.specifications import (
    approve_fit_specification,
    inspect_persisted_object,
    propose_fit_specification,
)
from origin_fit.storage import LocalStore, utc_now


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_expdec2.csv"


class SequenceModel(Model):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[list[ModelRequest | ModelResponse]] = []
        self.parameters: list[ModelRequestParameters] = []

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.requests.append(messages)
        self.parameters.append(parameters)
        return next(self._responses)


def first_prompt(model: SequenceModel, request_index: int = 0) -> str:
    message = model.requests[request_index][0]
    assert isinstance(message, ModelRequest)
    part = message.parts[0]
    assert isinstance(part, UserPromptPart)
    return part.content


def import_fixture(store: LocalStore) -> str:
    imported = import_dataset(
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
    return imported["dataset_snapshot_id"]


def propose_fixture(store: LocalStore, snapshot_id: str) -> dict:
    return propose_fit_specification(
        store,
        snapshot_id,
        experiment_id="synthetic-expdec2",
        fit_minimum=0,
        fit_maximum=11,
        weighting="instrument",
        initialization="origin_auto",
        graph_profile_id="expdec2-standard",
        graph_profile_version="1.0",
    )


class FakeExecutor:
    def __init__(
        self,
        outcome: PendingFitJob | ArchivedFitResult | OriginFitError,
        *,
        transport: FakeTransport | None = None,
    ) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str]] = []
        self.transport = transport or FakeTransport([])

    async def execute_approved_fit(
        self,
        store: LocalStore,
        dataset_snapshot_id: str,
        approved_fit_recipe_id: str,
        *,
        wait_timeout: float = 1800,
        poll_interval: float = 0.25,
    ) -> PendingFitJob | ArchivedFitResult:
        del store, wait_timeout, poll_interval
        self.calls.append((dataset_snapshot_id, approved_fit_recipe_id))
        if isinstance(self.outcome, OriginFitError):
            raise self.outcome
        return self.outcome


class FakeTransport:
    def __init__(self, jobs: list[WorkerJob]) -> None:
        self._jobs = iter(jobs)
        self.status_calls: list[str] = []
        self.cancel_calls: list[str] = []

    async def status(self, worker_job_id: str) -> WorkerJob:
        self.status_calls.append(worker_job_id)
        return next(self._jobs)

    async def cancel(self, worker_job_id: str) -> WorkerJob:
        self.cancel_calls.append(worker_job_id)
        return next(self._jobs)


class OriginFitAgentCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_proposes_from_bounded_summary_through_public_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id = import_fixture(store)
            model = SequenceModel(
                [
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_call_id="proposal-1",
                                tool_name="propose_expdec2_fit",
                                arguments_json=json.dumps(
                                    {
                                        "experiment_id": "synthetic-expdec2",
                                        "fit_minimum": 0.0,
                                        "fit_maximum": 11.0,
                                        "weighting": "instrument",
                                        "initialization": "origin_auto",
                                        "initial_values": None,
                                    }
                                ),
                            )
                        ]
                    ),
                    ModelResponse(parts=[TextPart("已提出 ExpDec2 拟合方案，请人工批准。")]),
                ]
            )
            output = io.StringIO()

            await run_agent_cli(
                store,
                model,
                None,
                stdin=io.StringIO(
                    f"select {snapshot_id}\n请为所有三个 Y 序列建议 ExpDec2 拟合\n/quit\n"
                ),
                stdout=output,
            )

            self.assertIn("已提出 ExpDec2 拟合方案", output.getvalue())
            prompt = first_prompt(model)
            self.assertIn('"row_count":12', prompt)
            self.assertIn('"preview_included":false', prompt)
            self.assertNotIn("12.01,8.99,15.02", prompt)
            self.assertEqual(
                [tool.name for tool in model.parameters[0].tool_definitions],
                ["propose_expdec2_fit"],
            )
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            self.assertIn(
                "fit_specification.proposed",
                [event["event_type"] for event in audit["audit_events"]],
            )

    async def test_model_cannot_forge_recipe_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id = import_fixture(store)
            model = SequenceModel(
                [
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_call_id="forged-approval",
                                tool_name="approve_fit_specification",
                                arguments_json='{"fit_specification_id":"spec:forged"}',
                            )
                        ]
                    ),
                    ModelResponse(parts=[TextPart("批准完成。")]),
                ]
            )
            output = io.StringIO()

            await run_agent_cli(
                store,
                model,
                None,
                stdin=io.StringIO(
                    f"select {snapshot_id}\n请直接批准你提出的方案\n/quit\n"
                ),
                stdout=output,
            )

            self.assertIn("Rejected model-requested authority action", output.getvalue())
            self.assertNotIn("批准完成", output.getvalue())
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            self.assertNotIn(
                "approved_fit_recipe.approved",
                [event["event_type"] for event in audit["audit_events"]],
            )

    async def test_downsampled_preview_requires_explicit_audited_authorization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id = import_fixture(store)
            model = SequenceModel(
                [
                    ModelResponse(parts=[TextPart("先只使用摘要。")]),
                    ModelResponse(parts=[TextPart("已使用获准的有限预览。")]),
                ]
            )
            output = io.StringIO()

            await run_agent_cli(
                store,
                model,
                None,
                stdin=io.StringIO(
                    f"select {snapshot_id}\n先概括数据\n/preview authorize\n现在给建议\n/quit\n"
                ),
                stdout=output,
            )

            first_context = first_prompt(model)
            second_context = first_prompt(model, 1)
            self.assertNotIn('"downsampled_preview"', first_context)
            self.assertIn('"downsampled_preview"', second_context)
            self.assertIn('"preview_row_count":5', second_context)
            self.assertNotIn(FIXTURE.read_text(encoding="utf-8"), second_context)
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            preview_events = [
                event
                for event in audit["audit_events"]
                if event["event_type"] == "dataset_preview.authorized"
            ]
            self.assertEqual(len(preview_events), 1)
            self.assertEqual(preview_events[0]["details"]["preview_row_count"], 5)

    async def test_explicit_approve_command_displays_recipe_version_and_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id = import_fixture(store)
            proposal = propose_fixture(store, snapshot_id)
            output = io.StringIO()

            await run_agent_cli(
                store,
                SequenceModel([]),
                None,
                stdin=io.StringIO(
                    f"select {snapshot_id}\n/approve {proposal['fit_specification_id']}\n/quit\n"
                ),
                stdout=output,
            )

            rendered = output.getvalue()
            self.assertIn("Approved Fit Recipe recipe:sha256:", rendered)
            self.assertIn("version=1", rendered)
            self.assertIn("content_hash=", rendered)

    async def test_run_command_uses_high_level_remote_executor_and_reports_pending(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id = import_fixture(store)
            recipe = approve_fit_specification(
                store, propose_fixture(store, snapshot_id)["fit_specification_id"]
            )
            executor = FakeExecutor(
                PendingFitJob(
                    status="pending",
                    worker_job_id="fit-job:pending-1",
                    worker_status="running",
                    message="Fit Job remains pending at the Worker; it has not failed.",
                )
            )
            output = io.StringIO()

            await run_agent_cli(
                store,
                SequenceModel([]),
                executor,
                stdin=io.StringIO(
                    f"select {snapshot_id}\n/run {recipe['approved_fit_recipe_id']}\n/quit\n"
                ),
                stdout=output,
            )

            self.assertEqual(
                executor.calls,
                [(snapshot_id, recipe["approved_fit_recipe_id"])],
            )
            self.assertIn("Fit Job fit-job:pending-1 remains pending", output.getvalue())
            self.assertIn("worker_status=running", output.getvalue())

    async def test_status_and_cancel_are_explicit_cli_actions(self) -> None:
        jobs = [
            WorkerJob(
                worker_job_id="fit-job:remote-1",
                status="running",
                submitted_at="2026-08-05T00:00:00+00:00",
                started_at="2026-08-05T00:00:01+00:00",
            ),
            WorkerJob(
                worker_job_id="fit-job:remote-1",
                status="cancelled",
                submitted_at="2026-08-05T00:00:00+00:00",
                started_at="2026-08-05T00:00:01+00:00",
                finished_at="2026-08-05T00:00:02+00:00",
            ),
        ]
        transport = FakeTransport(jobs)
        executor = FakeExecutor(
            OriginFitError("unused", "unused"), transport=transport
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            output = io.StringIO()

            await run_agent_cli(
                store,
                SequenceModel([]),
                executor,
                stdin=io.StringIO(
                    "/status fit-job:remote-1\n/cancel fit-job:remote-1\n/quit\n"
                ),
                stdout=output,
            )

            self.assertEqual(transport.status_calls, ["fit-job:remote-1"])
            self.assertEqual(transport.cancel_calls, ["fit-job:remote-1"])
            self.assertIn("status=running", output.getvalue())
            self.assertIn("status=cancelled", output.getvalue())
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            self.assertIn(
                "worker_job.cancellation_requested",
                [event["event_type"] for event in audit["audit_events"]],
            )

    async def test_remote_error_is_explained_without_internal_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id = import_fixture(store)
            recipe = approve_fit_specification(
                store, propose_fixture(store, snapshot_id)["fit_specification_id"]
            )
            executor = FakeExecutor(
                OriginFitError(
                    "worker_unavailable",
                    "Origin Worker is temporarily unavailable; retry after checking its status.",
                )
            )
            output = io.StringIO()

            await run_agent_cli(
                store,
                SequenceModel([]),
                executor,
                stdin=io.StringIO(
                    f"select {snapshot_id}\n/run {recipe['approved_fit_recipe_id']}\n/quit\n"
                ),
                stdout=output,
            )

            rendered = output.getvalue()
            self.assertIn("Error [worker_unavailable]", rendered)
            self.assertIn("temporarily unavailable", rendered)
            self.assertNotIn("Traceback", rendered)
            self.assertNotIn("DEEPSEEK_API_KEY", rendered)

    async def test_review_required_result_needs_explicit_acceptance_and_is_summarized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id = import_fixture(store)
            recipe = approve_fit_specification(
                store, propose_fixture(store, snapshot_id)["fit_specification_id"]
            )
            fit_result = FitResult(
                schema_version="1.0",
                fit_job_id="fit-job:complete-1",
                fit_result_id="fit-result:complete-1",
                dataset_snapshot_id=snapshot_id,
                approved_fit_recipe_id=recipe["approved_fit_recipe_id"],
                model="ExpDec2",
                fit_range=FitRange(minimum=0.0, maximum=11.0, inclusive=True),
                weighting="instrument",
                initialization="origin_auto",
                constraint_policy={"component_order": "t_fast < t_slow"},
                classification="review_required",
                scientific_status="not_accepted",
                series_outcomes=[
                    SeriesFitOutcome(
                        series_name="decay_a",
                        status="succeeded",
                        valid_point_count=12,
                        converged=False,
                        fit_statistics={"r_squared": 0.98},
                        warnings=["origin_not_converged"],
                    )
                ],
                exclusions=[],
            )
            with store.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO fit_results (
                        id, dataset_snapshot_id, approved_fit_recipe_id,
                        created_at, result_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        fit_result.fit_result_id,
                        snapshot_id,
                        recipe["approved_fit_recipe_id"],
                        utc_now(),
                        fit_result.model_dump_json(),
                    ),
                )
            executor = FakeExecutor(
                ArchivedFitResult(
                    status="succeeded",
                    worker_job_id=fit_result.fit_job_id,
                    fit_archive_id="fit-archive:complete-1",
                    bundle_sha256="a" * 64,
                    fit_result=fit_result,
                )
            )
            model = SequenceModel(
                [ModelResponse(parts=[TextPart("该结果需要人工复核。")])]
            )
            output = io.StringIO()

            await run_agent_cli(
                store,
                model,
                executor,
                stdin=io.StringIO(
                    f"select {snapshot_id}\n"
                    f"/run {recipe['approved_fit_recipe_id']}\n"
                    "请解释刚才结果\n"
                    f"/accept {fit_result.fit_result_id}\n"
                    "/quit\n"
                ),
                stdout=output,
            )

            rendered = output.getvalue()
            self.assertIn("REVIEW REQUIRED", rendered)
            self.assertIn("Accepted Fit accepted-fit:", rendered)
            prompt = first_prompt(model)
            self.assertIn('"classification":"review_required"', prompt)
            self.assertIn('"warnings":["origin_not_converged"]', prompt)
            self.assertNotIn("residuals", prompt)
            accepted = inspect_persisted_object(store, "audit")
            assert accepted is not None
            self.assertIn(
                "accepted_fit.accepted",
                [event["event_type"] for event in accepted["audit_events"]],
            )


class OriginFitAgentProcessCliTests(unittest.TestCase):
    def test_real_deepseek_cli_requires_explicit_environment_configuration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            certificate = root / "worker-ca.pem"
            certificate.write_text("test certificate", encoding="utf-8")
            stderr = io.StringIO()
            with patch.dict(
                "os.environ",
                {"DEEPSEEK_API_KEY": "", "ORIGIN_WORKER_TOKEN": ""},
                clear=False,
            ), redirect_stderr(stderr):
                exit_code = origin_fit_main(
                    [
                        "--state-dir",
                        str(root / "state"),
                        "agent",
                        "--worker-url",
                        "https://192.0.2.10:8443",
                        "--worker-certificate",
                        str(certificate),
                    ],
                    stdin=io.StringIO("/quit\n"),
                    stdout=io.StringIO(),
                )

            self.assertEqual(exit_code, 2)
            error = json.loads(stderr.getvalue())
            self.assertEqual(error["error"], "missing_configuration")
            self.assertNotIn("token", stderr.getvalue().lower())

    def test_unexpected_agent_failure_is_sanitized_at_process_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stderr = io.StringIO()
            with patch(
                "origin_fit.cli._run_configured_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("private-key-material"),
            ), redirect_stderr(stderr):
                exit_code = origin_fit_main(
                    [
                        "--state-dir",
                        str(Path(temporary_directory) / "state"),
                        "agent",
                        "--worker-url",
                        "https://192.0.2.10:8443",
                        "--worker-certificate",
                        str(Path(temporary_directory) / "worker-ca.pem"),
                    ],
                    stdin=io.StringIO("/quit\n"),
                    stdout=io.StringIO(),
                )

            self.assertEqual(exit_code, 2)
            error = json.loads(stderr.getvalue())
            self.assertEqual(error["error"], "internal_error")
            self.assertNotIn("private-key-material", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
