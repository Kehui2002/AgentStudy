from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from mini_agent import (
    Agent,
    Model,
    ModelRequest,
    ModelRequestParameters,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from origin_fit.datasets import ImportSelection, import_dataset
from origin_fit.errors import OriginFitError
from origin_fit.cli import main as origin_fit_main
from origin_fit.execution import (
    FakeOriginAdapter,
    OriginSeriesResponse,
    execute_approved_fit,
    make_execute_approved_fit_tool,
)
from origin_fit.specifications import (
    approve_fit_specification,
    inspect_persisted_object,
    propose_fit_specification,
)
from origin_fit.storage import LocalStore


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_expdec2.csv"


class SequenceModel(Model):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.parameters: list[ModelRequestParameters] = []

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.parameters.append(parameters)
        return next(self._responses)


def approved_fixture(
    store: LocalStore,
    *,
    weighting: str = "instrument",
    initialization: str = "origin-auto",
    fit_maximum: float = 11,
) -> tuple[str, str]:
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
    proposed = propose_fit_specification(
        store,
        snapshot["dataset_snapshot_id"],
        experiment_id="synthetic-expdec2",
        fit_minimum=0,
        fit_maximum=fit_maximum,
        weighting=weighting,
        initialization=initialization,
        graph_profile_id="expdec2-standard",
        graph_profile_version="1.0",
        initial_values=(
            {
                name: {
                    "y0": 1.0,
                    "A_fast": 6.0,
                    "t_fast": 1.0,
                    "A_slow": 3.0,
                    "t_slow": 6.0,
                }
                for name in ("decay_a", "decay_b", "decay_c")
            }
            if initialization == "explicit"
            else None
        ),
    )
    approved = approve_fit_specification(store, proposed["fit_specification_id"])
    return snapshot["dataset_snapshot_id"], approved["approved_fit_recipe_id"]


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
        covariance=[[1.0 if row == column else 0.0 for column in range(5)] for row in range(5)],
        correlations={"A1:t1": 0.2},
        fit_statistics={"reduced_chi_square": 1.02},
        actual_initial_values=parameters,
    )


class OriginFitExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_an_approved_independent_fit_and_canonicalizes_components(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            swapped_response = successful_response(
                "decay_b", y0=0.8, a1=3.2, t1=6.0, a2=5.0, t2=1.0
            )
            covariance = swapped_response.covariance
            assert covariance is not None
            covariance[:] = [
                [float(row * 10 + column) for column in range(5)]
                for row in range(5)
            ]
            swapped_response.correlations.clear()
            swapped_response.correlations.update(
                {"A1:t1": 0.2, "A2:t2": 0.3}
            )
            adapter = FakeOriginAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    # Origin returned the slow component under its first labels.
                    swapped_response,
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            self.assertEqual(result.classification, "completed")
            self.assertEqual(result.scientific_status, "not_accepted")
            self.assertTrue(result.fit_job_id.startswith("fit-job:"))
            self.assertEqual(result.model, "ExpDec2")
            self.assertEqual(
                result.fit_range.model_dump(),
                {"minimum": 0.0, "maximum": 11.0, "inclusive": True},
            )
            self.assertEqual(result.weighting, "instrument")
            self.assertEqual(
                [
                    (item.series_name, item.row_number, item.reason)
                    for item in result.exclusions
                ],
                [("decay_b", 5, "missing_y"), ("decay_c", 8, "missing_y")],
            )

            by_series = {
                outcome.series_name: outcome for outcome in result.series_outcomes
            }
            swapped = by_series["decay_b"]
            parameters = swapped.parameters
            assert parameters is not None
            self.assertEqual(
                parameters.model_dump(),
                {
                    "y0": 0.8,
                    "A_fast": 5.0,
                    "t_fast": 1.0,
                    "A_slow": 3.2,
                    "t_slow": 6.0,
                },
            )
            self.assertEqual(
                swapped.raw_origin_parameters,
                {"y0": 0.8, "A1": 3.2, "t1": 6.0, "A2": 5.0, "t2": 1.0},
            )
            self.assertEqual(
                swapped.covariance_parameter_order,
                ["y0", "A_fast", "t_fast", "A_slow", "t_slow"],
            )
            self.assertEqual(
                swapped.covariance,
                [
                    [0.0, 3.0, 4.0, 1.0, 2.0],
                    [30.0, 33.0, 34.0, 31.0, 32.0],
                    [40.0, 43.0, 44.0, 41.0, 42.0],
                    [10.0, 13.0, 14.0, 11.0, 12.0],
                    [20.0, 23.0, 24.0, 21.0, 22.0],
                ],
            )
            self.assertEqual(
                swapped.correlations,
                {"A_slow:t_slow": 0.2, "A_fast:t_fast": 0.3},
            )
            self.assertEqual(
                set(swapped.actual_initial_values),
                {"y0", "A_fast", "t_fast", "A_slow", "t_slow"},
            )
            fast_fraction = swapped.fast_amplitude_fraction
            slow_fraction = swapped.slow_amplitude_fraction
            assert fast_fraction is not None
            assert slow_fraction is not None
            self.assertAlmostEqual(fast_fraction, 5.0 / 8.2)
            self.assertAlmostEqual(slow_fraction, 3.2 / 8.2)
            self.assertNotIn("average_lifetime", result.model_dump_json())
            self.assertEqual(
                [outcome.valid_point_count for outcome in result.series_outcomes],
                [12, 11, 11],
            )
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            event_types = [event["event_type"] for event in audit["audit_events"]]
            self.assertIn("fit_job.started", event_types)
            self.assertIn("fit_job.completed", event_types)

    async def test_model_facing_async_tool_accepts_only_persisted_identifiers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
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
            model = SequenceModel(
                [
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_call_id="fit-1",
                                tool_name="execute_approved_fit",
                                arguments_json=(
                                    '{"dataset_snapshot_id":"'
                                    + snapshot_id
                                    + '","approved_fit_recipe_id":"'
                                    + recipe_id
                                    + '"}'
                                ),
                            )
                        ]
                    ),
                    ModelResponse(parts=[TextPart("拟合结果已返回，尚待人工验收。")]),
                ]
            )

            result = await Agent(
                model, tools=[make_execute_approved_fit_tool(store, adapter)]
            ).run("执行已经批准的拟合")

            self.assertEqual(result.output, "拟合结果已返回，尚待人工验收。")
            schema = model.parameters[0].tool_definitions[0].parameters_json_schema
            self.assertEqual(
                set(schema["properties"]),
                {"dataset_snapshot_id", "approved_fit_recipe_id"},
            )
            self.assertFalse(schema["additionalProperties"])

    async def test_preserves_partial_outcomes_and_marks_quality_issues_for_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            high_error = successful_response(
                "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
            )
            high_error.standard_errors["A1"] = 8.0
            high_error.correlations["A1:t1"] = 0.96
            high_error = replace(high_error, boundary_parameters=("A1",))
            not_converged = replace(
                successful_response(
                    "decay_c", y0=1.5, a1=9.0, t1=1.0, a2=4.5, t2=1.4
                ),
                converged=False,
                covariance_status="singular",
            )
            adapter = FakeOriginAdapter(
                [
                    high_error,
                    OriginSeriesResponse.failed(
                        "decay_b",
                        code="origin_runtime_error",
                        message="Origin could not complete this series.",
                    ),
                    not_converged,
                ]
            )

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            self.assertEqual(result.classification, "review_required")
            by_series = {outcome.series_name: outcome for outcome in result.series_outcomes}
            self.assertEqual(by_series["decay_a"].status, "succeeded")
            self.assertEqual(
                set(by_series["decay_a"].warnings),
                {
                    "parameter_near_boundary",
                    "high_relative_standard_error",
                    "high_parameter_correlation",
                },
            )
            self.assertEqual(by_series["decay_b"].status, "failed")
            self.assertEqual(by_series["decay_b"].error_code, "origin_runtime_error")
            self.assertIsNone(by_series["decay_b"].parameters)
            self.assertEqual(
                set(by_series["decay_c"].warnings),
                {"not_converged", "components_not_separated", "covariance_singular"},
            )
            self.assertEqual(result.scientific_status, "not_accepted")

    async def test_explicit_external_acceptance_creates_a_separate_accepted_fit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
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
            fit_result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            persisted_before = inspect_persisted_object(
                store, fit_result.fit_result_id
            )
            assert persisted_before is not None
            self.assertEqual(
                persisted_before["fit_result"]["scientific_status"], "not_accepted"
            )

            output = io.StringIO()
            exit_code = origin_fit_main(
                [
                    "--state-dir",
                    str(store.state_dir),
                    "accept",
                    fit_result.fit_result_id,
                ],
                stdout=output,
            )
            self.assertEqual(exit_code, 0)
            accepted = json.loads(output.getvalue())

            self.assertTrue(accepted["accepted_fit_id"].startswith("accepted-fit:"))
            persisted_after = inspect_persisted_object(store, fit_result.fit_result_id)
            assert persisted_after is not None
            self.assertEqual(
                persisted_after["fit_result"]["scientific_status"], "not_accepted"
            )
            accepted_object = inspect_persisted_object(
                store, accepted["accepted_fit_id"]
            )
            assert accepted_object is not None
            self.assertEqual(
                accepted_object["accepted_fit"]["fit_result_id"],
                fit_result.fit_result_id,
            )
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            self.assertIn(
                "accepted_fit.accepted",
                [event["event_type"] for event in audit["audit_events"]],
            )

    async def test_applies_one_contract_with_unweighted_explicit_initial_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(
                store, weighting="none", initialization="explicit"
            )
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

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            request = adapter.requests[0]
            self.assertEqual(request.model, "ExpDec2")
            self.assertEqual((request.fit_minimum, request.fit_maximum), (0, 11))
            self.assertEqual(request.weighting, "none")
            self.assertEqual(request.x_unit, "s")
            self.assertEqual(
                request.y_units,
                {
                    "decay_a": "dimensionless",
                    "decay_b": "dimensionless",
                    "decay_c": "dimensionless",
                },
            )
            self.assertEqual(request.initialization["mode"], "explicit")
            values_by_y = request.initialization["values_by_y"]
            assert isinstance(values_by_y, dict)
            self.assertEqual(
                set(values_by_y),
                {"decay_a", "decay_b", "decay_c"},
            )
            self.assertTrue(all(series.uncertainties is None for series in request.series))
            self.assertEqual(
                request.constraints["component_order"], "t_fast < t_slow"
            )
            self.assertEqual(result.initialization, "explicit")
            self.assertEqual(result.weighting, "none")

    async def test_six_to_nine_valid_points_require_review_without_rejection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store, fit_maximum=8)
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

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            self.assertEqual(result.classification, "review_required")
            self.assertEqual(
                [outcome.valid_point_count for outcome in result.series_outcomes],
                [9, 8, 8],
            )
            self.assertTrue(
                all(
                    "low_observation_count" in outcome.warnings
                    for outcome in result.series_outcomes
                )
            )
            outside_range = [
                exclusion
                for exclusion in result.exclusions
                if exclusion.reason == "outside_fit_range"
            ]
            self.assertEqual(len(outside_range), 9)
            self.assertEqual(
                {(item.series_name, item.row_number) for item in outside_range},
                {
                    (series_name, row_number)
                    for series_name in ("decay_a", "decay_b", "decay_c")
                    for row_number in (11, 12, 13)
                },
            )

    async def test_non_finite_origin_values_are_preserved_safely_for_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            non_finite = successful_response(
                "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
            )
            non_finite.raw_parameters["t2"] = float("inf")
            adapter = FakeOriginAdapter(
                [
                    non_finite,
                    successful_response(
                        "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                    ),
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            outcome = result.series_outcomes[0]
            self.assertEqual(result.classification, "review_required")
            self.assertIn("non_finite_value", outcome.warnings)
            self.assertEqual(outcome.raw_origin_parameters["t2"], "Infinity")
            self.assertIsNone(outcome.parameters)
            serialized = result.model_dump_json()
            self.assertNotIn(":Infinity", serialized)
            self.assertEqual(json.loads(serialized)["classification"], "review_required")

    async def test_preflight_is_atomic_and_starts_no_job_for_a_changed_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            object_path = store.objects_dir / snapshot_id.removeprefix("sha256:")
            object_path.chmod(0o644)
            object_path.write_text(
                object_path.read_text(encoding="utf-8").replace(
                    "1,8.117907", "1,8.117908", 1
                ),
                encoding="utf-8",
            )
            adapter = FakeOriginAdapter([])

            with self.assertRaises(OriginFitError) as raised:
                await execute_approved_fit(store, snapshot_id, recipe_id, adapter)

            self.assertEqual(raised.exception.code, "dataset_integrity_error")
            self.assertEqual(adapter.requests, [])
            audit = inspect_persisted_object(store, "audit")
            assert audit is not None
            self.assertNotIn(
                "fit_job.started",
                [event["event_type"] for event in audit["audit_events"]],
            )

    async def test_preflight_rejects_a_damaged_initialization_before_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            with store.connect() as connection:
                row = connection.execute(
                    "SELECT recipe_json FROM approved_fit_recipes WHERE id = ?",
                    (recipe_id,),
                ).fetchone()
                assert row is not None
                recipe = json.loads(row["recipe_json"])
                recipe["fit_specification"]["initialization"] = {"mode": "unknown"}
                connection.execute(
                    "UPDATE approved_fit_recipes SET recipe_json = ? WHERE id = ?",
                    (json.dumps(recipe), recipe_id),
                )
            adapter = FakeOriginAdapter([])

            with self.assertRaises(OriginFitError) as raised:
                await execute_approved_fit(store, snapshot_id, recipe_id, adapter)

            self.assertEqual(raised.exception.code, "invalid_approved_recipe")
            self.assertEqual(adapter.requests, [])

    async def test_preflight_rejects_a_damaged_instrument_weight_mapping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            with store.connect() as connection:
                row = connection.execute(
                    "SELECT recipe_json FROM approved_fit_recipes WHERE id = ?",
                    (recipe_id,),
                ).fetchone()
                assert row is not None
                recipe = json.loads(row["recipe_json"])
                recipe["fit_specification"]["y_series"][1]["uncertainty"] = None
                connection.execute(
                    "UPDATE approved_fit_recipes SET recipe_json = ? WHERE id = ?",
                    (json.dumps(recipe), recipe_id),
                )
            adapter = FakeOriginAdapter([])

            with self.assertRaises(OriginFitError) as raised:
                await execute_approved_fit(store, snapshot_id, recipe_id, adapter)

            self.assertEqual(raised.exception.code, "invalid_approved_recipe")
            self.assertEqual(adapter.requests, [])

    async def test_nonpositive_origin_components_fail_only_the_affected_series(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            invalid = successful_response(
                "decay_b", y0=-0.8, a1=-5.0, t1=1.0, a2=3.2, t2=6.0
            )
            adapter = FakeOriginAdapter(
                [
                    successful_response(
                        "decay_a", y0=-1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    invalid,
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            self.assertEqual(result.classification, "review_required")
            self.assertEqual(result.series_outcomes[0].status, "succeeded")
            invalid_outcome = result.series_outcomes[1]
            self.assertEqual(invalid_outcome.status, "failed")
            self.assertEqual(
                invalid_outcome.error_code, "origin_constraint_violation"
            )
            self.assertEqual(invalid_outcome.raw_origin_parameters["A1"], -5.0)
            self.assertEqual(result.series_outcomes[2].status, "succeeded")

    async def test_missing_actual_initial_values_fail_only_the_affected_series(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            incomplete = replace(
                successful_response(
                    "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                ),
                actual_initial_values={},
            )
            adapter = FakeOriginAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    incomplete,
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            self.assertEqual(result.classification, "review_required")
            self.assertEqual(result.series_outcomes[1].status, "failed")
            self.assertEqual(
                result.series_outcomes[1].error_code, "origin_result_incomplete"
            )

    async def test_nonconverged_response_can_omit_uncertainty_statistics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory) / "state")
            snapshot_id, recipe_id = approved_fixture(store)
            not_converged = replace(
                successful_response(
                    "decay_b", y0=0.8, a1=5.0, t1=1.0, a2=3.2, t2=6.0
                ),
                converged=False,
                standard_errors={},
                confidence_intervals={},
                covariance=None,
                covariance_status="unavailable",
            )
            adapter = FakeOriginAdapter(
                [
                    successful_response(
                        "decay_a", y0=1.0, a1=7.0, t1=1.5, a2=4.0, t2=8.0
                    ),
                    not_converged,
                    successful_response(
                        "decay_c", y0=1.5, a1=9.0, t1=0.8, a2=4.5, t2=5.0
                    ),
                ]
            )

            result = await execute_approved_fit(
                store, snapshot_id, recipe_id, adapter
            )

            outcome = result.series_outcomes[1]
            self.assertEqual(outcome.status, "succeeded")
            self.assertEqual(
                set(outcome.warnings),
                {
                    "not_converged",
                    "covariance_unavailable",
                    "standard_errors_unavailable",
                    "confidence_intervals_unavailable",
                },
            )
            self.assertIsNone(outcome.standard_errors)
            self.assertEqual(outcome.confidence_intervals, {})


if __name__ == "__main__":
    unittest.main()
