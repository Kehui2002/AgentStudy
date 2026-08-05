from __future__ import annotations

import csv
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from typing import Literal, Protocol
import uuid

from pydantic import BaseModel, ConfigDict, Field

from mini_agent import ToolError

from .errors import OriginFitError
from .storage import LocalStore, utc_now


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


JsonNumber = float | Literal["NaN", "Infinity", "-Infinity"]


class FitRange(_StrictModel):
    minimum: float
    maximum: float
    inclusive: bool


class ExpDec2Parameters(_StrictModel):
    y0: float
    A_fast: float
    t_fast: float
    A_slow: float
    t_slow: float


class ConfidenceInterval(_StrictModel):
    lower: JsonNumber
    upper: JsonNumber


class ExclusionRecord(_StrictModel):
    series_name: str
    row_number: int
    reason: Literal["missing_y", "outside_fit_range"]


class SeriesFitOutcome(_StrictModel):
    series_name: str
    status: Literal["succeeded", "failed"]
    valid_point_count: int
    converged: bool
    parameters: ExpDec2Parameters | None = None
    raw_origin_parameters: dict[str, JsonNumber] = Field(default_factory=dict)
    standard_errors: ExpDec2Parameters | None = None
    confidence_intervals: dict[str, ConfidenceInterval] = Field(default_factory=dict)
    covariance: list[list[JsonNumber]] | None = None
    covariance_parameter_order: list[
        Literal["y0", "A_fast", "t_fast", "A_slow", "t_slow"]
    ] = Field(default_factory=list)
    correlations: dict[str, JsonNumber] = Field(default_factory=dict)
    fit_statistics: dict[str, JsonNumber] = Field(default_factory=dict)
    actual_initial_values: dict[str, JsonNumber] = Field(default_factory=dict)
    raw_origin_initial_values: dict[str, JsonNumber] = Field(default_factory=dict)
    fast_amplitude_fraction: float | None = None
    slow_amplitude_fraction: float | None = None
    warnings: list[str]
    error_code: str | None = None
    error_message: str | None = None


class FitResult(_StrictModel):
    schema_version: Literal["1.0"]
    fit_job_id: str
    fit_result_id: str
    dataset_snapshot_id: str
    approved_fit_recipe_id: str
    model: Literal["ExpDec2"]
    fit_range: FitRange
    weighting: Literal["none", "instrument"]
    initialization: Literal["origin_auto", "explicit"]
    constraint_policy: dict[str, object]
    classification: Literal["completed", "review_required"]
    scientific_status: Literal["not_accepted"]
    series_outcomes: list[SeriesFitOutcome]
    exclusions: list[ExclusionRecord]


@dataclass(frozen=True, slots=True)
class OriginSeriesInput:
    series_name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    uncertainties: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class OriginExecutionRequest:
    model: str
    fit_minimum: float
    fit_maximum: float
    constraints: dict[str, object]
    weighting: Literal["none", "instrument"]
    initialization: dict[str, object]
    series: tuple[OriginSeriesInput, ...]


@dataclass(frozen=True, slots=True)
class OriginSeriesResponse:
    series_name: str
    status: Literal["succeeded", "failed"] = "succeeded"
    converged: bool = False
    raw_parameters: dict[str, float] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    covariance: list[list[float]] | None = None
    correlations: dict[str, float] = field(default_factory=dict)
    fit_statistics: dict[str, float] = field(default_factory=dict)
    actual_initial_values: dict[str, float] = field(default_factory=dict)
    boundary_parameters: tuple[str, ...] = field(default_factory=tuple)
    covariance_status: Literal["available", "unavailable", "singular"] = "available"
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def failed(
        cls, series_name: str, *, code: str, message: str
    ) -> OriginSeriesResponse:
        return cls(
            series_name=series_name,
            status="failed",
            error_code=code,
            error_message=message,
        )


class OriginAdapter(Protocol):
    async def execute(
        self, request: OriginExecutionRequest
    ) -> tuple[OriginSeriesResponse, ...]: ...


class FakeOriginAdapter:
    """Deterministic Origin execution seam for Linux tests."""

    adapter_name = "fake-origin-adapter/1.0"
    originpro_version = "fake-originpro-2025"

    def __init__(self, responses: list[OriginSeriesResponse]) -> None:
        self._responses = tuple(responses)
        self.requests: list[OriginExecutionRequest] = []

    async def execute(
        self, request: OriginExecutionRequest
    ) -> tuple[OriginSeriesResponse, ...]:
        self.requests.append(request)
        return self._responses


class DeterministicFakeOriginAdapter:
    """Data-derived Fake Adapter for demos that never claims real Origin coverage."""

    adapter_name = "deterministic-fake-origin-adapter/1.0"
    originpro_version = "fake-originpro-2025"

    def __init__(self) -> None:
        self.requests: list[OriginExecutionRequest] = []

    async def execute(
        self, request: OriginExecutionRequest
    ) -> tuple[OriginSeriesResponse, ...]:
        self.requests.append(request)
        responses: list[OriginSeriesResponse] = []
        span = request.fit_maximum - request.fit_minimum
        for series in request.series:
            baseline = min(series.y)
            amplitude = max(series.y) - baseline
            parameters = {
                "y0": baseline,
                "A1": max(amplitude * 0.6, 1e-12),
                "t1": max(span / 6, 1e-12),
                "A2": max(amplitude * 0.4, 1e-12),
                "t2": max(span / 2, 2e-12),
            }
            errors = {
                name: max(abs(value) * 0.01, 1e-12)
                for name, value in parameters.items()
            }
            responses.append(
                OriginSeriesResponse(
                    series_name=series.series_name,
                    converged=True,
                    raw_parameters=parameters,
                    standard_errors=errors,
                    confidence_intervals={
                        name: (value - 1.96 * errors[name], value + 1.96 * errors[name])
                        for name, value in parameters.items()
                    },
                    covariance=[
                        [1.0 if row == column else 0.0 for column in range(5)]
                        for row in range(5)
                    ],
                    correlations={},
                    fit_statistics={"fake_objective": 0.0},
                    actual_initial_values=parameters,
                )
            )
        return tuple(responses)


def load_approved_fit_execution_request(
    store: LocalStore,
    snapshot_id: str,
    recipe_id: str,
) -> tuple[OriginExecutionRequest, list[ExclusionRecord], dict[str, int]]:
    with store.connect() as connection:
        snapshot = connection.execute(
            "SELECT content_hash, metadata_json FROM dataset_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        recipe_row = connection.execute(
            "SELECT recipe_json FROM approved_fit_recipes WHERE id = ?",
            (recipe_id,),
        ).fetchone()
    if snapshot is None:
        raise OriginFitError("not_found", f"Dataset Snapshot '{snapshot_id}' not found.")
    if recipe_row is None:
        raise OriginFitError(
            "approval_required", f"Approved Fit Recipe '{recipe_id}' not found."
        )

    metadata = json.loads(snapshot["metadata_json"])
    recipe = json.loads(recipe_row["recipe_json"])
    specification = recipe["fit_specification"]
    if (
        specification.get("dataset_snapshot_id") != snapshot_id
        or specification.get("dataset_content_hash") != snapshot["content_hash"]
    ):
        raise OriginFitError(
            "dataset_contract_mismatch",
            "The Dataset Snapshot does not match the Approved Fit Recipe.",
        )

    expected_constraints = {
        "y0": {"lower": None, "upper": None},
        "A_fast": {"exclusive_lower": 0.0},
        "t_fast": {"exclusive_lower": 0.0},
        "A_slow": {"exclusive_lower": 0.0},
        "t_slow": {"exclusive_lower": 0.0},
        "component_order": "t_fast < t_slow",
    }
    model = specification.get("model", {})
    fit_range = specification.get("fit_range", {})
    y_series = specification.get("y_series", [])
    y_names = tuple(metadata["y_columns"])
    expected_y_series = [
        {
            "name": name,
            "unit": metadata["units"][name],
            "uncertainty": (
                {
                    "name": metadata["uncertainty_columns"][name],
                    "unit": metadata["units"][
                        metadata["uncertainty_columns"][name]
                    ],
                }
                if name in metadata["uncertainty_columns"]
                else None
            ),
        }
        for name in y_names
    ]
    weighting_mode = specification.get("weighting", {}).get("mode")
    if (
        model.get("name") != "ExpDec2"
        or model.get("x_offset_fitted") is not False
        or specification.get("shared_x_column") != metadata["x_column"]
        or y_series != expected_y_series
        or specification.get("units") != metadata["units"]
        or specification.get("constraints") != expected_constraints
        or fit_range.get("inclusive") is not True
        or weighting_mode not in ("none", "instrument")
        or (
            weighting_mode == "instrument"
            and any(series["uncertainty"] is None for series in expected_y_series)
        )
        or specification.get("data_handling")
        != {
            "missing_y": "exclude_per_series",
            "record_exclusions": True,
            "automatic_cleaning": False,
        }
    ):
        raise OriginFitError(
            "invalid_approved_recipe",
            "Approved Fit Recipe does not contain the supported ExpDec2 contract.",
        )

    path = store.objects_dir / snapshot["content_hash"]
    try:
        content_bytes = path.read_bytes()
    except OSError as error:
        raise OriginFitError(
            "dataset_integrity_error", "Dataset Snapshot content is unavailable."
        ) from error
    if hashlib.sha256(content_bytes).hexdigest() != snapshot["content_hash"]:
        raise OriginFitError(
            "dataset_integrity_error",
            "Dataset Snapshot content no longer matches its content identifier.",
        )
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OriginFitError(
            "dataset_integrity_error", "Dataset Snapshot is no longer valid UTF-8."
        ) from error
    reader = csv.reader(io.StringIO(content, newline=""))
    try:
        headers = next(reader)
    except StopIteration as error:
        raise OriginFitError(
            "invalid_dataset_contract", "Dataset Snapshot has no header row."
        ) from error
    indexes = {name: index for index, name in enumerate(headers)}
    required_columns = {
        metadata["x_column"],
        *metadata["y_columns"],
        *metadata["uncertainty_columns"].values(),
    }
    if len(indexes) != len(headers) or not required_columns.issubset(indexes):
        raise OriginFitError(
            "invalid_dataset_contract",
            "Dataset Snapshot columns no longer satisfy the Approved Fit Recipe.",
        )
    values: dict[str, list[tuple[float, float, float | None]]] = {
        name: [] for name in y_names
    }
    exclusions: list[ExclusionRecord] = []
    previous_x: float | None = None
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(headers):
            raise OriginFitError(
                "invalid_dataset_contract",
                f"Dataset Snapshot row {row_number} has an invalid field count.",
            )
        try:
            x = float(row[indexes[specification["shared_x_column"]]])
        except ValueError as error:
            raise OriginFitError(
                "invalid_dataset_contract", "X observations must be numeric."
            ) from error
        if not math.isfinite(x) or (previous_x is not None and x <= previous_x):
            raise OriginFitError(
                "invalid_dataset_contract",
                "X observations must remain finite, unique, and strictly increasing.",
            )
        previous_x = x
        if not fit_range["minimum"] <= x <= fit_range["maximum"]:
            exclusions.extend(
                ExclusionRecord(
                    series_name=name,
                    row_number=row_number,
                    reason="outside_fit_range",
                )
                for name in y_names
            )
            continue
        for series in y_series:
            name = series["name"]
            raw_y = row[indexes[name]].strip()
            if raw_y == "" or raw_y.lower() == "nan":
                exclusions.append(
                    ExclusionRecord(
                        series_name=name,
                        row_number=row_number,
                        reason="missing_y",
                    )
                )
                continue
            uncertainty = series["uncertainty"]
            try:
                y_value = float(raw_y)
                uncertainty_value = (
                    float(row[indexes[uncertainty["name"]]])
                    if uncertainty is not None
                    else None
                )
            except ValueError as error:
                raise OriginFitError(
                    "invalid_dataset_contract",
                    "Y observations and uncertainties must be numeric.",
                ) from error
            if not math.isfinite(y_value) or (
                uncertainty_value is not None
                and (
                    not math.isfinite(uncertainty_value)
                    or uncertainty_value <= 0
                )
            ):
                raise OriginFitError(
                    "invalid_dataset_contract",
                    "Y observations must be finite and uncertainties positive and finite.",
                )
            values[name].append(
                (
                    x,
                    y_value,
                    uncertainty_value,
                )
            )

    insufficient = [name for name, series_values in values.items() if len(series_values) < 6]
    if insufficient:
        raise OriginFitError(
            "invalid_dataset_contract",
            "ExpDec2 requires at least 6 valid points for every Y before execution: "
            + ", ".join(insufficient),
        )

    initialization = specification.get("initialization")
    initialization_is_valid = initialization == {"mode": "origin_auto"}
    if isinstance(initialization, dict) and initialization.get("mode") == "explicit":
        values_by_y = initialization.get("values_by_y")
        if isinstance(values_by_y, dict) and set(values_by_y) == set(y_names):
            initialization_is_valid = True
            parameter_names = {"y0", "A_fast", "t_fast", "A_slow", "t_slow"}
            for parameters in values_by_y.values():
                if (
                    not isinstance(parameters, dict)
                    or set(parameters) != parameter_names
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in parameters.values()
                    )
                    or parameters["A_fast"] <= 0
                    or parameters["t_fast"] <= 0
                    or parameters["A_slow"] <= 0
                    or parameters["t_slow"] <= parameters["t_fast"]
                ):
                    initialization_is_valid = False
                    break
        else:
            initialization_is_valid = False
    if not initialization_is_valid:
        raise OriginFitError(
            "invalid_approved_recipe",
            "Approved Fit Recipe has an invalid initialization contract.",
        )

    series_inputs = tuple(
        OriginSeriesInput(
            series_name=name,
            x=tuple(item[0] for item in values[name]),
            y=tuple(item[1] for item in values[name]),
            uncertainties=(
                tuple(float(item[2]) for item in values[name] if item[2] is not None)
                if specification["weighting"]["mode"] == "instrument"
                else None
            ),
        )
        for name in y_names
    )
    request = OriginExecutionRequest(
        model=specification["model"]["name"],
        fit_minimum=fit_range["minimum"],
        fit_maximum=fit_range["maximum"],
        constraints=specification["constraints"],
        weighting=specification["weighting"]["mode"],
        initialization=specification["initialization"],
        series=series_inputs,
    )
    return request, exclusions, {item.series_name: len(item.x) for item in series_inputs}


def _canonical_parameters(values: dict[str, float]) -> tuple[ExpDec2Parameters, bool]:
    required = {"y0", "A1", "t1", "A2", "t2"}
    if set(values) != required or any(not math.isfinite(value) for value in values.values()):
        raise OriginFitError(
            "invalid_origin_result", "Origin returned invalid ExpDec2 parameters."
        )
    swapped = values["t2"] < values["t1"]
    return _remap_parameter_values(values, swapped), swapped


def _component_mapping(swapped: bool) -> dict[str, str]:
    fast = "2" if swapped else "1"
    slow = "1" if swapped else "2"
    return {
        "y0": "y0",
        "A_fast": f"A{fast}",
        "t_fast": f"t{fast}",
        "A_slow": f"A{slow}",
        "t_slow": f"t{slow}",
    }


def _remap_parameter_values(
    values: dict[str, float], swapped: bool
) -> ExpDec2Parameters:
    mapping = _component_mapping(swapped)
    return ExpDec2Parameters(
        y0=values[mapping["y0"]],
        A_fast=values[mapping["A_fast"]],
        t_fast=values[mapping["t_fast"]],
        A_slow=values[mapping["A_slow"]],
        t_slow=values[mapping["t_slow"]],
    )


def _safe_number(value: float) -> JsonNumber:
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "Infinity"
    if value == -math.inf:
        return "-Infinity"
    return value


def _safe_mapping(values: dict[str, float]) -> dict[str, JsonNumber]:
    return {name: _safe_number(value) for name, value in values.items()}


def _canonicalize_named_values(
    values: dict[str, float], swapped: bool
) -> dict[str, JsonNumber]:
    mapping = _component_mapping(swapped)
    return {
        canonical_name: _safe_number(values[raw_name])
        for canonical_name, raw_name in mapping.items()
        if raw_name in values
    }


def _canonicalize_correlations(
    correlations: dict[str, float], swapped: bool
) -> dict[str, JsonNumber]:
    raw_to_canonical = {
        raw_name: canonical_name
        for canonical_name, raw_name in _component_mapping(swapped).items()
    }
    return {
        ":".join(raw_to_canonical.get(name, name) for name in pair.split(":")):
        _safe_number(value)
        for pair, value in correlations.items()
    }


def _canonicalize_covariance(
    covariance: list[list[float]] | None, swapped: bool
) -> list[list[JsonNumber]] | None:
    if covariance is None or len(covariance) != 5 or any(
        len(row) != 5 for row in covariance
    ):
        return None
    raw_order = ("y0", "A1", "t1", "A2", "t2")
    mapping = _component_mapping(swapped)
    order = tuple(raw_order.index(mapping[name]) for name in mapping)
    return [
        [_safe_number(covariance[raw_row][raw_column]) for raw_column in order]
        for raw_row in order
    ]


def _has_non_finite(response: OriginSeriesResponse) -> bool:
    numbers = [
        *response.raw_parameters.values(),
        *response.standard_errors.values(),
        *(bound for interval in response.confidence_intervals.values() for bound in interval),
        *response.correlations.values(),
        *response.fit_statistics.values(),
        *response.actual_initial_values.values(),
        *(
            value
            for row in (response.covariance or [])
            for value in row
        ),
    ]
    return any(not math.isfinite(value) for value in numbers)


async def execute_approved_fit(
    store: LocalStore,
    dataset_snapshot_id: str,
    approved_fit_recipe_id: str,
    adapter: OriginAdapter,
    *,
    fit_job_id: str | None = None,
) -> FitResult:
    request, exclusions, valid_counts = load_approved_fit_execution_request(
        store, dataset_snapshot_id, approved_fit_recipe_id
    )
    fit_job_id = fit_job_id or f"fit-job:{uuid.uuid4()}"
    with store.connect() as connection:
        connection.execute(
            """
            INSERT INTO fit_jobs (
                id, dataset_snapshot_id, approved_fit_recipe_id, status, started_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                fit_job_id,
                dataset_snapshot_id,
                approved_fit_recipe_id,
                "running",
                utc_now(),
            ),
        )
        store.audit(
            connection,
            "fit_job.started",
            fit_job_id,
            {
                "dataset_snapshot_id": dataset_snapshot_id,
                "approved_fit_recipe_id": approved_fit_recipe_id,
            },
        )
    responses = await adapter.execute(request)
    if tuple(response.series_name for response in responses) != tuple(
        item.series_name for item in request.series
    ):
        raise OriginFitError(
            "invalid_origin_result",
            "Origin must return one ordered outcome for every selected Y series.",
        )

    outcomes: list[SeriesFitOutcome] = []
    review_required = False
    for response in responses:
        if response.status == "failed":
            if not response.error_code or not response.error_message:
                raise OriginFitError(
                    "invalid_origin_result",
                    "A failed Origin series outcome requires a safe code and message.",
                )
            review_required = True
            outcomes.append(
                SeriesFitOutcome(
                    series_name=response.series_name,
                    status="failed",
                    valid_point_count=valid_counts[response.series_name],
                    converged=False,
                    warnings=["series_failed"],
                    error_code=response.error_code,
                    error_message=response.error_message,
                )
            )
            continue
        warnings: list[str] = []
        required_parameters = {"y0", "A1", "t1", "A2", "t2"}
        if set(response.raw_parameters) != required_parameters:
            raise OriginFitError(
                "invalid_origin_result",
                "Origin returned incomplete ExpDec2 parameters.",
            )
        if set(response.actual_initial_values) != required_parameters:
            review_required = True
            outcomes.append(
                SeriesFitOutcome(
                    series_name=response.series_name,
                    status="failed",
                    valid_point_count=valid_counts[response.series_name],
                    converged=response.converged,
                    raw_origin_parameters=_safe_mapping(response.raw_parameters),
                    raw_origin_initial_values=_safe_mapping(
                        response.actual_initial_values
                    ),
                    warnings=["series_failed"],
                    error_code="origin_result_incomplete",
                    error_message=(
                        "Origin did not report the complete actual initial values."
                    ),
                )
            )
            continue
        raw_parameters_finite = all(
            math.isfinite(value) for value in response.raw_parameters.values()
        )
        if _has_non_finite(response):
            warnings.append("non_finite_value")
        parameters: ExpDec2Parameters | None
        standard_errors: ExpDec2Parameters | None
        swapped: bool | None
        fast_fraction: float | None
        slow_fraction: float | None
        if raw_parameters_finite:
            if (
                response.raw_parameters["A1"] <= 0
                or response.raw_parameters["A2"] <= 0
                or response.raw_parameters["t1"] <= 0
                or response.raw_parameters["t2"] <= 0
                or response.raw_parameters["t1"] == response.raw_parameters["t2"]
            ):
                review_required = True
                outcomes.append(
                    SeriesFitOutcome(
                        series_name=response.series_name,
                        status="failed",
                        valid_point_count=valid_counts[response.series_name],
                        converged=response.converged,
                        raw_origin_parameters=_safe_mapping(
                            response.raw_parameters
                        ),
                        actual_initial_values=_safe_mapping(
                            response.actual_initial_values
                        ),
                        raw_origin_initial_values=_safe_mapping(
                            response.actual_initial_values
                        ),
                        warnings=["series_failed"],
                        error_code="origin_constraint_violation",
                        error_message=(
                            "Origin returned a nonpositive or non-distinct "
                            "ExpDec2 decay component."
                        ),
                    )
                )
                continue
            parameters, swapped = _canonical_parameters(response.raw_parameters)
            amplitude_sum = parameters.A_fast + parameters.A_slow
            fast_fraction = parameters.A_fast / amplitude_sum
            slow_fraction = parameters.A_slow / amplitude_sum
            if set(response.standard_errors) != required_parameters:
                standard_errors = None
                warnings.append("standard_errors_unavailable")
            elif all(
                math.isfinite(value) for value in response.standard_errors.values()
            ):
                standard_errors = _remap_parameter_values(
                    response.standard_errors, swapped
                )
            else:
                standard_errors = None
        else:
            parameters = None
            standard_errors = None
            swapped = None
            fast_fraction = None
            slow_fraction = None
        if not response.converged:
            warnings.append("not_converged")
        if parameters is not None and parameters.t_slow / parameters.t_fast < 1.5:
            warnings.append("components_not_separated")
        if response.boundary_parameters:
            warnings.append("parameter_near_boundary")
        canonical_covariance = (
            _canonicalize_covariance(response.covariance, swapped)
            if swapped is not None
            else None
        )
        if (
            response.covariance_status == "unavailable"
            or canonical_covariance is None
        ):
            warnings.append("covariance_unavailable")
        elif response.covariance_status == "singular":
            warnings.append("covariance_singular")
        if valid_counts[response.series_name] < 10:
            warnings.append("low_observation_count")
        if standard_errors is not None and parameters is not None and any(
            error > abs(value)
            for error, value in zip(
                standard_errors.model_dump().values(),
                parameters.model_dump().values(),
            )
        ):
            warnings.append("high_relative_standard_error")
        if any(
            math.isfinite(value) and abs(value) > 0.95
            for value in response.correlations.values()
        ):
            warnings.append("high_parameter_correlation")
        review_required = review_required or bool(warnings)
        interval_names: dict[str, str] = {}
        if swapped is not None:
            interval_names = _component_mapping(swapped)
        if set(response.confidence_intervals) != required_parameters:
            warnings.append("confidence_intervals_unavailable")
            review_required = True
        outcomes.append(
            SeriesFitOutcome(
                series_name=response.series_name,
                status="succeeded",
                valid_point_count=valid_counts[response.series_name],
                converged=response.converged,
                parameters=parameters,
                raw_origin_parameters=_safe_mapping(response.raw_parameters),
                standard_errors=standard_errors,
                confidence_intervals={
                    canonical: ConfidenceInterval(
                        lower=_safe_number(response.confidence_intervals[raw][0]),
                        upper=_safe_number(response.confidence_intervals[raw][1]),
                    )
                    for canonical, raw in interval_names.items()
                    if raw in response.confidence_intervals
                },
                covariance=canonical_covariance,
                covariance_parameter_order=(
                    ["y0", "A_fast", "t_fast", "A_slow", "t_slow"]
                    if canonical_covariance is not None
                    else []
                ),
                correlations=(
                    _canonicalize_correlations(response.correlations, swapped)
                    if swapped is not None
                    else _safe_mapping(response.correlations)
                ),
                fit_statistics=_safe_mapping(response.fit_statistics),
                actual_initial_values=(
                    _canonicalize_named_values(
                        response.actual_initial_values, swapped
                    )
                    if swapped is not None
                    else _safe_mapping(response.actual_initial_values)
                ),
                raw_origin_initial_values=_safe_mapping(
                    response.actual_initial_values
                ),
                fast_amplitude_fraction=fast_fraction,
                slow_amplitude_fraction=slow_fraction,
                warnings=warnings,
            )
        )

    initialization_mode: Literal["origin_auto", "explicit"] = (
        "origin_auto"
        if request.initialization["mode"] == "origin_auto"
        else "explicit"
    )
    result = FitResult(
        schema_version="1.0",
        fit_job_id=fit_job_id,
        fit_result_id=f"fit-result:{uuid.uuid4()}",
        dataset_snapshot_id=dataset_snapshot_id,
        approved_fit_recipe_id=approved_fit_recipe_id,
        model="ExpDec2",
        fit_range=FitRange(
            minimum=request.fit_minimum,
            maximum=request.fit_maximum,
            inclusive=True,
        ),
        weighting=request.weighting,
        initialization=initialization_mode,
        constraint_policy=request.constraints,
        classification="review_required" if review_required else "completed",
        scientific_status="not_accepted",
        series_outcomes=outcomes,
        exclusions=exclusions,
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
                result.fit_result_id,
                dataset_snapshot_id,
                approved_fit_recipe_id,
                utc_now(),
                result.model_dump_json(),
            ),
        )
        store.audit(
            connection,
            "fit_result.created",
            result.fit_result_id,
            {
                "dataset_snapshot_id": dataset_snapshot_id,
                "approved_fit_recipe_id": approved_fit_recipe_id,
                "classification": result.classification,
            },
        )
        completed_at = utc_now()
        connection.execute(
            """
            UPDATE fit_jobs
            SET status = ?, completed_at = ?, fit_result_id = ?
            WHERE id = ?
            """,
            ("completed", completed_at, result.fit_result_id, fit_job_id),
        )
        store.audit(
            connection,
            "fit_job.completed",
            fit_job_id,
            {
                "fit_result_id": result.fit_result_id,
                "classification": result.classification,
            },
        )
    return result


def accept_fit_result(store: LocalStore, fit_result_id: str) -> dict[str, str]:
    """Explicitly promote a reviewed Fit Result outside model control."""
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        fit_result = connection.execute(
            "SELECT id FROM fit_results WHERE id = ?", (fit_result_id,)
        ).fetchone()
        if fit_result is None:
            raise OriginFitError(
                "not_found", f"Fit Result '{fit_result_id}' not found."
            )
        existing = connection.execute(
            "SELECT id, accepted_at FROM accepted_fits WHERE fit_result_id = ?",
            (fit_result_id,),
        ).fetchone()
        if existing is not None:
            return {
                "accepted_fit_id": existing["id"],
                "fit_result_id": fit_result_id,
                "accepted_at": existing["accepted_at"],
            }
        accepted_at = utc_now()
        accepted_fit_id = f"accepted-fit:{uuid.uuid4()}"
        accepted_fit = {
            "schema_version": "1.0",
            "accepted_fit_id": accepted_fit_id,
            "fit_result_id": fit_result_id,
            "accepted_by": store.operator,
            "accepted_at": accepted_at,
        }
        connection.execute(
            """
            INSERT INTO accepted_fits (
                id, fit_result_id, accepted_by, accepted_at, accepted_fit_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                accepted_fit_id,
                fit_result_id,
                store.operator,
                accepted_at,
                json.dumps(accepted_fit, sort_keys=True, separators=(",", ":")),
            ),
        )
        store.audit(
            connection,
            "accepted_fit.accepted",
            accepted_fit_id,
            {"fit_result_id": fit_result_id},
        )
    return {
        "accepted_fit_id": accepted_fit_id,
        "fit_result_id": fit_result_id,
        "accepted_at": accepted_at,
    }


def make_execute_approved_fit_tool(
    store: LocalStore, adapter: OriginAdapter
) -> Callable[[str, str], Awaitable[FitResult]]:
    """Create the sole model-facing operation for authorized Origin fitting."""

    async def model_execute_approved_fit(
        dataset_snapshot_id: str,
        approved_fit_recipe_id: str,
    ) -> FitResult:
        """Execute one stored Dataset Snapshot with one stored Approved Fit Recipe."""
        try:
            return await execute_approved_fit(
                store,
                dataset_snapshot_id,
                approved_fit_recipe_id,
                adapter,
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
