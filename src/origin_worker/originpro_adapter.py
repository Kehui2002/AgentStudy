"""OriginPro 2025 production adapter.

The Windows-only dependency is imported lazily when a real adapter is created,
so importing :mod:`origin_worker` remains safe on Linux.  Tests inject a small
``originpro``-compatible module at the Origin execution seam.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any, Literal, cast

from origin_fit.execution import (
    OriginExecutionRequest,
    OriginSeriesInput,
    OriginSeriesResponse,
)


_RAW_PARAMETER_ORDER = ("y0", "A1", "t1", "A2", "t2")
_POSITIVE_PARAMETERS = ("A1", "t1", "A2", "t2")
_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
_GRAPH_PROFILE = "expdec2-standard@1.0"
_CONSTRAINT_POLICY = {
    "y0": {"lower": None, "upper": None},
    "A_fast": {"exclusive_lower": 0.0},
    "t_fast": {"exclusive_lower": 0.0},
    "A_slow": {"exclusive_lower": 0.0},
    "t_slow": {"exclusive_lower": 0.0},
    "component_order": "t_fast < t_slow",
}


class OriginProAdapterError(RuntimeError):
    """A Worker-facing OriginPro installation or artifact failure."""


@dataclass(frozen=True, slots=True)
class OriginGraphArtifacts:
    """Origin-authored graph and project files for one completed execution."""

    graph_profile: str
    png: bytes
    pdf: bytes
    opju: bytes


@dataclass(frozen=True, slots=True)
class _ColumnLayout:
    x: int
    y_by_series: dict[str, int]
    error_by_series: dict[str, int]
    next_column: int


class OriginProAdapter:
    """Execute independent ExpDec2 fits in one owned OriginPro 2025 instance.

    The adapter never calls ``originpro.attach``.  The external-Python package's
    default ``Application`` automation server therefore belongs to this adapter
    and is the only process that :meth:`terminate` closes.
    """

    adapter_name = "originpro-2025-adapter/1.0"
    originpro_version = "OriginPro 2025"

    def __init__(
        self,
        *,
        visible: bool = False,
        originpro_module: ModuleType | Any | None = None,
    ) -> None:
        if originpro_module is None:
            if sys.platform != "win32":
                raise OriginProAdapterError(
                    "The OriginPro production adapter is available only on Windows."
                )
            try:
                originpro_module = importlib.import_module("originpro")
            except ImportError as error:
                raise OriginProAdapterError(
                    "The Windows Worker requires the 'originpro' optional dependency."
                ) from error
        self._op = originpro_module
        self._visible = visible
        self._owns_instance = False
        self._artifacts: OriginGraphArtifacts | None = None

    async def execute(
        self, request: OriginExecutionRequest
    ) -> tuple[OriginSeriesResponse, ...]:
        """Run every Y independently and preserve an ordered partial outcome."""

        self._validate_request(request)
        if self._owns_instance:
            self._op.new(asksave=False)
        else:
            self._start_owned_instance()
        self._artifacts = None
        worksheet, columns = self._write_input_worksheet(request)
        outcomes: list[OriginSeriesResponse] = []
        fitted_parameters: dict[str, dict[str, float]] = {}
        for series in request.series:
            try:
                response = self._fit_series(
                    request, series, worksheet, columns
                )
            except Exception:
                response = OriginSeriesResponse.failed(
                    series.series_name,
                    code="origin_fit_failed",
                    message="OriginPro could not fit this series.",
                )
            outcomes.append(response)
            if response.status == "succeeded":
                fitted_parameters[series.series_name] = response.raw_parameters

        self._artifacts = self._render_artifacts(
            request, worksheet, columns, fitted_parameters
        )
        return tuple(outcomes)

    def take_artifacts(self) -> OriginGraphArtifacts | None:
        """Transfer the most recent job's artifacts exactly once."""

        artifacts = self._artifacts
        self._artifacts = None
        return artifacts

    def preflight(self) -> None:
        """Start and validate the dedicated automation server before serving."""

        self._start_owned_instance()

    def terminate(self) -> None:
        """Close the automation server owned by this adapter, if started."""

        if not self._owns_instance:
            return
        self._owns_instance = False
        self._op.exit()

    close = terminate

    def _start_owned_instance(self) -> None:
        if self._owns_instance:
            return
        # originpro's default external-Python Application starts a new server;
        # attach() is deliberately never used here.
        self._op.new(asksave=False)
        self._owns_instance = True
        self._op.set_show(self._visible)
        version = float(self._op.org_ver())
        if not 10.2 <= version < 10.3:
            self.terminate()
            raise OriginProAdapterError(
                "OriginPro 2025 (version 10.2 or 10.25) is required."
            )
        self.originpro_version = f"OriginPro {version:g}"

    @staticmethod
    def _validate_request(request: OriginExecutionRequest) -> None:
        if request.model != "ExpDec2":
            raise OriginProAdapterError("Only the approved ExpDec2 model is supported.")
        if request.weighting not in ("none", "instrument"):
            raise OriginProAdapterError("Unsupported Origin weighting mode.")
        if request.constraints != _CONSTRAINT_POLICY:
            raise OriginProAdapterError("Unsupported ExpDec2 constraint policy.")
        if request.initialization.get("mode") not in ("origin_auto", "explicit"):
            raise OriginProAdapterError("Unsupported Origin initialization mode.")
        for series in request.series:
            if len(series.x) != len(series.y):
                raise OriginProAdapterError("Origin series X and Y lengths differ.")
            if request.weighting == "instrument" and (
                series.uncertainties is None
                or len(series.uncertainties) != len(series.y)
            ):
                raise OriginProAdapterError(
                    "Instrument weighting requires one uncertainty per observation."
                )

    def _write_input_worksheet(
        self, request: OriginExecutionRequest
    ) -> tuple[Any, _ColumnLayout]:
        worksheet = self._op.new_sheet(
            type="w", lname="Origin Integration Data", hidden=not self._visible
        )
        if worksheet is None:
            raise OriginProAdapterError(
                "OriginPro could not create an input worksheet."
            )

        shared_x = sorted(
            {
                x
                for series in request.series
                for x in series.x
                if request.fit_minimum <= x <= request.fit_maximum
            }
        )
        worksheet.from_list(
            0, shared_x, lname="X", units=request.x_unit, axis="X"
        )
        y_by_series: dict[str, int] = {}
        error_by_series: dict[str, int] = {}
        column = 1
        for series in request.series:
            values = dict(zip(series.x, series.y))
            y_values: list[object] = [values.get(x, "") for x in shared_x]
            y_by_series[series.series_name] = column
            worksheet.from_list(
                column,
                y_values,
                lname=series.series_name,
                units=request.y_units.get(series.series_name, ""),
                axis="Y",
            )
            column += 1
            if request.weighting == "instrument":
                assert series.uncertainties is not None
                errors = dict(zip(series.x, series.uncertainties))
                error_values: list[object] = [errors.get(x, "") for x in shared_x]
                error_by_series[series.series_name] = column
                worksheet.from_list(
                    column,
                    error_values,
                    lname=f"{series.series_name} uncertainty",
                    axis="E",
                )
                column += 1
        return worksheet, _ColumnLayout(
            x=0,
            y_by_series=y_by_series,
            error_by_series=error_by_series,
            next_column=column,
        )

    def _fit_series(
        self,
        request: OriginExecutionRequest,
        series: OriginSeriesInput,
        worksheet: Any,
        columns: _ColumnLayout,
    ) -> OriginSeriesResponse:
        model = self._op.NLFit("ExpDec2")
        yerr: int | str = columns.error_by_series.get(series.series_name, "")
        model.set_data(
            worksheet,
            columns.x,
            columns.y_by_series[series.series_name],
            yerr=yerr,
        )
        for parameter in _POSITIVE_PARAMETERS:
            model.set_lbound(parameter, ">", 0.0)
            set_tree_value = getattr(model, "_set", None)
            if callable(set_tree_value):
                # originpro 1.1.12's wrapper skips assigning a falsy numeric
                # bound, so retain the approved strict-zero bound explicitly.
                set_tree_value(f"lb_{parameter}", 0.0)

        explicit = self._explicit_initial_values(request, series.series_name)
        for parameter, value in explicit.items():
            model.set_param(parameter, value)
        initial_values = {
            parameter: float(model._get(parameter))
            for parameter in _RAW_PARAMETER_ORDER
        }

        # Ask Origin to author 95% parameter intervals and both matrices in the
        # report.  This affects reporting only, never the fitted objective.
        self._op.lt_exec(
            "nlgui __PY_ORIGIN_FIT_OUTPUT 1;"
            "__PY_ORIGIN_FIT_OUTPUT.Quantities.Parameters.LCL=1;"
            "__PY_ORIGIN_FIT_OUTPUT.Quantities.Parameters.UCL=1;"
            "__PY_ORIGIN_FIT_OUTPUT.Quantities.Parameters.Confidence=95;"
            "__PY_ORIGIN_FIT_OUTPUT.Quantities.mCov=1;"
            "__PY_ORIGIN_FIT_OUTPUT.Quantities.mCor=1;"
            "nlgui __PY_ORIGIN_FIT_OUTPUT 0;"
        )
        model.fit()
        model.report(autoupdate=False)
        raw_result = model.result()
        if not isinstance(raw_result, dict):
            raise OriginProAdapterError("OriginPro returned an invalid NLFit result.")
        tree_name = model._get_tree_name()

        parameters = {
            name: self._result_number(raw_result, name, model)
            for name in _RAW_PARAMETER_ORDER
        }
        errors = {
            name: self._result_number(raw_result, f"e_{name}", model)
            for name in _RAW_PARAMETER_ORDER
        }
        intervals = {
            name: (
                self._result_number(raw_result, f"l_{name}", model),
                self._result_number(raw_result, f"u_{name}", model),
            )
            for name in _RAW_PARAMETER_ORDER
        }
        covariance = self._origin_matrix(tree_name, "covar1")
        correlation = self._origin_matrix(tree_name, "corr1")
        correlations = {
            f"{row_name}:{column_name}": correlation[row][column]
            for row, row_name in enumerate(_RAW_PARAMETER_ORDER)
            for column, column_name in enumerate(_RAW_PARAMETER_ORDER)
            if row < column
        }
        status = raw_result.get("fitstatus")
        status_text = str(status).strip().lower()
        converged = status == 100 or (
            ("converged" in status_text or "success" in status_text)
            and "not" not in status_text
        )
        covariance_status: Literal["available", "unavailable", "singular"] = (
            "available"
        )
        if "singular" in status_text:
            covariance_status = "singular"
        statistics_names = {
            "chisqr": "origin_reduced_chi_square",
            "dof": "degrees_of_freedom",
            "pts": "point_count",
            "ssr": "residual_sum_of_squares",
            "adjr": "adjusted_r_squared",
            "cod": "r_squared",
            "rmse": "root_mean_square_error",
            "niter": "iteration_count",
        }
        statistics = {
            output_name: float(raw_result[origin_name])
            for origin_name, output_name in statistics_names.items()
            if _is_number(raw_result.get(origin_name))
        }
        near_boundary = tuple(
            name
            for name in _POSITIVE_PARAMETERS
            if parameters[name] <= max(1e-12, abs(parameters[name]) * 1e-9)
        )
        return OriginSeriesResponse(
            series_name=series.series_name,
            converged=converged,
            raw_parameters=parameters,
            standard_errors=errors,
            confidence_intervals=intervals,
            covariance=covariance,
            correlations=correlations,
            fit_statistics=statistics,
            actual_initial_values=initial_values,
            boundary_parameters=near_boundary,
            covariance_status=covariance_status,
        )

    @staticmethod
    def _explicit_initial_values(
        request: OriginExecutionRequest, series_name: str
    ) -> dict[str, float]:
        if request.initialization.get("mode") != "explicit":
            return {}
        by_series = request.initialization.get("values_by_y")
        if not isinstance(by_series, dict):
            raise OriginProAdapterError("Explicit initialization is incomplete.")
        values = by_series.get(series_name)
        if not isinstance(values, dict):
            raise OriginProAdapterError("Explicit initialization is incomplete.")
        mapping = {
            "y0": "y0",
            "A_fast": "A1",
            "t_fast": "t1",
            "A_slow": "A2",
            "t_slow": "t2",
        }
        result: dict[str, float] = {}
        for canonical, raw in mapping.items():
            value = values.get(canonical)
            if not _is_number(value):
                raise OriginProAdapterError("Explicit initialization is incomplete.")
            result[raw] = float(cast(int | float, value))
        return result

    @staticmethod
    def _result_number(raw_result: dict[str, Any], name: str, model: Any) -> float:
        value = raw_result.get(name)
        if not _is_number(value):
            value = model._get(name)
        if not _is_number(value):
            raise OriginProAdapterError(f"OriginPro omitted NLFit quantity '{name}'.")
        return float(cast(int | float, value))

    def _origin_matrix(self, tree_name: str, matrix_name: str) -> list[list[float]]:
        return [
            [
                float(self._op.lt_float(f"{tree_name}.{matrix_name}[{row},{column}]"))
                for column in range(1, 6)
            ]
            for row in range(1, 6)
        ]

    def _render_artifacts(
        self,
        request: OriginExecutionRequest,
        worksheet: Any,
        columns: _ColumnLayout,
        fitted_parameters: dict[str, dict[str, float]],
    ) -> OriginGraphArtifacts:
        graph = self._op.new_graph(
            lname="ExpDec2 Combined Fit", template="Origin", hidden=not self._visible
        )
        if graph is None:
            raise OriginProAdapterError(
                "OriginPro could not create the combined graph."
            )
        layer = graph[0]
        fitted_column = columns.next_column
        shared_x = sorted(
            {
                x
                for series in request.series
                for x in series.x
                if request.fit_minimum <= x <= request.fit_maximum
            }
        )
        for index, series in enumerate(request.series):
            color = _COLORS[index % len(_COLORS)]
            observed = layer.add_plot(
                worksheet,
                columns.y_by_series[series.series_name],
                columns.x,
                type="s",
            )
            observed.color = color
            observed.symbol_kind = 3
            observed.symbol_size = 5
            parameters = fitted_parameters.get(series.series_name)
            if parameters is None:
                continue
            fit_y = [_expdec2(x, parameters) for x in shared_x]
            worksheet.from_list(
                fitted_column,
                fit_y,
                lname=f"{series.series_name} — ExpDec2 fit",
                axis="Y",
            )
            fitted = layer.add_plot(
                worksheet, fitted_column, columns.x, type="l"
            )
            fitted.color = color
            fitted_column += 1
        layer.axis("x").title = _axis_title("X", request.x_unit)
        layer.axis("y").title = _y_axis_title(request)
        layer.rescale()
        layer.lt_exec("legend -c")

        with tempfile.TemporaryDirectory(prefix="origin-fit-artifacts-") as directory:
            root = Path(directory)
            png_path = root / "combined.png"
            pdf_path = root / "combined.pdf"
            opju_path = root / "project.opju"
            if not graph.save_fig(str(png_path), type="png", replace=True, width=1800):
                raise OriginProAdapterError("OriginPro could not export the PNG graph.")
            if not graph.save_fig(str(pdf_path), type="pdf", replace=True):
                raise OriginProAdapterError("OriginPro could not export the PDF graph.")
            if not self._op.save(str(opju_path)):
                raise OriginProAdapterError(
                    "OriginPro could not save the OPJU project."
                )
            try:
                return OriginGraphArtifacts(
                    graph_profile=_GRAPH_PROFILE,
                    png=png_path.read_bytes(),
                    pdf=pdf_path.read_bytes(),
                    opju=opju_path.read_bytes(),
                )
            except OSError as error:
                raise OriginProAdapterError(
                    "OriginPro did not create all required graph artifacts."
                ) from error


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expdec2(x: float, parameters: dict[str, float]) -> float:
    """Evaluate, but never optimize, Origin's authoritative fitted parameters."""

    return (
        parameters["y0"]
        + parameters["A1"] * math.exp(-x / parameters["t1"])
        + parameters["A2"] * math.exp(-x / parameters["t2"])
    )


def _axis_title(name: str, unit: str) -> str:
    return f"{name} ({unit})" if unit else name


def _y_axis_title(request: OriginExecutionRequest) -> str:
    units = {
        unit for series in request.series
        if (unit := request.y_units.get(series.series_name, ""))
    }
    if len(units) == 1:
        return _axis_title("Y", next(iter(units)))
    if not units:
        return "Y"
    details = "; ".join(
        f"{series.series_name}: {request.y_units[series.series_name]}"
        for series in request.series
        if request.y_units.get(series.series_name)
    )
    return f"Y ({details})"


__all__ = [
    "OriginGraphArtifacts",
    "OriginProAdapter",
    "OriginProAdapterError",
]
