from __future__ import annotations

import asyncio
from pathlib import Path
import io
import tempfile
import sys
import threading
import unittest
from unittest.mock import patch
import zipfile

from origin_fit.execution import (
    DeterministicFakeOriginAdapter,
    OriginExecutionRequest,
    OriginFittedCurve,
    OriginSeriesInput,
)
from origin_fit.remote import InProcessWorkerTransport, RemoteOriginExecutor
from origin_fit.storage import LocalStore
from origin_worker.service import OriginWorker
from tests.test_origin_remote import approved_fixture


PARAMETERS = {
    "y0": 0.5,
    "A1": 6.0,
    "t1": 1.5,
    "A2": 3.0,
    "t2": 5.0,
}


class _FakeWorksheet:
    def __init__(self) -> None:
        self.columns: list[tuple[int, list[object], str, str, str]] = []

    def from_list(
        self,
        column: int,
        values: list[object],
        lname: str = "",
        units: str = "",
        comments: str = "",
        axis: str = "",
    ) -> None:
        del comments
        self.columns.append((column, values, lname, units, axis))

    def to_list(self, column: int) -> list[object]:
        return next(values for index, values, *_ in self.columns if index == column)


class _FakePlot:
    def __init__(self, plot_type: str) -> None:
        self.plot_type = plot_type
        self.color: object = None
        self.symbol_kind = 0
        self.symbol_size = 0.0


class _FakeAxis:
    def __init__(self) -> None:
        self.title = ""


class _FakeLayer:
    def __init__(self) -> None:
        self.plots: list[tuple[int, int, str, _FakePlot]] = []
        self.plot_worksheets: list[_FakeWorksheet] = []
        self.axes = {"x": _FakeAxis(), "y": _FakeAxis()}
        self.legend_text = ""

    def add_plot(
        self,
        worksheet: _FakeWorksheet,
        coly: int,
        colx: int,
        type: str,
    ) -> _FakePlot:
        self.plot_worksheets.append(worksheet)
        plot = _FakePlot(type)
        self.plots.append((coly, colx, type, plot))
        return plot

    def axis(self, name: str) -> _FakeAxis:
        return self.axes[name]

    def rescale(self) -> None:
        pass

    def lt_exec(self, command: str) -> None:
        if command.startswith("legend -c"):
            self.legend_text = command


class _FakeGraph:
    def __init__(self, module: _FakeOriginPro) -> None:
        self._module = module
        self.layer = _FakeLayer()

    def __getitem__(self, index: int) -> _FakeLayer:
        if index != 0:
            raise IndexError(index)
        return self.layer

    def save_fig(self, path: str, **kwargs: object) -> str:
        del kwargs
        suffix = path.rsplit(".", 1)[-1].upper()
        content = f"REAL-{suffix}".encode()
        self._module.exported[path] = content
        Path(path).write_bytes(content)
        return path


class _FakeNLFit:
    def __init__(self, module: _FakeOriginPro, function: str) -> None:
        self.module = module
        self.function = function
        self.y_column = -1
        self.parameters = dict(PARAMETERS)
        self.lower_bounds: list[tuple[str, str, float]] = []
        self.explicit: dict[str, float] = {}
        self._tree_name = f"tree{len(module.fits) + 1}"
        module.fits.append(self)

    def set_data(
        self,
        worksheet: _FakeWorksheet,
        x: int,
        y: int,
        yerr: int | str = "",
    ) -> None:
        del worksheet
        self.module.data_calls.append((x, y, yerr))
        self.y_column = y

    def set_lbound(self, parameter: str, control: str, value: float) -> None:
        self.lower_bounds.append((parameter, control, value))

    def set_param(self, parameter: str, value: float) -> None:
        self.parameters[parameter] = value
        self.explicit[parameter] = value

    def _set(self, property_name: str, value: float) -> None:
        self.module.tree_values.append((property_name, value))

    def _get(self, property_name: str) -> float:
        if property_name in self.parameters:
            return self.parameters[property_name]
        prefix, _, parameter = property_name.partition("_")
        if prefix == "e":
            return 0.1
        if prefix == "l":
            if self.module.omit_confidence_intervals:
                raise KeyError(property_name)
            return self.parameters[parameter] - 0.2
        if prefix == "u":
            if self.module.omit_confidence_intervals:
                raise KeyError(property_name)
            return self.parameters[parameter] + 0.2
        raise KeyError(property_name)

    def _get_tree_name(self) -> str:
        return self._tree_name

    def fit(self) -> None:
        if self.module.block_first_fit and not self.module.blocked_once:
            self.module.blocked_once = True
            self.module.fit_started.set()
            self.module.fit_release.wait(timeout=0.3)
        if self.y_column == self.module.failed_y_column:
            raise RuntimeError("sensitive Origin detail")

    def report(self, autoupdate: bool = False) -> tuple[str, str]:
        del autoupdate
        index = len(self.module.report_calls) + 1
        report_range = f"[Report{index}]Fit!"
        curves_range = f"[Curves{index}]Fit!"
        curves = _FakeWorksheet()
        curve_x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        curves.from_list(0, curve_x, lname="X", axis="X")
        curves.from_list(
            1,
            [900.0 + index - row for row, _ in enumerate(curve_x)],
            lname="Fit Curve",
            axis="Y",
        )
        self.module.report_calls.append((report_range, curves_range))
        self.module.report_sheets[report_range] = _FakeWorksheet()
        self.module.report_sheets[curves_range] = curves
        return report_range, curves_range

    def result(self) -> dict[str, float]:
        values = {
            **self.parameters,
            **{f"e_{name}": 0.1 for name in PARAMETERS},
            "chisqr": 1.25,
            "dof": 7.0,
            "pts": 12.0,
            "ssr": 8.75,
            "adjr": 0.98,
            "cod": 0.99,
            "rmse": 0.4,
            "niter": 8.0,
            "fitstatus": 100.0,
        }
        if not self.module.omit_confidence_intervals:
            values.update(
                {f"l_{name}": value - 0.2 for name, value in self.parameters.items()}
            )
            values.update(
                {f"u_{name}": value + 0.2 for name, value in self.parameters.items()}
            )
        return values


class _FakeOriginPro:
    def __init__(self) -> None:
        self.new_calls: list[bool] = []
        self.show_calls: list[bool] = []
        self.exit_calls = 0
        self.attach_calls = 0
        self.worksheets: list[_FakeWorksheet] = []
        self.fits: list[_FakeNLFit] = []
        self.data_calls: list[tuple[int, int, int | str]] = []
        self.graphs: list[_FakeGraph] = []
        self.exported: dict[str, bytes] = {}
        self.saved_projects: list[str] = []
        self.failed_y_column: int | None = None
        self.tree_values: list[tuple[str, float]] = []
        self.report_calls: list[tuple[str, str]] = []
        self.report_sheets: dict[str, _FakeWorksheet] = {}
        self.find_sheet_calls: list[tuple[str, str]] = []
        self.block_first_fit = False
        self.blocked_once = False
        self.fit_started = threading.Event()
        self.fit_release = threading.Event()
        self.omit_confidence_intervals = False
        self.omit_covariance = False

    def new(self, asksave: bool = False) -> None:
        self.new_calls.append(asksave)

    def set_show(self, visible: bool) -> None:
        self.show_calls.append(visible)

    def attach(self) -> None:
        self.attach_calls += 1

    def exit(self) -> None:
        self.exit_calls += 1
        self.fit_release.set()

    def org_ver(self) -> float:
        return 10.2

    def new_sheet(self, **kwargs: object) -> _FakeWorksheet:
        del kwargs
        worksheet = _FakeWorksheet()
        self.worksheets.append(worksheet)
        return worksheet

    def NLFit(self, function: str) -> _FakeNLFit:
        return _FakeNLFit(self, function)

    def lt_exec(self, command: str) -> bool:
        self.last_labtalk = command
        return True

    def lt_float(self, expression: str) -> float:
        if self.omit_covariance and ".covar1" in expression:
            raise RuntimeError("covariance unavailable")
        row, column = expression.rsplit("[", 1)[1].removesuffix("]").split(",")
        if ".corr1" in expression:
            return 1.0 if row == column else 0.05
        return 1.0 if row == column else 0.0

    def find_sheet(self, type: str, range: str) -> _FakeWorksheet:
        self.find_sheet_calls.append((type, range))
        return self.report_sheets[range]

    def new_graph(self, **kwargs: object) -> _FakeGraph:
        del kwargs
        graph = _FakeGraph(self)
        self.graphs.append(graph)
        return graph

    def save(self, path: str) -> bool:
        self.saved_projects.append(path)
        self.exported[path] = b"REAL-OPJU"
        Path(path).write_bytes(b"REAL-OPJU")
        return True


def _request(*, explicit: bool = False) -> OriginExecutionRequest:
    initialization: dict[str, object]
    if explicit:
        initialization = {
            "mode": "explicit",
            "values_by_y": {
                "sample-a": {
                    "y0": 0.25,
                    "A_fast": 5.5,
                    "t_fast": 1.25,
                    "A_slow": 2.5,
                    "t_slow": 4.5,
                },
                "sample-b": {
                    "y0": 0.75,
                    "A_fast": 4.5,
                    "t_fast": 1.75,
                    "A_slow": 3.5,
                    "t_slow": 6.5,
                },
            },
        }
    else:
        initialization = {"mode": "origin_auto"}
    return OriginExecutionRequest(
        model="ExpDec2",
        fit_minimum=0.0,
        fit_maximum=5.0,
        constraints={
            "y0": {"lower": None, "upper": None},
            "A_fast": {"exclusive_lower": 0.0},
            "t_fast": {"exclusive_lower": 0.0},
            "A_slow": {"exclusive_lower": 0.0},
            "t_slow": {"exclusive_lower": 0.0},
            "component_order": "t_fast < t_slow",
        },
        weighting="instrument",
        initialization=initialization,
        x_unit="ns",
        y_units={"sample-a": "counts", "sample-b": "counts"},
        series=(
            OriginSeriesInput(
                series_name="sample-a",
                x=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                y=(9.5, 7.0, 5.0, 3.5, 2.5, 1.8),
                uncertainties=(0.1,) * 6,
            ),
            OriginSeriesInput(
                series_name="sample-b",
                x=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                y=(10.0, 7.5, 5.5, 4.0, 2.8, 2.0),
                uncertainties=(0.2,) * 6,
            ),
        ),
    )


class OriginProAdapterContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_independent_weighted_fits_and_extracts_origin_results(
        self,
    ) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        adapter = OriginProAdapter(originpro_module=originpro)

        responses = await adapter.execute(_request())

        self.assertEqual(originpro.new_calls, [False])
        self.assertEqual(originpro.show_calls, [False])
        self.assertEqual(originpro.attach_calls, 0)
        self.assertEqual([fit.function for fit in originpro.fits], ["ExpDec2"] * 2)
        self.assertEqual(originpro.data_calls, [(0, 1, 2), (0, 3, 4)])
        for fit in originpro.fits:
            self.assertEqual(
                fit.lower_bounds,
                [
                    ("A1", ">", 0.0),
                    ("t1", ">", 0.0),
                    ("A2", ">", 0.0),
                    ("t2", ">", 0.0),
                ],
            )
        self.assertEqual(
            originpro.tree_values[:4],
            [(f"lb_{name}", 0.0) for name in ("A1", "t1", "A2", "t2")],
        )
        self.assertEqual([item.status for item in responses], ["succeeded"] * 2)
        first = responses[0]
        self.assertTrue(first.converged)
        self.assertEqual(first.raw_parameters, PARAMETERS)
        self.assertEqual(first.standard_errors, {name: 0.1 for name in PARAMETERS})
        self.assertEqual(first.confidence_intervals["A1"], (5.8, 6.2))
        self.assertEqual(
            first.covariance,
            [
                [1.0 if row == column else 0.0 for column in range(5)]
                for row in range(5)
            ],
        )
        self.assertIn("A1:A2", first.correlations)
        self.assertEqual(first.fit_statistics["origin_reduced_chi_square"], 1.25)
        self.assertEqual(first.actual_initial_values, PARAMETERS)

    async def test_applies_explicit_initial_values_and_preserves_partial_failure(
        self,
    ) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        originpro.failed_y_column = 3
        adapter = OriginProAdapter(originpro_module=originpro, visible=True)

        responses = await adapter.execute(_request(explicit=True))

        self.assertEqual(originpro.show_calls, [True])
        self.assertEqual(
            originpro.fits[0].explicit,
            {"y0": 0.25, "A1": 5.5, "t1": 1.25, "A2": 2.5, "t2": 4.5},
        )
        self.assertEqual([item.status for item in responses], ["succeeded", "failed"])
        self.assertEqual(responses[1].error_code, "origin_fit_failed")
        self.assertNotIn("sensitive", responses[1].error_message or "")

    async def test_missing_origin_uncertainty_outputs_preserve_successful_fit(
        self,
    ) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        originpro.omit_confidence_intervals = True
        originpro.omit_covariance = True
        adapter = OriginProAdapter(originpro_module=originpro)

        responses = await adapter.execute(_request())

        self.assertEqual([item.status for item in responses], ["succeeded"] * 2)
        self.assertEqual(responses[0].confidence_intervals, {})
        self.assertIsNone(responses[0].covariance)
        self.assertEqual(responses[0].covariance_status, "unavailable")

    async def test_builds_versioned_combined_graph_and_real_origin_artifacts(
        self,
    ) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        adapter = OriginProAdapter(originpro_module=originpro)

        await adapter.execute(_request())
        artifacts = adapter.take_artifacts()

        self.assertIsNotNone(artifacts)
        assert artifacts is not None
        self.assertEqual(artifacts.graph_profile, "expdec2-standard@1.0")
        self.assertEqual(artifacts.png, b"REAL-PNG")
        self.assertEqual(artifacts.pdf, b"REAL-PDF")
        self.assertEqual(artifacts.opju, b"REAL-OPJU")
        layer = originpro.graphs[0].layer
        self.assertEqual([plot[2] for plot in layer.plots], ["s", "l", "s", "l"])
        self.assertEqual(layer.plots[0][3].color, layer.plots[1][3].color)
        self.assertEqual(layer.axes["x"].title, "X (ns)")
        self.assertEqual(layer.axes["y"].title, "Y (counts)")
        self.assertEqual(layer.legend_text, "legend -c")

    async def test_graph_uses_the_fitted_curves_authored_by_origin(self) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        adapter = OriginProAdapter(originpro_module=originpro)

        await adapter.execute(_request())
        artifacts = adapter.take_artifacts()

        self.assertEqual(
            originpro.find_sheet_calls,
            [("w", "[Curves1]Fit!"), ("w", "[Curves2]Fit!")],
        )
        layer = originpro.graphs[0].layer
        self.assertIs(
            layer.plot_worksheets[1], originpro.report_sheets["[Curves1]Fit!"]
        )
        self.assertIs(
            layer.plot_worksheets[3], originpro.report_sheets["[Curves2]Fit!"]
        )
        self.assertEqual(len(originpro.worksheets[0].columns), 5)
        assert artifacts is not None
        self.assertEqual(artifacts.fitted_curves[0].x, _request().series[0].x)
        self.assertEqual(
            artifacts.fitted_curves[0].y,
            (901.0, 900.0, 899.0, 898.0, 897.0, 896.0),
        )

    async def test_terminate_closes_only_the_owned_origin_instance(self) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        adapter = OriginProAdapter(originpro_module=originpro)
        await adapter.execute(_request())

        adapter.terminate()
        adapter.terminate()

        self.assertEqual(originpro.exit_calls, 1)
        self.assertEqual(originpro.attach_calls, 0)

    async def test_reuses_owned_instance_but_starts_a_clean_project_per_job(
        self,
    ) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        adapter = OriginProAdapter(originpro_module=originpro)

        await adapter.execute(_request())
        await adapter.execute(_request())
        adapter.terminate()

        self.assertEqual(originpro.new_calls, [False, False])
        self.assertEqual(originpro.exit_calls, 1)

    async def test_timeout_can_interrupt_blocking_origin_and_next_job_is_clean(
        self,
    ) -> None:
        from origin_worker.originpro_adapter import OriginProAdapter

        originpro = _FakeOriginPro()
        originpro.block_first_fit = True
        adapter = OriginProAdapter(originpro_module=originpro)

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(adapter.execute(_request()), timeout=0.03)
        adapter.terminate()

        responses = await asyncio.wait_for(adapter.execute(_request()), timeout=0.5)

        self.assertTrue(originpro.fit_started.is_set())
        self.assertEqual([item.status for item in responses], ["succeeded"] * 2)
        self.assertEqual(originpro.new_calls, [False, False])
        self.assertEqual(originpro.exit_calls, 1)

    def test_importing_adapter_does_not_import_originpro_on_linux(self) -> None:
        sys.modules.pop("originpro", None)

        __import__("origin_worker.originpro_adapter")

        self.assertNotIn("originpro", sys.modules)

    async def test_worker_places_production_graph_artifacts_in_the_bundle(self) -> None:
        from origin_worker.originpro_adapter import OriginGraphArtifacts

        class ArtifactAdapter(DeterministicFakeOriginAdapter):
            adapter_name = "originpro-2025-adapter/1.0"
            originpro_version = "OriginPro 10.2"

            def __init__(self) -> None:
                super().__init__()
                self.take_count = 0

            def take_artifacts(self) -> OriginGraphArtifacts:
                self.take_count += 1
                request = self.requests[-1]
                return OriginGraphArtifacts(
                    graph_profile="expdec2-standard@1.0",
                    png=b"PRODUCTION-PNG",
                    pdf=b"PRODUCTION-PDF",
                    opju=b"PRODUCTION-OPJU",
                    fitted_curves=tuple(
                        OriginFittedCurve(
                            series_name=series.series_name,
                            x=series.x,
                            y=tuple(1234.0 + index for index, _ in enumerate(series.x)),
                        )
                        for series in request.series
                    ),
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            adapter = ArtifactAdapter()
            worker = OriginWorker(root / "worker", adapter)
            submission = RemoteOriginExecutor(
                InProcessWorkerTransport(worker)
            ).prepare_submission(store, snapshot_id, recipe_id)
            submitted = worker.submit(submission, "production-artifacts")

            await worker.run_queued()
            with zipfile.ZipFile(
                io.BytesIO(worker.get_bundle(submitted.worker_job_id))
            ) as archive:
                self.assertEqual(archive.read("combined.png"), b"PRODUCTION-PNG")
                self.assertEqual(archive.read("combined.pdf"), b"PRODUCTION-PDF")
                self.assertEqual(archive.read("project.opju"), b"PRODUCTION-OPJU")
                self.assertIn(b",1234.0\n", archive.read("fitted-data.csv"))
            self.assertEqual(adapter.take_count, 1)

    def test_worker_cli_selects_visible_production_adapter_by_default(self) -> None:
        from origin_worker.cli import main as worker_main

        constructed: list[bool] = []

        class StubProductionAdapter:
            def __init__(self, *, visible: bool = False) -> None:
                constructed.append(visible)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            certificate = root / "worker.crt"
            key = root / "worker.key"
            certificate.write_text("certificate", encoding="utf-8")
            key.write_text("key", encoding="utf-8")
            with (
                patch.dict("os.environ", {"ORIGIN_WORKER_TOKEN": "t" * 32}),
                patch(
                    "origin_worker.cli.OriginProAdapter",
                    StubProductionAdapter,
                    create=True,
                ),
                patch("uvicorn.run"),
            ):
                status = worker_main(
                    [
                        "serve",
                        "--state-dir",
                        str(root / "state"),
                        "--host",
                        "192.168.56.1",
                        "--host-only-network",
                        "192.168.56.0/24",
                        "--linux-guest-address",
                        "192.168.56.2",
                        "--certfile",
                        str(certificate),
                        "--keyfile",
                        str(key),
                        "--origin-visible",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertEqual(constructed, [True])


if __name__ == "__main__":
    unittest.main()
