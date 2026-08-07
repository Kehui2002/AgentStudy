"""Explicitly enabled real OriginPro 2025 automation coverage.

This is the only test module that claims coverage of a real Origin install.
Run on Windows with ``ORIGINPRO_2025_INTEGRATION=1`` after installing the
``origin-worker`` optional dependency group.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

from origin_fit.execution import (
    OriginExecutionRequest,
    OriginGraphTemplate,
    OriginSeriesInput,
)
from origin_worker.originpro_adapter import OriginProAdapter
from origin_worker.templates import GraphTemplateRegistry


_REAL_ORIGIN_ENABLED = (
    sys.platform == "win32"
    and os.environ.get("ORIGINPRO_2025_INTEGRATION") == "1"
)


def _request() -> OriginExecutionRequest:
    x = tuple(index * 0.5 for index in range(25))

    def series(
        name: str, y0: float, a1: float, t1: float, a2: float, t2: float
    ) -> OriginSeriesInput:
        y = tuple(
            y0
            + a1 * math.exp(-value / t1)
            + a2 * math.exp(-value / t2)
            + (0.015 if index % 2 else -0.015)
            for index, value in enumerate(x)
        )
        return OriginSeriesInput(name, x, y, None)

    return OriginExecutionRequest(
        model="ExpDec2",
        fit_minimum=x[0],
        fit_maximum=x[-1],
        constraints={
            "y0": {"lower": None, "upper": None},
            "A_fast": {"exclusive_lower": 0.0},
            "t_fast": {"exclusive_lower": 0.0},
            "A_slow": {"exclusive_lower": 0.0},
            "t_slow": {"exclusive_lower": 0.0},
            "component_order": "t_fast < t_slow",
        },
        weighting="none",
        initialization={"mode": "origin_auto"},
        series=(
            series("sample-a", 0.4, 6.0, 1.2, 3.0, 4.8),
            series("sample-b", 0.7, 4.5, 1.8, 2.5, 6.2),
        ),
        x_unit="ns",
        y_units={"sample-a": "counts", "sample-b": "counts"},
    )


def _create_graph_template_with_origin(target: Path) -> bool:
    """Export a scratch graph as an Origin template; False when unsupported."""

    try:
        op = OriginProAdapter()._op
        op.new_sheet("w", lname="template-probe")
        graph = op.new_graph(template="Origin")
        save_template = getattr(graph, "save_template", None)
        if save_template is None:
            return False
        save_template(str(target))
        return target.is_file()
    except Exception:
        return False


@unittest.skipUnless(
    _REAL_ORIGIN_ENABLED,
    "requires Windows and ORIGINPRO_2025_INTEGRATION=1",
)
class OriginPro2025IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_expdec2_fit_graph_and_project_workflow(self) -> None:
        adapter = OriginProAdapter()
        try:
            responses = await adapter.execute(_request())
            artifacts = adapter.take_artifacts()

            self.assertEqual([item.status for item in responses], ["succeeded"] * 2)
            for response in responses:
                self.assertEqual(
                    set(response.raw_parameters), {"y0", "A1", "t1", "A2", "t2"}
                )
                self.assertEqual(
                    set(response.standard_errors), set(response.raw_parameters)
                )
                self.assertEqual(
                    set(response.confidence_intervals), set(response.raw_parameters)
                )
                self.assertEqual(len(response.covariance or []), 5)
            self.assertIsNotNone(artifacts)
            assert artifacts is not None
            self.assertTrue(artifacts.png.startswith(b"\x89PNG"))
            self.assertTrue(artifacts.pdf.startswith(b"%PDF"))
            self.assertGreater(len(artifacts.opju), 100)
            self.assertIn(
                adapter.originpro_version, {"OriginPro 10.2", "OriginPro 10.25"}
            )
        finally:
            adapter.terminate()

    async def test_real_expdec2_fit_uses_a_registered_graph_template(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="origin-fit-template-"
        ) as directory:
            state_dir = Path(directory)
            workspace = state_dir / "workspace"
            workspace.mkdir()

            source = os.environ.get("ORIGIN_GRAPH_TEMPLATE")
            if source:
                template_bytes = Path(source).read_bytes()
            else:
                with tempfile.TemporaryDirectory(
                    prefix="origin-fit-template-probe-"
                ) as probe_directory:
                    probe = Path(probe_directory) / "probe.otpu"
                    if not _create_graph_template_with_origin(probe):
                        self.skipTest(
                            "set ORIGIN_GRAPH_TEMPLATE to a saved Origin graph "
                            "template; this Origin build has no graph.save_template"
                        )
                    template_bytes = probe.read_bytes()

            record = GraphTemplateRegistry(state_dir).register(
                name="integration",
                content=template_bytes,
                filename="integration.otpu",
                graph_profile_id="expdec2-standard",
                graph_profile_version="1.0",
                originpro_min_version=10.2,
                originpro_max_version=10.3,
            )
            template_copy = workspace / "graph-template-integration.otpu"
            template_copy.write_bytes(template_bytes)
            graph_template = OriginGraphTemplate(
                template_id=record["template_id"],
                version=record["version"],
                sha256=record["sha256"],
                graph_profile="expdec2-standard@1.0",
                path=template_copy,
            )

            adapter = OriginProAdapter()
            try:
                responses = await adapter.execute(_request(), graph_template)
                artifacts = adapter.take_artifacts()

                self.assertEqual(
                    [item.status for item in responses], ["succeeded"] * 2
                )
                self.assertIsNotNone(artifacts)
                assert artifacts is not None
                self.assertTrue(artifacts.png.startswith(b"\x89PNG"))
                self.assertTrue(artifacts.pdf.startswith(b"%PDF"))
                self.assertGreater(len(artifacts.opju), 100)
            finally:
                adapter.terminate()


if __name__ == "__main__":
    unittest.main()
