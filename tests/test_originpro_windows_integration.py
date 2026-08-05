"""Explicitly enabled real OriginPro 2025 automation coverage.

This is the only test module that claims coverage of a real Origin install.
Run on Windows with ``ORIGINPRO_2025_INTEGRATION=1`` after installing the
``origin-worker`` optional dependency group.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

from origin_fit.execution import OriginExecutionRequest, OriginSeriesInput
from origin_worker.originpro_adapter import OriginProAdapter


_REAL_ORIGIN_ENABLED = (
    sys.platform == "win32"
    and os.environ.get("ORIGINPRO_2025_INTEGRATION") == "1"
)


@unittest.skipUnless(
    _REAL_ORIGIN_ENABLED,
    "requires Windows and ORIGINPRO_2025_INTEGRATION=1",
)
class OriginPro2025IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_expdec2_fit_graph_and_project_workflow(self) -> None:
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

        request = OriginExecutionRequest(
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
        adapter = OriginProAdapter()
        try:
            responses = await adapter.execute(request)
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


if __name__ == "__main__":
    unittest.main()
