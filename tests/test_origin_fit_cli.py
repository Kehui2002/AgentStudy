from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_expdec2.csv"


class OriginFitCliTests(unittest.TestCase):
    def run_cli(self, state_dir: Path, *arguments: str, succeeds: bool = True) -> dict:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "origin_fit",
                "--state-dir",
                str(state_dir),
                *arguments,
            ],
            cwd=ROOT,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")),
                    )
                ),
                "ORIGIN_FIT_OPERATOR": "researcher@example.test",
            },
            text=True,
        )
        if succeeds and completed.returncode != 0:
            self.fail(f"CLI failed: {completed.stderr}")
        if not succeeds and completed.returncode == 0:
            self.fail(f"CLI unexpectedly succeeded: {completed.stdout}")
        output = completed.stdout if completed.returncode == 0 else completed.stderr
        return json.loads(output)

    def import_fixture(self, state_dir: Path, csv_path: Path = FIXTURE) -> dict:
        return self.run_cli(
            state_dir,
            "import",
            str(csv_path),
            "--x",
            "time_s",
            "--y",
            "decay_a",
            "--y",
            "decay_b",
            "--y",
            "decay_c",
            "--uncertainty",
            "decay_a=decay_a_error",
            "--uncertainty",
            "decay_b=decay_b_error",
            "--uncertainty",
            "decay_c=decay_c_error",
            "--unit",
            "time_s=s",
            "--unit",
            "decay_a=dimensionless",
            "--unit",
            "decay_b=dimensionless",
            "--unit",
            "decay_c=dimensionless",
            "--unit",
            "decay_a_error=dimensionless",
            "--unit",
            "decay_b_error=dimensionless",
            "--unit",
            "decay_c_error=dimensionless",
        )

    def import_two_columns(
        self, state_dir: Path, csv_path: Path, *, succeeds: bool = True
    ) -> dict:
        return self.run_cli(
            state_dir,
            "import",
            str(csv_path),
            "--x",
            "x",
            "--y",
            "y",
            "--unit",
            "x=s",
            "--unit",
            "y=dimensionless",
            succeeds=succeeds,
        )

    def propose_fixture(
        self,
        state_dir: Path,
        snapshot_id: str,
        *,
        fit_max: str = "11",
    ) -> dict:
        return self.run_cli(
            state_dir,
            "propose",
            snapshot_id,
            "--experiment-id",
            "synthetic-expdec2",
            "--fit-min",
            "0",
            "--fit-max",
            fit_max,
            "--weighting",
            "instrument",
            "--initialization",
            "origin-auto",
            "--graph-profile",
            "expdec2-standard@1.0",
        )

    def test_import_creates_an_immutable_snapshot_with_a_bounded_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            imported = self.import_fixture(state_dir)

            expected_digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
            self.assertEqual(imported["dataset_snapshot_id"], f"sha256:{expected_digest}")

            inspected = self.run_cli(
                state_dir,
                "inspect",
                imported["dataset_snapshot_id"],
            )
            self.assertEqual(
                inspected["summary"],
                {
                    "schema_version": "1.0",
                    "row_count": 12,
                    "column_count": 7,
                    "x": {
                        "name": "time_s",
                        "unit": "s",
                        "minimum": 0.0,
                        "maximum": 11.0,
                    },
                    "y_series": [
                        {
                            "name": "decay_a",
                            "unit": "dimensionless",
                            "valid_point_count": 12,
                            "missing_point_count": 0,
                            "minimum": 2.008932,
                            "maximum": 12.01,
                            "uncertainty": {
                                "name": "decay_a_error",
                                "unit": "dimensionless",
                            },
                        },
                        {
                            "name": "decay_b",
                            "unit": "dimensionless",
                            "valid_point_count": 11,
                            "missing_point_count": 1,
                            "minimum": 1.319699,
                            "maximum": 8.99,
                            "uncertainty": {
                                "name": "decay_b_error",
                                "unit": "dimensionless",
                            },
                        },
                        {
                            "name": "decay_c",
                            "unit": "dimensionless",
                            "valid_point_count": 11,
                            "missing_point_count": 1,
                            "minimum": 1.993624,
                            "maximum": 15.02,
                            "uncertainty": {
                                "name": "decay_c_error",
                                "unit": "dimensionless",
                            },
                        },
                    ],
                    "preview_included": False,
                },
            )
            self.assertNotIn("rows", inspected)
            self.assertLess(len(json.dumps(inspected["summary"])), 4096)

            object_path = state_dir / "objects" / "sha256" / expected_digest
            self.assertEqual(object_path.read_bytes(), FIXTURE.read_bytes())

    def test_import_rejects_duplicate_columns_and_non_monotonic_x(self) -> None:
        invalid_datasets = {
            "duplicate_columns": "x,y,y\n0,1,2\n1,2,3\n2,3,4\n3,4,5\n4,5,6\n5,6,7\n",
            "duplicate_x": "x,y\n0,1\n1,2\n2,3\n2,4\n4,5\n5,6\n",
            "unordered_x": "x,y\n0,1\n1,2\n3,3\n2,4\n4,5\n5,6\n",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, csv_text in invalid_datasets.items():
                with self.subTest(name=name):
                    csv_path = root / f"{name}.csv"
                    csv_path.write_text(csv_text, encoding="utf-8")
                    error = self.import_two_columns(root / name, csv_path, succeeds=False)
                    self.assertEqual(error["error"], "invalid_dataset_contract")

    def test_import_rejects_invalid_y_uncertainties_and_too_few_points(self) -> None:
        invalid_datasets = {
            "nonnumeric_y": "x,y,error\n0,1,.1\n1,2,.1\n2,bad,.1\n3,4,.1\n4,5,.1\n5,6,.1\n",
            "infinite_y": "x,y,error\n0,1,.1\n1,2,.1\n2,inf,.1\n3,4,.1\n4,5,.1\n5,6,.1\n",
            "zero_uncertainty": "x,y,error\n0,1,.1\n1,2,.1\n2,3,0\n3,4,.1\n4,5,.1\n5,6,.1\n",
            "negative_uncertainty": "x,y,error\n0,1,.1\n1,2,.1\n2,3,-.1\n3,4,.1\n4,5,.1\n5,6,.1\n",
            "five_valid_points": "x,y,error\n0,1,.1\n1,2,.1\n2,,.1\n3,4,.1\n4,5,.1\n5,6,.1\n",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, csv_text in invalid_datasets.items():
                with self.subTest(name=name):
                    csv_path = root / f"{name}.csv"
                    csv_path.write_text(csv_text, encoding="utf-8")
                    error = self.run_cli(
                        root / name,
                        "import",
                        str(csv_path),
                        "--x",
                        "x",
                        "--y",
                        "y",
                        "--uncertainty",
                        "y=error",
                        "--unit",
                        "x=s",
                        "--unit",
                        "y=dimensionless",
                        "--unit",
                        "error=dimensionless",
                        succeeds=False,
                    )
                    self.assertEqual(error["error"], "invalid_dataset_contract")

    def test_import_requires_utf8_known_columns_and_explicit_units(self) -> None:
        valid_text = "x,y\n0,1\n1,2\n2,3\n3,4\n4,5\n5,6\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            latin1_path = root / "latin1.csv"
            latin1_path.write_bytes(
                "x,y\n0,olá\n1,2\n2,3\n3,4\n4,5\n5,6\n".encode("latin-1")
            )
            encoding_error = self.import_two_columns(
                root / "latin-state", latin1_path, succeeds=False
            )
            self.assertEqual(encoding_error["error"], "invalid_encoding")

            csv_path = root / "valid.csv"
            csv_path.write_text(valid_text, encoding="utf-8")
            missing_unit = self.run_cli(
                root / "unit-state",
                "import",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--unit",
                "x=s",
                succeeds=False,
            )
            self.assertEqual(missing_unit["error"], "invalid_dataset_contract")

            unknown_column = self.run_cli(
                root / "column-state",
                "import",
                str(csv_path),
                "--x",
                "time",
                "--y",
                "y",
                "--unit",
                "time=s",
                "--unit",
                "y=dimensionless",
                succeeds=False,
            )
            self.assertEqual(unknown_column["error"], "invalid_dataset_contract")

            first_import = self.import_two_columns(root / "immutable-state", csv_path)
            conflicting_metadata = self.run_cli(
                root / "immutable-state",
                "import",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--unit",
                "x=s",
                "--unit",
                "y=V",
                succeeds=False,
            )
            self.assertEqual(
                conflicting_metadata["error"], "snapshot_metadata_conflict"
            )
            unchanged = self.run_cli(
                root / "immutable-state",
                "inspect",
                first_import["dataset_snapshot_id"],
            )
            self.assertEqual(unchanged["summary"]["y_series"][0]["unit"], "dimensionless")

    def test_import_bounds_summary_metadata_and_keeps_column_roles_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "valid.csv"
            csv_path.write_text(
                "x,y,error\n0,6,.1\n1,5,.1\n2,4,.1\n3,3,.1\n4,2,.1\n5,1,.1\n",
                encoding="utf-8",
            )
            long_unit = self.run_cli(
                root / "long-unit",
                "import",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--unit",
                "x=s",
                "--unit",
                f"y={'u' * 129}",
                succeeds=False,
            )
            self.assertEqual(long_unit["error"], "invalid_dataset_contract")

            role_conflict = self.run_cli(
                root / "role-conflict",
                "import",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--uncertainty",
                "y=y",
                "--unit",
                "x=s",
                "--unit",
                "y=dimensionless",
                succeeds=False,
            )
            self.assertEqual(role_conflict["error"], "invalid_dataset_contract")

    def test_import_enforces_deployment_resource_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            oversized = root / "oversized.csv"
            with oversized.open("wb") as csv_file:
                csv_file.truncate(100 * 1024 * 1024 + 1)
            size_error = self.import_two_columns(
                root / "size-state", oversized, succeeds=False
            )
            self.assertEqual(size_error["error"], "resource_limit_exceeded")

            too_many_rows = root / "too-many-rows.csv"
            with too_many_rows.open("w", encoding="utf-8", newline="") as csv_file:
                csv_file.write("x,y\n")
                csv_file.writelines(
                    f"{row},{row + 1}\n" for row in range(1_000_001)
                )
            row_error = self.import_two_columns(
                root / "row-state", too_many_rows, succeeds=False
            )
            self.assertEqual(row_error["error"], "resource_limit_exceeded")

            y_columns = [f"y_{number}" for number in range(21)]
            too_many_y = root / "too-many-y.csv"
            too_many_y.write_text(
                "x," + ",".join(y_columns) + "\n"
                + "\n".join(
                    f"{x}," + ",".join(str(x + number + 1) for number in range(21))
                    for x in range(6)
                )
                + "\n",
                encoding="utf-8",
            )
            arguments = ["import", str(too_many_y), "--x", "x"]
            for y_name in y_columns:
                arguments.extend(("--y", y_name))
            arguments.extend(("--unit", "x=s"))
            for y_name in y_columns:
                arguments.extend(("--unit", f"{y_name}=dimensionless"))
            y_error = self.run_cli(
                root / "y-state", *arguments, succeeds=False
            )
            self.assertEqual(y_error["error"], "resource_limit_exceeded")

    def test_propose_and_approve_create_versioned_immutable_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            snapshot = self.import_fixture(state_dir)

            proposed = self.propose_fixture(
                state_dir, snapshot["dataset_snapshot_id"]
            )
            specification = self.run_cli(
                state_dir, "inspect", proposed["fit_specification_id"]
            )
            self.assertFalse(specification["fit_specification"]["execution_authorized"])
            self.assertEqual(
                specification["fit_specification"]["model"],
                {
                    "name": "ExpDec2",
                    "formula": (
                        "y = y0 + A_fast*exp(-x/t_fast) + "
                        "A_slow*exp(-x/t_slow)"
                    ),
                    "x_offset_fitted": False,
                },
            )
            self.assertEqual(
                specification["fit_specification"]["weighting"],
                {"mode": "instrument"},
            )
            self.assertEqual(
                specification["fit_specification"]["initialization"],
                {"mode": "origin_auto"},
            )
            self.assertEqual(
                specification["fit_specification"]["fit_range"],
                {"minimum": 0.0, "maximum": 11.0, "inclusive": True},
            )
            self.assertEqual(
                specification["fit_specification"]["graph_profile"],
                {"id": "expdec2-standard", "version": "1.0"},
            )
            self.assertEqual(
                specification["fit_specification"]["constraints"],
                {
                    "y0": {"lower": None, "upper": None},
                    "A_fast": {"exclusive_lower": 0.0},
                    "t_fast": {"exclusive_lower": 0.0},
                    "A_slow": {"exclusive_lower": 0.0},
                    "t_slow": {"exclusive_lower": 0.0},
                    "component_order": "t_fast < t_slow",
                },
            )
            self.assertEqual(
                specification["fit_specification"]["data_handling"],
                {
                    "missing_y": "exclude_per_series",
                    "record_exclusions": True,
                    "automatic_cleaning": False,
                },
            )
            self.assertEqual(
                specification["fit_specification"]["experimental_data_contract"][
                    "data_handling"
                ],
                specification["fit_specification"]["data_handling"],
            )

            first_recipe = self.run_cli(
                state_dir,
                "approve",
                proposed["fit_specification_id"],
            )
            self.assertEqual(first_recipe["version"], 1)
            repeated_approval = self.run_cli(
                state_dir,
                "approve",
                proposed["fit_specification_id"],
            )
            self.assertEqual(repeated_approval, first_recipe)

            changed = self.propose_fixture(
                state_dir, snapshot["dataset_snapshot_id"], fit_max="10"
            )
            self.assertNotEqual(
                changed["fit_specification_id"], proposed["fit_specification_id"]
            )
            second_recipe = self.run_cli(
                state_dir,
                "approve",
                changed["fit_specification_id"],
            )
            self.assertEqual(second_recipe["version"], 2)

            original = self.run_cli(
                state_dir, "inspect", first_recipe["approved_fit_recipe_id"]
            )
            self.assertEqual(original["approved_fit_recipe"]["version"], 1)
            self.assertEqual(
                original["approved_fit_recipe"]["approved_by"],
                "researcher@example.test",
            )
            self.assertEqual(
                original["approved_fit_recipe"]["fit_specification"]["fit_range"],
                {"minimum": 0.0, "maximum": 11.0, "inclusive": True},
            )

            audit = self.run_cli(state_dir, "inspect", "audit")
            audit_text = json.dumps(audit, sort_keys=True)
            self.assertIn("fit_specification.proposed", audit_text)
            self.assertIn("approved_fit_recipe.approved", audit_text)
            self.assertNotIn("12.010", audit_text)
            self.assertNotIn("model_prompt", audit_text)
            self.assertNotIn("token", audit_text.lower())

    def test_propose_supports_explicit_initial_values_for_every_y(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            snapshot = self.import_fixture(state_dir)
            initial_values = {
                "decay_a": {
                    "y0": 1.0,
                    "A_fast": 7.0,
                    "t_fast": 0.8,
                    "A_slow": 4.0,
                    "t_slow": 5.0,
                },
                "decay_b": {
                    "y0": 1.0,
                    "A_fast": 5.0,
                    "t_fast": 1.0,
                    "A_slow": 3.0,
                    "t_slow": 6.0,
                },
                "decay_c": {
                    "y0": 2.0,
                    "A_fast": 9.0,
                    "t_fast": 0.7,
                    "A_slow": 4.0,
                    "t_slow": 4.5,
                },
            }
            values_path = Path(temporary_directory) / "initial-values.json"
            values_path.write_text(json.dumps(initial_values), encoding="utf-8")

            proposed = self.run_cli(
                state_dir,
                "propose",
                snapshot["dataset_snapshot_id"],
                "--experiment-id",
                "synthetic-expdec2",
                "--fit-min",
                "0",
                "--fit-max",
                "11",
                "--weighting",
                "none",
                "--initialization",
                "explicit",
                "--initial-values",
                str(values_path),
                "--graph-profile",
                "expdec2-standard@1.0",
            )
            inspected = self.run_cli(
                state_dir, "inspect", proposed["fit_specification_id"]
            )
            self.assertEqual(
                inspected["fit_specification"]["initialization"],
                {"mode": "explicit", "values_by_y": initial_values},
            )

            del initial_values["decay_c"]
            values_path.write_text(json.dumps(initial_values), encoding="utf-8")
            error = self.run_cli(
                state_dir,
                "propose",
                snapshot["dataset_snapshot_id"],
                "--experiment-id",
                "synthetic-expdec2",
                "--fit-min",
                "0",
                "--fit-max",
                "11",
                "--weighting",
                "none",
                "--initialization",
                "explicit",
                "--initial-values",
                str(values_path),
                "--graph-profile",
                "expdec2-standard@1.0",
                succeeds=False,
            )
            self.assertEqual(error["error"], "invalid_fit_specification")

    def test_propose_rejects_an_invalid_fit_range_or_weighting_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state_dir = root / "state"
            snapshot = self.import_fixture(state_dir)
            range_error = self.run_cli(
                state_dir,
                "propose",
                snapshot["dataset_snapshot_id"],
                "--experiment-id",
                "synthetic-expdec2",
                "--fit-min",
                "0",
                "--fit-max",
                "4",
                "--weighting",
                "none",
                "--initialization",
                "origin-auto",
                "--graph-profile",
                "expdec2-standard@1.0",
                succeeds=False,
            )
            self.assertEqual(range_error["error"], "invalid_fit_specification")

            csv_path = root / "without-uncertainty.csv"
            csv_path.write_text(
                "x,y\n0,6\n1,5\n2,4\n3,3\n4,2\n5,1\n", encoding="utf-8"
            )
            unweighted_snapshot = self.import_two_columns(state_dir, csv_path)
            weighting_error = self.run_cli(
                state_dir,
                "propose",
                unweighted_snapshot["dataset_snapshot_id"],
                "--experiment-id",
                "unweighted",
                "--fit-min",
                "0",
                "--fit-max",
                "5",
                "--weighting",
                "instrument",
                "--initialization",
                "origin-auto",
                "--graph-profile",
                "expdec2-standard@1.0",
                succeeds=False,
            )
            self.assertEqual(weighting_error["error"], "invalid_fit_specification")


if __name__ == "__main__":
    unittest.main()
