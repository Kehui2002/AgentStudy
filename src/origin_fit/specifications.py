from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .errors import OriginFitError
from .datasets import valid_points_in_range
from .storage import LocalStore, utc_now


EXPDEC2_FORMULA = (
    "y = y0 + A_fast*exp(-x/t_fast) + A_slow*exp(-x/t_slow)"
)
REQUIRED_OUTPUTS = [
    "result.json",
    "fitted-data.csv",
    "residuals.csv",
    "exclusions.csv",
    "combined.png",
    "combined.pdf",
    "project.opju",
    "manifest.json",
]
APPROVED_GRAPH_PROFILES = {("expdec2-standard", "1.0")}
_TEMPLATE_ID_PATTERN = re.compile(r"^template:[a-z0-9][a-z0-9-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: dict) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def propose_fit_specification(
    store: LocalStore,
    snapshot_id: str,
    *,
    experiment_id: str,
    fit_minimum: float,
    fit_maximum: float,
    weighting: str,
    initialization: str,
    graph_profile_id: str,
    graph_profile_version: str,
    template_id: str,
    template_version: int,
    template_sha256: str,
    initial_values: dict | None = None,
) -> dict:
    initialization_contract: dict[str, Any]
    with store.connect() as connection:
        snapshot = connection.execute(
            """
            SELECT id, content_hash, metadata_json, summary_json
            FROM dataset_snapshots WHERE id = ?
            """,
            (snapshot_id,),
        ).fetchone()
        if snapshot is None:
            raise OriginFitError(
                "not_found", f"Dataset Snapshot '{snapshot_id}' not found."
            )
        metadata = json.loads(snapshot["metadata_json"])
        summary = json.loads(snapshot["summary_json"])
        if not experiment_id.strip():
            raise OriginFitError(
                "invalid_fit_specification", "Experiment identity must not be empty."
            )
        if (graph_profile_id, graph_profile_version) not in APPROVED_GRAPH_PROFILES:
            raise OriginFitError(
                "invalid_fit_specification",
                "Graph Profile is not present in the human-maintained approved profile registry.",
            )
        if (
            not _TEMPLATE_ID_PATTERN.fullmatch(template_id)
            or isinstance(template_version, bool)
            or not isinstance(template_version, int)
            or template_version < 1
            or not _SHA256_PATTERN.fullmatch(template_sha256)
        ):
            raise OriginFitError(
                "invalid_fit_specification",
                "A Registered Origin Graph Template selection (id, version, sha256) is required.",
            )
        x_minimum = summary["x"]["minimum"]
        x_maximum = summary["x"]["maximum"]
        if (
            fit_minimum >= fit_maximum
            or fit_minimum < x_minimum
            or fit_maximum > x_maximum
        ):
            raise OriginFitError(
                "invalid_fit_specification",
                "The inclusive fit range must lie within the Dataset Snapshot X range.",
            )
        valid_counts = valid_points_in_range(
            store,
            snapshot["content_hash"],
            metadata,
            fit_minimum,
            fit_maximum,
        )
        insufficient = [name for name, count in valid_counts.items() if count < 6]
        if insufficient:
            raise OriginFitError(
                "invalid_fit_specification",
                "The inclusive fit range must contain at least 6 valid points for every Y: "
                + ", ".join(insufficient),
            )
        if weighting == "instrument" and any(
            y_name not in metadata["uncertainty_columns"]
            for y_name in metadata["y_columns"]
        ):
            raise OriginFitError(
                "invalid_fit_specification",
                "Instrument weighting requires an uncertainty column for every Y series.",
            )
        initialization_mode = initialization.replace("-", "_")
        if initialization_mode == "origin_auto":
            if initial_values is not None:
                raise OriginFitError(
                    "invalid_fit_specification",
                    "Origin automatic initialization does not accept explicit values.",
                )
            initialization_contract = {"mode": "origin_auto"}
        else:
            if not isinstance(initial_values, dict) or set(initial_values) != set(
                metadata["y_columns"]
            ):
                raise OriginFitError(
                    "invalid_fit_specification",
                    "Explicit initialization requires values for every selected Y series.",
                )
            parameter_names = {"y0", "A_fast", "t_fast", "A_slow", "t_slow"}
            for y_name, parameters in initial_values.items():
                if not isinstance(parameters, dict) or set(parameters) != parameter_names:
                    raise OriginFitError(
                        "invalid_fit_specification",
                        f"Explicit initialization for '{y_name}' must define all ExpDec2 parameters.",
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in parameters.values()
                ):
                    raise OriginFitError(
                        "invalid_fit_specification",
                        f"Explicit initialization for '{y_name}' must contain finite numbers.",
                    )
                if (
                    parameters["A_fast"] <= 0
                    or parameters["t_fast"] <= 0
                    or parameters["A_slow"] <= 0
                    or parameters["t_slow"] <= parameters["t_fast"]
                ):
                    raise OriginFitError(
                        "invalid_fit_specification",
                        f"Explicit initialization for '{y_name}' must satisfy ExpDec2 constraints.",
                    )
            initialization_contract = {
                "mode": "explicit",
                "values_by_y": initial_values,
            }

        y_series = [
            {
                "name": y_name,
                "unit": metadata["units"][y_name],
                "uncertainty": (
                    {
                        "name": metadata["uncertainty_columns"][y_name],
                        "unit": metadata["units"][
                            metadata["uncertainty_columns"][y_name]
                        ],
                    }
                    if y_name in metadata["uncertainty_columns"]
                    else None
                ),
            }
            for y_name in metadata["y_columns"]
        ]
        constraints = {
            "y0": {"lower": None, "upper": None},
            "A_fast": {"exclusive_lower": 0.0},
            "t_fast": {"exclusive_lower": 0.0},
            "A_slow": {"exclusive_lower": 0.0},
            "t_slow": {"exclusive_lower": 0.0},
            "component_order": "t_fast < t_slow",
        }
        data_handling = {
            "missing_y": "exclude_per_series",
            "record_exclusions": True,
            "automatic_cleaning": False,
        }
        specification = {
            "schema_version": "1.0",
            "dataset_snapshot_id": snapshot["id"],
            "dataset_content_hash": snapshot["content_hash"],
            "execution_authorized": False,
            "model": {
                "name": "ExpDec2",
                "formula": EXPDEC2_FORMULA,
                "x_offset_fitted": False,
            },
            "shared_x_column": metadata["x_column"],
            "y_series": y_series,
            "fit_range": {
                "minimum": fit_minimum,
                "maximum": fit_maximum,
                "inclusive": True,
            },
            "initialization": initialization_contract,
            "constraints": constraints,
            "data_handling": data_handling,
            "weighting": {"mode": weighting},
            "units": metadata["units"],
            "graph_profile": {
                "id": graph_profile_id,
                "version": graph_profile_version,
            },
            "graph_template": {
                "template_id": template_id,
                "version": template_version,
                "sha256": template_sha256,
            },
            "output_requirements": REQUIRED_OUTPUTS,
            "experimental_data_contract": {
                "experiment_id": experiment_id,
                "dimensionality": "one_shared_x_multiple_independent_y",
                "x_column": metadata["x_column"],
                "y_series": y_series,
                "units": metadata["units"],
                "independent_variable_range": {
                    "minimum": fit_minimum,
                    "maximum": fit_maximum,
                    "inclusive": True,
                },
                "model": "ExpDec2",
                "initialization": initialization_contract,
                "weighting": weighting,
                "constraints": constraints,
                "data_handling": data_handling,
                "graph_profile": {
                    "id": graph_profile_id,
                    "version": graph_profile_version,
                },
            },
        }
        digest = _digest(specification)
        specification_id = f"spec:sha256:{digest}"
        connection.execute(
            """
            INSERT OR IGNORE INTO fit_specifications (
                id, content_hash, dataset_snapshot_id, created_at, specification_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                specification_id,
                digest,
                snapshot_id,
                utc_now(),
                _canonical_bytes(specification).decode("utf-8"),
            ),
        )
        store.audit(
            connection,
            "fit_specification.proposed",
            specification_id,
            {"content_hash": digest, "dataset_snapshot_id": snapshot_id},
        )
    return {"fit_specification_id": specification_id, "content_hash": digest}


def approve_fit_specification(store: LocalStore, specification_id: str) -> dict:
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT id, content_hash, version FROM approved_fit_recipes
            WHERE fit_specification_id = ?
            """,
            (specification_id,),
        ).fetchone()
        if existing is not None:
            return {
                "approved_fit_recipe_id": existing["id"],
                "content_hash": existing["content_hash"],
                "version": existing["version"],
            }
        specification_row = connection.execute(
            """
            SELECT id, content_hash, specification_json FROM fit_specifications
            WHERE id = ?
            """,
            (specification_id,),
        ).fetchone()
        if specification_row is None:
            raise OriginFitError(
                "not_found", f"Fit Specification '{specification_id}' not found."
            )
        specification = json.loads(specification_row["specification_json"])
        template = specification.get("graph_template")
        if not (
            isinstance(template, dict)
            and _TEMPLATE_ID_PATTERN.fullmatch(str(template.get("template_id", "")))
            and isinstance(template.get("version"), int)
            and not isinstance(template.get("version"), bool)
            and int(template["version"]) >= 1
            and _SHA256_PATTERN.fullmatch(str(template.get("sha256", "")))
        ):
            raise OriginFitError(
                "invalid_fit_specification",
                "A Fit Specification must explicitly select a Registered Origin Graph Template before approval.",
            )
        latest = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM approved_fit_recipes"
        ).fetchone()
        version = int(latest["version"]) + 1
        approved_at = utc_now()
        recipe = {
            "schema_version": "1.0",
            "version": version,
            "fit_specification_id": specification_row["id"],
            "fit_specification_hash": specification_row["content_hash"],
            "fit_specification": specification,
            "approved_by": store.operator,
            "approved_at": approved_at,
        }
        digest = _digest(recipe)
        recipe_id = f"recipe:sha256:{digest}"
        connection.execute(
            """
            INSERT INTO approved_fit_recipes (
                id, content_hash, version, fit_specification_id,
                approved_by, approved_at, recipe_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recipe_id,
                digest,
                version,
                specification_id,
                store.operator,
                approved_at,
                _canonical_bytes(recipe).decode("utf-8"),
            ),
        )
        store.audit(
            connection,
            "approved_fit_recipe.approved",
            recipe_id,
            {
                "content_hash": digest,
                "fit_specification_id": specification_id,
                "version": version,
            },
        )
    return {
        "approved_fit_recipe_id": recipe_id,
        "content_hash": digest,
        "version": version,
    }


def inspect_persisted_object(store: LocalStore, object_id: str) -> dict | None:
    with store.connect() as connection:
        if object_id == "audit":
            rows = connection.execute(
                """
                SELECT event_type, occurred_at, actor, object_id, details_json
                FROM audit_events ORDER BY id
                """
            ).fetchall()
            return {
                "audit_events": [
                    {
                        "event_type": row["event_type"],
                        "occurred_at": row["occurred_at"],
                        "actor": row["actor"],
                        "object_id": row["object_id"],
                        "details": json.loads(row["details_json"]),
                    }
                    for row in rows
                ]
            }
        if object_id.startswith("spec:"):
            row = connection.execute(
                """
                SELECT id, content_hash, specification_json
                FROM fit_specifications WHERE id = ?
                """,
                (object_id,),
            ).fetchone()
            if row is not None:
                return {
                    "fit_specification_id": row["id"],
                    "content_hash": row["content_hash"],
                    "fit_specification": json.loads(row["specification_json"]),
                }
        if object_id.startswith("recipe:"):
            row = connection.execute(
                """
                SELECT id, content_hash, recipe_json
                FROM approved_fit_recipes WHERE id = ?
                """,
                (object_id,),
            ).fetchone()
            if row is not None:
                return {
                    "approved_fit_recipe_id": row["id"],
                    "content_hash": row["content_hash"],
                    "approved_fit_recipe": json.loads(row["recipe_json"]),
                }
        if object_id.startswith("fit-result:"):
            row = connection.execute(
                "SELECT id, result_json FROM fit_results WHERE id = ?",
                (object_id,),
            ).fetchone()
            if row is not None:
                return {
                    "fit_result_id": row["id"],
                    "fit_result": json.loads(row["result_json"]),
                }
        if object_id.startswith("accepted-fit:"):
            row = connection.execute(
                "SELECT id, accepted_fit_json FROM accepted_fits WHERE id = ?",
                (object_id,),
            ).fetchone()
            if row is not None:
                return {
                    "accepted_fit_id": row["id"],
                    "accepted_fit": json.loads(row["accepted_fit_json"]),
                }
        if object_id.startswith("fit-archive:"):
            row = connection.execute(
                """
                SELECT id, dataset_snapshot_id, approved_fit_recipe_id,
                       worker_job_id, fit_result_id, bundle_hash,
                       archived_at, manifest_json
                FROM fit_archives WHERE id = ?
                """,
                (object_id,),
            ).fetchone()
            if row is not None:
                return {
                    "fit_archive_id": row["id"],
                    "dataset_snapshot_id": row["dataset_snapshot_id"],
                    "approved_fit_recipe_id": row["approved_fit_recipe_id"],
                    "worker_job_id": row["worker_job_id"],
                    "fit_result_id": row["fit_result_id"],
                    "bundle_sha256": row["bundle_hash"],
                    "archived_at": row["archived_at"],
                    "manifest": json.loads(row["manifest_json"]),
                }
        if object_id.startswith("fit-job:"):
            row = connection.execute(
                """
                SELECT worker_job_id, dataset_snapshot_id,
                       approved_fit_recipe_id, status, submitted_at,
                       updated_at, error_code, bundle_hash, fit_archive_id
                FROM worker_job_mappings WHERE worker_job_id = ?
                """,
                (object_id,),
            ).fetchone()
            if row is not None:
                return {
                    "worker_job_id": row["worker_job_id"],
                    "dataset_snapshot_id": row["dataset_snapshot_id"],
                    "approved_fit_recipe_id": row["approved_fit_recipe_id"],
                    "status": row["status"],
                    "submitted_at": row["submitted_at"],
                    "updated_at": row["updated_at"],
                    "error_code": row["error_code"],
                    "bundle_sha256": row["bundle_hash"],
                    "fit_archive_id": row["fit_archive_id"],
                }
    return None
