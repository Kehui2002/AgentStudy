from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from typing import TYPE_CHECKING, Any
import zipfile

from origin_fit.contracts import (
    FIT_RESULT_SCHEMA_VERSION,
    FIT_SPECIFICATION_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    FitResultManifest,
    GraphProfileCapability,
    ManifestObject,
    ManifestRecipe,
    ManifestSchemas,
    ManifestSoftware,
    ManifestSpecification,
    WorkerSubmission,
)
from origin_fit.execution import FitResult, OriginExecutionRequest
from origin_fit.storage import utc_now

if TYPE_CHECKING:
    from .originpro_adapter import OriginGraphArtifacts


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _csv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _fit_value(x: float, parameters: dict[str, float]) -> float:
    return (
        parameters["y0"]
        + parameters["A_fast"] * math.exp(-x / parameters["t_fast"])
        + parameters["A_slow"] * math.exp(-x / parameters["t_slow"])
    )


def _data_artifacts(
    request: OriginExecutionRequest, result: FitResult
) -> tuple[bytes, bytes]:
    outcomes = {item.series_name: item for item in result.series_outcomes}
    fitted_rows: list[list[object]] = []
    residual_rows: list[list[object]] = []
    for series in request.series:
        outcome = outcomes[series.series_name]
        parameters = outcome.parameters
        canonical = parameters.model_dump() if parameters is not None else None
        for x, observed in zip(series.x, series.y):
            fitted = _fit_value(x, canonical) if canonical is not None else ""
            residual = observed - fitted if isinstance(fitted, float) else ""
            fitted_rows.append([series.series_name, x, observed, fitted])
            residual_rows.append([series.series_name, x, residual])
    return (
        _csv_bytes(
            ["series_name", "x", "observed_y", "fitted_y"], fitted_rows
        ),
        _csv_bytes(["series_name", "x", "residual"], residual_rows),
    )


def _fake_graph_artifacts(result: FitResult) -> tuple[bytes, bytes, bytes]:
    description = _json_bytes(
        {
            "graph_profile": "expdec2-standard@1.0",
            "model": result.model,
            "series": [item.series_name for item in result.series_outcomes],
            "rule": "observed points and fitted line share one color per series",
        }
    )
    png = b"\x89PNG\r\n\x1a\nFAKE-ORIGIN-GRAPH\n" + description
    pdf = b"%PDF-1.4\n% Fake Origin combined graph\n" + description + b"\n%%EOF\n"
    opju = b"FAKE-ORIGINPRO-2025-OPJU\n" + description
    return png, pdf, opju


def build_result_bundle(
    *,
    worker_job_id: str,
    request: OriginExecutionRequest,
    result: FitResult,
    submission: WorkerSubmission,
    adapter_name: str,
    originpro_version: str,
    graph_artifacts: OriginGraphArtifacts | None = None,
) -> tuple[bytes, FitResultManifest]:
    approved_fit_recipe = submission.approved_fit_recipe
    specification = approved_fit_recipe["fit_specification"]
    fitted_data, residuals = _data_artifacts(request, result)
    if adapter_name.startswith("originpro-2025-adapter/") and graph_artifacts is None:
        raise ValueError(
            "The Production Origin Adapter did not provide graph artifacts."
        )
    if graph_artifacts is None:
        png, pdf, opju = _fake_graph_artifacts(result)
    else:
        expected_profile = (
            f"{specification['graph_profile']['id']}@"
            f"{specification['graph_profile']['version']}"
        )
        if graph_artifacts.graph_profile != expected_profile:
            raise ValueError(
                "Origin graph artifacts do not match the approved profile."
            )
        png, pdf, opju = (
            graph_artifacts.png,
            graph_artifacts.pdf,
            graph_artifacts.opju,
        )
    exclusions = _csv_bytes(
        ["series_name", "row_number", "reason"],
        [
            [item.series_name, item.row_number, item.reason]
            for item in result.exclusions
        ],
    )
    artifacts = {
        "result.json": _json_bytes(result.model_dump(mode="json")),
        "fitted-data.csv": fitted_data,
        "residuals.csv": residuals,
        "exclusions.csv": exclusions,
        "combined.png": png,
        "combined.pdf": pdf,
        "project.opju": opju,
    }
    manifest = FitResultManifest(
        schema_version="1.0",
        worker_job_id=worker_job_id,
        status="succeeded",
        created_at=utc_now(),
        dataset_snapshot=ManifestObject(
            id=submission.dataset_snapshot_id,
            sha256=submission.dataset_content_hash,
        ),
        approved_fit_recipe=ManifestRecipe(
            id=submission.approved_fit_recipe_id,
            sha256=submission.approved_fit_recipe_hash,
            version=approved_fit_recipe["version"],
        ),
        fit_specification=ManifestSpecification(
            id=approved_fit_recipe["fit_specification_id"],
            sha256=approved_fit_recipe["fit_specification_hash"],
            schema_version=specification["schema_version"],
        ),
        software=ManifestSoftware(
            worker="origin-worker/0.1.0",
            adapter=adapter_name,
            originpro=originpro_version,
        ),
        graph_profile=GraphProfileCapability.model_validate(
            specification["graph_profile"], strict=True
        ),
        schemas=ManifestSchemas(
            fit_specification=FIT_SPECIFICATION_SCHEMA_VERSION,
            fit_result=FIT_RESULT_SCHEMA_VERSION,
            manifest=MANIFEST_SCHEMA_VERSION,
        ),
        files={
            name: hashlib.sha256(content).hexdigest()
            for name, content in artifacts.items()
        },
    )
    members = {**artifacts, "manifest.json": _json_bytes(manifest.model_dump())}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100444 << 16
            archive.writestr(info, members[name])
    return output.getvalue(), manifest
