from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TRANSPORT_SCHEMA_VERSION = "1.0"
FIT_SPECIFICATION_SCHEMA_VERSION = "1.0"
FIT_RESULT_SCHEMA_VERSION = "1.0"
MANIFEST_SCHEMA_VERSION = "1.0"

BUNDLE_ARTIFACTS = frozenset(
    {
        "result.json",
        "fitted-data.csv",
        "residuals.csv",
        "exclusions.csv",
        "combined.png",
        "combined.pdf",
        "project.opju",
    }
)
BUNDLE_FILES = BUNDLE_ARTIFACTS | {"manifest.json"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class GraphProfileCapability(StrictModel):
    id: str
    version: str


class WorkerCapabilities(StrictModel):
    transport_schema_version: str
    fit_specification_schema_versions: list[str]
    fit_result_schema_versions: list[str]
    manifest_schema_versions: list[str]
    models: list[str]
    graph_profiles: list[GraphProfileCapability]
    max_dataset_bytes: int
    max_rows: int
    max_y_series: int


class WorkerSubmission(StrictModel):
    transport_schema_version: Literal["1.0"]
    dataset_snapshot_id: str
    dataset_content_hash: str
    dataset_base64: str
    dataset_metadata: dict[str, Any]
    dataset_summary: dict[str, Any]
    approved_fit_recipe_id: str
    approved_fit_recipe_hash: str
    approved_fit_recipe: dict[str, Any]


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class WorkerJob(StrictModel):
    worker_job_id: str
    status: JobStatus
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    bundle_sha256: str | None = None


class ManifestObject(StrictModel):
    id: str
    sha256: str


class ManifestRecipe(ManifestObject):
    version: int


class ManifestSpecification(ManifestObject):
    schema_version: str


class ManifestSoftware(StrictModel):
    worker: str
    adapter: str
    originpro: str


class ManifestSchemas(StrictModel):
    fit_specification: str
    fit_result: str
    manifest: str


class FitResultManifest(StrictModel):
    schema_version: Literal["1.0"]
    worker_job_id: str
    status: Literal["succeeded"]
    created_at: str
    dataset_snapshot: ManifestObject
    approved_fit_recipe: ManifestRecipe
    fit_specification: ManifestSpecification
    software: ManifestSoftware
    graph_profile: GraphProfileCapability
    schemas: ManifestSchemas
    files: dict[str, str] = Field(min_length=len(BUNDLE_ARTIFACTS))
