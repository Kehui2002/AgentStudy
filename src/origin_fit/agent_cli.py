"""Researcher-facing interactive Agent CLI for the Origin Integration Application."""

from __future__ import annotations

from collections.abc import Callable
import csv
import json
import shlex
from typing import Literal, Protocol, TextIO

from mini_agent import Agent, Model, ModelResponse, ToolCallPart, ToolError

from .contracts import WorkerJob
from .datasets import inspect_dataset
from .errors import OriginFitError
from .execution import FitResult, accept_fit_result
from .remote import PendingFitJob, RemoteFitOutcome
from .specifications import approve_fit_specification, propose_fit_specification
from .storage import LocalStore


class FitExecutor(Protocol):
    @property
    def transport(self) -> WorkerControl: ...

    async def execute_approved_fit(
        self,
        store: LocalStore,
        dataset_snapshot_id: str,
        approved_fit_recipe_id: str,
        *,
        wait_timeout: float = 1800,
        poll_interval: float = 0.25,
    ) -> RemoteFitOutcome: ...


class WorkerControl(Protocol):
    async def status(self, worker_job_id: str) -> WorkerJob: ...

    async def cancel(self, worker_job_id: str) -> WorkerJob: ...


def _proposal_tool(
    store: LocalStore, dataset_snapshot_id: str
) -> Callable[..., dict[str, str]]:
    def propose_expdec2_fit(
        experiment_id: str,
        fit_minimum: float,
        fit_maximum: float,
        weighting: Literal["none", "instrument"],
        initialization: Literal["origin_auto", "explicit"],
        initial_values: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, str]:
        """Propose, but never approve, an ExpDec2 Fit Specification for the selected dataset."""
        try:
            return propose_fit_specification(
                store,
                dataset_snapshot_id,
                experiment_id=experiment_id,
                fit_minimum=fit_minimum,
                fit_maximum=fit_maximum,
                weighting=weighting,
                initialization=initialization,
                graph_profile_id="expdec2-standard",
                graph_profile_version="1.0",
                initial_values=initial_values,
            )
        except OriginFitError as error:
            raise ToolError(code=error.code, message=error.message) from error

    return propose_expdec2_fit


def _downsampled_preview(store: LocalStore, dataset: dict) -> dict:
    """Read a deterministic subset only after the CLI has received authorization."""
    content_path = store.objects_dir / dataset["content_hash"]
    source_row_count = int(dataset["summary"]["row_count"])
    preview_count = min(5, max(source_row_count - 1, 1))
    if preview_count == 1:
        indexes = [0]
    else:
        indexes = [
            index * (source_row_count - 1) // (preview_count - 1)
            for index in range(preview_count)
        ]
    selected_indexes = set(indexes)
    try:
        with content_path.open(encoding="utf-8", newline="") as dataset_file:
            reader = csv.DictReader(dataset_file)
            preview_rows = {
                index: row
                for index, row in enumerate(reader)
                if index in selected_indexes
            }
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise OriginFitError(
            "dataset_unavailable", "Dataset Snapshot preview could not be read."
        ) from error
    if len(preview_rows) != len(indexes):
        raise OriginFitError(
            "dataset_integrity_error",
            "Dataset Snapshot does not match its persisted summary.",
        )
    columns = [
        dataset["metadata"]["x_column"],
        *dataset["metadata"]["y_columns"],
    ]
    return {
        "preview_row_count": len(indexes),
        "source_row_count": source_row_count,
        "columns": columns,
        "rows": [
            {
                "source_row_number": index + 2,
                "values": {
                    column: preview_rows[index][column] for column in columns
                },
            }
            for index in indexes
        ],
    }


def _agent_prompt(
    request: str,
    dataset: dict,
    downsampled_preview: dict | None,
    latest_result_summary: dict | None,
) -> str:
    summary = dict(dataset["summary"])
    summary["preview_included"] = downsampled_preview is not None
    context = {
        "dataset_snapshot_id": dataset["dataset_snapshot_id"],
        "dataset_content_hash": dataset["content_hash"],
        "dataset_summary": summary,
        "allowed_domain_options": {
            "model": "ExpDec2",
            "fit_kind": "independent_multi_series",
            "weighting": ["none", "instrument"],
            "initialization": ["origin_auto", "explicit"],
            "graph_profile": "expdec2-standard@1.0",
        },
    }
    if downsampled_preview is not None:
        context["downsampled_preview"] = downsampled_preview
    if latest_result_summary is not None:
        context["latest_fit_result_summary"] = latest_result_summary
    return (
        "You assist a researcher with an Origin ExpDec2 fit. You may only propose a "
        "Fit Specification by using propose_expdec2_fit. You cannot import data, approve "
        "or change a recipe, run or cancel a Fit Job, or accept a Fit Result. Those are "
        "explicit CLI actions outside model control. Never claim that one of those actions "
        "has occurred. Context:"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
        + "\nResearcher request: "
        + request
    )


def _compressed_result(result: FitResult) -> dict:
    statistic_names = ("r_squared", "reduced_chi_square", "sse", "dof")
    return {
        "fit_result_id": result.fit_result_id,
        "classification": result.classification,
        "scientific_status": result.scientific_status,
        "series": [
            {
                "name": outcome.series_name,
                "status": outcome.status,
                "converged": outcome.converged,
                "valid_point_count": outcome.valid_point_count,
                "warnings": [warning[:160] for warning in outcome.warnings[:8]],
                "fit_statistics": {
                    name: outcome.fit_statistics[name]
                    for name in statistic_names
                    if name in outcome.fit_statistics
                },
            }
            for outcome in result.series_outcomes
        ],
    }


async def run_agent_cli(
    store: LocalStore,
    model: Model,
    executor: FitExecutor | None,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> None:
    """Run an interactive CLI while keeping every authority action outside the model."""
    selected: dict | None = None
    downsampled_preview: dict | None = None
    latest_result_summary: dict | None = None
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line == "/quit":
            return
        if line.startswith("select "):
            snapshot_id = line.removeprefix("select ").strip()
            selected = None
            downsampled_preview = None
            latest_result_summary = None
            try:
                selected = inspect_dataset(store, snapshot_id)
            except OriginFitError as error:
                print(f"Error [{error.code}]: {error.message}", file=stdout)
            else:
                print(
                    "Selected Dataset Snapshot "
                    f"{selected['dataset_snapshot_id']} "
                    f"(content_hash={selected['content_hash']}).",
                    file=stdout,
                )
            continue
        if line == "/preview authorize":
            if selected is None:
                print(
                    "Error [dataset_required]: Select a Dataset Snapshot first.",
                    file=stdout,
                )
                continue
            try:
                downsampled_preview = _downsampled_preview(store, selected)
            except OriginFitError as error:
                print(f"Error [{error.code}]: {error.message}", file=stdout)
                continue
            with store.connect() as connection:
                store.audit(
                    connection,
                    "dataset_preview.authorized",
                    selected["dataset_snapshot_id"],
                    {
                        "preview_row_count": downsampled_preview["preview_row_count"],
                        "source_row_count": downsampled_preview["source_row_count"],
                    },
                )
            print(
                "Authorized a bounded downsampled preview "
                f"({downsampled_preview['preview_row_count']} rows).",
                file=stdout,
            )
            continue
        if line.startswith("/approve"):
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = []
            if len(parts) != 2 or parts[0] != "/approve":
                print(
                    "Error [invalid_command]: Usage: /approve FIT_SPECIFICATION_ID",
                    file=stdout,
                )
                continue
            try:
                approved = approve_fit_specification(store, parts[1])
            except OriginFitError as error:
                print(f"Error [{error.code}]: {error.message}", file=stdout)
                continue
            print(
                f"Approved Fit Recipe {approved['approved_fit_recipe_id']} "
                f"(version={approved['version']}, content_hash={approved['content_hash']}).",
                file=stdout,
            )
            continue
        if line.startswith("/accept"):
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = []
            if len(parts) != 2 or parts[0] != "/accept":
                print(
                    "Error [invalid_command]: Usage: /accept FIT_RESULT_ID",
                    file=stdout,
                )
                continue
            try:
                accepted = accept_fit_result(store, parts[1])
            except OriginFitError as error:
                print(f"Error [{error.code}]: {error.message}", file=stdout)
                continue
            print(
                f"Accepted Fit {accepted['accepted_fit_id']} "
                f"for Fit Result {accepted['fit_result_id']} "
                f"at {accepted['accepted_at']}.",
                file=stdout,
            )
            continue
        if line.startswith("/status") or line.startswith("/cancel"):
            command = "/status" if line.startswith("/status") else "/cancel"
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = []
            if len(parts) != 2 or parts[0] != command:
                print(
                    f"Error [invalid_command]: Usage: {command} WORKER_JOB_ID",
                    file=stdout,
                )
                continue
            if executor is None:
                print(
                    "Error [worker_unavailable]: Origin Worker is not configured.",
                    file=stdout,
                )
                continue
            try:
                if command == "/status":
                    job = await executor.transport.status(parts[1])
                    event_type = "worker_job.status_observed"
                else:
                    job = await executor.transport.cancel(parts[1])
                    event_type = "worker_job.cancellation_requested"
            except OriginFitError as error:
                print(f"Error [{error.code}]: {error.message}", file=stdout)
                continue
            with store.connect() as connection:
                store.audit(
                    connection,
                    event_type,
                    job.worker_job_id,
                    {"status": job.status, "error_code": job.error_code},
                )
            details = [f"status={job.status}"]
            if job.bundle_sha256 is not None:
                details.append(f"bundle_sha256={job.bundle_sha256}")
            if job.error_code is not None:
                details.append(f"error={job.error_code}")
            print(
                f"Fit Job {job.worker_job_id}: " + ", ".join(details) + ".",
                file=stdout,
            )
            continue
        if line.startswith("/run"):
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = []
            if len(parts) != 2 or parts[0] != "/run":
                print(
                    "Error [invalid_command]: Usage: /run APPROVED_FIT_RECIPE_ID",
                    file=stdout,
                )
                continue
            if selected is None:
                print(
                    "Error [dataset_required]: Select a Dataset Snapshot first.",
                    file=stdout,
                )
                continue
            if executor is None:
                print(
                    "Error [worker_unavailable]: Origin Worker is not configured.",
                    file=stdout,
                )
                continue
            try:
                outcome = await executor.execute_approved_fit(
                    store,
                    selected["dataset_snapshot_id"],
                    parts[1],
                )
            except OriginFitError as error:
                print(f"Error [{error.code}]: {error.message}", file=stdout)
                continue
            if isinstance(outcome, PendingFitJob):
                print(
                    f"Fit Job {outcome.worker_job_id} remains pending "
                    f"(worker_status={outcome.worker_status}). {outcome.message}",
                    file=stdout,
                )
            else:
                latest_result_summary = _compressed_result(outcome.fit_result)
                classification = outcome.fit_result.classification
                warning = (
                    " REVIEW REQUIRED: inspect all diagnostics before acceptance."
                    if classification == "review_required"
                    else ""
                )
                print(
                    f"Fit Job {outcome.worker_job_id} succeeded; "
                    f"Fit Result {outcome.fit_result.fit_result_id}; "
                    f"Fit Archive {outcome.fit_archive_id}; "
                    f"bundle_sha256={outcome.bundle_sha256}; "
                    f"classification={classification}.{warning}",
                    file=stdout,
                )
            continue
        if line.startswith("/"):
            print("Error [unknown_command]: Unknown interactive command.", file=stdout)
            continue
        if selected is None:
            print(
                "Error [dataset_required]: Select a Dataset Snapshot first.",
                file=stdout,
            )
            continue
        tool = _proposal_tool(store, selected["dataset_snapshot_id"])
        result = await Agent(model, tools=[tool]).run(
            _agent_prompt(
                line,
                selected,
                downsampled_preview,
                latest_result_summary,
            )
        )
        rejected_tools = [
            part.tool_name
            for message in result.all_messages()
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart)
            and part.tool_name != "propose_expdec2_fit"
        ]
        if rejected_tools:
            with store.connect() as connection:
                store.audit(
                    connection,
                    "model.authority_action.rejected",
                    selected["dataset_snapshot_id"],
                    {"tool_names": sorted(set(rejected_tools))},
                )
            print(
                "Rejected model-requested authority action; use an explicit CLI command.",
                file=stdout,
            )
        else:
            print(result.output, file=stdout)


__all__ = ("FitExecutor", "run_agent_cli")
