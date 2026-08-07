"""Researcher-facing interactive Agent CLI for the Origin Integration Application."""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass, field
import json
import shlex
from typing import Literal, Protocol, TextIO

from mini_agent import Agent, Model, ModelResponse, ToolCallPart, ToolError

from .contracts import WorkerCapabilities, WorkerJob
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
    async def capabilities(self) -> WorkerCapabilities: ...

    async def status(self, worker_job_id: str) -> WorkerJob: ...

    async def cancel(self, worker_job_id: str) -> WorkerJob: ...


def _proposal_tool(
    store: LocalStore,
    dataset_snapshot_id: str,
    selected_template: dict | None,
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
            if selected_template is None:
                raise OriginFitError(
                    "template_selection_required",
                    "Select a Registered Origin Graph Template explicitly before proposing a fit.",
                )
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
                template_id=selected_template["template_id"],
                template_version=selected_template["version"],
                template_sha256=selected_template["sha256"],
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
    available_templates: list[dict] | None = None,
) -> str:
    summary = dict(dataset["summary"])
    summary["preview_included"] = downsampled_preview is not None
    registered_templates = [
        {
            "template_id": template["template_id"],
            "version": template["version"],
            "sha256": template["sha256"],
        }
        for template in (available_templates or [])
    ]
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
            "graph_template": {
                "selection": "explicit_user_action_required",
                "registered_templates": registered_templates,
            },
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
    statistic_names = (
        "origin_reduced_chi_square",
        "degrees_of_freedom",
        "residual_sum_of_squares",
        "adjusted_r_squared",
        "r_squared",
        "root_mean_square_error",
        "iteration_count",
    )
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


@dataclass
class _AgentCliSession:
    """Own one interactive session while authority remains in explicit handlers."""

    store: LocalStore
    model: Model
    executor: FitExecutor | None
    stdout: TextIO
    selected: dict | None = None
    downsampled_preview: dict | None = None
    latest_result_summary: dict | None = None
    selected_template: dict | None = None
    available_templates: list[dict] = field(default_factory=list)

    async def run(self, stdin: TextIO) -> None:
        for raw_line in stdin:
            line = raw_line.strip()
            if line and not await self._handle_line(line):
                return

    async def _handle_line(self, line: str) -> bool:
        if line == "/quit":
            return False
        if line.startswith("select "):
            self._select_dataset(line.removeprefix("select ").strip())
            return True
        if line == "/templates":
            await self._show_templates()
            return True
        if line.startswith("/template "):
            self._select_template(line)
            return True
        if line == "/preview authorize":
            self._authorize_preview()
            return True
        command_handlers = (
            ("/approve", self._approve),
            ("/accept", self._accept),
            ("/status", self._status),
            ("/cancel", self._cancel),
            ("/run", self._run_fit),
        )
        for command, handler in command_handlers:
            if line.startswith(command):
                await handler(line)
                return True
        if line.startswith("/"):
            print(
                "Error [unknown_command]: Unknown interactive command.",
                file=self.stdout,
            )
            return True
        await self._ask_agent(line)
        return True

    def _print_error(self, error: OriginFitError) -> None:
        print(f"Error [{error.code}]: {error.message}", file=self.stdout)

    def _single_argument(
        self, line: str, command: str, argument_name: str
    ) -> str | None:
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = []
        if len(parts) != 2 or parts[0] != command:
            print(
                f"Error [invalid_command]: Usage: {command} {argument_name}",
                file=self.stdout,
            )
            return None
        return parts[1]

    def _require_dataset(self) -> dict | None:
        if self.selected is None:
            print(
                "Error [dataset_required]: Select a Dataset Snapshot first.",
                file=self.stdout,
            )
        return self.selected

    def _require_executor(self) -> FitExecutor | None:
        if self.executor is None:
            print(
                "Error [worker_unavailable]: Origin Worker is not configured.",
                file=self.stdout,
            )
        return self.executor

    def _select_dataset(self, snapshot_id: str) -> None:
        self.selected = None
        self.downsampled_preview = None
        self.latest_result_summary = None
        try:
            self.selected = inspect_dataset(self.store, snapshot_id)
        except OriginFitError as error:
            self._print_error(error)
            return
        print(
            "Selected Dataset Snapshot "
            f"{self.selected['dataset_snapshot_id']} "
            f"(content_hash={self.selected['content_hash']}).",
            file=self.stdout,
        )

    def _authorize_preview(self) -> None:
        selected = self._require_dataset()
        if selected is None:
            return
        try:
            self.downsampled_preview = _downsampled_preview(self.store, selected)
        except OriginFitError as error:
            self._print_error(error)
            return
        with self.store.connect() as connection:
            self.store.audit(
                connection,
                "dataset_preview.authorized",
                selected["dataset_snapshot_id"],
                {
                    "preview_row_count": self.downsampled_preview["preview_row_count"],
                    "source_row_count": self.downsampled_preview["source_row_count"],
                },
            )
        print(
            "Authorized a bounded downsampled preview "
            f"({self.downsampled_preview['preview_row_count']} rows).",
            file=self.stdout,
        )

    async def _show_templates(self) -> None:
        executor = self._require_executor()
        if executor is None:
            return
        try:
            capabilities = await executor.transport.capabilities()
        except OriginFitError as error:
            self._print_error(error)
            return
        self.available_templates = [
            template.model_dump(mode="json")
            for template in capabilities.graph_templates
        ]
        suggestion = self.available_templates[0] if self.available_templates else None
        with self.store.connect() as connection:
            self.store.audit(
                connection,
                "graph_template.suggested",
                "worker",
                {
                    "suggestion": (
                        {
                            "template_id": suggestion["template_id"],
                            "version": suggestion["version"],
                            "sha256": suggestion["sha256"],
                        }
                        if suggestion is not None
                        else None
                    ),
                    "template_count": len(self.available_templates),
                },
            )
        for template in self.available_templates:
            print(
                f"Template {template['template_id']}@{template['version']} "
                f"sha256={template['sha256']} "
                f"profile={template['graph_profile']['id']}@"
                f"{template['graph_profile']['version']}.",
                file=self.stdout,
            )
        if suggestion is None:
            print(
                "No registered Origin graph templates are available.",
                file=self.stdout,
            )
            return
        print(
            f"Suggestion only: {suggestion['template_id']}@"
            f"{suggestion['version']}; you must still select it explicitly.",
            file=self.stdout,
        )

    def _select_template(self, line: str) -> None:
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = []
        if len(parts) != 3 or parts[0] != "/template":
            print(
                "Error [invalid_command]: Usage: /template TEMPLATE_ID@VERSION SHA256",
                file=self.stdout,
            )
            return
        template_reference = parts[1]
        template_sha256 = parts[2]
        if "@" not in template_reference:
            print(
                "Error [invalid_command]: Usage: /template TEMPLATE_ID@VERSION SHA256",
                file=self.stdout,
            )
            return
        template_id, raw_version = template_reference.rsplit("@", 1)
        try:
            version = int(raw_version)
        except ValueError:
            version = 0
        reference = {
            "template_id": template_id,
            "version": version,
            "sha256": template_sha256,
        }
        matches = [
            template
            for template in self.available_templates
            if template["template_id"] == template_id
            and template["version"] == version
            and template["sha256"] == template_sha256
        ]
        if self.available_templates and not matches:
            print(
                "Error [template_selection_rejected]: Selection must match a listed "
                "active template from /templates.",
                file=self.stdout,
            )
            return
        self.selected_template = reference
        with self.store.connect() as connection:
            self.store.audit(
                connection,
                "graph_template.selected",
                template_id,
                {"version": version, "sha256": template_sha256},
            )
        print(
            f"Selected Registered Origin Graph Template {template_id}@{version} "
            f"(sha256={template_sha256}).",
            file=self.stdout,
        )

    async def _approve(self, line: str) -> None:
        specification_id = self._single_argument(
            line, "/approve", "FIT_SPECIFICATION_ID"
        )
        if specification_id is None:
            return
        try:
            approved = approve_fit_specification(self.store, specification_id)
        except OriginFitError as error:
            self._print_error(error)
            return
        print(
            f"Approved Fit Recipe {approved['approved_fit_recipe_id']} "
            f"(version={approved['version']}, content_hash={approved['content_hash']}).",
            file=self.stdout,
        )

    async def _accept(self, line: str) -> None:
        fit_result_id = self._single_argument(line, "/accept", "FIT_RESULT_ID")
        if fit_result_id is None:
            return
        try:
            accepted = accept_fit_result(self.store, fit_result_id)
        except OriginFitError as error:
            self._print_error(error)
            return
        print(
            f"Accepted Fit {accepted['accepted_fit_id']} "
            f"for Fit Result {accepted['fit_result_id']} "
            f"at {accepted['accepted_at']}.",
            file=self.stdout,
        )

    async def _status(self, line: str) -> None:
        await self._control_worker(line, "/status")

    async def _cancel(self, line: str) -> None:
        await self._control_worker(line, "/cancel")

    async def _control_worker(self, line: str, command: str) -> None:
        worker_job_id = self._single_argument(line, command, "WORKER_JOB_ID")
        if worker_job_id is None:
            return
        executor = self._require_executor()
        if executor is None:
            return
        try:
            if command == "/status":
                job = await executor.transport.status(worker_job_id)
                event_type = "worker_job.status_observed"
            else:
                job = await executor.transport.cancel(worker_job_id)
                event_type = "worker_job.cancellation_requested"
        except OriginFitError as error:
            self._print_error(error)
            return
        with self.store.connect() as connection:
            self.store.audit(
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
            file=self.stdout,
        )

    async def _run_fit(self, line: str) -> None:
        recipe_id = self._single_argument(
            line, "/run", "APPROVED_FIT_RECIPE_ID"
        )
        if recipe_id is None:
            return
        selected = self._require_dataset()
        if selected is None:
            return
        executor = self._require_executor()
        if executor is None:
            return
        try:
            outcome = await executor.execute_approved_fit(
                self.store,
                selected["dataset_snapshot_id"],
                recipe_id,
            )
        except OriginFitError as error:
            self._print_error(error)
            return
        if isinstance(outcome, PendingFitJob):
            print(
                f"Fit Job {outcome.worker_job_id} remains pending "
                f"(worker_status={outcome.worker_status}). {outcome.message}",
                file=self.stdout,
            )
            return
        self.latest_result_summary = _compressed_result(outcome.fit_result)
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
            file=self.stdout,
        )

    async def _ask_agent(self, line: str) -> None:
        selected = self._require_dataset()
        if selected is None:
            return
        tool = _proposal_tool(
            self.store, selected["dataset_snapshot_id"], self.selected_template
        )
        result = await Agent(self.model, tools=[tool]).run(
            _agent_prompt(
                line,
                selected,
                self.downsampled_preview,
                self.latest_result_summary,
                self.available_templates,
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
            with self.store.connect() as connection:
                self.store.audit(
                    connection,
                    "model.authority_action.rejected",
                    selected["dataset_snapshot_id"],
                    {"tool_names": sorted(set(rejected_tools))},
                )
            print(
                "Rejected model-requested authority action; use an explicit CLI command.",
                file=self.stdout,
            )
        else:
            print(result.output, file=self.stdout)


async def run_agent_cli(
    store: LocalStore,
    model: Model,
    executor: FitExecutor | None,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> None:
    """Run an interactive CLI while keeping every authority action outside the model."""
    await _AgentCliSession(store, model, executor, stdout).run(stdin)


__all__ = ("FitExecutor", "run_agent_cli")
