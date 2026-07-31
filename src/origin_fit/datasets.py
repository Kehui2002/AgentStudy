from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path

from .errors import OriginFitError
from .storage import LocalStore, utc_now

MAX_DATASET_BYTES = 100 * 1024 * 1024
MAX_DATASET_ROWS = 1_000_000
MAX_SELECTED_Y = 20
MAX_ROLE_TEXT_BYTES = 128


@dataclass(frozen=True)
class ImportSelection:
    x: str
    ys: tuple[str, ...]
    uncertainties: dict[str, str]
    units: dict[str, str]


def _number(value: str, *, column: str, row: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise OriginFitError(
            "invalid_dataset_contract", f"Column '{column}' row {row} must be numeric."
        ) from error
    if not math.isfinite(parsed):
        raise OriginFitError(
            "invalid_dataset_contract", f"Column '{column}' row {row} must be finite."
        )
    return parsed


def import_dataset(
    store: LocalStore, csv_path: Path, selection: ImportSelection
) -> dict:
    if len(selection.ys) > MAX_SELECTED_Y:
        raise OriginFitError(
            "resource_limit_exceeded",
            f"At most {MAX_SELECTED_Y} Y series may be selected.",
        )
    try:
        size = csv_path.stat().st_size
    except OSError as error:
        raise OriginFitError("dataset_unavailable", "Dataset could not be read.") from error
    if size > MAX_DATASET_BYTES:
        raise OriginFitError(
            "resource_limit_exceeded",
            f"Dataset exceeds the {MAX_DATASET_BYTES}-byte import limit.",
        )
    try:
        content = csv_path.read_bytes()
    except OSError as error:
        raise OriginFitError("dataset_unavailable", "Dataset could not be read.") from error
    if len(content) > MAX_DATASET_BYTES:
        raise OriginFitError(
            "resource_limit_exceeded",
            f"Dataset exceeds the {MAX_DATASET_BYTES}-byte import limit.",
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OriginFitError("invalid_encoding", "Dataset must be UTF-8 CSV.") from error

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        headers = next(reader)
    except StopIteration as error:
        raise OriginFitError("invalid_csv", "Dataset must contain a header row.") from error
    if len(headers) != len(set(headers)):
        raise OriginFitError(
            "invalid_dataset_contract", "CSV column names must be unique."
        )
    if any(not header for header in headers):
        raise OriginFitError(
            "invalid_dataset_contract", "CSV column names must not be empty."
        )
    indexes = {name: index for index, name in enumerate(headers)}
    selected_columns = {selection.x, *selection.ys}
    if (
        not selection.ys
        or len(selection.ys) != len(set(selection.ys))
        or selection.x in selection.ys
    ):
        raise OriginFitError(
            "invalid_dataset_contract", "Select one X and one or more unique Y columns."
        )
    if set(selection.uncertainties) - set(selection.ys):
        raise OriginFitError(
            "invalid_dataset_contract", "Uncertainty mappings must reference a selected Y."
        )
    uncertainty_columns = tuple(selection.uncertainties.values())
    if (
        len(uncertainty_columns) != len(set(uncertainty_columns))
        or set(uncertainty_columns) & selected_columns
    ):
        raise OriginFitError(
            "invalid_dataset_contract",
            "X, Y, and uncertainty columns must have distinct roles.",
        )
    selected_columns.update(selection.uncertainties.values())
    unknown_columns = selected_columns - set(headers)
    if unknown_columns:
        raise OriginFitError(
            "invalid_dataset_contract",
            "Selected columns are absent from the CSV: " + ", ".join(sorted(unknown_columns)),
        )
    missing_units = [
        name for name in sorted(selected_columns) if not selection.units.get(name, "").strip()
    ]
    if missing_units:
        raise OriginFitError(
            "invalid_dataset_contract",
            "Explicit units are required for X, Y, and uncertainty columns: "
            + ", ".join(missing_units),
        )
    if set(selection.units) != selected_columns:
        raise OriginFitError(
            "invalid_dataset_contract",
            "Unit mappings must exactly match the selected X, Y, and uncertainty columns.",
        )
    bounded_text = [*selected_columns, *selection.units.values()]
    if any(len(value.encode("utf-8")) > MAX_ROLE_TEXT_BYTES for value in bounded_text):
        raise OriginFitError(
            "invalid_dataset_contract",
            f"Selected column names and units must be at most {MAX_ROLE_TEXT_BYTES} UTF-8 bytes.",
        )
    row_count = 0
    x_minimum: float | None = None
    x_maximum: float | None = None
    previous_x: float | None = None
    valid_counts: dict[str, int] = {name: 0 for name in selection.ys}
    y_minimum: dict[str, float | None] = {name: None for name in selection.ys}
    y_maximum: dict[str, float | None] = {name: None for name in selection.ys}
    missing: dict[str, int] = {name: 0 for name in selection.ys}
    for row_number, row in enumerate(reader, start=2):
        row_count += 1
        if row_count > MAX_DATASET_ROWS:
            raise OriginFitError(
                "resource_limit_exceeded",
                f"Dataset exceeds the {MAX_DATASET_ROWS}-row import limit.",
            )
        if len(row) != len(headers):
            raise OriginFitError(
                "invalid_dataset_contract",
                f"CSV row {row_number} has {len(row)} fields; expected {len(headers)}.",
            )
        x_value = _number(
            row[indexes[selection.x]], column=selection.x, row=row_number
        )
        if previous_x is not None and x_value <= previous_x:
            raise OriginFitError(
                "invalid_dataset_contract",
                "X values must be finite, unique, and strictly increasing.",
            )
        if x_minimum is None:
            x_minimum = x_value
        x_maximum = x_value
        previous_x = x_value
        for y_name in selection.ys:
            raw_value = row[indexes[y_name]].strip()
            if raw_value == "" or raw_value.lower() == "nan":
                missing[y_name] += 1
            else:
                y_value = _number(raw_value, column=y_name, row=row_number)
                valid_counts[y_name] += 1
                current_minimum = y_minimum[y_name]
                current_maximum = y_maximum[y_name]
                y_minimum[y_name] = (
                    y_value if current_minimum is None else min(current_minimum, y_value)
                )
                y_maximum[y_name] = (
                    y_value if current_maximum is None else max(current_maximum, y_value)
                )
        for uncertainty_name in selection.uncertainties.values():
            uncertainty = _number(
                row[indexes[uncertainty_name]],
                column=uncertainty_name,
                row=row_number,
            )
            if uncertainty <= 0:
                raise OriginFitError(
                    "invalid_dataset_contract",
                    f"Uncertainty column '{uncertainty_name}' row {row_number} "
                    "must be positive and finite.",
                )

    insufficient = [name for name, count in valid_counts.items() if count < 6]
    if insufficient:
        raise OriginFitError(
            "invalid_dataset_contract",
            "ExpDec2 requires at least 6 valid points for every Y series: "
            + ", ".join(insufficient),
        )

    summary = {
        "schema_version": "1.0",
        "row_count": row_count,
        "column_count": len(headers),
        "x": {
            "name": selection.x,
            "unit": selection.units[selection.x],
            "minimum": x_minimum,
            "maximum": x_maximum,
        },
        "y_series": [
            {
                "name": y_name,
                "unit": selection.units[y_name],
                "valid_point_count": valid_counts[y_name],
                "missing_point_count": missing[y_name],
                "minimum": y_minimum[y_name],
                "maximum": y_maximum[y_name],
                "uncertainty": (
                    {
                        "name": selection.uncertainties[y_name],
                        "unit": selection.units[selection.uncertainties[y_name]],
                    }
                    if y_name in selection.uncertainties
                    else None
                ),
            }
            for y_name in selection.ys
        ],
        "preview_included": False,
    }
    digest = hashlib.sha256(content).hexdigest()
    snapshot_id = f"sha256:{digest}"
    metadata = {
        "schema_version": "1.0",
        "x_column": selection.x,
        "y_columns": list(selection.ys),
        "uncertainty_columns": selection.uncertainties,
        "units": selection.units,
    }
    with store.connect() as connection:
        existing = connection.execute(
            "SELECT metadata_json FROM dataset_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        canonical_metadata = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )
        if existing is not None and existing["metadata_json"] != canonical_metadata:
            raise OriginFitError(
                "snapshot_metadata_conflict",
                "This Dataset Snapshot was already imported with different column roles or units.",
            )
        store.put_object(digest, content)
        connection.execute(
            """
            INSERT OR IGNORE INTO dataset_snapshots (
                id, content_hash, imported_at, metadata_json, summary_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                digest,
                utc_now(),
                canonical_metadata,
                json.dumps(summary, sort_keys=True, separators=(",", ":")),
            ),
        )
        store.audit(
            connection,
            "dataset_snapshot.imported",
            snapshot_id,
            {"content_hash": digest, "row_count": row_count},
        )
    return {"dataset_snapshot_id": snapshot_id, "content_hash": digest}


def inspect_dataset(store: LocalStore, snapshot_id: str) -> dict:
    with store.connect() as connection:
        row = connection.execute(
            """
            SELECT id, content_hash, metadata_json, summary_json
            FROM dataset_snapshots WHERE id = ?
            """,
            (snapshot_id,),
        ).fetchone()
    if row is None:
        raise OriginFitError("not_found", f"Dataset Snapshot '{snapshot_id}' not found.")
    return {
        "dataset_snapshot_id": row["id"],
        "content_hash": row["content_hash"],
        "metadata": json.loads(row["metadata_json"]),
        "summary": json.loads(row["summary_json"]),
    }


def valid_points_in_range(
    store: LocalStore,
    digest: str,
    metadata: dict,
    minimum: float,
    maximum: float,
) -> dict[str, int]:
    """Count valid selected Y observations in an inclusive X range."""
    content = (store.objects_dir / digest).read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(content, newline=""))
    headers = next(reader)
    indexes = {name: index for index, name in enumerate(headers)}
    counts = {name: 0 for name in metadata["y_columns"]}
    for row in reader:
        x_value = float(row[indexes[metadata["x_column"]]])
        if minimum <= x_value <= maximum:
            for y_name in metadata["y_columns"]:
                raw_value = row[indexes[y_name]].strip()
                if raw_value != "" and raw_value.lower() != "nan":
                    counts[y_name] += 1
    return counts
