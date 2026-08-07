"""Worker-side registry for versioned, immutable Origin graph templates.

Registered Origin Graph Templates are Origin-native rendering assets that a
researcher explicitly selects for a Fit Specification.  The registry stores
template content in a content-addressed file store and keeps only metadata in
SQLite.  Template actions live in a Worker management CLI that the model cannot
reach.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import re
from pathlib import Path
import sqlite3
from typing import Iterator
import zipfile

from origin_fit.storage import utc_now


TEMPLATE_EXTENSIONS = frozenset({".otpu", ".otp"})
DEFAULT_MAX_TEMPLATE_BYTES = 20 * 1024 * 1024
_TEMPLATE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SCRIPT_MARKERS = (
    b"#!",
    b"MZ",
    b"\x7fELF",
    b"<?php",
    b"<script",
)
_EXECUTABLE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyc",
        ".js",
        ".jse",
        ".vbs",
        ".vbe",
        ".ps1",
        ".bat",
        ".cmd",
        ".exe",
        ".dll",
        ".com",
        ".sh",
        ".bash",
        ".ojs",
        ".ogs",
        ".opx",
        ".oadd",
        ".msi",
        ".scr",
        ".pif",
        ".lnk",
        ".url",
        ".hta",
        ".wsf",
        ".wsh",
    }
)
_REMOTE_REFERENCE_MARKERS = (
    b"http://",
    b"https://",
    b"ftp://",
    b"file://",
    b"<script",
    b"lt_exec",
    b"system(",
)


class TemplateError(Exception):
    """A fail-closed Worker template registry error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class GraphTemplateRegistry:
    """Content-addressed registry for versioned Origin graph templates."""

    def __init__(
        self,
        state_dir: Path,
        *,
        max_template_bytes: int = DEFAULT_MAX_TEMPLATE_BYTES,
    ) -> None:
        if max_template_bytes <= 0:
            raise ValueError("max_template_bytes must be positive")
        self.state_dir = state_dir
        self.max_template_bytes = max_template_bytes
        self.database_path = state_dir / "templates.sqlite3"
        self.templates_dir = state_dir / "templates" / "sha256"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_templates (
                    template_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    graph_profile_id TEXT NOT NULL,
                    graph_profile_version TEXT NOT NULL,
                    originpro_min_version REAL NOT NULL,
                    originpro_max_version REAL NOT NULL,
                    registered_at TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    PRIMARY KEY (template_id, version),
                    UNIQUE (template_id, sha256)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def register(
        self,
        *,
        name: str,
        content: bytes,
        filename: str,
        graph_profile_id: str,
        graph_profile_version: str,
        originpro_min_version: float,
        originpro_max_version: float,
        max_template_bytes: int | None = None,
    ) -> dict:
        """Register one immutable template version, idempotent for same content."""
        self._validate_name(name)
        self._validate_filename(filename)
        self._validate_content(content, filename, max_template_bytes)
        if not graph_profile_id.strip() or not graph_profile_version.strip():
            raise TemplateError(
                "invalid_template_metadata",
                "A Graph Profile id and version are required.",
            )
        if (
            isinstance(originpro_min_version, bool)
            or isinstance(originpro_max_version, bool)
            or not (
                0 < originpro_min_version <= originpro_max_version
            )
        ):
            raise TemplateError(
                "invalid_template_metadata",
                "OriginPro version compatibility range is invalid.",
            )
        digest = hashlib.sha256(content).hexdigest()
        template_id = f"template:{name}"
        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM graph_templates
                WHERE template_id = ? AND sha256 = ?
                """,
                (template_id, digest),
            ).fetchone()
            if existing is not None:
                self._audit(
                    connection,
                    "graph_template.registration_idempotent",
                    template_id,
                    {
                        "version": existing["version"],
                        "sha256": digest,
                        "graph_profile": {
                            "id": existing["graph_profile_id"],
                            "version": existing["graph_profile_version"],
                        },
                    },
                )
                return self._record(existing)
            latest = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS version
                FROM graph_templates WHERE template_id = ?
                """,
                (template_id,),
            ).fetchone()
            version = int(latest["version"]) + 1
            registered_at = utc_now()
            connection.execute(
                """
                INSERT INTO graph_templates (
                    template_id, version, sha256, filename,
                    graph_profile_id, graph_profile_version,
                    originpro_min_version, originpro_max_version,
                    registered_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    template_id,
                    version,
                    digest,
                    filename,
                    graph_profile_id,
                    graph_profile_version,
                    originpro_min_version,
                    originpro_max_version,
                    registered_at,
                ),
            )
            self._audit(
                connection,
                "graph_template.registered",
                template_id,
                {
                    "version": version,
                    "sha256": digest,
                    "filename": filename,
                    "graph_profile": {
                        "id": graph_profile_id,
                        "version": graph_profile_version,
                    },
                    "originpro_min_version": originpro_min_version,
                    "originpro_max_version": originpro_max_version,
                },
            )
        self._put_content(digest, content)
        row = self.get(template_id, version)
        assert row is not None
        return row

    def get(self, template_id: str, version: int) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM graph_templates
                WHERE template_id = ? AND version = ?
                """,
                (template_id, version),
            ).fetchone()
        return self._record(row) if row is not None else None

    def list_templates(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_templates
                ORDER BY template_id, version
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def active_versions(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_templates
                WHERE active = 1
                ORDER BY template_id, version
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def deactivate(self, template_id: str, version: int) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM graph_templates
                WHERE template_id = ? AND version = ?
                """,
                (template_id, version),
            ).fetchone()
            if row is None:
                raise TemplateError(
                    "template_not_found",
                    f"Registered Origin Graph Template '{template_id}@{version}' not found.",
                )
            if row["active"]:
                connection.execute(
                    """
                    UPDATE graph_templates SET active = 0
                    WHERE template_id = ? AND version = ?
                    """,
                    (template_id, version),
                )
                self._audit(
                    connection,
                    "graph_template.deactivated",
                    template_id,
                    {"version": version, "sha256": row["sha256"]},
                )
            row = connection.execute(
                """
                SELECT * FROM graph_templates
                WHERE template_id = ? AND version = ?
                """,
                (template_id, version),
            ).fetchone()
            assert row is not None
            return self._record(row)

    def content(self, template_id: str, version: int) -> bytes:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT sha256 FROM graph_templates
                WHERE template_id = ? AND version = ?
                """,
                (template_id, version),
            ).fetchone()
        if row is None:
            raise TemplateError(
                "template_not_found",
                f"Registered Origin Graph Template '{template_id}@{version}' not found.",
            )
        digest = row["sha256"]
        path = self.templates_dir / digest
        try:
            content = path.read_bytes()
        except OSError as error:
            raise TemplateError(
                "template_integrity_error",
                "Registered Origin Graph Template content is unavailable.",
            ) from error
        if hashlib.sha256(content).hexdigest() != digest:
            raise TemplateError(
                "template_integrity_error",
                "Registered Origin Graph Template content no longer matches its content identifier.",
            )
        return content

    def inspect_audit_events(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT event_type, occurred_at, object_id, details_json
                FROM audit_events ORDER BY id
                """
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "occurred_at": row["occurred_at"],
                "object_id": row["object_id"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    def _put_content(self, digest: str, content: bytes) -> None:
        path = self.templates_dir / digest
        try:
            with path.open("xb") as template_file:
                template_file.write(content)
            path.chmod(0o444)
        except FileExistsError:
            if path.read_bytes() != content:
                raise TemplateError(
                    "template_integrity_error",
                    "Content-addressed template store collided with different content.",
                )

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _TEMPLATE_ID_PATTERN.fullmatch(name):
            raise TemplateError(
                "invalid_template_name",
                "Template name must match [a-z0-9][a-z0-9-]{0,63}.",
            )

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if (
            not filename
            or filename.startswith(("\\\\", "//"))
            or "/" in filename
            or "\\" in filename
            or ".." in filename
            or Path(filename).name != filename
            or Path(filename).suffix.lower() not in TEMPLATE_EXTENSIONS
        ):
            raise TemplateError(
                "invalid_template_filename",
                "Template file must be a local basename ending in .otpu or .otp.",
            )

    def _validate_content(
        self,
        content: bytes,
        filename: str,
        max_template_bytes: int | None,
    ) -> None:
        limit = max_template_bytes or self.max_template_bytes
        if not content:
            raise TemplateError(
                "invalid_template_content", "Template content must not be empty."
            )
        if len(content) > limit:
            raise TemplateError(
                "template_too_large", "Template content exceeds the Worker limit."
            )
        if any(content.startswith(marker) for marker in _SCRIPT_MARKERS):
            raise TemplateError(
                "invalid_template_content",
                "Template content has an executable or script marker.",
            )
        if content.startswith(b"PK"):
            self._validate_zip_container(content)

    @staticmethod
    def _validate_zip_container(content: bytes) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                for info in archive.infolist():
                    name = info.filename
                    if name.endswith("/"):
                        continue
                    parts = name.replace("\\", "/").split("/")
                    if (
                        not name
                        or name.startswith(("/", "\\\\", "//"))
                        or "\\" in name
                        or re.match(r"^[A-Za-z]:", name)
                        or ".." in parts
                        or any(not part for part in parts)
                    ):
                        raise TemplateError(
                            "invalid_template_content",
                            "Template archive contains an unsafe member path.",
                        )
                    file_type = (info.external_attr >> 28) & 0xF
                    if file_type in (1, 2, 3, 4, 5, 6, 7, 10):
                        raise TemplateError(
                            "invalid_template_content",
                            "Template archive contains a link or device member.",
                        )
                    suffix = Path(name).suffix.lower()
                    if suffix in _EXECUTABLE_EXTENSIONS:
                        raise TemplateError(
                            "invalid_template_content",
                            "Template archive contains an executable member.",
                        )
                    if suffix in (".xml", ".txt", ".json", ".ini"):
                        member = archive.read(name)
                        if any(
                            marker in member for marker in _REMOTE_REFERENCE_MARKERS
                        ):
                            raise TemplateError(
                                "invalid_template_content",
                                "Template archive contains an external reference or script call.",
                            )
        except (OSError, zipfile.BadZipFile, KeyError) as error:
            raise TemplateError(
                "invalid_template_content",
                "Template archive is not a valid zip container.",
            ) from error

    @staticmethod
    def _record(row: sqlite3.Row) -> dict:
        return {
            "template_id": row["template_id"],
            "version": row["version"],
            "sha256": row["sha256"],
            "filename": row["filename"],
            "graph_profile": {
                "id": row["graph_profile_id"],
                "version": row["graph_profile_version"],
            },
            "originpro_min_version": row["originpro_min_version"],
            "originpro_max_version": row["originpro_max_version"],
            "registered_at": row["registered_at"],
            "active": bool(row["active"]),
        }

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        event_type: str,
        object_id: str,
        details: dict,
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events (
                event_type, occurred_at, object_id, details_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                event_type,
                utc_now(),
                object_id,
                json.dumps(details, sort_keys=True, separators=(",", ":")),
            ),
        )


__all__ = ("GraphTemplateRegistry", "TemplateError")
