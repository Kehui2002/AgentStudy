from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from origin_fit.datasets import ImportSelection, import_dataset
from origin_fit.execution import DeterministicFakeOriginAdapter
from origin_fit.remote import (
    ArchivedFitResult,
    InProcessWorkerTransport,
    RemoteOriginExecutor,
)
from origin_fit.specifications import (
    approve_fit_specification,
    propose_fit_specification,
)
from origin_fit.storage import LocalStore
from origin_worker.service import OriginWorker, WorkerError
from origin_worker.cli import main as worker_main
from origin_worker.templates import GraphTemplateRegistry, TemplateError
from tests.test_support import (
    TEMPLATE_ID,
    TEMPLATE_SHA256,
    TEMPLATE_VERSION,
    make_worker,
    register_standard_template,
)
from tests.test_origin_remote import approved_fixture
from tests.test_origin_remote import FIXTURE


def _zip_bytes(members: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return output.getvalue()


class GraphTemplateRegistryTests(unittest.TestCase):
    def register(
        self,
        registry: GraphTemplateRegistry,
        *,
        name: str = "standard",
        content: bytes = b"ORIGIN-GRAPH-TEMPLATE-CONTENT\n",
        filename: str = "standard.otpu",
        graph_profile: str = "expdec2-standard@1.0",
    ) -> dict:
        profile_id, profile_version = graph_profile.split("@", 1)
        return registry.register(
            name=name,
            content=content,
            filename=filename,
            graph_profile_id=profile_id,
            graph_profile_version=profile_version,
            originpro_min_version=10.2,
            originpro_max_version=10.3,
        )

    def test_register_stores_an_immutable_content_addressed_copy_with_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            content = b"ORIGIN-GRAPH-TEMPLATE-CONTENT\n"
            registered = self.register(registry, content=content)

            digest = hashlib.sha256(content).hexdigest()
            self.assertEqual(registered["template_id"], "template:standard")
            self.assertEqual(registered["version"], 1)
            self.assertEqual(registered["sha256"], digest)
            self.assertEqual(
                registered["graph_profile"],
                {"id": "expdec2-standard", "version": "1.0"},
            )
            self.assertEqual(registered["originpro_min_version"], 10.2)
            self.assertEqual(registered["originpro_max_version"], 10.3)
            self.assertEqual(registered["filename"], "standard.otpu")
            self.assertTrue(registered["active"])

            stored = (
                registry.templates_dir / digest
            )
            self.assertEqual(stored.read_bytes(), content)
            self.assertEqual(stored.stat().st_mode & 0o444, 0o444)
            self.assertEqual(
                registry.content("template:standard", 1), content
            )

    def test_same_content_is_idempotent_and_changed_content_creates_a_new_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            first = self.register(registry)
            second = self.register(registry)
            self.assertEqual(second, first)
            self.assertEqual(second["version"], 1)

            changed = self.register(
                registry, content=b"CHANGED-ORIGIN-TEMPLATE-CONTENT\n"
            )
            self.assertEqual(changed["template_id"], "template:standard")
            self.assertEqual(changed["version"], 2)
            self.assertNotEqual(changed["sha256"], first["sha256"])

            first_record = registry.get("template:standard", 1)
            assert first_record is not None
            self.assertEqual(
                registry.content("template:standard", 1),
                b"ORIGIN-GRAPH-TEMPLATE-CONTENT\n",
            )
            self.assertEqual(
                registry.content("template:standard", 2),
                b"CHANGED-ORIGIN-TEMPLATE-CONTENT\n",
            )
            listings = {
                (record["template_id"], record["version"]): record
                for record in registry.list_templates()
            }
            self.assertEqual(set(listings), {("template:standard", 1), ("template:standard", 2)})
            self.assertTrue(listings[("template:standard", 1)]["active"])
            self.assertTrue(listings[("template:standard", 2)]["active"])

    def test_deactivation_keeps_history_and_content_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            first = self.register(registry)
            deactivated = registry.deactivate("template:standard", 1)
            self.assertFalse(deactivated["active"])
            self.assertEqual(
                registry.content("template:standard", 1),
                b"ORIGIN-GRAPH-TEMPLATE-CONTENT\n",
            )
            records = registry.list_templates()
            self.assertEqual(
                {
                    (record["template_id"], record["version"]): record["active"]
                    for record in records
                },
                {("template:standard", 1): False},
            )
            self.assertEqual(
                [record["template_id"] for record in registry.active_versions()],
                [],
            )
            unchanged = registry.deactivate("template:standard", 1)
            self.assertEqual(unchanged, deactivated)

    def test_show_returns_the_requested_version_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            registered = self.register(registry)
            shown = registry.get("template:standard", 1)
            self.assertEqual(shown, registered)
            self.assertIsNone(registry.get("template:standard", 99))
            self.assertIsNone(registry.get("template:missing", 1))

    def test_register_rejects_unsafe_names_filenames_and_script_content(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            invalid_names = ("Standard", "std template", "a" * 65, "../std", "")
            for name in invalid_names:
                with self.subTest(name=name):
                    with self.assertRaises(TemplateError) as raised:
                        self.register(registry, name=name)
                    self.assertEqual(raised.exception.code, "invalid_template_name")

            invalid_filenames = (
                "../standard.otpu",
                "/absolute/standard.otpu",
                "C:\\absolute\\standard.otpu",
                "\\\\server\\share\\standard.otpu",
                "standard.py",
                "standard.otp.exe",
                "",
            )
            for filename in invalid_filenames:
                with self.subTest(filename=filename):
                    with self.assertRaises(TemplateError) as raised:
                        self.register(registry, filename=filename)
                    self.assertEqual(
                        raised.exception.code, "invalid_template_filename"
                    )

            script_contents = (
                b"#!/bin/sh\nrm -rf /\n",
                b"MZ\x90\x00executable\n",
                b"\x7fELF\x02\x01\x01binary\n",
                b"<?php system('id'); ?>\n",
                b"<script>evil()</script>\n",
            )
            for index, content in enumerate(script_contents):
                with self.subTest(content=content[:4]):
                    with self.assertRaises(TemplateError) as raised:
                        self.register(registry, content=content)
                    self.assertEqual(
                        raised.exception.code, "invalid_template_content"
                    )

            with self.assertRaises(TemplateError) as raised:
                self.register(registry, content=b"")
            self.assertEqual(raised.exception.code, "invalid_template_content")

            with self.assertRaises(TemplateError) as raised:
                registry.register(
                    name="big",
                    content=b"x" * 100,
                    filename="big.otpu",
                    graph_profile_id="expdec2-standard",
                    graph_profile_version="1.0",
                    originpro_min_version=10.2,
                    originpro_max_version=10.3,
                    max_template_bytes=50,
                )
            self.assertEqual(raised.exception.code, "template_too_large")

    def test_register_rejects_unsafe_zip_members_and_remote_references(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            unsafe_members = (
                "../escape.otpu",
                "/absolute.xml",
                "C:/drive.xml",
                "windows\\path.xml",
                "member.py",
                "payload.ojs",
                "payload.dll",
            )
            for member in unsafe_members:
                with self.subTest(member=member):
                    with self.assertRaises(TemplateError) as raised:
                        self.register(
                            registry,
                            content=_zip_bytes([(member, b"<xml/>")]),
                            filename="standard.otpu",
                        )
                    self.assertEqual(
                        raised.exception.code, "invalid_template_content"
                    )

            remote_reference = _zip_bytes(
                [
                    ("template.xml", b"<xml src=\"http://evil.example/x\"/>"),
                ]
            )
            with self.assertRaises(TemplateError) as raised:
                self.register(
                    registry,
                    content=remote_reference,
                    filename="standard.otpu",
                )
            self.assertEqual(raised.exception.code, "invalid_template_content")

            labtalk = _zip_bytes(
                [("template.xml", b"<xml>lt_exec system(\"whoami\");</xml>")]
            )
            with self.assertRaises(TemplateError) as raised:
                self.register(
                    registry,
                    content=labtalk,
                    filename="standard.otpu",
                )
            self.assertEqual(raised.exception.code, "invalid_template_content")

    def test_content_verification_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            registered = self.register(registry)
            stored = registry.templates_dir / registered["sha256"]
            stored.chmod(0o600)
            stored.write_bytes(b"TAMPERED-CONTENT\n")
            with self.assertRaises(TemplateError) as raised:
                registry.content("template:standard", 1)
            self.assertEqual(raised.exception.code, "template_integrity_error")

    def test_audit_events_cover_registration_and_deactivation_without_content(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry = GraphTemplateRegistry(Path(temporary_directory) / "state")
            registered = self.register(registry)
            registry.deactivate("template:standard", 1)
            audit = registry.inspect_audit_events()
            event_types = [event["event_type"] for event in audit]
            self.assertIn("graph_template.registered", event_types)
            self.assertIn("graph_template.deactivated", event_types)
            audit_text = json.dumps(audit, sort_keys=True)
            self.assertNotIn("ORIGIN-GRAPH-TEMPLATE-CONTENT", audit_text)
            self.assertIn(registered["sha256"], audit_text)
            self.assertNotIn("secret", audit_text.lower())


class OriginWorkerTemplateCliTests(unittest.TestCase):
    def run_template_command(self, *arguments: str) -> dict:
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                worker_main(["template", *arguments])
        except SystemExit:
            self.fail(f"template CLI failed: {output.getvalue()}")
        return json.loads(output.getvalue())

    def test_cli_registers_lists_shows_and_deactivates_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            template_path = state_dir.parent / "standard.otpu"
            template_path.write_bytes(b"ORIGIN-GRAPH-TEMPLATE-CONTENT\n")
            digest = hashlib.sha256(template_path.read_bytes()).hexdigest()

            registered = self.run_template_command(
                "register",
                "--state-dir",
                str(state_dir),
                "--name",
                "standard",
                "--file",
                str(template_path),
                "--graph-profile",
                "expdec2-standard@1.0",
                "--originpro-min",
                "10.2",
                "--originpro-max",
                "10.3",
            )
            self.assertEqual(registered["template_id"], "template:standard")
            self.assertEqual(registered["version"], 1)
            self.assertEqual(registered["sha256"], digest)

            listed = self.run_template_command("list", "--state-dir", str(state_dir))
            self.assertEqual(
                [record["template_id"] for record in listed["graph_templates"]],
                ["template:standard"],
            )
            shown = self.run_template_command(
                "show", "--state-dir", str(state_dir), "template:standard@1"
            )
            self.assertEqual(shown["sha256"], digest)
            deactivated = self.run_template_command(
                "deactivate", "--state-dir", str(state_dir), "template:standard@1"
            )
            self.assertFalse(deactivated["active"])
            after = self.run_template_command(
                "list", "--state-dir", str(state_dir)
            )
            self.assertEqual(after["graph_templates"][0]["active"], False)

    def test_cli_rejects_an_invalid_template_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            script_path = state_dir.parent / "evil.py"
            script_path.write_text("#!/usr/bin/python\nprint('evil')\n")
            with self.assertRaises(SystemExit):
                worker_main(
                    [
                        "template",
                        "register",
                        "--state-dir",
                        str(state_dir),
                        "--name",
                        "evil",
                        "--file",
                        str(script_path),
                        "--graph-profile",
                        "expdec2-standard@1.0",
                        "--originpro-min",
                        "10.2",
                        "--originpro-max",
                        "10.3",
                    ]
                )

    def test_cli_show_and_deactivate_fail_for_missing_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            with self.assertRaises(SystemExit):
                worker_main(
                    [
                        "template",
                        "show",
                        "--state-dir",
                        str(state_dir),
                        "template:missing@1",
                    ]
                )
            with self.assertRaises(SystemExit):
                worker_main(
                    [
                        "template",
                        "deactivate",
                        "--state-dir",
                        str(state_dir),
                        "template:missing@1",
                    ]
                )


class OriginWorkerTemplateCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_capabilities_list_only_active_registered_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory) / "state"
            registry = GraphTemplateRegistry(state_dir)
            first = register_standard_template(registry)
            registry.deactivate(TEMPLATE_ID, 1)
            second = registry.register(
                name="standard",
                content=b"SECOND-ORIGIN-TEMPLATE-VERSION\n",
                filename="standard.otpu",
                graph_profile_id="expdec2-standard",
                graph_profile_version="1.0",
                originpro_min_version=10.2,
                originpro_max_version=10.3,
            )
            worker = OriginWorker(
                state_dir,
                DeterministicFakeOriginAdapter(),
                template_registry=registry,
            )

            capabilities = worker.capabilities()

            self.assertEqual(
                [item.template_id for item in capabilities.graph_templates],
                [TEMPLATE_ID],
            )
            self.assertEqual(
                capabilities.graph_templates[0].version, second["version"]
            )
            self.assertEqual(
                capabilities.graph_templates[0].sha256, second["sha256"]
            )
            self.assertEqual(
                capabilities.graph_templates[0].graph_profile.id,
                "expdec2-standard",
            )
            self.assertNotEqual(first["sha256"], second["sha256"])
            audit = worker.inspect_audit_events()
            reported = [
                event
                for event in audit
                if event["event_type"] == "worker.capabilities.reported"
            ]
            self.assertEqual(reported[-1]["details"]["graph_template_count"], 1)

    async def test_submit_fails_closed_for_missing_or_unknown_templates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            worker = make_worker(
                root / "worker", DeterministicFakeOriginAdapter()
            )
            snapshot_id, recipe_id = approved_fixture(store)
            submission = RemoteOriginExecutor.prepare_submission(
                store, snapshot_id, recipe_id
            )

            recipe = dict(submission.approved_fit_recipe)
            specification = dict(recipe["fit_specification"])
            specification["graph_template"] = None
            recipe["fit_specification"] = specification
            missing = submission.model_copy(
                update={"approved_fit_recipe": recipe}
            )
            with self.assertRaises(WorkerError) as raised:
                worker.submit(missing, "missing-template")
            self.assertEqual(raised.exception.code, "template_selection_required")

            specification["graph_template"] = {
                "template_id": "template:unknown",
                "version": 1,
                "sha256": "0" * 64,
            }
            recipe["fit_specification"] = specification
            unknown = submission.model_copy(
                update={"approved_fit_recipe": recipe}
            )
            with self.assertRaises(WorkerError) as raised:
                worker.submit(unknown, "unknown-template")
            self.assertEqual(raised.exception.code, "template_not_found")
            self.assertEqual(tuple(worker.jobs_dir.iterdir()), ())

    async def test_submit_fails_closed_for_deactivated_or_mismatched_templates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            registry = GraphTemplateRegistry(root / "worker")
            register_standard_template(registry)
            registry.deactivate(TEMPLATE_ID, 1)
            worker = OriginWorker(
                root / "worker",
                DeterministicFakeOriginAdapter(),
                template_registry=registry,
            )
            snapshot_id, recipe_id = approved_fixture(store)
            submission = RemoteOriginExecutor.prepare_submission(
                store, snapshot_id, recipe_id
            )

            with self.assertRaises(WorkerError) as raised:
                worker.submit(submission, "deactivated-template")
            self.assertEqual(raised.exception.code, "template_deactivated")

            re_registered = registry.register(
                name="standard",
                content=b"DIFFERENT-ORIGIN-TEMPLATE-CONTENT\n",
                filename="standard.otpu",
                graph_profile_id="expdec2-standard",
                graph_profile_version="1.0",
                originpro_min_version=10.2,
                originpro_max_version=10.3,
            )
            self.assertEqual(re_registered["version"], 2)
            recipe = dict(submission.approved_fit_recipe)
            specification = dict(recipe["fit_specification"])
            specification["graph_template"] = {
                "template_id": TEMPLATE_ID,
                "version": 2,
                "sha256": TEMPLATE_SHA256,
            }
            recipe["fit_specification"] = specification
            hash_mismatched = submission.model_copy(
                update={"approved_fit_recipe": recipe}
            )
            with self.assertRaises(WorkerError) as raised:
                worker.submit(hash_mismatched, "hash-mismatch")
            self.assertEqual(raised.exception.code, "template_hash_mismatch")

            other_profile = registry.register(
                name="other-profile",
                content=b"OTHER-PROFILE-ORIGIN-TEMPLATE\n",
                filename="other-profile.otpu",
                graph_profile_id="different-profile",
                graph_profile_version="1.0",
                originpro_min_version=10.2,
                originpro_max_version=10.3,
            )
            specification["graph_template"] = {
                "template_id": other_profile["template_id"],
                "version": other_profile["version"],
                "sha256": other_profile["sha256"],
            }
            recipe["fit_specification"] = specification
            profile_mismatched = submission.model_copy(
                update={"approved_fit_recipe": recipe}
            )
            with self.assertRaises(WorkerError) as raised:
                worker.submit(profile_mismatched, "profile-mismatch")
            self.assertEqual(raised.exception.code, "template_hash_mismatch")

    async def test_submit_detects_tampered_template_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            worker = make_worker(
                root / "worker", DeterministicFakeOriginAdapter()
            )
            snapshot_id, recipe_id = approved_fixture(store)
            submission = RemoteOriginExecutor.prepare_submission(
                store, snapshot_id, recipe_id
            )
            stored = worker.template_registry.templates_dir / TEMPLATE_SHA256
            stored.chmod(0o600)
            stored.write_bytes(b"TAMPERED-TEMPLATE-CONTENT\n")

            with self.assertRaises(WorkerError) as raised:
                worker.submit(submission, "tampered-template")
            self.assertEqual(raised.exception.code, "template_integrity_error")
            self.assertEqual(tuple(worker.jobs_dir.iterdir()), ())

    async def test_valid_template_selection_submits_and_capabilities_negotiate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            worker = make_worker(
                root / "worker", DeterministicFakeOriginAdapter()
            )
            snapshot_id, recipe_id = approved_fixture(store)
            submission = RemoteOriginExecutor.prepare_submission(
                store, snapshot_id, recipe_id
            )
            capabilities = worker.capabilities()
            self.assertEqual(capabilities.graph_templates[0].version, 1)

            job = worker.submit(submission, "valid-template-selection")
            self.assertEqual(job.status, "queued")
            await worker.run_queued()
            self.assertEqual(worker.get_job(job.worker_job_id).status, "succeeded")

    async def test_template_selection_does_not_change_request_or_numeric_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot = import_dataset(
                store,
                FIXTURE,
                ImportSelection(
                    x="time_s",
                    ys=("decay_a", "decay_b", "decay_c"),
                    uncertainties={
                        "decay_a": "decay_a_error",
                        "decay_b": "decay_b_error",
                        "decay_c": "decay_c_error",
                    },
                    units={
                        "time_s": "s",
                        "decay_a": "dimensionless",
                        "decay_b": "dimensionless",
                        "decay_c": "dimensionless",
                        "decay_a_error": "dimensionless",
                        "decay_b_error": "dimensionless",
                        "decay_c_error": "dimensionless",
                    },
                ),
            )
            snapshot_id = snapshot["dataset_snapshot_id"]
            registry = GraphTemplateRegistry(root / "worker")
            first = register_standard_template(registry)
            second = registry.register(
                name="standard",
                content=b"SECOND-ORIGIN-TEMPLATE-VERSION\n",
                filename="standard.otpu",
                graph_profile_id="expdec2-standard",
                graph_profile_version="1.0",
                originpro_min_version=10.2,
                originpro_max_version=10.3,
            )
            adapter = DeterministicFakeOriginAdapter()
            worker = OriginWorker(
                root / "worker", adapter, template_registry=registry
            )
            executor = RemoteOriginExecutor(InProcessWorkerTransport(worker))

            def recipe_for(template_id: str, version: int, sha256: str) -> str:
                proposed = propose_fit_specification(
                    store,
                    snapshot_id,
                    experiment_id="synthetic-expdec2",
                    fit_minimum=0,
                    fit_maximum=11,
                    weighting="instrument",
                    initialization="origin_auto",
                    graph_profile_id="expdec2-standard",
                    graph_profile_version="1.0",
                    template_id=template_id,
                    template_version=version,
                    template_sha256=sha256,
                )
                return approve_fit_specification(
                    store, proposed["fit_specification_id"]
                )["approved_fit_recipe_id"]

            first_recipe = recipe_for(
                TEMPLATE_ID, first["version"], first["sha256"]
            )
            second_recipe = recipe_for(
                TEMPLATE_ID, second["version"], second["sha256"]
            )
            first_outcome = await executor.execute_approved_fit(
                store,
                snapshot_id,
                first_recipe,
                wait_timeout=2,
                poll_interval=0.01,
            )
            second_outcome = await executor.execute_approved_fit(
                store,
                snapshot_id,
                second_recipe,
                wait_timeout=2,
                poll_interval=0.01,
            )

            self.assertIsInstance(first_outcome, ArchivedFitResult)
            self.assertIsInstance(second_outcome, ArchivedFitResult)
            self.assertEqual(len(adapter.requests), 2)
            self.assertEqual(adapter.requests[0], adapter.requests[1])
            assert isinstance(first_outcome, ArchivedFitResult)
            assert isinstance(second_outcome, ArchivedFitResult)
            first_json = json.loads(first_outcome.fit_result.model_dump_json())
            second_json = json.loads(second_outcome.fit_result.model_dump_json())
            self.assertEqual(
                first_json["series_outcomes"], second_json["series_outcomes"]
            )
            self.assertNotEqual(first_json["fit_job_id"], second_json["fit_job_id"])

    async def test_template_render_failure_fails_without_bundle_or_archive(
        self,
    ) -> None:
        from origin_worker.originpro_adapter import OriginProAdapterError

        class RenderFailingAdapter(DeterministicFakeOriginAdapter):
            async def execute(self, request, graph_template=None):  # type: ignore[override]
                assert graph_template is not None
                raise OriginProAdapterError(
                    "template render failed", code="template_render_failed"
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "linux")
            snapshot_id, recipe_id = approved_fixture(store)
            worker = make_worker(root / "worker", RenderFailingAdapter())
            submission = RemoteOriginExecutor.prepare_submission(
                store, snapshot_id, recipe_id
            )
            job = worker.submit(submission, "render-failure")

            await worker.run_queued()

            current = worker.get_job(job.worker_job_id)
            self.assertEqual(current.status, "failed")
            self.assertEqual(current.error_code, "template_render_failed")
            with self.assertRaises(WorkerError) as raised:
                worker.get_bundle(job.worker_job_id)
            self.assertEqual(raised.exception.code, "bundle_unavailable")
            workspace = worker.jobs_dir / job.worker_job_id.removeprefix("fit-job:")
            self.assertTrue((workspace / "diagnostic.json").is_file())
            self.assertTrue(
                (workspace / f"graph-template-{TEMPLATE_SHA256}.otpu").is_file()
            )
            self.assertFalse((workspace / "fit-result-bundle.zip").exists())
            with store.connect() as connection:
                count = connection.execute(
                    "SELECT COUNT(*) AS count FROM fit_archives"
                ).fetchone()
            assert count is not None
            self.assertEqual(count["count"], 0)


if __name__ == "__main__":
    unittest.main()
