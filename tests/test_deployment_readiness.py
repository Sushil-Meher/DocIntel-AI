import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


PYTHON_SOURCE_DIRS = [ROOT_DIR / "src", ROOT_DIR]


def read_python_files():
    for directory in PYTHON_SOURCE_DIRS:
        for path in directory.glob("*.py"):
            yield path, path.read_text(encoding="utf-8")


class HardwarePortabilityTests(unittest.TestCase):

    def test_generator_does_not_hardcode_gpu_device(self):
        source = (ROOT_DIR / "src" / "generator.py").read_text(encoding="utf-8")

        self.assertNotIn("device=0", source)
        self.assertIn("torch.cuda.is_available", source)


class NoHardcodedLocalPathsTests(unittest.TestCase):

    def test_no_hardcoded_windows_drive_paths_in_source(self):
        drive_path = re.compile(r"[A-Za-z]:\\")

        for path, content in read_python_files():
            self.assertIsNone(
                drive_path.search(content),
                f"{path} appears to contain a hardcoded Windows path"
            )


class NoEagerArtifactLoadingTests(unittest.TestCase):

    def test_rag_module_does_not_load_artifacts_at_import_time(self):
        # Regression guard for the Task 4 fix: loading a shared
        # artifacts/faiss.index at module import time meant importing
        # src.rag could crash on a fresh clone/deployment with no
        # artifacts yet, and was dead weight since app.py never used it.
        source = (ROOT_DIR / "src" / "rag.py").read_text(encoding="utf-8")

        guard_index = source.index('if __name__ == "__main__":')
        before_guard = source[:guard_index]

        self.assertNotIn("load_index(", before_guard)
        self.assertNotIn("load_chunks(", before_guard)


class DependencyDeclarationTests(unittest.TestCase):

    IMPORT_TO_PACKAGE = {
        "pymupdf": "pymupdf",
        "sentence_transformers": "sentence-transformers",
        "faiss": "faiss-cpu",
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "transformers": "transformers",
        "torch": "torch",
        "streamlit": "streamlit",
        "numpy": "numpy",
    }

    def test_all_imported_third_party_packages_are_declared(self):
        requirements = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
        declared = {
            line.split(">=")[0].split("==")[0].strip().lower()
            for line in requirements.splitlines()
            if line.strip()
        }

        imported_modules = set()
        import_pattern = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w]*)")

        for _, content in read_python_files():
            for line in content.splitlines():
                match = import_pattern.match(line)
                if match:
                    imported_modules.add(match.group(1))

        expected_packages = {
            self.IMPORT_TO_PACKAGE[module]
            for module in imported_modules
            if module in self.IMPORT_TO_PACKAGE
        }

        # Sanity check: the project does use every package in the map,
        # so this test would silently pass trivially if imports moved.
        self.assertTrue(expected_packages)

        missing = {pkg for pkg in expected_packages if pkg.lower() not in declared}
        self.assertEqual(missing, set())


class EnvExampleTests(unittest.TestCase):

    def test_env_example_has_no_real_looking_secret(self):
        env_example = ROOT_DIR / ".env.example"
        self.assertTrue(env_example.exists())

        content = env_example.read_text(encoding="utf-8")

        for line in content.splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue

            _, _, value = line.partition("=")
            self.assertEqual(
                value.strip(), "",
                f".env.example should only contain placeholder values, got: {line}"
            )


class TempFileRobustnessTests(unittest.TestCase):

    def test_pdf_upload_survives_temp_file_cleanup_failure(self):
        from streamlit.testing.v1 import AppTest

        fake_index, fake_chunks = MagicMock(name="index"), ["chunk"]

        with patch("src.ingestion.ingest_pdf", return_value=(fake_index, fake_chunks)), \
             patch("os.remove", side_effect=PermissionError("file in use")):

            at = AppTest.from_file(str(ROOT_DIR / "app.py"))
            at.run(timeout=15)
            at.sidebar.file_uploader[0].upload("a.pdf", b"%PDF-1.4 fake", "application/pdf")
            at.run(timeout=15)
            at.sidebar.button[0].click()
            at.run(timeout=15)

        self.assertEqual(at.exception.len, 0)
        self.assertEqual(at.session_state.source, "a.pdf")


if __name__ == "__main__":
    unittest.main()
