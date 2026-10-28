"""Keep the manual-install dependency bounds aligned with package metadata."""

from pathlib import Path
import tomllib
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class DependencyMetadataTests(unittest.TestCase):
    def test_manual_requirements_match_project_coreml_diffusion_spec(self):
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        project_spec = next(
            requirement
            for requirement in project["project"]["dependencies"]
            if requirement.lower().startswith("coreml-diffusion")
        )
        manual_specs = [
            line.strip()
            for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
            if line.strip().lower().startswith("coreml-diffusion")
        ]

        self.assertEqual(len(manual_specs), 1)
        self.assertEqual(manual_specs[0], project_spec)
