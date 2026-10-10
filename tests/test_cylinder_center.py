"""Exercise numerical fit cases in the pre-existing isolated NumPy environment."""
import subprocess
from pathlib import Path


def test_numerical_cylinder_fit_and_unchanged_gates():
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [str(root / ".venv-perception/Scripts/python.exe"),
         str(root / "tests/numerical_cylinder_center_cases.py")],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
