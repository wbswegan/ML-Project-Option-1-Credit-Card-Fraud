from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(script_name: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)]
    subprocess.run(command, check=True)


def main() -> None:
    run_step("run_eda.py")
    run_step("run_preprocessing.py")
    run_step("run_training.py")
    print("All steps complete.")


if __name__ == "__main__":
    main()

