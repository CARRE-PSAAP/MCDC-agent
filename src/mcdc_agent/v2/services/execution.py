import ast
import os
import subprocess
import sys
import time
from pathlib import Path

from mcdc_agent.v2.types import ExecutionResult


class ExecutionService:
    """Run simulation scripts and capture stdout, stderr, and output files."""

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout

    def run_script(
        self,
        script_path: str | Path,
        *,
        working_dir: str | Path | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        script_path = Path(script_path).resolve()
        run_dir = Path(working_dir).resolve() if working_dir else script_path.parent.resolve()
        script_text = script_path.read_text(encoding="utf-8")
        expected_output = self._expected_output_h5(script_text, run_dir)
        before = self._snapshot_h5(run_dir)
        env = self._build_env(run_dir)
        start = time.perf_counter()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=run_dir,
                env=env,
                timeout=timeout or self.timeout,
            )
            output_h5 = self._detect_output_h5(run_dir, before, expected_output)
            return ExecutionResult(
                script_path=script_path,
                working_dir=run_dir,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_h5=output_h5,
                timed_out=False,
                duration_seconds=time.perf_counter() - start,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = self._coerce_stream(exc.stdout)
            stderr = self._coerce_stream(exc.stderr)
            output_h5 = self._detect_output_h5(run_dir, before, expected_output)
            return ExecutionResult(
                script_path=script_path,
                working_dir=run_dir,
                returncode=None,
                stdout=stdout,
                stderr=stderr,
                output_h5=output_h5,
                timed_out=True,
                duration_seconds=time.perf_counter() - start,
            )

    @staticmethod
    def _build_env(run_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        run_dir_text = str(run_dir)
        env["PYTHONPATH"] = f"{run_dir_text}:{pythonpath}" if pythonpath else run_dir_text
        return env

    @staticmethod
    def _snapshot_h5(run_dir: Path) -> dict[Path, float]:
        return {path: path.stat().st_mtime for path in run_dir.glob("*.h5")}

    def _detect_output_h5(
        self,
        run_dir: Path,
        before: dict[Path, float],
        expected_output: Path,
    ) -> Path | None:
        if expected_output.exists():
            return expected_output

        candidates = []
        for path in run_dir.glob("*.h5"):
            previous_mtime = before.get(path)
            current_mtime = path.stat().st_mtime
            if previous_mtime is None or current_mtime > previous_mtime:
                candidates.append(path)

        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
        return None

    def _expected_output_h5(self, script_text: str, run_dir: Path) -> Path:
        output_name = self._extract_output_name(script_text)
        return run_dir / f"{output_name}.h5"

    @staticmethod
    def _extract_output_name(script_text: str) -> str:
        try:
            tree = ast.parse(script_text)
        except SyntaxError:
            return "output"

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if ExecutionService._is_output_name_target(target):
                    return node.value.value

        return "output"

    @staticmethod
    def _is_output_name_target(node: ast.AST) -> bool:
        if not isinstance(node, ast.Attribute) or node.attr != "output_name":
            return False
        value = node.value
        return (
            isinstance(value, ast.Attribute)
            and value.attr == "settings"
            and isinstance(value.value, ast.Name)
            and value.value.id == "mcdc"
        )

    @staticmethod
    def _coerce_stream(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode()
        return value
