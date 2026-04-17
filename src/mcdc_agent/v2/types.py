import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GeneratorConfig:
    """Configuration shared by full-generation and interactive sessions."""

    model: str | None = None
    provider: str = "openrouter"
    context_method: str = "api_examples_plan_geom"
    generation_mode: str = "auto"
    max_fix_attempts: int = 0
    temperature: float = 0.1
    trace_dir: Path | None = None


@dataclass(slots=True)
class GenerationArtifacts:
    """Normalized generation result for the new v2 pipeline."""

    prompt: str
    script: str | None
    mode: str
    plan: dict[str, Any] = field(default_factory=dict)
    geometry_plan: dict[str, Any] = field(default_factory=dict)
    phase_scripts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    """Captured result of running a generated simulation script."""

    script_path: Path
    working_dir: Path
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    output_h5: Path | None = None
    timed_out: bool = False
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def stdout_display(self) -> str:
        """Condensed stdout for terminal display.

        Prefers the final output tail starting at the HDF generation message.
        Otherwise strips repeated progress-bar lines to keep the output readable.
        """
        text = self.stdout or ""
        marker = "Generating output HDF5 files..."
        marker_index = text.find(marker)
        if marker_index != -1:
            return text[marker_index:].strip()

        lines = text.splitlines()
        progress_pattern = re.compile(r"^\s*\[[=\s]+\]\s*\d+%\s*$")
        filtered = [line for line in lines if not progress_pattern.match(line)]
        cleaned = "\n".join(filtered)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
