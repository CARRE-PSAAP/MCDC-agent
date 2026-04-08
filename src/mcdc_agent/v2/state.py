from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcdc_agent.v2.types import GenerationArtifacts


@dataclass(slots=True)
class AppState:
    """Shared interactive state for the simplified v2 application."""

    current_prompt: str = ""
    current_script: str = ""
    current_script_path: Path | None = None
    last_generation: GenerationArtifacts | None = None
    last_run_returncode: int | None = None
    last_run_stdout: str = ""
    last_run_stderr: str = ""
    last_output_path: Path | None = None
    last_output_h5: Path | None = None
    last_output_summary: dict[str, Any] = field(default_factory=dict)
