from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GeneratorConfig:
    """Configuration shared by full-generation and interactive sessions."""

    model: str | None = None
    provider: str = "openrouter"
    context_method: str = "api_examples_plan_geom"
    generation_mode: str = "phased"
    max_fix_attempts: int = 0
    temperature: float = 0.1


@dataclass(slots=True)
class GenerationArtifacts:
    """Normalized generation result for the new v2 pipeline."""

    prompt: str
    script: str | None
    mode: str
    plan: dict[str, Any] = field(default_factory=dict)
    geometry_plan: dict[str, Any] = field(default_factory=dict)
    phase_scripts: dict[str, str] = field(default_factory=dict)
