from dataclasses import dataclass, field
from pathlib import Path

from mcdc_agent.v2.llm import DEFAULT_OPENROUTER_MODEL
from mcdc_agent.v2.types import GeneratorConfig


TOP_LEVEL_MENU = [
    ("1", "Learn MCDC"),
    ("2", "Ask Questions"),
    ("3", "Generate and Run Simulation"),
    ("4", "View/Edit Current Script"),
    ("5", "Results and Diagnostics"),
]

LEARN_MENU = [
    ("1", "Materials"),
    ("2", "Surfaces and Regions"),
    ("3", "Cells"),
    ("4", "Universes and Lattices"),
    ("5", "Sources"),
    ("6", "Tallies"),
    ("7", "Settings and Run"),
    ("8", "Example Walkthrough"),
]

DIAGNOSTICS_MENU = [
    ("1", "Visualize"),
    ("2", "Analyze / Diagnose"),
]


@dataclass(slots=True)
class AppConfig:
    """User-facing v2 configuration with sensible defaults for the interactive app."""

    provider: str = "openrouter"
    model: str = DEFAULT_OPENROUTER_MODEL
    context_method: str = "api_examples_plan_geom"
    generation_mode: str = "auto"
    max_fix_attempts: int = 3
    temperature: float = 0.1
    output_path: Path = field(default_factory=lambda: Path("mcdc_input.py"))

    def generator_config(self) -> GeneratorConfig:
        return GeneratorConfig(
            model=self.model,
            provider=self.provider,
            context_method=self.context_method,
            generation_mode=self.generation_mode,
            max_fix_attempts=self.max_fix_attempts,
            temperature=self.temperature,
        )
