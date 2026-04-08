from pathlib import Path

from mcdc_agent.v2.config import AppConfig
from mcdc_agent.v2.mcdc import MCDCGeneratorV2
from mcdc_agent.v2.types import GenerationArtifacts


class GenerationService:
    """High-level script generation service for the simplified v2 app."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._generator = MCDCGeneratorV2(config.generator_config())

    def plan(self, prompt: str) -> GenerationArtifacts:
        return self._generator.generate(prompt, plan_only=True)

    def generate(self, prompt: str) -> GenerationArtifacts:
        return self._generator.generate(prompt, plan_only=False)

    def save_script(self, script: str, output_path: Path | None = None) -> Path:
        destination = output_path or self.config.output_path
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(script)
        return destination
