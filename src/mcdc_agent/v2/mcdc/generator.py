from mcdc_agent.mcdc.generators import SmallModelGenerator
from mcdc_agent.v2.llm import load_llm
from mcdc_agent.v2.types import GenerationArtifacts, GeneratorConfig


class MCDCGeneratorV2:
    """Thin v2 wrapper around the proven small-model pipeline."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.llm = None
        self.generator = None

    def _get_generator(self) -> SmallModelGenerator:
        if self.generator is None:
            self.llm = load_llm(
                temperature=self.config.temperature,
                model=self.config.model,
                provider=self.config.provider,
            )
            self.generator = SmallModelGenerator(
                self.llm,
                max_fix_attempts=self.config.max_fix_attempts,
                generation_mode=self.config.generation_mode,
                context_method=self.config.context_method,
            )
        return self.generator

    @staticmethod
    def _build_prompt(prompt: str, extra_context: str = "") -> str:
        if not extra_context.strip():
            return prompt
        return (
            f"{prompt.strip()}\n\n"
            "Additional reference context:\n"
            "Use this only when relevant to the user's request.\n\n"
            f"{extra_context.strip()}"
        )

    def generate(
        self,
        prompt: str,
        *,
        plan_only: bool = False,
        extra_context: str = "",
    ) -> GenerationArtifacts:
        generator = self._get_generator()
        effective_prompt = self._build_prompt(prompt, extra_context)
        script = generator.generate(effective_prompt, plan_only=plan_only)
        return GenerationArtifacts(
            prompt=prompt,
            script=script,
            mode=generator.last_mode,
            plan=generator.last_plan,
            geometry_plan=generator.last_geometry_plan,
            phase_scripts=generator.last_phase_scripts,
        )
