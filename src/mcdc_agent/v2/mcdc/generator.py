from mcdc_agent.mcdc.generators import SmallModelGenerator
from mcdc_agent.v2.llm import load_llm
from mcdc_agent.v2.types import GenerationArtifacts, GeneratorConfig


class MCDCGeneratorV2:
    """Thin v2 wrapper around the proven small-model pipeline."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.llm = None
        self.generators = {}

    def _get_llm(self):
        if self.llm is None:
            self.llm = load_llm(
                temperature=self.config.temperature,
                model=self.config.model,
                provider=self.config.provider,
            )
        return self.llm

    def _get_generator(self, generation_mode: str | None = None) -> SmallModelGenerator:
        mode = generation_mode or self.config.generation_mode
        if mode == "auto":
            mode = "phased"
        if mode not in self.generators:
            self.generators[mode] = SmallModelGenerator(
                self._get_llm(),
                max_fix_attempts=self.config.max_fix_attempts,
                generation_mode=mode,
                context_method=self.config.context_method,
                artifact_output_dir=self.config.trace_dir,
            )
        return self.generators[mode]

    def _select_generation_mode(self, prompt: str) -> str:
        if self.config.generation_mode != "auto":
            return self.config.generation_mode

        detector = self._get_generator("phased")
        complexity = detector._detect_complexity(prompt)
        return "full" if complexity == "simple" else "phased"

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
        effective_prompt = self._build_prompt(prompt, extra_context)
        selected_mode = self._select_generation_mode(effective_prompt)
        generator = self._get_generator(selected_mode)
        script = generator.generate(effective_prompt, plan_only=plan_only)
        return GenerationArtifacts(
            prompt=prompt,
            script=script,
            mode=generator.last_mode,
            plan=generator.last_plan,
            geometry_plan=generator.last_geometry_plan,
            phase_scripts=generator.last_phase_scripts,
        )
