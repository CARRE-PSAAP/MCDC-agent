from mcdc_agent.v2.materials import MaterialCatalog
from mcdc_agent.v2.mcdc import MCDCGeneratorV2
from mcdc_agent.v2.types import GenerationArtifacts, GeneratorConfig


class InteractiveGenerationSession:
    """Simple non-LangChain interactive controller for the new generator stack."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.materials = MaterialCatalog()
        self.generator = MCDCGeneratorV2(config)

    def lookup_material(self, query: str) -> str:
        return self.materials.format_context(query)

    def plan(self, prompt: str, *, extra_context: str = "") -> GenerationArtifacts:
        return self.generator.generate(prompt, plan_only=True, extra_context=extra_context)

    def generate(self, prompt: str, *, extra_context: str = "") -> GenerationArtifacts:
        return self.generator.generate(prompt, plan_only=False, extra_context=extra_context)
