"""Second-generation MCDC agent architecture."""

from .llm import DEFAULT_OPENROUTER_MODEL, OpenRouterLLM, load_llm
from .types import GenerationArtifacts, GeneratorConfig

__all__ = [
    "DEFAULT_OPENROUTER_MODEL",
    "GenerationArtifacts",
    "GeneratorConfig",
    "OpenRouterLLM",
    "load_llm",
]
