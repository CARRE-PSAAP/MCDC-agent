import json
from pathlib import Path
from typing import Any

from mcdc_agent.mcdc.generators import SmallModelGenerator
from mcdc_agent.mcdc.utils import load_llm


def build_direct_generator(
    *,
    model: str | None = None,
    provider: str | None = None,
    context_method: str = "api_examples_plan_geom",
    generation_mode: str = "phased",
    max_fix_attempts: int = 0,
    temperature: float = 0.1,
) -> SmallModelGenerator:
    llm = load_llm(
        temperature=temperature,
        model=model,
        provider=provider or "openrouter",
    )
    return SmallModelGenerator(
        llm,
        max_fix_attempts=max_fix_attempts,
        generation_mode=generation_mode,
        context_method=context_method,
    )


def save_generation_trace(
    *,
    generator: SmallModelGenerator,
    prompt: str,
    script: str | None,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "prompt.txt").write_text(prompt)
    if generator.last_plan:
        (output_path / "general_plan.json").write_text(
            json.dumps(generator.last_plan, indent=2)
        )
    if generator.last_geometry_plan:
        (output_path / "geometry_plan.json").write_text(
            json.dumps(generator.last_geometry_plan, indent=2)
        )
    for phase, phase_script in generator.last_phase_scripts.items():
        (output_path / f"{phase}.py").write_text(phase_script)
    if script is not None:
        (output_path / "final_script.py").write_text(script)

    summary: dict[str, Any] = {
        "mode": generator.last_mode,
        "phases": list(generator.last_phase_scripts),
        "has_plan": bool(generator.last_plan),
        "has_geometry_plan": bool(generator.last_geometry_plan),
    }
    (output_path / "trace_summary.json").write_text(json.dumps(summary, indent=2))
    return output_path
