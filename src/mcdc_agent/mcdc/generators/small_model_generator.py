"""Generator optimized for small LLMs using complexity routing and phased generation.

Flow:
  1. Classify complexity (simple/complex)
  2. Generate JSON plan (response_format=json_object)
  3. If complex: generate dedicated geometry plan
  4. Generate code (phased or full-script, based on generation_mode)
  5. Dry-run validation + fix loop (repeated-error detection)
"""

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mc_agent.core.utils import extract_response_text, extract_code
from mc_agent.mcdc.tools.api_reference import APIReference
from mc_agent.mcdc.tools.validator import DryRunValidator, ValidationStatus

from .shared import ERROR_HINTS
from .small_model_prompts import (
    CLASSIFY_PROMPT,
    SIMPLE_PLANNING_PROMPT,
    COMPLEX_PLANNING_PROMPT,
    COMPLEX_GEOMETRY_PLANNING_PROMPT,
    SETUP_PROMPT,
    GEOMETRY_PROMPT,
    COMPLEX_GEOMETRY_INSTRUCTIONS,
    FINALIZE_PROMPT,
    FULL_SCRIPT_PROMPT,
    FIX_PROMPT,
)

PHASES = ["setup", "geometry", "finalize"]
NO_REASONING = {"effort": "none", "exclude": True}
CONTEXT_METHODS = {
    "api_only": {
        "api": True,
        "examples": False,
        "plan": False,
        "geometry_plan": False,
    },
    "api_examples": {
        "api": True,
        "examples": True,
        "plan": False,
        "geometry_plan": False,
    },
    "api_examples_plan": {
        "api": True,
        "examples": True,
        "plan": True,
        "geometry_plan": False,
    },
    "api_examples_plan_geom": {
        "api": True,
        "examples": True,
        "plan": True,
        "geometry_plan": True,
    },
    "no_context": {
        "api": False,
        "examples": False,
        "plan": False,
        "geometry_plan": False,
    },
}


class SmallModelGenerator:
    """Generator optimized for small LLMs (8-14B+) using complexity routing.

    Uses LLM classification to route to appropriate prompts:
    - simple: Basic geometries without hierarchy or complex CSG
    - complex: CSG unions/complements, universe/lattice hierarchy

    All problems use phased generation (setup → geometry → finalize)
    with separate general and geometry plans injected into prompts.
    """

    def __init__(
        self,
        llm,
        max_fix_attempts: int = 0,
        validation_timeout: float = 15.0,
        generation_mode: str = "phased",
        context_method: str = "api_examples_plan_geom",
    ):
        self.llm = llm
        self.max_fix_attempts = max_fix_attempts
        self.validation_timeout = validation_timeout
        self.generation_mode = self._normalize_generation_mode(generation_mode)
        self.context_method = self._normalize_context_method(context_method)
        self.context_config = CONTEXT_METHODS[self.context_method]
        self.api_reference = APIReference()
        self.validator = DryRunValidator(timeout=validation_timeout)
        self.examples_registry = self._discover_examples()

        # Tracking (public, for plan-only mode)
        self.last_raw_response: str = ""
        self.last_mode: str = ""
        self.last_plan: Dict = {}
        self.last_geometry_plan: Dict = {}
        self.last_phase_scripts: Dict[str, str] = {}
        self.last_script: str = ""

    # =========================================================================
    # LLM INVOCATION HELPERS
    # =========================================================================

    def _invoke(self, prompt: str) -> str:
        """Invoke the LLM and return the text response."""
        response = self.llm.invoke({
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": copy.deepcopy(NO_REASONING),
        })
        text = extract_response_text(response)
        self.last_raw_response = text
        return text

    def _invoke_json(self, prompt: str) -> Dict:
        """Invoke the LLM with JSON response format and parse the result."""
        response = self.llm.invoke({
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": copy.deepcopy(NO_REASONING),
            "response_format": {"type": "json_object"},
        })
        text = extract_response_text(response)
        self.last_raw_response = text
        return self._extract_json(text)

    # =========================================================================
    # MAIN GENERATION FLOW
    # =========================================================================

    def generate(self, prompt: str, plan_only: bool = False) -> Optional[str]:
        """Generate MCDC script from a natural language prompt.

        Args:
            prompt: Natural language simulation description
            plan_only: If True, only generate plans and return None.

        Returns:
            Complete MCDC script, or None if plan_only=True
        """
        self.last_plan = {}
        self.last_geometry_plan = {}
        self.last_phase_scripts = {}
        self.last_script = ""

        # 1. Classify complexity
        mode = self._detect_complexity(prompt)
        self.last_mode = mode
        print(f"[Generator] Mode: {mode}")

        # 2. Generate general plan (JSON)
        plan = {}
        if plan_only or self._should_include("plan") or self._should_include("geometry_plan"):
            plan = self._plan(prompt, mode)
        self.last_plan = plan

        # 3. Generate geometry plan (JSON, complex only)
        geometry_plan = {}
        if mode == "complex" and (plan_only or self._should_include("geometry_plan")):
            geometry_plan = self._plan_geometry(prompt, plan)
            self.last_geometry_plan = geometry_plan

        if plan_only:
            return None

        # 4. Generate code
        if self.generation_mode == "full":
            print("[Generator] Full-script generation")
            script = self._generate_full_script(prompt, plan, geometry_plan, mode)
            self.last_phase_scripts["full"] = script
        else:
            # Phased generation (default)
            script = ""
            for phase in PHASES:
                print(f"[Generator] Phase: {phase}")
                script = self._run_phase(phase, prompt, plan, geometry_plan, script, mode)
                self.last_phase_scripts[phase] = script

        # 5. Dry-run validation + fix loop
        script = self._final_validation(prompt, plan, geometry_plan, script, mode)
        self.last_script = script

        return script

    @staticmethod
    def _normalize_generation_mode(generation_mode: str) -> str:
        """Normalize generation mode while keeping 'full' backward compatible."""
        if generation_mode == "one_shot":
            return "full"
        if generation_mode in {"phased", "full"}:
            return generation_mode
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")

    @staticmethod
    def _normalize_context_method(context_method: str) -> str:
        """Validate supported context methods."""
        if context_method not in CONTEXT_METHODS:
            valid = ", ".join(sorted(CONTEXT_METHODS))
            raise ValueError(f"Unsupported context_method '{context_method}'. Valid values: {valid}")
        return context_method

    def _should_include(self, key: str) -> bool:
        """Return whether a context component should be included in codegen prompts."""
        return self.context_config.get(key, False)

    def generate_plans(
        self,
        prompts: List[Dict[str, str]],
        output_path: Path,
        model_name: str = "",
    ) -> List[Dict]:
        """Run planning-only generation for a batch of prompts and save results.

        Args:
            prompts: List of dicts with keys ``name`` and ``prompt``.
            output_path: File path (without extension) to write the JSON output.
            model_name: Optional model name string to include in the output.

        Returns:
            List of plan result dicts.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        plan_results: List[Dict] = []

        total = len(prompts)
        for i, item in enumerate(prompts, 1):
            name = item.get("name", f"prompt_{i}")
            prompt_text = item.get("prompt", "")
            print(f"[generate_plans] ({i}/{total}) Planning: {name}")

            result: Dict = {
                "name": name,
                "prompt": prompt_text,
                "mode": "",
                "plan": {},
                "geometry_plan": {},
                "error": None,
            }

            try:
                self.generate(prompt_text, plan_only=True)
                result["mode"] = self.last_mode
                result["plan"] = self.last_plan
                result["geometry_plan"] = self.last_geometry_plan
            except Exception as exc:
                result["error"] = str(exc)

            plan_results.append(result)

        # Write JSON output
        output_doc = {
            "model": model_name,
            "timestamp": timestamp,
            "total": total,
            "plans": plan_results,
        }
        output_file = Path(str(output_path)).with_suffix("")
        output_file = Path(str(output_file) + "_plans.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(output_doc, indent=2))
        print(f"[generate_plans] Saved plans to: {output_file}")

        return plan_results

    # =========================================================================
    # COMPLEXITY DETECTION
    # =========================================================================

    def _detect_complexity(self, prompt: str) -> str:
        """Detect complexity using LLM classification.

        Returns:
            "simple" or "complex"
        """
        response_text = self._invoke(CLASSIFY_PROMPT.format(prompt=prompt))
        return "complex" if "complex" in response_text.strip().lower() else "simple"

    # =========================================================================
    # PLANNING (JSON output)
    # =========================================================================

    def _plan(self, prompt: str, mode: str) -> Dict:
        """Generate a general simulation plan as JSON."""
        planning_prompt = (
            COMPLEX_PLANNING_PROMPT.format(prompt=prompt)
            if mode == "complex"
            else SIMPLE_PLANNING_PROMPT.format(prompt=prompt)
        )
        return self._invoke_json(planning_prompt)

    def _plan_geometry(self, prompt: str, plan: Dict) -> Dict:
        """Generate a dedicated geometry plan for complex problems."""
        geometry_prompt = COMPLEX_GEOMETRY_PLANNING_PROMPT.format(
            prompt=prompt,
            general_plan_json=json.dumps(plan, indent=2),
        )
        return self._invoke_json(geometry_prompt)

    # =========================================================================
    # FULL SCRIPT GENERATION
    # =========================================================================

    def _generate_full_script(
        self, prompt: str, plan: Dict, geometry_plan: Dict, mode: str
    ) -> str:
        """Generate the complete MCDC script in a single LLM call."""
        general_plan_section = self._format_plan_section(
            "## General Plan",
            plan,
            include=self._should_include("plan"),
        )
        geometry_section = self._format_plan_section(
            "## Geometry Plan",
            geometry_plan,
            include=self._should_include("geometry_plan"),
        )
        api_section = self._format_markdown_section(
            "## Full API Reference",
            self.api_reference.get_full_text() if self._should_include("api") else "",
        )
        examples_section = self._format_examples_section(
            "## Reference Examples",
            "Study these complete working scripts. Use them to learn the API patterns,\n"
            "but use values from YOUR prompt and any provided plan, not from the examples.",
            self._get_full_examples(mode, prompt) if self._should_include("examples") else "",
        )

        full_prompt = FULL_SCRIPT_PROMPT.format(
            prompt=prompt,
            general_plan_section=general_plan_section,
            geometry_plan_section=geometry_section,
            api_reference_section=api_section,
            examples_section=examples_section,
        )

        response_text = self._invoke(full_prompt)
        return extract_code(response_text)

    def _get_full_examples(self, mode: str, prompt: str) -> str:
        """Get 3 full reference examples scored by keyword relevance."""
        all_examples = self.examples_registry.get(mode, [])
        if not all_examples:
            return "# See documentation"

        prompt_lower = prompt.lower()

        scored = sorted(
            all_examples,
            key=lambda ex: sum(1 for kw in ex["keywords"] if kw in prompt_lower),
            reverse=True,
        )
        selected = scored[:3]

        selected_info = ", ".join(ex["name"] for ex in selected)
        print(f"[Generator] Full examples: {selected_info}")

        parts = []
        for i, ex in enumerate(selected, 1):
            parts.append(f"### Example {i} ({ex['name']})\n```python\n{ex['content']}\n```")
        return "\n\n".join(parts)

    # =========================================================================
    # PHASE GENERATION
    # =========================================================================

    def _run_phase(
        self,
        phase: str,
        prompt: str,
        plan: Dict,
        geometry_plan: Dict,
        current_script: str,
        mode: str,
    ) -> str:
        """Run a single generation phase (no per-phase validation)."""
        return self._generate_phase(
            phase, prompt, plan, geometry_plan, current_script, mode
        )

    def _generate_phase(
        self,
        phase: str,
        prompt: str,
        plan: Dict,
        geometry_plan: Dict,
        current_script: str,
        mode: str,
    ) -> str:
        """Generate code for a specific phase."""
        general_plan_section = self._format_plan_section(
            "## General Plan",
            plan,
            include=self._should_include("plan"),
        )
        geometry_section = self._format_plan_section(
            "## Geometry Plan",
            geometry_plan,
            include=self._should_include("geometry_plan"),
        )
        examples = (
            self._get_examples(mode, prompt, phase=phase)
            if self._should_include("examples")
            else ""
        )

        if phase == "setup":
            api_sections = self._format_markdown_section(
                "## API Reference",
                "\n\n".join(filter(None, [
                    self.api_reference.get_section("## Materials"),
                    self.api_reference.get_section("## Surfaces"),
                ])) if self._should_include("api") else "",
            )
            examples_section = self._format_examples_section(
                "## Reference Snippets",
                "Study the MATERIALS and SURFACES sections from these examples.",
                examples,
            )
            phase_prompt = SETUP_PROMPT.format(
                prompt=prompt,
                general_plan_section=general_plan_section,
                geometry_plan_section=geometry_section,
                api_reference_section=api_sections,
                examples_section=examples_section,
            )
        elif phase == "geometry":
            api_sections_list = [self.api_reference.get_section("## Cells and Regions")]
            if mode == "complex":
                api_sections_list.append(
                    self.api_reference.get_section("## Universe & Lattice")
                )
            api_sections = self._format_markdown_section(
                "## API Reference",
                "\n\n".join(filter(None, api_sections_list)) if self._should_include("api") else "",
            )
            mode_instructions = COMPLEX_GEOMETRY_INSTRUCTIONS if mode == "complex" else ""
            examples_section = self._format_examples_section(
                "## Reference Snippets",
                "Study the GEOMETRY (cells, regions, universes, lattices) sections from these examples.",
                examples,
            )
            phase_prompt = GEOMETRY_PROMPT.format(
                prompt=prompt,
                general_plan_section=general_plan_section,
                geometry_plan_section=geometry_section,
                current_script=current_script,
                mode_instructions=mode_instructions,
                api_reference_section=api_sections,
                examples_section=examples_section,
            )
        elif phase == "finalize":
            api_sections = self._format_markdown_section(
                "## API Reference",
                "\n\n".join(filter(None, [
                    self.api_reference.get_section("## Source"),
                    self.api_reference.get_section("## Tallies"),
                    self.api_reference.get_section("## Settings"),
                    self.api_reference.get_section("## Techniques (Variance Reduction)"),
                ])) if self._should_include("api") else "",
            )
            examples_section = self._format_examples_section(
                "## Reference Snippets",
                "Study the SOURCE, TALLY, and SETTINGS sections from these examples.",
                examples,
            )
            phase_prompt = FINALIZE_PROMPT.format(
                prompt=prompt,
                general_plan_section=general_plan_section,
                geometry_plan_section=geometry_section,
                current_script=current_script,
                api_reference_section=api_sections,
                examples_section=examples_section,
            )
        else:
            raise ValueError(f"Unknown phase: {phase}")

        response_text = self._invoke(phase_prompt)
        return extract_code(response_text)

    def _fix_script(
        self,
        prompt: str,
        plan: Dict,
        geometry_plan: Dict,
        script: str,
        error: str,
        mode: str,
    ) -> str:
        """Fix a script that failed validation."""
        hints = []
        error_lower = error.lower()
        for pattern, hint in ERROR_HINTS.items():
            if re.search(pattern, error_lower, re.IGNORECASE):
                hints.append(hint)

        hints_section = ""
        if hints:
            hints_section = "## Hints\n" + "\n".join(hints) + "\n"
        general_plan_section = self._format_plan_section(
            "## General Plan",
            plan,
            include=self._should_include("plan"),
        )
        geometry_section = self._format_plan_section(
            "## Geometry Plan",
            geometry_plan,
            include=self._should_include("geometry_plan"),
        )
        api_section = self._format_markdown_section(
            "## Relevant API Reference",
            self.api_reference.get_full_text() if self._should_include("api") else "",
        )
        examples_section = self._format_examples_section(
            "## Reference Examples",
            "Study these working examples for correct patterns:",
            self._get_examples(mode, prompt) if self._should_include("examples") else "",
        )

        fix_prompt = FIX_PROMPT.format(
            prompt=prompt,
            general_plan_section=general_plan_section,
            geometry_plan_section=geometry_section,
            hints_section=hints_section,
            error=error[:4000],
            api_reference_section=api_section,
            examples_section=examples_section,
            script=script,
        )

        response_text = self._invoke(fix_prompt)
        return extract_code(response_text)

    # =========================================================================
    # EXAMPLES
    # =========================================================================

    def _discover_examples(self) -> Dict[str, list]:
        """Discover and index all examples from the examples directory.

        Returns:
            Dict mapping mode -> list of example dicts with
            {name, content, keywords, phases: {setup, geometry, finalize}}
        """
        registry = {"simple": [], "complex": []}
        examples_dir = Path(__file__).parent / "examples"

        if not examples_dir.exists():
            return registry

        COMMON_KEYWORDS = {
            "sphere", "cylinder", "slab", "plane", "box", "cube",
            "eigenvalue", "criticality", "k-effective", "k_eff",
            "lattice", "universe", "translation", "rotation",
            "source", "tally", "current", "void", "streaming",
            "pin", "assembly", "core", "checkerboard", "csg",
            "union", "intersection", "complement", "carve", "simple", "slab"
        }

        for path in examples_dir.glob("*.py"):
            if path.name.startswith("__"):
                continue

            content = path.read_text()
            doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            doc_text = doc_match.group(1).lower() if doc_match else ""

            # Determine mode
            mode = "simple"
            complexity_match = re.search(r'complexity:\s*(simple|csg|hierarchy|complex)', doc_text)
            if complexity_match:
                raw_mode = complexity_match.group(1)
                mode = "simple" if raw_mode == "simple" else "complex"
            elif any(kw in doc_text for kw in ("hierarchy", "lattice", "csg")):
                mode = "complex"

            # Extract keywords
            found_keywords = {k for k in COMMON_KEYWORDS if k in doc_text}

            # Extract per-phase snippets (complex mode only)
            phases = {}
            if mode == "complex":
                for phase_name in ("setup", "geometry", "finalize"):
                    snippet = self._extract_phase_content(content, phase_name)
                    if snippet:
                        phases[phase_name] = snippet

            registry.get(mode, registry["simple"]).append({
                "name": path.name,
                "content": content,
                "keywords": found_keywords,
                "phases": phases,
            })

        return registry

    @staticmethod
    def _extract_phase_content(content: str, phase: str) -> str:
        """Extract code between PHASE_<PHASE> START/END markers."""
        tag = phase.upper()
        pattern = rf"# === PHASE_{tag} START ===(.*?)# === PHASE_{tag} END ==="
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _get_examples(self, mode: str, prompt: str = "", phase: str = None) -> str:
        """Get formatted example string for a mode using keyword scoring.

        For complex mode with a phase, returns phase-scoped snippets.
        For simple mode or phase=None, returns full examples.
        """
        all_examples = self.examples_registry.get(mode, [])
        if not all_examples:
            return "# See documentation"

        prompt_lower = prompt.lower()

        # Score and sort by keyword relevance
        scored = sorted(
            all_examples,
            key=lambda ex: sum(1 for kw in ex["keywords"] if kw in prompt_lower),
            reverse=True,
        )
        selected = scored[:5]

        selected_info = ", ".join(ex["name"] for ex in selected)
        print(f"[Generator] Examples (phase={phase}): {selected_info}")

        # Phase-scoped snippets for complex mode
        if phase and mode == "complex":
            parts = []
            for i, ex in enumerate(selected, 1):
                snippet = ex.get("phases", {}).get(phase, "")
                if snippet:
                    parts.append(f"### Snippet {i} ({ex['name']})\n```python\n{snippet}\n```")
                else:
                    parts.append(f"### Example {i} ({ex['name']})\n```python\n{ex['content']}\n```")
            return "\n\n".join(parts) if parts else "# See documentation"

        # Full examples
        parts = []
        for i, ex in enumerate(selected, 1):
            parts.append(f"### Example {i} ({ex['name']})\n```python\n{ex['content']}\n```")
        return "\n\n".join(parts)

    # =========================================================================
    # PLAN HELPERS
    # =========================================================================

    @staticmethod
    def _clean_plan_for_codegen(plan: Dict) -> Dict:
        """Remove non-API metadata (like 'role') from cells before sending to model."""
        cleaned = copy.deepcopy(plan)
        allowed_cell_keys = {"cell", "material", "fill", "region", "translation", "rotation"}
        if "cells" in cleaned:
            cleaned["cells"] = [
                {k: v for k, v in cell.items() if k in allowed_cell_keys}
                if isinstance(cell, dict) else cell
                for cell in cleaned["cells"]
            ]
        return cleaned

    def _format_plan_section(self, title: str, plan: Dict, include: bool) -> str:
        """Format a JSON plan section for prompt injection."""
        if not include or not plan:
            return ""
        cleaned = self._clean_plan_for_codegen(plan)
        return f"{title}\n```json\n{json.dumps(cleaned, indent=2)}\n```"

    @staticmethod
    def _format_markdown_section(title: str, content: str) -> str:
        """Format a markdown section if content is available."""
        content = content.strip()
        if not content:
            return ""
        return f"{title}\n{content}"

    @staticmethod
    def _format_examples_section(title: str, intro: str, examples: str) -> str:
        """Format an examples section if examples are enabled."""
        examples = examples.strip()
        if not examples:
            return ""
        return f"{title}\n{intro}\n\n{examples}"

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_syntax(self, script: str) -> Tuple[bool, str]:
        """Validate script syntax with compile + lightweight API checks."""
        try:
            compile(script, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        # Check for invalid API usage patterns
        invalid_setting = re.search(
            r'mcdc\.settings\.(?!N_particle|N_batch|N_census|output_name|'
            r'active_bank_buffer|census_bank_buffer_ratio|source_bank_buffer_ratio|'
            r'time_boundary)[a-zA-Z_0-9]+\s*=',
            script
        )
        if invalid_setting:
            attr = invalid_setting.group(0).strip(' =')
            return False, f"APIError: Invalid assignment to {attr}. Use constructors instead."

        if re.search(r'mcdc\.settings\..*\.append\(', script):
            return False, "APIError: mcdc.settings.*.append() is not valid. Use constructors directly."

        return True, ""

    def _validate_full(self, script: str) -> Tuple[bool, str]:
        """Run full dry-run validation including MCDC execution."""
        success, error = self._validate_syntax(script)
        if not success:
            return success, error

        result = self.validator.validate(script)

        if result.status == ValidationStatus.SUCCESS and not result.warnings:
            return True, ""

        # Check for geometry issues
        is_geometry_issue = any(
            pattern in (result.error_message or "").lower() or
            "particle is lost" in (result.raw_output or "").lower()
            for pattern in ["particle", "lost", "geometry", "gap"]
        )

        if is_geometry_issue:
            numba_result = self.validator.validate_numba(script, timeout=self.validation_timeout)
            if numba_result.lost_particle_coords:
                coords = numba_result.lost_particle_coords
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                zs = [c[2] for c in coords]
                diagnosis = (
                    f"GEOMETRY GAP - Particles escaping between:\n"
                    f"  X: [{min(xs):.2f}, {max(xs):.2f}]\n"
                    f"  Y: [{min(ys):.2f}, {max(ys):.2f}]\n"
                    f"  Z: [{min(zs):.2f}, {max(zs):.2f}]\n\n"
                    f"FIX: Ensure bounding surfaces match geometry exactly.\n"
                    f"For lattices: x=(start, pitch, N) covers [start, start+pitch*N]"
                )
                return False, f"{diagnosis}\n\n{result.error_message or ''}"

        # Build error message
        error_parts = [f"Status: {result.status.value.upper()}"]
        if result.runtime_report:
            error_parts.append(f"Runtime: {result.runtime_report}")
        if result.error_message:
            error_parts.append(f"Error: {result.error_message}")
        if result.warnings:
            error_parts.append(f"Warnings: {'; '.join(result.warnings)}")
        if result.raw_output:
            # Extract traceback and lost particle info
            text = result.raw_output
            if "Traceback" in text:
                idx = text.find("Traceback")
                error_parts.append(f"\nTRACEBACK:\n{text[idx:idx+2000]}")
            lines = text.splitlines()
            lost = [l for l in lines if "particle is lost" in l]
            if lost:
                error_parts.append("\nLOST PARTICLES:\n" + "\n".join(lost[:5]))
            elif result.status in (ValidationStatus.ERROR, ValidationStatus.TIMEOUT):
                tail = "\n".join(lines[-10:])
                if tail.strip():
                    error_parts.append("\nOUTPUT TAIL:\n" + tail)

        return False, "\n".join(error_parts) if error_parts else "Validation failed"

    def _final_validation(
        self, prompt: str, plan: Dict, geometry_plan: Dict, script: str, mode: str
    ) -> str:
        """Run full dry-run validation with fix loop and repeated-error detection."""
        success, error = self._validate_full(script)

        if success:
            print("[Generator] Final validation passed")
            return script

        print("[Generator] Final validation failed")
        print(self._format_validation_error(error))

        if self.max_fix_attempts <= 0:
            print("[Generator] Final validation failed; fix attempts disabled")
            return script

        for attempt in range(self.max_fix_attempts):
            print(f"[Generator] Final fix attempt {attempt + 1}/{self.max_fix_attempts}")
            script = self._fix_script(prompt, plan, geometry_plan, script, error, mode)
            success, error = self._validate_full(script)
            if success:
                print(f"[Generator] Fixed after {attempt + 1} attempt(s)")
                return script
            print(f"[Generator] Fix attempt {attempt + 1} failed")
            print(self._format_validation_error(error))

        print("[Generator] All fix attempts exhausted, returning best effort")
        return script

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @staticmethod
    def _format_validation_error(error: str, max_chars: int = 3000) -> str:
        """Trim long validation output before logging it."""
        text = (error or "Validation failed").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """Extract JSON object from response text."""
        # Try code block first
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try raw JSON
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            return json.loads(text[first_brace:last_brace + 1])

        raise ValueError("No JSON object found in response")
