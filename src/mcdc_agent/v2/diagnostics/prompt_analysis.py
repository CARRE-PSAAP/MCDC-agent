import re
from dataclasses import dataclass, field

from mcdc_agent.v2.materials.catalog import MaterialCatalog


@dataclass(slots=True)
class PromptIntent:
    raw_prompt: str
    expects_ce: bool = False
    named_materials: list[str] = field(default_factory=list)
    geometry_features: set[str] = field(default_factory=set)
    requested_tally_kinds: set[str] = field(default_factory=set)
    requested_scores: set[str] = field(default_factory=set)
    requested_boundaries: set[str] = field(default_factory=set)
    source_style: str | None = None
    run_mode: str = "fixed"
    lattice_dims: list[tuple[int, int]] = field(default_factory=list)
    interval_hints: dict[str, tuple[float, float]] = field(default_factory=dict)
    radius_hints: list[float] = field(default_factory=list)


_SCORE_KEYWORDS = {
    "flux": "flux",
    "fission": "fission",
    "capture": "capture",
    "density": "density",
    "collision": "collision",
    "net current": "net-current",
    "net-current": "net-current",
    "current": "net-current",
}


def extract_prompt_intent(prompt: str) -> PromptIntent:
    text = prompt.strip()
    lowered = text.lower()
    intent = PromptIntent(raw_prompt=text)

    intent.expects_ce = any(token in lowered for token in ("continuous-energy", "continuous energy", " ce "))
    intent.named_materials = _find_named_materials(lowered)
    intent.geometry_features = _find_geometry_features(lowered)
    intent.requested_tally_kinds = _find_tally_kinds(lowered)
    intent.requested_scores = _find_scores(lowered)
    intent.requested_boundaries = {
        boundary for boundary in ("vacuum", "reflective") if boundary in lowered
    }
    intent.source_style = _find_source_style(lowered)
    intent.run_mode = "eigenvalue" if any(token in lowered for token in ("eigenvalue", "k-eigenvalue", "k-effective", "criticality", "keff")) else "fixed"
    intent.lattice_dims = _find_lattice_dims(lowered)
    intent.interval_hints = _find_interval_hints(lowered)
    intent.radius_hints = _find_radius_hints(lowered)
    return intent


def format_prompt_intent(intent: PromptIntent) -> str:
    lines = [
        f"Run mode: {intent.run_mode}",
        f"Continuous-energy requested: {intent.expects_ce}",
        f"Source style: {intent.source_style or 'unspecified'}",
    ]
    if intent.named_materials:
        lines.append("Named materials: " + ", ".join(intent.named_materials))
    if intent.geometry_features:
        lines.append("Geometry features: " + ", ".join(sorted(intent.geometry_features)))
    if intent.requested_tally_kinds:
        lines.append("Requested tally kinds: " + ", ".join(sorted(intent.requested_tally_kinds)))
    if intent.requested_scores:
        lines.append("Requested tally scores: " + ", ".join(sorted(intent.requested_scores)))
    if intent.requested_boundaries:
        lines.append("Requested boundaries: " + ", ".join(sorted(intent.requested_boundaries)))
    if intent.lattice_dims:
        dims = ", ".join(f"{x}x{y}" for x, y in intent.lattice_dims)
        lines.append("Lattice dimensions: " + dims)
    if intent.interval_hints:
        lines.append(
            "Axis intervals: "
            + ", ".join(f"{axis}=[{low}, {high}]" for axis, (low, high) in sorted(intent.interval_hints.items()))
        )
    if intent.radius_hints:
        lines.append("Radius hints: " + ", ".join(str(value) for value in intent.radius_hints))
    return "\n".join(lines)


def _find_named_materials(lowered_prompt: str) -> list[str]:
    catalog = MaterialCatalog()
    found: list[str] = []
    for row in catalog.materials:
        names = [row["name"].lower()]
        aliases = [alias.strip().lower() for alias in row.get("aliases", "").split(";") if alias.strip()]
        for candidate in names + aliases:
            if not candidate:
                continue
            if len(candidate) < 3:
                continue
            pattern = r"(?<![a-z0-9_])" + re.escape(candidate) + r"(?![a-z0-9_])"
            if re.search(pattern, lowered_prompt):
                canonical = row["name"]
                if canonical not in found:
                    found.append(canonical)
                break
    return found


def _find_geometry_features(lowered_prompt: str) -> set[str]:
    features = set()
    keyword_map = {
        "sphere": "sphere",
        "cylinder": "cylinder",
        "slab": "slab",
        "lattice": "lattice",
        "assembly": "assembly",
        "core": "core",
        "pin": "pin",
        "checkerboard": "checkerboard",
        "hierarchy": "hierarchy",
        "universe": "hierarchy",
        "channel": "csg",
        "duct": "csg",
        "shell": "csg",
        "ring": "csg",
    }
    for keyword, feature in keyword_map.items():
        if keyword in lowered_prompt:
            features.add(feature)
    return features


def _find_tally_kinds(lowered_prompt: str) -> set[str]:
    kinds = set()
    if "mesh tally" in lowered_prompt or "mesh" in lowered_prompt:
        kinds.add("TallyMesh")
    if "surface tally" in lowered_prompt or "through the surface" in lowered_prompt:
        kinds.add("TallySurface")
    if "cell tally" in lowered_prompt:
        kinds.add("TallyCell")
    if "global tally" in lowered_prompt:
        kinds.add("TallyGlobal")
    return kinds


def _find_scores(lowered_prompt: str) -> set[str]:
    scores = set()
    for keyword, score in _SCORE_KEYWORDS.items():
        if keyword in lowered_prompt:
            scores.add(score)
    return scores


def _find_source_style(lowered_prompt: str) -> str | None:
    if "point source" in lowered_prompt or "point at" in lowered_prompt:
        return "point"
    if "beam" in lowered_prompt:
        return "beam"
    if "across the entire" in lowered_prompt or "volumetric" in lowered_prompt or "uniform source" in lowered_prompt:
        return "volumetric"
    if "source across" in lowered_prompt:
        return "volumetric"
    return None


def _find_lattice_dims(lowered_prompt: str) -> list[tuple[int, int]]:
    dims = []
    for match in re.finditer(r"(\d+)\s*x\s*(\d+)", lowered_prompt):
        dims.append((int(match.group(1)), int(match.group(2))))
    return dims


def _find_interval_hints(lowered_prompt: str) -> dict[str, tuple[float, float]]:
    hints: dict[str, tuple[float, float]] = {}
    patterns = [
        r"between\s+([xyz])\s*=\s*([-+]?\d*\.?\d+)\s+and\s+([-+]?\d*\.?\d+)",
        r"([xyz])\s+from\s+([-+]?\d*\.?\d+)\s+to\s+([-+]?\d*\.?\d+)",
        r"([xyz])\s*=\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered_prompt):
            axis = match.group(1)
            low = float(match.group(2))
            high = float(match.group(3))
            hints[axis] = (min(low, high), max(low, high))
    return hints


def _find_radius_hints(lowered_prompt: str) -> list[float]:
    hints = []
    for match in re.finditer(r"radius\s*(?:=|of)?\s*([-+]?\d*\.?\d+)", lowered_prompt):
        hints.append(float(match.group(1)))
    return hints
