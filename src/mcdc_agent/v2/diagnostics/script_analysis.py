import ast
from dataclasses import dataclass, field
from typing import Any

from .prompt_analysis import PromptIntent, extract_prompt_intent


@dataclass(slots=True)
class MaterialDef:
    name: str
    kind: str
    is_ce: bool
    declared_name: str | None = None


@dataclass(slots=True)
class SurfaceDef:
    name: str
    kind: str
    axis: str | None = None
    value: float | None = None
    boundary_condition: str | None = None
    center: tuple[float, ...] | None = None
    radius: float | None = None


@dataclass(slots=True)
class MeshDef:
    name: str
    kind: str
    axes: dict[str, tuple[float | None, float | None, int | None]] = field(default_factory=dict)


@dataclass(slots=True)
class RegionDef:
    name: str
    text: str
    expr: ast.AST


@dataclass(slots=True)
class CellDef:
    name: str
    fill: str | None
    region_text: str
    region_expr: ast.AST | None
    translation: tuple[float, ...] | None = None
    rotation: tuple[float, ...] | None = None


@dataclass(slots=True)
class UniverseDef:
    name: str
    cells: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LatticeDef:
    name: str
    axes: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    universes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceDef:
    name: str
    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None
    position: tuple[float, ...] | None = None
    energy_group: int | None = None


@dataclass(slots=True)
class TallyDef:
    name: str
    kind: str
    target_name: str | None = None
    scores: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SettingsDef:
    n_particle: int | None = None
    n_batch: int | None = None
    n_cycle: int | None = None
    eigenmode: bool = False
    eigen_args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScriptModel:
    parse_error: str | None = None
    materials: dict[str, MaterialDef] = field(default_factory=dict)
    surfaces: dict[str, SurfaceDef] = field(default_factory=dict)
    regions: dict[str, RegionDef] = field(default_factory=dict)
    cells: dict[str, CellDef] = field(default_factory=dict)
    universes: dict[str, UniverseDef] = field(default_factory=dict)
    lattices: dict[str, LatticeDef] = field(default_factory=dict)
    meshes: dict[str, MeshDef] = field(default_factory=dict)
    sources: list[SourceDef] = field(default_factory=list)
    tallies: list[TallyDef] = field(default_factory=list)
    settings: SettingsDef = field(default_factory=SettingsDef)
    root_universe_cells: list[str] = field(default_factory=list)
    scalar_values: dict[str, float] = field(default_factory=dict)
    symbol_kinds: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CheckIssue:
    code: str
    severity: str
    message: str
    evidence: str = ""


def extract_script_model(script_text: str) -> ScriptModel:
    model = ScriptModel()
    if not script_text.strip():
        return model

    try:
        tree = ast.parse(script_text)
    except SyntaxError as exc:
        model.parse_error = str(exc)
        return model

    analyzer = _ScriptAnalyzer(script_text, model)
    analyzer.visit(tree)
    return model


def run_script_checks(model: ScriptModel, output_summary: dict[str, Any] | None = None) -> list[CheckIssue]:
    return run_script_checks_with_plan(model, output_summary=output_summary)


def run_script_checks_with_plan(
    model: ScriptModel,
    *,
    output_summary: dict[str, Any] | None = None,
    original_prompt: str = "",
    plan: dict[str, Any] | None = None,
    geometry_plan: dict[str, Any] | None = None,
) -> list[CheckIssue]:
    issues: list[CheckIssue] = []

    if model.parse_error:
        issues.append(CheckIssue("parse_error", "error", f"Script could not be parsed: {model.parse_error}"))
        return issues

    summary = output_summary or {}
    issues.extend(_check_root_universe(model))
    issues.extend(_check_lattice_extents(model))
    issues.extend(_check_source_bounds(model))
    issues.extend(_check_source_coverage(model))
    issues.extend(_check_reference_targets(model))
    issues.extend(_check_run_mode_consistency(model, summary))
    if original_prompt.strip():
        issues.extend(_check_prompt_alignment(model, extract_prompt_intent(original_prompt)))
    issues.extend(_check_plan_alignment(model, plan or {}, geometry_plan or {}))

    return issues


def format_script_model(model: ScriptModel) -> str:
    if model.parse_error:
        return f"Script parse error: {model.parse_error}"

    lines = [
        f"Materials: {len(model.materials)}",
        f"Surfaces: {len(model.surfaces)}",
        f"Regions: {len(model.regions)}",
        f"Cells: {len(model.cells)}",
        f"Universes: {len(model.universes)}",
        f"Lattices: {len(model.lattices)}",
        f"Sources: {len(model.sources)}",
        f"Tallies: {len(model.tallies)}",
    ]

    if model.materials:
        lines.append("Material modes:")
        for material in model.materials.values():
            mode = "CE" if material.is_ce else "MG"
            label = material.declared_name or material.name
            lines.append(f"  - {material.name}: {mode} ({label})")

    if model.lattices:
        lines.append("Lattice extents:")
        for lattice in model.lattices.values():
            axis_text = ", ".join(
                f"{axis}=({start}, {pitch}, {count})"
                for axis, (start, pitch, count) in sorted(lattice.axes.items())
            )
            lines.append(f"  - {lattice.name}: {axis_text or 'none'}")

    if model.meshes:
        lines.append("Meshes:")
        for mesh in model.meshes.values():
            axis_text = ", ".join(
                f"{axis}=({low}, {high}, {count})"
                for axis, (low, high, count) in sorted(mesh.axes.items())
            )
            lines.append(f"  - {mesh.name}: {mesh.kind} {axis_text or 'none'}")

    if model.root_universe_cells:
        lines.append("Root universe cells: " + ", ".join(model.root_universe_cells))
    else:
        lines.append("Root universe cells: none")

    return "\n".join(lines)


def format_check_issues(issues: list[CheckIssue]) -> str:
    if not issues:
        return "No structured script issues detected."

    lines = []
    for issue in issues:
        line = f"- [{issue.severity}] {issue.code}: {issue.message}"
        if issue.evidence:
            line += f" | evidence: {issue.evidence}"
        lines.append(line)
    return "\n".join(lines)


def format_plan_comparison(plan: dict[str, Any], geometry_plan: dict[str, Any]) -> str:
    if not plan and not geometry_plan:
        return "No generation plan is available."

    lines = []
    if plan:
        lines.append(f"General plan keys: {', '.join(sorted(plan.keys()))}")
        settings = plan.get("settings")
        tally = plan.get("tally")
        source = plan.get("source")
        if settings:
            lines.append(f"Plan settings: {settings}")
        if source:
            lines.append(f"Plan source: {source}")
        if tally:
            lines.append(f"Plan tally: {tally}")
    if geometry_plan:
        mode = geometry_plan.get("geometry_mode")
        if mode:
            lines.append(f"Geometry mode: {mode}")
        if geometry_plan.get("hierarchy"):
            lines.append("Geometry plan includes hierarchy.")
        if geometry_plan.get("csg_operations"):
            lines.append("Geometry plan includes CSG operations.")
        if geometry_plan.get("root"):
            lines.append(f"Geometry root: {geometry_plan.get('root')}")
    return "\n".join(lines)


class _ScriptAnalyzer(ast.NodeVisitor):
    def __init__(self, script_text: str, model: ScriptModel):
        self.script_text = script_text
        self.model = model
        self._cell_counter = 0
        self._source_counter = 0
        self._tally_counter = 0

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            self.generic_visit(node)
            return

        target = node.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
            scalar_value = _eval_numeric(node.value, self.model.scalar_values)
            if scalar_value is not None:
                self.model.scalar_values[name] = scalar_value
                self.model.symbol_kinds[name] = "scalar"
                return

            if _is_region_expr(node.value):
                self.model.regions[name] = RegionDef(name=name, text=_safe_unparse(node.value), expr=node.value)
                self.model.symbol_kinds[name] = "region"
                return

            if isinstance(node.value, ast.Call):
                self._handle_call_assignment(name, node.value)
                return

        if isinstance(target, ast.Attribute):
            self._handle_settings_assignment(target, node.value)
            return

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self._handle_expr_call(node.value)
            return
        self.generic_visit(node)

    def _handle_call_assignment(self, name: str, call: ast.Call) -> None:
        dotted = _call_name(call)

        if dotted in {"mcdc.MaterialMG", "mcdc.Material"}:
            material = MaterialDef(
                name=name,
                kind=dotted.split(".")[-1],
                is_ce=dotted.endswith("Material"),
                declared_name=_literal_value(_keyword_value(call, "name"), self.model.scalar_values),
            )
            self.model.materials[name] = material
            self.model.symbol_kinds[name] = "material"
            return

        if dotted.startswith("mcdc.Surface."):
            surface = _surface_from_call(name, call, self.model.scalar_values)
            self.model.surfaces[name] = surface
            self.model.symbol_kinds[name] = "surface"
            return

        if dotted in {"mcdc.MeshUniform", "mcdc.MeshStructured"}:
            self.model.meshes[name] = _mesh_from_call(name, dotted, call, self.model.scalar_values)
            self.model.symbol_kinds[name] = "mesh"
            return

        if dotted == "mcdc.Universe":
            cells = _extract_name_list(_keyword_value(call, "cells"))
            self.model.universes[name] = UniverseDef(name=name, cells=cells)
            self.model.symbol_kinds[name] = "universe"
            return

        if dotted == "mcdc.Lattice":
            lattice = LatticeDef(name=name)
            for axis in ("x", "y", "z"):
                axis_value = _keyword_value(call, axis)
                parsed_axis = _parse_lattice_axis(axis_value, self.model.scalar_values)
                if parsed_axis is not None:
                    lattice.axes[axis] = parsed_axis
            lattice.universes = _extract_name_list(_keyword_value(call, "universes"))
            self.model.lattices[name] = lattice
            self.model.symbol_kinds[name] = "lattice"
            return

        if dotted == "mcdc.Cell":
            cell = _cell_from_call(name, call, self.model.scalar_values)
            self.model.cells[name] = cell
            self.model.symbol_kinds[name] = "cell"
            return

        if dotted == "mcdc.Source":
            self.model.sources.append(_source_from_call(name, call, self.model.scalar_values))
            self.model.symbol_kinds[name] = "source"
            return

        if dotted.startswith("mcdc.Tally"):
            self.model.tallies.append(_tally_from_call(name, dotted, call))
            self.model.symbol_kinds[name] = "tally"
            return

    def _handle_expr_call(self, call: ast.Call) -> None:
        dotted = _call_name(call)

        if dotted == "mcdc.Cell":
            name = f"cell_{self._cell_counter}"
            self._cell_counter += 1
            self.model.cells[name] = _cell_from_call(name, call, self.model.scalar_values)
            return

        if dotted == "mcdc.Source":
            name = f"source_{self._source_counter}"
            self._source_counter += 1
            self.model.sources.append(_source_from_call(name, call, self.model.scalar_values))
            return

        if dotted.startswith("mcdc.Tally"):
            name = f"tally_{self._tally_counter}"
            self._tally_counter += 1
            self.model.tallies.append(_tally_from_call(name, dotted, call))
            return

        if dotted == "mcdc.simulation.set_root_universe":
            self.model.root_universe_cells = _extract_name_list(_keyword_value(call, "cells"))
            return

        if dotted == "mcdc.settings.set_eigenmode":
            self.model.settings.eigenmode = True
            self.model.settings.eigen_args = {
                kw.arg: _literal_value(kw.value, self.model.scalar_values)
                for kw in call.keywords
                if kw.arg
            }
            return

    def _handle_settings_assignment(self, target: ast.Attribute, value: ast.AST) -> None:
        dotted = _attribute_name(target)
        parsed = _literal_value(value, self.model.scalar_values)

        if dotted == "mcdc.settings.N_particle" and isinstance(parsed, (int, float)):
            self.model.settings.n_particle = int(parsed)
        elif dotted == "mcdc.settings.N_batch" and isinstance(parsed, (int, float)):
            self.model.settings.n_batch = int(parsed)
        elif dotted == "mcdc.settings.N_cycle" and isinstance(parsed, (int, float)):
            self.model.settings.n_cycle = int(parsed)


def _check_root_universe(model: ScriptModel) -> list[CheckIssue]:
    issues: list[CheckIssue] = []

    has_hierarchy = bool(model.universes or model.lattices)
    if has_hierarchy and not model.root_universe_cells:
        issues.append(
            CheckIssue(
                "missing_root_universe",
                "error",
                "The script defines universes or lattices but never calls mcdc.simulation.set_root_universe(...).",
            )
        )

    for cell_name in model.root_universe_cells:
        if cell_name not in model.cells:
            issues.append(
                CheckIssue(
                    "unknown_root_cell",
                    "error",
                    f"Root universe references cell '{cell_name}', but that cell is not defined.",
                    evidence=cell_name,
                )
            )
    return issues


def _check_lattice_extents(model: ScriptModel) -> list[CheckIssue]:
    issues: list[CheckIssue] = []

    for cell in model.cells.values():
        if not cell.fill or cell.fill not in model.lattices:
            continue

        lattice = model.lattices[cell.fill]
        bounds = _extract_region_bounds(cell.region_expr, model)
        if not bounds:
            continue

        translation = cell.translation or ()
        for axis, axis_spec in lattice.axes.items():
            if axis not in bounds:
                continue

            start, pitch, count = axis_spec
            shift = translation[_axis_index(axis)] if len(translation) > _axis_index(axis) else 0.0
            expected_low = start + shift
            expected_high = start + pitch * count + shift
            actual_low, actual_high = bounds[axis]
            if actual_low is None or actual_high is None:
                continue

            if abs(actual_low - expected_low) > 1e-6 or abs(actual_high - expected_high) > 1e-6:
                issues.append(
                    CheckIssue(
                        "lattice_extent_mismatch",
                        "error",
                        f"Cell '{cell.name}' bounds for lattice '{lattice.name}' do not match the lattice extent on axis {axis}.",
                        evidence=(
                            f"expected {axis}=[{expected_low}, {expected_high}], "
                            f"actual {axis}=[{actual_low}, {actual_high}]"
                        ),
                    )
                )
    return issues


def _check_source_bounds(model: ScriptModel) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    global_bounds = _global_boundary_bounds(model)

    for source in model.sources:
        if source.energy_group is not None and any(material.is_ce for material in model.materials.values()):
            issues.append(
                CheckIssue(
                    "ce_energy_group_mismatch",
                    "warning",
                    f"Source '{source.name}' uses energy_group even though the script defines at least one CE material.",
                    evidence=f"energy_group={source.energy_group}",
                )
            )

        for axis in ("x", "y", "z"):
            if axis not in global_bounds:
                continue
            low, high = global_bounds[axis]
            source_range = getattr(source, axis)
            if source_range is None:
                if source.position and len(source.position) > _axis_index(axis):
                    value = source.position[_axis_index(axis)]
                    if value <= low or value >= high:
                        issues.append(
                            CheckIssue(
                                "source_outside_bounds",
                                "error",
                                f"Source '{source.name}' position lies on or outside the global {axis}-bounds.",
                                evidence=f"{axis}={value}, bounds=[{low}, {high}]",
                            )
                        )
                continue

            src_low, src_high = source_range
            if src_low <= low or src_high >= high:
                issues.append(
                    CheckIssue(
                        "source_touches_boundary",
                        "warning",
                        f"Source '{source.name}' touches or exceeds the global {axis}-bounds.",
                        evidence=f"{axis}=[{src_low}, {src_high}], bounds=[{low}, {high}]",
                    )
                )
    return issues


def _check_source_coverage(model: ScriptModel) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    fill_boxes: list[tuple[str, list[dict[str, tuple[float | None, float | None]]]]] = []

    for cell in model.cells.values():
        if not cell.fill:
            continue
        boxes = _extract_region_boxes(cell.region_expr, model)
        if not boxes:
            continue
        fill_boxes.append((cell.name, boxes))

    if not fill_boxes:
        return issues

    for source in model.sources:
        if source.position:
            if not any(_point_in_any_box(source.position, boxes) for _, boxes in fill_boxes):
                issues.append(
                    CheckIssue(
                        "source_not_in_filled_region",
                        "warning",
                        f"Point source '{source.name}' is not inside any filled region that could be bounded deterministically.",
                        evidence=f"position={source.position}",
                    )
                )
            continue

        source_box = {}
        for axis in ("x", "y", "z"):
            axis_range = getattr(source, axis)
            if axis_range is not None:
                source_box[axis] = axis_range

        if source_box and not any(_box_overlaps_any_box(source_box, boxes) for _, boxes in fill_boxes):
            issues.append(
                CheckIssue(
                    "source_not_in_filled_region",
                    "warning",
                    f"Source '{source.name}' does not overlap any deterministically bounded filled region.",
                    evidence=str(source_box),
                )
            )

    return issues


def _check_reference_targets(model: ScriptModel) -> list[CheckIssue]:
    issues: list[CheckIssue] = []

    valid_fill_names = set(model.materials) | set(model.universes) | set(model.lattices)
    for cell in model.cells.values():
        if cell.fill and cell.fill not in valid_fill_names:
            issues.append(
                CheckIssue(
                    "unknown_cell_fill",
                    "error",
                    f"Cell '{cell.name}' fills with '{cell.fill}', but that symbol is not a known material, universe, or lattice.",
                    evidence=cell.fill,
                )
            )

    for tally in model.tallies:
        if tally.kind == "TallyMesh" and tally.target_name and tally.target_name not in model.meshes:
            issues.append(
                CheckIssue(
                    "unknown_mesh_tally_target",
                    "error",
                    f"Tally '{tally.name}' references mesh '{tally.target_name}', but that mesh is not defined.",
                    evidence=tally.target_name,
                )
            )
        if tally.kind == "TallyCell" and tally.target_name and tally.target_name not in model.cells:
            issues.append(
                CheckIssue(
                    "unknown_cell_tally_target",
                    "error",
                    f"Tally '{tally.name}' references cell '{tally.target_name}', but that cell is not defined.",
                    evidence=tally.target_name,
                )
            )
        if tally.kind == "TallySurface" and tally.target_name and tally.target_name not in model.surfaces:
            issues.append(
                CheckIssue(
                    "unknown_surface_tally_target",
                    "error",
                    f"Tally '{tally.name}' references surface '{tally.target_name}', but that surface is not defined.",
                    evidence=tally.target_name,
                )
            )
    return issues


def _check_run_mode_consistency(model: ScriptModel, summary: dict[str, Any]) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    if model.settings.eigenmode and not summary.get("has_eigenvalue") and summary.get("output_present", True):
        issues.append(
            CheckIssue(
                "missing_eigenvalue_output",
                "warning",
                "The script calls set_eigenmode(...), but the output summary does not contain eigenvalue datasets.",
            )
        )

    if model.tallies and not summary.get("has_tallies") and summary.get("output_present", True):
        issues.append(
            CheckIssue(
                "missing_tally_output",
                "warning",
                "The script defines tallies, but no tally datasets were found in the output summary.",
            )
        )
    return issues


def _check_plan_alignment(model: ScriptModel, plan: dict[str, Any], geometry_plan: dict[str, Any]) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    if not plan and not geometry_plan:
        return issues

    has_hierarchy = bool(model.universes or model.lattices)
    has_csg_ops = _model_uses_csg_ops(model)

    geometry_mode = str(geometry_plan.get("geometry_mode", "")).lower()
    if geometry_mode == "hierarchy" and not has_hierarchy:
        issues.append(
            CheckIssue(
                "plan_hierarchy_missing",
                "error",
                "The geometry plan expected hierarchy, but the script does not define universes or lattices.",
            )
        )
    if geometry_mode == "hybrid":
        if not has_hierarchy:
            issues.append(
                CheckIssue(
                    "plan_hybrid_missing_hierarchy",
                    "error",
                    "The geometry plan expected a hybrid geometry, but the script has no hierarchy.",
                )
            )
        if not has_csg_ops:
            issues.append(
                CheckIssue(
                    "plan_hybrid_missing_csg",
                    "warning",
                    "The geometry plan expected CSG operations, but the script has no detected union/complement regions.",
                )
            )

    if geometry_plan.get("hierarchy") and not has_hierarchy:
        issues.append(
            CheckIssue(
                "plan_hierarchy_section_unused",
                "error",
                "The geometry plan contains hierarchy instructions, but the script never built hierarchy objects.",
            )
        )

    if geometry_plan.get("csg_operations") and not has_csg_ops:
        issues.append(
            CheckIssue(
                "plan_csg_section_unused",
                "warning",
                "The geometry plan contains CSG operations, but the script has no detected union/complement region logic.",
            )
        )

    plan_settings = str(plan.get("settings", "")).lower()
    if "eigen" in plan_settings and not model.settings.eigenmode:
        issues.append(
            CheckIssue(
                "plan_settings_mismatch",
                "error",
                "The plan requested eigenmode settings, but the script does not call set_eigenmode(...).",
            )
        )
    if "eigen" not in plan_settings and model.settings.eigenmode and plan_settings:
        issues.append(
            CheckIssue(
                "unexpected_eigenmode",
                "warning",
                "The script enables eigenmode even though the saved plan settings do not mention it.",
            )
        )

    plan_tally = str(plan.get("tally", "")).lower()
    tally_kinds = {t.kind for t in model.tallies}
    if "surface" in plan_tally and "TallySurface" not in tally_kinds:
        issues.append(
            CheckIssue(
                "plan_surface_tally_missing",
                "warning",
                "The plan mentions a surface tally, but the script does not define TallySurface.",
            )
        )
    if "mesh" in plan_tally and "TallyMesh" not in tally_kinds:
        issues.append(
            CheckIssue(
                "plan_mesh_tally_missing",
                "warning",
                "The plan mentions a mesh tally, but the script does not define TallyMesh.",
            )
        )
    if "cell tally" in plan_tally and "TallyCell" not in tally_kinds:
        issues.append(
            CheckIssue(
                "plan_cell_tally_missing",
                "warning",
                "The plan mentions a cell tally, but the script does not define TallyCell.",
            )
        )

    plan_source = str(plan.get("source", "")).lower()
    if "point" in plan_source and any(source.position is None for source in model.sources):
        issues.append(
            CheckIssue(
                "plan_source_style_mismatch",
                "warning",
                "The plan describes a point source, but the script appears to use a ranged source definition.",
            )
        )
    if ("volumetric" in plan_source or "x=[" in plan_source) and any(source.position is not None for source in model.sources):
        issues.append(
            CheckIssue(
                "plan_source_style_mismatch",
                "warning",
                "The plan describes a volumetric source, but the script appears to use a point source.",
            )
        )

    return issues


def _check_prompt_alignment(model: ScriptModel, intent: PromptIntent) -> list[CheckIssue]:
    issues: list[CheckIssue] = []

    if intent.expects_ce and not any(material.is_ce for material in model.materials.values()):
        issues.append(
            CheckIssue(
                "prompt_ce_missing",
                "error",
                "The original prompt requests continuous-energy materials, but the script does not define any CE materials.",
            )
        )

    if intent.named_materials:
        declared_names = {material.declared_name.lower() for material in model.materials.values() if material.declared_name}
        for material_name in intent.named_materials:
            if material_name.lower() not in declared_names and not intent.expects_ce:
                continue
            if material_name.lower() not in declared_names:
                issues.append(
                    CheckIssue(
                        "prompt_material_name_missing",
                        "warning",
                        f"The prompt names material '{material_name}', but no script material carries that name.",
                        evidence=material_name,
                    )
                )

    if intent.run_mode == "eigenvalue" and not model.settings.eigenmode:
        issues.append(
            CheckIssue(
                "prompt_eigenmode_missing",
                "error",
                "The original prompt requests an eigenvalue/criticality run, but the script does not call set_eigenmode(...).",
            )
        )
    if intent.run_mode == "fixed" and model.settings.eigenmode:
        issues.append(
            CheckIssue(
                "prompt_unexpected_eigenmode",
                "warning",
                "The script enables eigenmode even though the original prompt does not appear to request it.",
            )
        )

    tally_kinds = {tally.kind for tally in model.tallies}
    for requested_kind in intent.requested_tally_kinds:
        if requested_kind not in tally_kinds:
            issues.append(
                CheckIssue(
                    "prompt_tally_kind_missing",
                    "warning",
                    f"The original prompt requests {requested_kind}, but the script does not define it.",
                    evidence=requested_kind,
                )
            )

    script_scores = {score for tally in model.tallies for score in tally.scores}
    for score in intent.requested_scores:
        if score not in script_scores:
            issues.append(
                CheckIssue(
                    "prompt_tally_score_missing",
                    "warning",
                    f"The original prompt requests tally score '{score}', but the script does not include it.",
                    evidence=score,
                )
            )

    if intent.source_style == "point" and any(source.position is None for source in model.sources):
        issues.append(
            CheckIssue(
                "prompt_source_style_mismatch",
                "warning",
                "The original prompt suggests a point source, but the script uses a ranged source definition.",
            )
        )
    if intent.source_style == "volumetric" and any(source.position is not None for source in model.sources):
        issues.append(
            CheckIssue(
                "prompt_source_style_mismatch",
                "warning",
                "The original prompt suggests a volumetric source, but the script uses a point source.",
            )
        )

    if "hierarchy" in intent.geometry_features or {"assembly", "core", "lattice", "pin", "checkerboard"} & intent.geometry_features:
        if not (model.universes or model.lattices):
            issues.append(
                CheckIssue(
                    "prompt_hierarchy_missing",
                    "error",
                    "The original prompt describes hierarchy/lattices, but the script does not define universes or lattices.",
                )
            )

    if "sphere" in intent.geometry_features and not any(surface.kind == "Sphere" for surface in model.surfaces.values()):
        issues.append(
            CheckIssue(
                "prompt_sphere_missing",
                "warning",
                "The original prompt mentions a sphere, but the script does not define any Sphere surface.",
            )
        )
    if "cylinder" in intent.geometry_features and not any("Cylinder" in surface.kind for surface in model.surfaces.values()):
        issues.append(
            CheckIssue(
                "prompt_cylinder_missing",
                "warning",
                "The original prompt mentions a cylinder, but the script does not define any cylinder surface.",
            )
        )
    if "slab" in intent.geometry_features:
        plane_axes = {surface.axis for surface in model.surfaces.values() if surface.kind.startswith("Plane")}
        if not plane_axes:
            issues.append(
                CheckIssue(
                    "prompt_slab_missing_planes",
                    "warning",
                    "The original prompt mentions a slab, but the script does not define planar slab boundaries.",
                )
            )

    if intent.requested_boundaries:
        declared_bcs = {surface.boundary_condition for surface in model.surfaces.values() if surface.boundary_condition}
        for boundary in intent.requested_boundaries:
            if boundary not in declared_bcs:
                issues.append(
                    CheckIssue(
                        "prompt_boundary_missing",
                        "warning",
                        f"The original prompt requests {boundary} boundaries, but the script does not declare any {boundary} surface.",
                        evidence=boundary,
                    )
                )

    if intent.lattice_dims and model.lattices:
        lattice_dim_pairs = {
            (lattice.axes["x"][2], lattice.axes["y"][2])
            for lattice in model.lattices.values()
            if "x" in lattice.axes and "y" in lattice.axes
        }
        for dims in intent.lattice_dims:
            if dims not in lattice_dim_pairs:
                issues.append(
                    CheckIssue(
                        "prompt_lattice_dims_missing",
                        "warning",
                        f"The original prompt mentions a {dims[0]}x{dims[1]} layout, but no script lattice matches those dimensions.",
                        evidence=f"{dims[0]}x{dims[1]}",
                    )
                )

    for axis, (low, high) in intent.interval_hints.items():
        axis_values = sorted(
            surface.value for surface in model.surfaces.values()
            if surface.axis == axis and surface.value is not None
        )
        if axis_values and not _has_value_pair(axis_values, low, high):
            issues.append(
                CheckIssue(
                    "prompt_interval_mismatch",
                    "warning",
                    f"The original prompt specifies {axis}=[{low}, {high}], but the script surfaces do not expose both values on that axis.",
                    evidence=f"script {axis} values={axis_values}",
                )
            )

    if intent.radius_hints:
        script_radii = [surface.radius for surface in model.surfaces.values() if surface.radius is not None]
        for radius in intent.radius_hints:
            if script_radii and not any(abs(script_radius - radius) <= 1e-6 for script_radius in script_radii):
                issues.append(
                    CheckIssue(
                        "prompt_radius_mismatch",
                        "warning",
                        f"The original prompt mentions radius {radius}, but no script sphere/cylinder uses that radius.",
                        evidence=f"script radii={script_radii}",
                    )
                )

    return issues


def _surface_from_call(name: str, call: ast.Call, scalars: dict[str, float]) -> SurfaceDef:
    kind = _call_name(call).split(".")[-1]
    axis = None
    value = None
    boundary = _literal_value(_keyword_value(call, "boundary_condition"), scalars)
    center = _parse_numeric_sequence(_keyword_value(call, "center"), scalars)
    radius = _eval_numeric(_keyword_value(call, "radius"), scalars)

    if kind == "PlaneX":
        axis = "x"
        value = _eval_numeric(_keyword_value(call, "x"), scalars)
    elif kind == "PlaneY":
        axis = "y"
        value = _eval_numeric(_keyword_value(call, "y"), scalars)
    elif kind == "PlaneZ":
        axis = "z"
        value = _eval_numeric(_keyword_value(call, "z"), scalars)

    return SurfaceDef(
        name=name,
        kind=kind,
        axis=axis,
        value=value,
        boundary_condition=boundary,
        center=center,
        radius=radius,
    )


def _mesh_from_call(name: str, dotted: str, call: ast.Call, scalars: dict[str, float]) -> MeshDef:
    structured = dotted.endswith("MeshStructured")
    mesh = MeshDef(name=name, kind=dotted.split(".")[-1])
    for axis in ("x", "y", "z"):
        axis_value = _keyword_value(call, axis)
        parsed = _parse_mesh_axis(axis_value, scalars, structured=structured)
        if parsed is not None:
            mesh.axes[axis] = parsed
    return mesh


def _cell_from_call(name: str, call: ast.Call, scalars: dict[str, float]) -> CellDef:
    region_expr = _keyword_value(call, "region")
    fill_value = _keyword_value(call, "fill")
    fill_name = fill_value.id if isinstance(fill_value, ast.Name) else None
    translation = _parse_numeric_sequence(_keyword_value(call, "translation"), scalars)
    rotation = _parse_numeric_sequence(_keyword_value(call, "rotation"), scalars)
    return CellDef(
        name=name,
        fill=fill_name,
        region_text=_safe_unparse(region_expr),
        region_expr=region_expr,
        translation=translation,
        rotation=rotation,
    )


def _source_from_call(name: str, call: ast.Call, scalars: dict[str, float]) -> SourceDef:
    return SourceDef(
        name=name,
        x=_parse_range(_keyword_value(call, "x"), scalars),
        y=_parse_range(_keyword_value(call, "y"), scalars),
        z=_parse_range(_keyword_value(call, "z"), scalars),
        position=_parse_numeric_sequence(_keyword_value(call, "position"), scalars),
        energy_group=_literal_value(_keyword_value(call, "energy_group"), scalars),
    )


def _tally_from_call(name: str, dotted: str, call: ast.Call) -> TallyDef:
    kind = dotted.split(".")[-1]
    target_keyword = {
        "TallyMesh": "mesh",
        "TallyCell": "cell",
        "TallySurface": "surface",
    }.get(kind)
    target_expr = _keyword_value(call, target_keyword) if target_keyword else None
    target_name = target_expr.id if isinstance(target_expr, ast.Name) else None
    scores_expr = _keyword_value(call, "scores")
    scores = [item for item in _literal_value(scores_expr, {}) or [] if isinstance(item, str)]
    return TallyDef(name=name, kind=kind, target_name=target_name, scores=scores)


def _keyword_value(call: ast.Call, keyword: str | None) -> ast.AST | None:
    if keyword is None:
        return None
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _call_name(call: ast.Call) -> str:
    return _attribute_name(call.func)


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _literal_value(node: ast.AST | None, scalars: dict[str, float]) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in scalars:
        return scalars[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal_value(item, scalars) for item in node.elts]
    numeric = _eval_numeric(node, scalars)
    return numeric


def _eval_numeric(node: ast.AST | None, scalars: dict[str, float]) -> float | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return scalars.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_numeric(node.operand, scalars)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _eval_numeric(node.left, scalars)
        right = _eval_numeric(node.right, scalars)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right if right != 0 else None
    return None


def _parse_numeric_sequence(node: ast.AST | None, scalars: dict[str, float]) -> tuple[float, ...] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = []
    for item in node.elts:
        value = _eval_numeric(item, scalars)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _parse_range(node: ast.AST | None, scalars: dict[str, float]) -> tuple[float, float] | None:
    sequence = _parse_numeric_sequence(node, scalars)
    if sequence is None or len(sequence) != 2:
        return None
    return float(sequence[0]), float(sequence[1])


def _parse_lattice_axis(node: ast.AST | None, scalars: dict[str, float]) -> tuple[float, float, int] | None:
    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) != 3:
        return None
    start = _eval_numeric(node.elts[0], scalars)
    pitch = _eval_numeric(node.elts[1], scalars)
    count = _literal_value(node.elts[2], scalars)
    if start is None or pitch is None or not isinstance(count, (int, float)):
        return None
    return float(start), float(pitch), int(count)


def _parse_mesh_axis(
    node: ast.AST | None,
    scalars: dict[str, float],
    *,
    structured: bool,
) -> tuple[float | None, float | None, int | None] | None:
    if node is None:
        return None

    if structured:
        if isinstance(node, ast.Call) and _call_name(node).endswith("linspace") and len(node.args) >= 3:
            start = _eval_numeric(node.args[0], scalars)
            stop = _eval_numeric(node.args[1], scalars)
            count = _literal_value(node.args[2], scalars)
            if start is not None and stop is not None and isinstance(count, (int, float)):
                return float(start), float(stop), max(int(count) - 1, 0)
        sequence = _parse_numeric_sequence(node, scalars)
        if sequence and len(sequence) >= 2:
            return float(sequence[0]), float(sequence[-1]), len(sequence) - 1
        return None

    if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) != 3:
        return None
    start = _eval_numeric(node.elts[0], scalars)
    width = _eval_numeric(node.elts[1], scalars)
    count = _literal_value(node.elts[2], scalars)
    if start is None or width is None or not isinstance(count, (int, float)):
        return None
    count_int = int(count)
    return float(start), float(start + width * count_int), count_int


def _extract_name_list(node: ast.AST | None) -> list[str]:
    names: list[str] = []

    def walk(value: ast.AST | None) -> None:
        if value is None:
            return
        if isinstance(value, ast.Name):
            names.append(value.id)
            return
        if isinstance(value, (ast.List, ast.Tuple)):
            for item in value.elts:
                walk(item)

    walk(node)
    return names


def _is_region_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        return True
    if isinstance(node, ast.Name):
        return True
    return False


def _extract_region_bounds(expr: ast.AST | None, model: ScriptModel) -> dict[str, tuple[float | None, float | None]]:
    terms = _collect_conjunctive_terms(expr, model)
    if terms is None:
        return {}

    bounds: dict[str, list[float | None]] = {}
    for surface_name, sign in terms:
        surface = model.surfaces.get(surface_name)
        if not surface or surface.axis is None or surface.value is None:
            continue
        if surface.axis not in bounds:
            bounds[surface.axis] = [None, None]
        if sign == "+":
            current = bounds[surface.axis][0]
            bounds[surface.axis][0] = surface.value if current is None else max(current, surface.value)
        else:
            current = bounds[surface.axis][1]
            bounds[surface.axis][1] = surface.value if current is None else min(current, surface.value)

    return {axis: (low, high) for axis, (low, high) in bounds.items()}


def _extract_region_boxes(
    expr: ast.AST | None,
    model: ScriptModel,
) -> list[dict[str, tuple[float | None, float | None]]] | None:
    if expr is None:
        return None

    if isinstance(expr, ast.Name):
        region = model.regions.get(expr.id)
        if region is None:
            return None
        return _extract_region_boxes(region.expr, model)

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
        if isinstance(expr.operand, ast.Name):
            surface = model.surfaces.get(expr.operand.id)
            if not surface or surface.axis is None or surface.value is None:
                return None
            if isinstance(expr.op, ast.UAdd):
                return [{surface.axis: (surface.value, None)}]
            return [{surface.axis: (None, surface.value)}]
        return None

    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitAnd):
        left_boxes = _extract_region_boxes(expr.left, model)
        right_boxes = _extract_region_boxes(expr.right, model)
        if left_boxes is None or right_boxes is None:
            return None
        merged: list[dict[str, tuple[float | None, float | None]]] = []
        for left in left_boxes:
            for right in right_boxes:
                box = _merge_boxes(left, right)
                if box is not None:
                    merged.append(box)
        return merged or None

    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        left_boxes = _extract_region_boxes(expr.left, model)
        right_boxes = _extract_region_boxes(expr.right, model)
        if left_boxes is None or right_boxes is None:
            return None
        return left_boxes + right_boxes

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Invert):
        return None

    return None


def _collect_conjunctive_terms(expr: ast.AST | None, model: ScriptModel) -> list[tuple[str, str]] | None:
    if expr is None:
        return None

    if isinstance(expr, ast.Name):
        region = model.regions.get(expr.id)
        if region is None:
            return None
        return _collect_conjunctive_terms(region.expr, model)

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
        if isinstance(expr.operand, ast.Name):
            return [(expr.operand.id, "+" if isinstance(expr.op, ast.UAdd) else "-")]
        return None

    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitAnd):
        left = _collect_conjunctive_terms(expr.left, model)
        right = _collect_conjunctive_terms(expr.right, model)
        if left is None or right is None:
            return None
        return left + right

    return None


def _global_boundary_bounds(model: ScriptModel) -> dict[str, tuple[float, float]]:
    by_axis: dict[str, list[float]] = {}
    for surface in model.surfaces.values():
        if surface.axis is None or surface.value is None:
            continue
        if not surface.boundary_condition or surface.boundary_condition == "none":
            continue
        by_axis.setdefault(surface.axis, []).append(surface.value)

    bounds: dict[str, tuple[float, float]] = {}
    for axis, values in by_axis.items():
        if values:
            bounds[axis] = (min(values), max(values))
    return bounds


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _merge_boxes(
    left: dict[str, tuple[float | None, float | None]],
    right: dict[str, tuple[float | None, float | None]],
) -> dict[str, tuple[float | None, float | None]] | None:
    merged: dict[str, tuple[float | None, float | None]] = {}
    for axis in set(left) | set(right):
        left_low, left_high = left.get(axis, (None, None))
        right_low, right_high = right.get(axis, (None, None))
        low = _max_optional(left_low, right_low)
        high = _min_optional(left_high, right_high)
        if low is not None and high is not None and low > high:
            return None
        merged[axis] = (low, high)
    return merged


def _point_in_any_box(point: tuple[float, ...], boxes: list[dict[str, tuple[float | None, float | None]]]) -> bool:
    for box in boxes:
        inside = True
        for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
            if idx >= len(point) or axis not in box:
                continue
            low, high = box[axis]
            value = point[idx]
            if low is not None and value < low:
                inside = False
                break
            if high is not None and value > high:
                inside = False
                break
        if inside:
            return True
    return False


def _box_overlaps_any_box(
    source_box: dict[str, tuple[float, float]],
    boxes: list[dict[str, tuple[float | None, float | None]]],
) -> bool:
    for box in boxes:
        overlap = True
        for axis, (src_low, src_high) in source_box.items():
            if axis not in box:
                continue
            box_low, box_high = box[axis]
            if box_low is not None and src_high < box_low:
                overlap = False
                break
            if box_high is not None and src_low > box_high:
                overlap = False
                break
        if overlap:
            return True
    return False


def _model_uses_csg_ops(model: ScriptModel) -> bool:
    for region in model.regions.values():
        if _expr_uses_csg_ops(region.expr):
            return True
    for cell in model.cells.values():
        if cell.region_expr is not None and _expr_uses_csg_ops(cell.region_expr):
            return True
    return False


def _expr_uses_csg_ops(expr: ast.AST | None) -> bool:
    if expr is None:
        return False
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Invert):
        return True
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        return True
    for child in ast.iter_child_nodes(expr):
        if _expr_uses_csg_ops(child):
            return True
    return False


def _max_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _min_optional(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _has_value_pair(values: list[float], low: float, high: float, tol: float = 1e-6) -> bool:
    has_low = any(abs(value - low) <= tol for value in values)
    has_high = any(abs(value - high) <= tol for value in values)
    return has_low and has_high
