"""Prompt templates for the Small Model Generator.

Optimized for 8-14B+ parameter models with:
- JSON-format planning prompts (uses response_format=json_object)
- Separate general + geometry plans injected into codegen
- Phase-scoped code generation (setup → geometry → finalize)
- Model-based plan and code review passes
"""

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

SHARED_ANTI_COPYING_WARNING = '''## Anti-Copying Warning
Use any examples above ONLY to learn the API structure and logic. 
DO NOT copy specific dimensions, material values, or surface names unless they exactly match the task. 
If a JSON plan is provided, use its values exactly. Otherwise use the values from the problem description.'''

SHARED_SURFACE_RULES = '''## Valid Surface Types (ONLY these exist in mcdc)
- **Planes**: `PlaneX(x=...)`, `PlaneY(y=...)`, `PlaneZ(z=...)`
- **Sphere**: `Sphere(center=[x,y,z], radius=r)`
- **Cylinders**: `CylinderX(center=[y,z], radius=r)`, `CylinderY(center=[x,z], radius=r)`, `CylinderZ(center=[x,y], radius=r)`

## INVALID — These do NOT exist in mcdc (NEVER use them in the plan)
- `Box(...)` / `mcdc.Surface.Box(...)`
- `Cone(...)` / `mcdc.Surface.Cone(...)`
- `Torus(...)` / `mcdc.Surface.Torus(...)`
- To represent a box/rectangular region: list its 6 bounding planes in `surfaces.boundary` or `surfaces.internal`.
  In the geometry phase the box region is built with CSG: `+x_lo & -x_hi & +y_lo & -y_hi & +z_lo & -z_hi`'''

SHARED_MATERIAL_MODE_RULES = '''## Material Mode Rules
- Default to single-group multigroup materials (`mcdc.MaterialMG`) unless the prompt explicitly asks for continuous-energy materials, named real materials, isotopic compositions, or CE nuclear-data usage.
- If the prompt explicitly asks for continuous-energy materials, use `mcdc.Material(...)` with `nuclide_composition={...}`.
- When CE materials are requested and only a material name or formula is given, call the material lookup tool/service to obtain the `nuclide_composition` values instead of inventing them.
- For CE materials, reuse the returned number densities exactly.
- For CE problems, usually omit `energy_group` from `mcdc.Source(...)`.
- For MG problems, keep using `energy_group` and prefer single-group MaterialMG unless the prompt explicitly asks for multiple energy groups.
'''

# =============================================================================
# COMPLEXITY CLASSIFICATION
# =============================================================================

CLASSIFY_PROMPT = '''Classify this MCDC simulation request into ONE category.

## Prompt
{prompt}

## Categories

**simple**: Single-level flat geometry. All cells defined directly with planes, spheres, or cylinders.
Includes:
- Layered slabs (any number of regions along one axis)
- Sphere or cylinder centered in a box/moderator
- Concentric shells or rings (regions with +inner & -outer)
Key rule: Using `~complement` to say "outside a cylinder inside a box" is STILL SIMPLE.

**complex**: Geometry that requires many cells/surfaces OR repeated units via universes/lattices.
Includes:
- Void channels (multiple connected void segments joined with | union)
- Any geometry with more than 10 cells
- Assemblies
- Multi-level lattice hierarchies (pins → assembly → core)
Key rule: Anything using a universes, lattices, or a large number of cells/surfaces is complex.

## Concrete Examples
- "sphere inside a scattering cube" → **simple**
- "two rooms separated by a barrier wall" → **simple**
- "eigenvalue with two fissile regions" → **simple**
- "concentric spherical shells" → **simple**
- "T-shaped void channel in a shield block" → **complex**
- "fuel pin assembly with 5x5 lattice" → **complex**

Respond with ONLY one word: simple or complex
'''

# =============================================================================
# PLANNING PROMPTS (JSON output via response_format)
# =============================================================================

SIMPLE_PLANNING_PROMPT = '''You are planning an MCDC Monte Carlo simulation. Analyze the request and create a structured plan.

## Input Prompt
{prompt}

## Example Plan 1 (1D Slab with Reflector)
**Prompt**: "Create a 1D slab with 2cm source region, 5cm shield, 1cm detector. Reflective left, vacuum right."

**Plan**:
```json
{{
  "complexity": "simple",
  "materials": [
    "source_mat: low scatter (0.1), low capture (0.05)",
    "shield_mat: high scatter (0.8), moderate capture (0.1)"
  ],
  "surfaces": {{
    "boundary": ["x=0 (reflective)", "x=8 (vacuum)"],
    "internal": ["x=2", "x=7"]
  }},
  "cells": [
    {{"cell": "source cell (x=[0,2])", "material": "source_mat"}},
    {{"cell": "shield cell (x=[2,7])", "material": "shield_mat"}}
  ],
  "source": "volumetric in x=[0,2], isotropic, group 0",
  "tally": "cell tally on detector cell (x=[7,8]), score: flux",
  "settings": "N_particle=1000, N_batch=2, fixed-source mode"
}}
```

## Example Plan 2 (Cylinder in Box)
**Prompt**: "Fuel cylinder (radius 2cm, height 10cm) inside a 12cm moderator box. Reflective bottom, vacuum elsewhere."

**Plan**:
```json
{{
  "complexity": "simple",
  "materials": [
    "fuel_mat: fission (0.05), nu_p (2.4), scatter (0.2), capture (0.02)",
    "moderator_mat: scatter (0.95), capture (0.01)"
  ],
  "surfaces": {{
    "boundary": ["z=0 (reflective)", "z=10 (vacuum)", "x=-6 (vacuum)", "x=6 (vacuum)", "y=-6 (vacuum)", "y=6 (vacuum)"],
    "internal": ["CylinderZ(center=[0,0], radius=2)"]
  }},
  "cells": [
    {{"cell": "fuel cylinder", "material": "fuel_mat", "region": "-cylinder & +z0 & -z10 (inside cylinder, bounded in z)"}},
    {{"cell": "moderator box", "material": "moderator_mat", "region": "inside box & +cylinder (outside cylinder)"}}
  ],
  "source": "volumetric in fuel cylinder, isotropic, group 0",
  "tally": "mesh tally covering geometry, score: flux",
  "settings": "N_particle=1000, N_batch=2, fixed-source mode"
}}
```

## Example Plan 3 (Eigenvalue Problem)
**Prompt**: "Two adjacent fissile regions (4cm HEU, 6cm LEU) with vacuum boundaries. Calculate k-eigenvalue."

**Plan**:
```json
{{
  "complexity": "simple",
  "materials": [
    "heu: fission (0.12), nu_p (2.8), scatter (0.3), capture (0.05)",
    "leu: fission (0.06), nu_p (2.2), scatter (0.25), capture (0.15)"
  ],
  "surfaces": {{
    "boundary": ["x=0 (vacuum)", "x=10 (vacuum)"],
    "internal": ["x=4"]
  }},
  "cells": [
    {{"cell": "HEU region (x=[0,4])", "material": "heu"}},
    {{"cell": "LEU region (x=[4,10])", "material": "leu"}}
  ],
  "source": "volumetric in x=[0,10], isotropic, group 0",
  "tally": "MeshStructured, scores: flux, fission",
  "settings": "N_particle=1000, eigenmode: N_inactive=2, N_active=4"
}}
```

## Example Plan 4 (Sphere with Surface Tally)
**Prompt**: "Fissile sphere (r=3cm) surrounded by moderator in a 12cm cube. Track leakage through sphere surface."

**Plan**:
```json
{{
  "complexity": "simple",
  "materials": [
    "fuel: fission (0.15), nu_p (2.5), scatter (0.2), capture (0.1)",
    "moderator: scatter (0.85), capture (0.02)"
  ],
  "surfaces": {{
    "boundary": ["x=-6 (vacuum)", "x=6 (vacuum)", "y=-6 (vacuum)", "y=6 (vacuum)", "z=-6 (vacuum)", "z=6 (vacuum)"],
    "internal": ["Sphere(center=[0,0,0], radius=3)"]
  }},
  "cells": [
    {{"cell": "fuel sphere", "material": "fuel", "region": "-sphere (inside sphere)"}},
    {{"cell": "moderator", "material": "moderator", "region": "+sphere & inside bounding box"}}
  ],
  "source": "volumetric in x=[-3,3], y=[-3,3], z=[-3,3], isotropic, group 0",
  "tally": "MeshStructured 3D + TallySurface on sphere, scores: flux, net-current",
  "settings": "N_particle=1000, N_batch=2, fixed-source mode"
}}
```

''' + SHARED_SURFACE_RULES + '''

## DO NOT ADD (unless explicitly in the prompt):
- Lattices or universes
- Hierarchy levels
- Time-dependent features
- Additional tally types not mentioned
- Extra dimensions not specified in the prompt

## Output Format
Return ONLY valid JSON matching this structure:
{{
  "complexity": "simple",
  "materials": [...],
  "surfaces": {{"boundary": [...], "internal": [...]}},
  "cells": [...],
  "source": "...",
  "tally": "...",
  "settings": "..."
}}

## Rules
1. Copy EXACT numbers from the prompt - do not generalize
2. Keep the plan minimal - only what the prompt asks for
3. For spheres/cylinders: describe the surface type and parameters
4. Use appropriate boundary conditions (vacuum vs reflective)
5. Must explicitly specify EVERY constraint from the prompt
6. Do NOT invent extra y or z boundaries just to make a box if the prompt does not explicitly ask for it. 1D and 2D simulations work well
7. Use the FEWEST surfaces needed to satisfy the prompt. Do not overbuild the geometry.
8. If the prompt is eigenvalue or criticality, include eigenmode settings explicitly.
9. Only list materials that actually fill a cell inside the modeled geometry.
10. Tallies do NOT create geometry. Do NOT add surfaces, cells, or materials just for a tally. If possible, use existing geometry.
11. Every source bullet must state its spatial location or host cell/region.
12. Cell regions must describe real volumes only. Never use the same surface on both sides in one region expression.
13. Use single-group multigroup materials unless the prompt explicitly asks for multiple energy groups.
14. Use continuous-energy materials only when the prompt explicitly asks for CE, named real materials, isotopic composition, or use of the material lookup tool/service.
15. If CE materials are requested and only material names/formulas are given, call the material lookup tool/service for `nuclide_composition` values instead of inventing them.
16. If you use multigroup materials and the prompt does NOT explicitly ask for eigenvalue, critical, or supercritical behavior, keep the system subcritical.
17. For single-group multigroup material values, prefer `fission * nu_p < capture + scatter`.
18. If the prompt DOES ask for eigenvalue or multiplying behavior, keep it only mildly multiplying unless stronger criticality is explicitly requested.
19. Use `N_particle` around 1000 unless the prompt explicitly says otherwise.
20. For eigenvalue runs, use `N_inactive=2` and `N_active=4` unless the prompt explicitly says otherwise.
'''

COMPLEX_PLANNING_PROMPT = '''You are planning an MCDC Monte Carlo simulation with complex geometry.

Complex geometry includes CSG operations (union/complement), universe/lattice hierarchy, or both.

## Input Prompt
{prompt}

## CONDITIONAL FIELDS — READ CAREFULLY
The JSON template has optional sections. Include ONLY what the problem actually requires:
- **"csg_operations"**: ONLY if geometry needs region union (|) or complement (~) to carve shapes
- **"hierarchy"**: ONLY if geometry uses mcdc.Universe() and/or mcdc.Lattice()
- **"dimensions"**: ONLY alongside "hierarchy" (shows lattice width/half-width calculations)

**Omitting a field is CORRECT when the problem doesn't need it. DO NOT invent structure.**

---

## Example 1: Pure CSG — Concentric Shells (no hierarchy)
```json
{{
  "complexity": "complex",
  "materials": [
    "core_mat: fission=0.10, nu_p=2.5, scatter=0.30, capture=0.05",
    "shell_mat: capture=0.40, scatter=0.50",
    "moderator: scatter=0.08, capture=0.02"
  ],
  "surfaces": {{
    "boundary": ["x=-6 (vacuum)", "x=6 (vacuum)", "y=-6 (vacuum)", "y=6 (vacuum)", "z=-6 (vacuum)", "z=6 (vacuum)"],
    "internal": ["Sphere(center=[0,0,0], radius=2)", "Sphere(center=[0,0,0], radius=4)"]
  }},
  "cells": [
    {{"cell": "core", "region": "-inner_sphere", "material": "core_mat"}},
    {{"cell": "shell", "region": "+inner_sphere & -outer_sphere", "material": "shell_mat"}},
    {{"cell": "moderator", "region": "+outer_sphere & inside bounding box", "material": "moderator"}}
  ],
  "csg_operations": {{
    "core_region": "-inner_sphere (inside r=2 sphere)",
    "shell_region": "+inner_sphere & -outer_sphere (annular shell between r=2 and r=4)",
    "outer_region": "bounding_box & +outer_sphere (outside r=4 sphere)"
  }},
  "source": "point source at (0,0,0), isotropic, group 0",
  "tally": "TallyCells on core and shell + MeshStructured 3D, scores: flux, fission",
  "settings": "N_particle=500, N_batch=2, fixed-source mode"
}}
```

---

## Example 2: Pure Hierarchy (Lattice) — 5×5 Pin Assembly (no CSG)
```json
{{
  "complexity": "complex",
  "materials": [
    "fuel: fission=0.12, nu_p=2.43, scatter=0.20, capture=0.08",
    "moderator: scatter=0.88, capture=0.02"
  ],
  "surfaces": {{
    "boundary": ["x=-3.15 (reflective)", "x=+3.15 (reflective)", "y=-3.15 (reflective)", "y=+3.15 (reflective)"],
    "internal": ["CylinderZ(center=[0,0], radius=0.4) — local to pin universe"]
  }},
  "cells": [],
  "hierarchy": {{
    "level_1_pin": {{
      "cells": ["fuel_cell: -pin_cyl, fill=fuel", "mod_cell: +pin_cyl, fill=moderator"],
      "universe": "pin = Universe([fuel_cell, mod_cell])"
    }},
    "level_2_assembly": {{
      "lattice": "assembly_lattice: 5x5 pins, x=(-3.15, 1.26, 5), y=(-3.15, 1.26, 5)",
      "pattern": "all pins use the pin universe (homogeneous)",
      "cell": "assembly_cell: bounded by reflective surfaces, fill=assembly_lattice",
      "root": "set_root_universe([assembly_cell])"
    }}
  }},
  "dimensions": {{
    "pin_pitch": 1.26,
    "assembly_half": "1.26 * 5 / 2 = 3.15",
    "assembly_width": "1.26 * 5 = 6.30"
  }},
  "source": "volumetric x=[-3.15,3.15], y=[-3.15,3.15], isotropic, group 0",
  "tally": "MeshStructured aligned with pins (5x5 bins), scores: flux, fission",
  "settings": "N_particle=1000, eigenmode: N_inactive=2, N_active=4"
}}
```

---

## Example 3: Pure Hierarchy (Translation-Only) — No Lattice, No CSG
```json
{{
  "complexity": "complex",
  "materials": [
    "fuel: fission=0.10, nu_p=2.4, scatter=0.25, capture=0.05",
    "moderator: scatter=0.85, capture=0.02"
  ],
  "surfaces": {{
    "boundary": ["x=-12 (vacuum)", "x=12 (vacuum)", "y=-5 (vacuum)", "y=5 (vacuum)", "z=-5 (vacuum)", "z=5 (vacuum)"],
    "internal": ["x=-2 (fuel left edge, local)", "x=2 (fuel right edge, local)", "x=-4 (divider)", "x=4 (divider)"]
  }},
  "cells": [],
  "hierarchy": {{
    "level_1_assembly": {{
      "cells": [
        "fuel_cell: +x_fuel_l & -x_fuel_r & inside_box, fill=fuel",
        "mod_cell: remaining region in 8cm box, fill=moderator"
      ],
      "universe": "assembly = Universe([fuel_cell, mod_cell])"
    }},
    "placement": {{
      "left":   "Cell(region=x=[-12,-4] box, fill=assembly, translation=[-8, 0, 0])",
      "center": "Cell(region=x=[-4,4] box,  fill=assembly, translation=[ 0, 0, 0])",
      "right":  "Cell(region=x=[4,12] box,  fill=assembly, translation=[+8, 0, 0])"
    }},
    "root": "set_root_universe([left_cell, center_cell, right_cell])"
  }},
  "source": "point source at (0,0,0), isotropic, group 0",
  "tally": "MeshStructured x=[-12,12], y=[-5,5], score: flux",
  "settings": "N_particle=1000, N_batch=2, fixed-source"
}}
```

---

## Example 4: Hybrid — CSG Bore in Shield Cylinder, Repeated in Lattice
```json
{{
  "complexity": "complex",
  "materials": [
    "shield: scatter=0.70, capture=0.20",
    "void_mat: scatter=0.001, capture=0.0001",
    "water: scatter=0.92, capture=0.008"
  ],
  "surfaces": {{
    "boundary": ["x=-7.5 (vacuum)", "x=7.5 (vacuum)", "y=-2.5 (vacuum)", "y=2.5 (vacuum)"],
    "internal": [
      "CylinderZ(center=[0,0], radius=2.0) — outer shield (local)",
      "CylinderZ(center=[0,0], radius=0.5) — inner bore (local)"
    ]
  }},
  "cells": [],
  "csg_operations": {{
    "bore_channel": "-inner_cyl (inside bore, void)",
    "shield_annulus": "+inner_cyl & -outer_cyl (solid shield with bore removed)",
    "water_region": "+outer_cyl & inside 5x5 cell (water around cylinder)"
  }},
  "hierarchy": {{
    "level_1_unit": {{
      "cells": [
        "bore_cell: -inner_cyl, fill=void_mat",
        "shield_cell: +inner_cyl & -outer_cyl, fill=shield",
        "water_cell: +outer_cyl & inside_box, fill=water"
      ],
      "universe": "unit = Universe([bore_cell, shield_cell, water_cell])"
    }},
    "level_2_row": {{
      "lattice": "row_lattice: 3x1, x=(-7.5, 5.0, 3), y=(-2.5, 5.0, 1)",
      "pattern": "a single 1D row of 3 identical unit universes",
      "cell": "row_cell: bounded by boundaries, fill=row_lattice",
      "root": "set_root_universe([row_cell])"
    }}
  }},
  "dimensions": {{
    "cell_pitch": 5.0,
    "row_half_x": "5.0 * 3 / 2 = 7.5",
    "row_half_y": "5.0 * 1 / 2 = 2.5"
  }},
  "source": "volumetric x=[-7.5,7.5], y=[-2.5,2.5], isotropic, group 0",
  "tally": "MeshStructured x=[-7.5,7.5], y=[-2.5,2.5], score: flux",
  "settings": "N_particle=1000, N_batch=2, fixed-source"
}}
```

---

## DO NOT ADD (unless explicitly in the prompt):
- "csg_operations" if no region union/complement is needed
- "hierarchy" / "dimensions" if no universes or lattices are used
- More hierarchy levels than described
- Additional tally types not mentioned

''' + SHARED_SURFACE_RULES + '''

## Output Format
Return ONLY valid JSON:
{{
  "complexity": "complex",
  "materials": [...],
  "surfaces": {{"boundary": [...], "internal": [...]}},
  "cells": [...],
  "csg_operations": {{...}},
  "hierarchy": {{...}},
  "dimensions": {{...}},
  "source": "...",
  "tally": "...",
  "settings": "..."
}}
IMPORTANT: Only return valid JSON. Do not take any shortcuts that may result in invalid JSON.

## Plan Completeness Rules
1. Copy EXACT dimensions from the prompt — do not generalize
2. ALL materials, surfaces, and cells from the prompt must appear in the plan
3. If eigenvalue/criticality is mentioned, settings MUST include eigenmode
4. Lattice size must match prompt exactly (e.g. 14×14 if specified)
5. Connected void regions must OVERLAP spatially (not just touch edges)
6. Must explicitly specify EVERY constraint from the prompt
7. The plan MUST be valid, static JSON. NEVER use Python list comprehensions or formulas.
8. For non-homogeneous lattices, describe the layout with a text string
9. For pure CSG geometry (no universe/lattice): OMIT the `hierarchy` field entirely.
10. Use single-group multigroup materials unless the prompt explicitly asks for multiple energy groups.
11. If you use multigroup materials and the prompt does NOT explicitly ask for critical or supercritical behavior, keep the system subcritical.
12. For single-group multigroup material values, prefer `fission * nu_p < capture + scatter` (subcritical).
13. If eigenvalue or multiplying behavior is requested, keep it only mildly multiplying unless stronger criticality is explicitly requested.
14. Use `N_particle` around 1000 unless the prompt explicitly says otherwise.
15. For eigenvalue runs, use `N_particle` around 1000 with `N_inactive=2` and `N_active=4` unless the prompt explicitly says otherwise.
'''

# =============================================================================
# GEOMETRY PLANNING PROMPT (JSON output)
# =============================================================================

COMPLEX_GEOMETRY_PLANNING_PROMPT = '''You are creating the DETAILED GEOMETRY PLAN for an MCDC simulation.

This step is ONLY for complex problems. Convert the general plan into explicit geometry actions that the code generator can follow.

## Original Prompt
{prompt}

## General Plan
```json
{general_plan_json}
```

''' + SHARED_SURFACE_RULES + '''

## Example 1 - CSG Streaming Channel
```json
{{
  "geometry_mode": "csg",
  "boundary_surfaces": [
    "x_min | PlaneX | value=0 | bc=reflective | role=box min x",
    "x_max | PlaneX | value=60 | bc=vacuum | role=box max x",
    "y_min | PlaneY | value=0 | bc=reflective | role=box min y",
    "y_max | PlaneY | value=100 | bc=vacuum | role=box max y",
    "z_min | PlaneZ | value=0 | bc=reflective | role=box min z",
    "z_max | PlaneZ | value=60 | bc=vacuum | role=box max z"
  ],
  "internal_surfaces": [
    "x10 | PlaneX | value=10 | role=shared source max x",
    "y10 | PlaneY | value=10 | role=shared source and channel face",
    "y20 | PlaneY | value=20 | role=channel bend face",
    "z10 | PlaneZ | value=10 | role=shared source and channel face",
    "z20 | PlaneZ | value=20 | role=channel bend face"
  ],
  "regions": [
    "bounding_box = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max | role=full domain",
    "source_region = +x_min & -x10 & +y_min & -y10 & +z_min & -z10 | role=source cell",
    "seg_1 = +x_min & -x_max & +y_min & -y10 & +z_min & -z10 | role=void segment",
    "seg_2 = +x_min & -x_max & +y10 & -y20 & +z_min & -z10 | role=void segment",
    "seg_3 = +x_min & -x_max & +y10 & -y20 & +z10 & -z20 | role=void segment",
    "seg_4 = +x_min & -x_max & +y_min & -y10 & +z10 & -z20 | role=void segment"
  ],
  "cells": [
    {{"cell": "source_cell", "fill": "void_mat", "region": "source_region", "role": "source region cell"}},
    {{"cell": "void_cell", "fill": "void_mat", "region": "channel_tail & ~source_region", "role": "remaining channel"}},
    {{"cell": "shield_cell", "fill": "shield", "region": "shield_region", "role": "solid shield complement"}}
  ],
  "csg_operations": {{
    "channel_tail": "seg_1 | seg_2 | seg_3 | seg_4",
    "shield_region": "bounding_box & ~channel_tail"
  }},
  "hierarchy": {{}},
  "dimension_checks": [
    "every segment must be bounded in all active axes",
    "consecutive segments must overlap by a 2D face",
    "if source_region sits inside seg_1 then void_cell must subtract source_region"
  ],
  "root": "none"
}}
```

## Streaming Duct Anti-Pattern - NEVER DO THIS
WRONG:
- `src_y | PlaneY | value=10` and `ch1_y | PlaneY | value=10` in the same plan
- `seg4 = ... +ch3_z & -ch4_z` when both surfaces are `PlaneZ(value=20)`

Why: duplicate same-coordinate surfaces create zero-thickness regions.
CORRECT: define one shared `y10`, one shared `z10`, reuse everywhere.

## Example 2 - Hierarchy
```json
{{
  "geometry_mode": "hierarchy",
  "boundary_surfaces": [
    "x0 | PlaneX | value=-3.15 | bc=reflective | role=assembly min x",
    "x1 | PlaneX | value=3.15 | bc=reflective | role=assembly max x",
    "y0 | PlaneY | value=-3.15 | bc=reflective | role=assembly min y",
    "y1 | PlaneY | value=3.15 | bc=reflective | role=assembly max y"
  ],
  "internal_surfaces": [
    "pin_cyl | CylinderZ | center=[0,0] | radius=0.4 | role=pin fuel boundary"
  ],
  "regions": [
    "fuel_region = -pin_cyl | role=inside pin cylinder",
    "moderator_region = +pin_cyl | role=outside pin cylinder",
    "assembly_region = +x0 & -x1 & +y0 & -y1 | role=assembly bounds"
  ],
  "cells": [
    {{"cell": "fuel_cell", "fill": "fuel", "region": "fuel_region", "role": "pin fuel cell"}},
    {{"cell": "mod_cell", "fill": "moderator", "region": "moderator_region", "role": "pin moderator cell"}},
    {{"cell": "assembly_cell", "fill": "assembly_lattice", "region": "assembly_region", "role": "outer lattice cell"}}
  ],
  "csg_operations": {{}},
  "hierarchy": {{
    "pin_universe": "Universe([fuel_cell, mod_cell])",
    "assembly_lattice": "Lattice with pin_universe entries and exact pitch/extent",
    "assembly_cell": "filled with assembly_lattice"
  }},
  "dimension_checks": [
    "lattice extent equals [start, start + pitch*N]",
    "boundary surfaces match the lattice extent exactly"
  ],
  "root": "set_root_universe(cells=[assembly_cell])"
}}
```

## Example 3 - Hybrid
```json
{{
  "geometry_mode": "hybrid",
  "boundary_surfaces": [
    "global_x_min | PlaneX | value=-x_bound_test | bc=vacuum",
    "global_x_max | PlaneX | value=+x_bound_test | bc=vacuum",
    "global_y_min | PlaneY | value=-y_bound_test | bc=vacuum",
    "global_y_max | PlaneY | value=+y_bound_test | bc=vacuum",
    "global_z_min | PlaneZ | value=-z_bound_test | bc=vacuum",
    "global_z_max | PlaneZ | value=+z_bound_test | bc=vacuum"
  ],
  "internal_surfaces": [
    "center_plane | PlaneX | value=0 | role=split between placements",
    "sphere_outer | Sphere | center=[0,0,0] | radius=R_test | role=assembly outer",
    "cyl_z | CylinderZ | center=[0,0] | radius=r_z_test",
    "cyl_x | CylinderX | center=[0,0] | radius=r_x_test"
  ],
  "regions": [
    "global_box = +global_x_min & -global_x_max & +global_y_min & -global_y_max & +global_z_min & -global_z_max",
    "left_region = +global_x_min & -center_plane & +global_y_min & -global_y_max & +global_z_min & -global_z_max",
    "right_region = +center_plane & -global_x_max & +global_y_min & -global_y_max & +global_z_min & -global_z_max",
    "star_region = (-cyl_z | -cyl_x) & -sphere_outer",
    "cover_region = -sphere_outer & ~star_region",
    "water_region = +sphere_outer"
  ],
  "cells": [
    {{"cell": "fuel_cell", "fill": "fuel", "region": "star_region"}},
    {{"cell": "cover_cell", "fill": "cover", "region": "cover_region"}},
    {{"cell": "water_cell", "fill": "water", "region": "water_region"}},
    {{"cell": "left_cell", "fill": "assembly_universe", "region": "left_region", "translation": "[-dx_test,0,0]"}},
    {{"cell": "right_cell", "fill": "assembly_universe", "region": "right_region", "translation": "[+dx_test,0,0]", "rotation": "[0,theta_small,0]"}}
  ],
  "csg_operations": {{
    "star_region": "(-cyl_z | -cyl_x) & -sphere_outer",
    "cover_region": "-sphere_outer & ~star_region"
  }},
  "hierarchy": {{
    "assembly_universe": "Universe([fuel_cell, cover_cell, water_cell])",
    "placement": "left and right cells fill the same universe with different transforms"
  }},
  "dimension_checks": [
    "both placed copies fit inside the outer boundaries",
    "translation and rotation are explicit on the placement cells"
  ],
  "root": "set_root_universe(cells=[left_cell, right_cell])"
}}
```

## Rules
1. Use exact dimensions from the prompt whenever given.
2. Name every boundary and internal surface needed to build the geometry.
3. Every region must be bounded in the axes that matter.
4. Every cell must clearly state its fill and region.
5. For CSG, explicitly show unions and complements.
6. For hierarchy, explicitly show the wrapping order: cells -> universe -> lattice -> outer cell -> root.
7. For hybrid, include BOTH the CSG steps and the hierarchy/placement steps.
8. Reuse a named surface when the coordinate and surface type are the same. Do NOT create duplicates.
9. In `root`, output only `set_root_universe(cells=[...])` or `none`.
10. `csg_operations` is only for named region algebra. Never put Universe/Lattice there.
11. `hierarchy` is only for universe, lattice, and placement steps.
12. For 2D problems, do NOT add z boundary planes unless the prompt explicitly asks for them.
13. If one cell is a strict subregion of a larger region with the same fill, subtract the smaller from the larger.
14. If dimensions are not specified, keep them symbolic with clear placeholder names.
15. Never define a zero-thickness region with `+surface_a & -surface_b` when both have the same coordinate.
16. Default to single-group multigroup materials unless the prompt explicitly asks for continuous-energy materials, named real materials, isotopic compositions, or CE nuclear-data usage.
17. If CE materials are requested and only material names/formulas are given, call the material lookup tool/service for `nuclide_composition` values instead of inventing them.
18. For MG problems, keep using `energy_group` in sources. For CE problems, usually omit `energy_group`.

Return ONLY valid JSON matching this structure:
{{
  "geometry_mode": "csg|hierarchy|hybrid",
  "boundary_surfaces": [...],
  "internal_surfaces": [...],
  "regions": [...],
  "cells": [...],
  "csg_operations": {{}},
  "hierarchy": {{}},
  "dimension_checks": [...],
  "root": "..."
}}
'''

# =============================================================================
# PHASE GENERATION PROMPTS
# =============================================================================

SETUP_PROMPT = '''# MCDC Setup Phase - Materials and Surfaces

## Problem
{prompt}

{general_plan_section}
{geometry_plan_section}

## Instructions
Generate ONLY: imports (numpy, mcdc), materials, and surfaces.
Do NOT include: cells, universes, lattices, source, tallies, settings, or mcdc.run().
Use only short factual comments in the code.

## === PHASE BOUNDARY ===
STOP HERE. Do NOT generate anything beyond surfaces.
The following will be added in later phases:
- Cells, Universes, Lattices → geometry phase
- Source, Tallies, Settings, mcdc.run() → finalize phase

## Common Mistakes to Avoid
- **`mcdc.Surface.Box(...)` does NOT exist** — boxes are built from 6 planes + CSG
- **`mcdc.Surface.Cone(...)` does NOT exist**
- Forgetting nu_p when using fission cross-sections (required for fissile materials)
- Use single-group multigroup materials unless the prompt explicitly asks for multiple energy groups
- For multigroup fissile materials, default to subcritical values unless the prompt explicitly requests critical or supercritical behavior
- For single-group multigroup material values, prefer `fission * nu_p < capture + scatter` (subcritical)
- Even for eigenmode prompts, avoid strongly supercritical materials unless the prompt explicitly asks for that (barely critical)
- Using wrong boundary_condition strings (must be lowercase: 'vacuum', 'reflective')
- CylinderZ center is 2D [x, y], not 3D
- Cylinders extend infinitely along axis - bound with planes if needed
- Scatter cross-section must be 2D array: np.array([[value]]) for 1-group
- Each surface on the same axis must have a DISTINCT coordinate value

''' + SHARED_MATERIAL_MODE_RULES + '''

{api_reference_section}
{examples_section}

''' + SHARED_ANTI_COPYING_WARNING + '''

Output ONLY the setup code in ```python``` blocks.
'''

GEOMETRY_PROMPT = '''# MCDC Geometry Phase - Cells, Universes, Lattices

## Problem
{prompt}

{general_plan_section}
{geometry_plan_section}

## Current Script
```python
{current_script}
```

## Instructions
Add cells, universes, and lattices to the existing script.
Do NOT modify materials or surfaces.
Do NOT include source, tallies, settings, or mcdc.run() yet.
If plan context is provided, treat it as authoritative implementation guidance.
Use only short factual comments in the code.

## === PHASE BOUNDARY ===
STOP after setting the root universe.
Do NOT generate:
- mcdc.Source(...)
- mcdc.TallyMesh(...) or mcdc.TallyCell(...)
- mcdc.MeshStructured(...)
- mcdc.settings.*
- mcdc.run()

{mode_instructions}

{api_reference_section}
{examples_section}

''' + SHARED_ANTI_COPYING_WARNING + '''

## Critical Rules
- Use +surface/-surface for regions: +surf means positive side
- For CSG: use | for union, & for intersection, ~ for complement
- Every surface reference inside a region must be explicitly signed with `+` or `-`
- Every region operand must use TWO DIFFERENT surface variables
- set_root_universe: ONLY call this if the plan contains a `hierarchy` field.
  For pure CSG (no hierarchy in plan): do NOT call set_root_universe
- Plan metadata keys such as `cell`, `role`, and `notes` are NOT API arguments.
  In `mcdc.Cell(...)`, use only valid kwargs like `region`, `fill`, and optional `translation` / `rotation`.
- Do NOT redefine a geometry variable that already exists in `Current Script`.

Output the COMPLETE script (including previous code) in ```python``` blocks.
'''

COMPLEX_GEOMETRY_INSTRUCTIONS = '''## Complex Geometry Construction

Apply ONLY the section(s) that match your plan:

### If Plan Has CSG Only (csg_operations present, NO hierarchy field)
1. Build each region using surfaces: +surf & -surf
2. Combine connected void segments with | (union) — regions must OVERLAP spatially
3. Use ~region for complement to carve voids from solid regions
4. Create one Cell per distinct material region
5. Do NOT call mcdc.simulation.set_root_universe() — it is not needed for flat CSG geometry

### If Plan Has Hierarchy (hierarchy field present)
1. Create innermost cells (e.g., pin cells) → wrap in Universe
2. For each level outward:
   - Create Lattice using universes from the previous level
   - Create bounding surfaces matching lattice extent exactly
   - Create Cell with region and fill=lattice
   - Create Universe containing that cell
3. For translation-only placement: Cell(region=..., fill=universe, translation=[x,y,z])
4. Final step: mcdc.simulation.set_root_universe(cells=[outermost_cell])

### Always Follow the Dedicated Geometry Plan Literally
1. Reuse the exact named surfaces and region formulas from the geometry plan
2. Do NOT change coordinates or extend segments
3. If the geometry plan already encodes overlap/connectivity, preserve it exactly
4. `role=` text in the plan is documentation only and must never appear in Python constructor arguments
5. If placement cells reuse one universe, create that universe once and reuse the variable

---

## RULE 1: Connected Segments Must Share a 2D Face (Not Just an Edge or Point)
When two void segments connect, they must OVERLAP spatially to form a shared face.

```python
# WRONG — segments only touch at the point (y=10, z=10), no flow possible:
seg_1 = +x_min & -x_max & +y_min & -y_10 & +z_min & -z_10
seg_2 = +x_min & -x_max & +y_10 & -y_max & +z_10 & -z_max

# CORRECT — extend seg_1's z-bound to z_max so they share a face:
seg_1 = +x_min & -x_max & +y_min & -y_10 & +z_min & -z_max
seg_2 = +x_min & -x_max & +y_10 & -y_max & +z_10 & -z_max
void_channel = seg_1 | seg_2
```

---

## RULE 2: Mutual Exclusivity — No Two Cells May Claim the Same Volume
Every point in space must belong to EXACTLY ONE cell.

```python
# WRONG — source_region is inside void_channel, so they both claim the same volume:
source_cell = mcdc.Cell(region=source_region, fill=void_mat)
void_cell   = mcdc.Cell(region=void_channel,  fill=void_mat)

# CORRECT — subtract source_region from void_cell:
source_cell = mcdc.Cell(region=source_region,                  fill=void_mat)
void_cell   = mcdc.Cell(region=void_channel & ~source_region,  fill=void_mat)
```

IMPORTANT: Lattices contain Universe objects, NOT other Lattice objects!
'''


FINALIZE_PROMPT = '''# MCDC Finalize Phase - Source, Tallies, Settings, Run

## Problem
{prompt}

{general_plan_section}
{geometry_plan_section}

## Current Script
```python
{current_script}
```

## Instructions
Add ONLY: source, tallies, settings, and mcdc.run()
Do NOT modify existing code.
Do NOT redefine or overwrite surfaces, regions, cells, universes, lattices, or geometry constants from earlier phases.
Use only short factual comments in the code.

## Common Mistakes to Avoid
- Only define the dimensions of Tallies and sources that are needed
- Source x/y/z must be within geometry bounds. DO NOT place sources exactly on the external boundary
- TallyCell requires cell=<cell_object>, NOT cells=
- TallyMesh requires mesh=<mesh_object>
- Valid scores ONLY: 'flux', 'density', 'collision', 'fission', 'capture', 'net-current'
- Eigenmode: use mcdc.settings.set_eigenmode(N_inactive=, N_active=)
- For eigenvalue runs, use `N_particle` around 1000 with `N_inactive=2` and `N_active=4` unless the prompt explicitly says otherwise
- Ordinary fixed-source problems: do NOT use eigenmode
- For fixed-source runs, use `N_particle` around 1000 unless the prompt explicitly says otherwise
- If the problem is eigenmode, multiplying, supercritical, or has time-census population growth, use `mcdc.simulation.population_control(active=True)`
- If particle banks may grow, consider `mcdc.settings.active_bank_buffer`
- Sources and Tallies are constructors called directly (mcdc.Source(...), mcdc.TallyMesh(...))
  They are NOT assigned to settings
- For translated universe copies, source and tally coordinates are GLOBAL coordinates
- If geometry constants were already defined earlier, reuse them as-is
- For CE problems, usually omit `energy_group` in `mcdc.Source(...)`
- **Techniques are function calls, NOT settings attributes!**
  RIGHT: `mcdc.simulation.implicit_capture()`
- **Rotation is 3 Euler angles in degrees**, NOT axis + angle:
  RIGHT: `rotation=[0, 5, 0]`  (5 degrees around Y axis)

''' + SHARED_MATERIAL_MODE_RULES + '''

{api_reference_section}
{examples_section}

''' + SHARED_ANTI_COPYING_WARNING + '''

Output the COMPLETE script (including previous code) in ```python``` blocks.
'''

# =============================================================================
# FULL SCRIPT GENERATION PROMPT
# =============================================================================

FULL_SCRIPT_PROMPT = '''# Generate Complete MCDC Simulation Script

## Problem Description
{prompt}

{general_plan_section}
{geometry_plan_section}

## Instructions
Generate the COMPLETE MCDC simulation script in a single pass.
The script must include ALL of the following sections in order:
1. **Imports**: `import numpy as np` and `import mcdc`
2. **Materials**: Define all materials from the plan
3. **Surfaces**: Define all boundary and internal surfaces
4. **Geometry**: Define regions, cells, universes, lattices as needed
5. **Root Universe**: Call `mcdc.simulation.set_root_universe(cells=[...])` if using hierarchy
6. **Source**: Define particle source(s)
7. **Tallies**: Define tallies and meshes
8. **Settings**: Set N_particle, N_batch, eigenmode if needed
9. **Techniques**: Apply variance reduction if specified (implicit_capture, etc.)
10. **Run**: Call `mcdc.run()`

Use only short factual comments. If plan context is provided, follow it exactly.

## Common Mistakes to Avoid
### Materials
- scatter must be 2D: `np.array([[value]])` for 1-group
- Fissile materials MUST have both `fission` AND `nu_p`
- Use single-group multigroup materials unless the prompt explicitly asks for multiple energy groups
- Use continuous-energy materials only when the prompt explicitly asks for CE, named real materials, isotopic composition, or use of the material lookup tool/service
- If CE is requested and only material names/formulas are given, call the material lookup tool/service for `nuclide_composition`
- Unless the prompt explicitly asks for critical, supercritical, or eigenvalue behavior, choose subcritical material values
- For multigroup material values, prefer `fission * nu_p < capture + scatter` (subcritical)
- Even when eigenmode is requested, avoid strongly supercritical material choices unless the prompt explicitly demands that. (barely critical)

### Surfaces
- `mcdc.Surface.Box(...)` does NOT exist — use 6 planes + CSG
- CylinderZ center is 2D `[x, y]`, NOT 3D
- Each surface on the same axis must have a DISTINCT coordinate

### Geometry
- Use `+surface` for positive side, `-surface` for negative side (inside)
- CSG: `&` for intersection, `|` for union, `~` for complement
- Every surface reference in a region must be explicitly signed
- For hierarchy: lattice cell surfaces must span `-pitch/2` to `+pitch/2` (LOCAL coordinates)
- For hierarchy: bounding surfaces must EXACTLY match lattice extent `[start, start + pitch * N]`
- Cells that share space MUST have a dividing surface to prevent overlap
- If the geometry plan has a `hierarchy` field: call `mcdc.simulation.set_root_universe(cells=[...])`
- If pure CSG (no hierarchy): do NOT call `set_root_universe`
- Plan metadata keys like `role` and `notes` are NOT API arguments

### Source, Tallies, Settings
- Source x/y/z must be INSIDE geometry bounds, NOT on the boundary
- For CE problems, usually omit `energy_group` in `mcdc.Source(...)`
- TallyCell: `cell=<cell_object>`, NOT `cells=`
- TallyMesh: `mesh=<mesh_object>`
- Valid scores: 'flux', 'density', 'collision', 'fission', 'capture', 'net-current'
- Eigenmode: `mcdc.settings.set_eigenmode(N_inactive=, N_active=)`
- For eigenvalue runs, use `N_particle` around 1000 with `N_inactive=2` and `N_active=4` unless the prompt explicitly says otherwise
- Ordinary fixed-source problems: do NOT use eigenmode
- For fixed-source runs, use `N_particle` around 1000 unless the prompt explicitly says otherwise
- If the problem is eigenmode, multiplying, supercritical, or has time-census population growth, use `mcdc.simulation.population_control(active=True)`
- If particle banks may grow, consider `mcdc.settings.active_bank_buffer`

### Techniques and API
- **Techniques are function calls, NOT settings attributes!**
  WRONG: `mcdc.settings.implicit_capture = True`
  RIGHT: `mcdc.simulation.implicit_capture()`
- **Rotation is 3 Euler angles in degrees**: `rotation=[rx, ry, rz]`
  WRONG: `rotation=[0, 1, 0, 0.1]` (axis + angle)
  RIGHT: `rotation=[0, 5, 0]` (5 degrees around Y axis)
- **Cell translation** is relative to the center of the filled universe

''' + SHARED_MATERIAL_MODE_RULES + '''

{api_reference_section}
{examples_section}

''' + SHARED_ANTI_COPYING_WARNING + '''

Output the COMPLETE script in ```python``` blocks.
'''

# =============================================================================
# FIX PROMPT
# =============================================================================

FIX_PROMPT = '''## Fix Error in MCDC Script

## Original Problem
{prompt}

{general_plan_section}
{geometry_plan_section}

{hints_section}

## Error
```
{error}
```

{api_reference_section}
{examples_section}

''' + SHARED_ANTI_COPYING_WARNING + '''

## Script with error
```python
{script}
```

## Rules
- Make ONLY the minimum change to fix the error
- Do NOT remove or rewrite working code
- Variable names must be valid Python identifiers (no decimals like x1.5)
- Use the plan, if provided, to understand what geometry/materials are needed
- Verify surface/cell variable names are defined before use
- For "bank is full" errors, do NOT just add buffers. First check whether the material data is too multiplying (fission * nu_p > capture + scatter, it should be subcritical or barely critical if eigenmode is used or criticality was requested).
- For multigroup material values, prefer `fission * nu_p < capture + scatter` unless the prompt explicitly asks for critical, supercritical, or eigenvalue behavior.
- If the original prompt requested CE materials, preserve CE and use the material lookup tool/service for `nuclide_composition` values instead of converting the script back to MG.
- If multiplying behavior is required, keep it only mildly multiplying (barely critical), use `mcdc.simulation.population_control(active=True)`, and consider `active_bank_buffer` if needed.
- For eigenvalue runs, use `N_particle` around 1000 with `N_inactive=2` and `N_active=4` unless the prompt explicitly says otherwise.
- For fixed-source runs, use `N_particle` around 1000 unless the prompt explicitly says otherwise.

Output the COMPLETE fixed script in ```python``` blocks.
'''
