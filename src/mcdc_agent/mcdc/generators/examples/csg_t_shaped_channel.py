"""Example: T-Shaped Void Channel in Shield Block
Demonstrates: CSG with multi-segment union and complement operations.
Complexity: complex
Distinct from benchmarks: T-shape (not L or dog-leg), different dimensions.
Keywords: box, plane, void, channel, duct, complement, union, shielding, absorber, source, tally, streaming, csg

PATTERNS DEMONSTRATED:
- Multiple rectangular regions defined with plane surfaces
- UNION (|) to combine connected void segments
- COMPLEMENT (~) to carve void from solid shield
- Overlapping regions for connectivity (critical for CSG)
- set_root_universe for flat CSG geometry
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Shield material — high scatter, moderate capture
shield_mat = mcdc.MaterialMG(
    capture=np.array([0.15]),
    scatter=np.array([[0.75]]),
)

# PATTERN: Void material — near-zero cross-sections (NOT actual zero)
void_mat = mcdc.MaterialMG(
    capture=np.array([0.001]),
    scatter=np.array([[0.001]]),
)

# =============================================================================
# Geometry — Surfaces
# =============================================================================

# PATTERN: 3D bounding box with mixed boundary conditions
x_min = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")
x_max = mcdc.Surface.PlaneX(x=20.0, boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=0.0, boundary_condition="reflective")
y_max = mcdc.Surface.PlaneY(y=20.0, boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="reflective")
z_max = mcdc.Surface.PlaneZ(z=20.0, boundary_condition="vacuum")

# Internal surfaces for T-shaped channel
x_8 = mcdc.Surface.PlaneX(x=8.0)
x_12 = mcdc.Surface.PlaneX(x=12.0)
y_6 = mcdc.Surface.PlaneY(y=6.0)
y_14 = mcdc.Surface.PlaneY(y=14.0)
z_4 = mcdc.Surface.PlaneZ(z=4.0)
# === PHASE_SETUP END ===

# === PHASE_GEOMETRY START ===
# =============================================================================
# Geometry — Regions and Cells
# =============================================================================

# PATTERN (RULE 1): All regions are bounded in ALL 3 axes using bounding_box surfaces
# PATTERN: Define bounding box region
bounding_box = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max

# Vertical stem of T: x=[8,12], y=[0,20], z=[0,4]
# RULE 1: x bounded by x_8/x_12, y by y_min/y_max, z by z_min/z_4 — all 3 axes explicit
stem_region = +x_8 & -x_12 & +y_min & -y_max & +z_min & -z_4

# Horizontal bar of T: x=[0,20], y=[6,14], z=[0,4]
# CRITICAL: This OVERLAPS with stem — required for physical connectivity (RULE 2)!
bar_region = +x_min & -x_max & +y_6 & -y_14 & +z_min & -z_4

# PATTERN: Source sub-region at bottom of stem
source_region = +x_8 & -x_12 & +y_min & -y_6 & +z_min & -z_4

# PATTERN: UNION (|) combines connected segments into one void channel
void_channel = stem_region | bar_region

# PATTERN: COMPLEMENT (~) carves void from solid shield
shield_region = bounding_box & ~void_channel

# PATTERN (RULE 3): Mutual exclusivity — source_region is inside void_channel,
# so subtract it from void_cell to prevent two cells overlapping the same volume.
source_cell = mcdc.Cell(region=source_region,                 fill=void_mat)
void_cell   = mcdc.Cell(region=void_channel & ~source_region, fill=void_mat)
shield_cell = mcdc.Cell(region=shield_region,                 fill=shield_mat)
# PATTERN: For pure CSG geometry (no Universe/Lattice), do NOT call set_root_universe
# === PHASE_GEOMETRY END ===

# === PHASE_FINALIZE START ===
# =============================================================================
# Source
# =============================================================================

# PATTERN: Volumetric source in a specific sub-region
mcdc.Source(
    x=[8.0, 12.0],
    y=[0.0, 2.0],
    z=[0.0, 4.0],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: 2D mesh tally for spatial distribution
mesh = mcdc.MeshUniform(x=(0.0, 1.0, 20), y=(0.0, 1.0, 20)) # start, step, num_intervals
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

# PATTERN: Cell tallies for integral quantities in each region
mcdc.TallyCell(cell=void_cell, scores=["flux"])
mcdc.TallyCell(cell=shield_cell, scores=["flux"])

mcdc.settings.N_particle = 500
mcdc.settings.N_batch = 2

mcdc.run()
# === PHASE_FINALIZE END ===
