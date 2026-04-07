"""Example: Concentric Spherical Shells
Demonstrates: nested spheres with CSG complement for shell regions.
Complexity: complex
Distinct from benchmarks: multiple concentric shells with annular regions, no void channels.
Keywords: sphere, box, plane, shell, annular, complement, fissile, moderator, source, tally, csg

PATTERNS DEMONSTRATED:
- Multiple Sphere surfaces at same center, different radii
- Shell region using +inner_sphere & -outer_sphere
- Outermost cell bounded by box & +outer_sphere
- TallyCell on specific shell regions
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Fissile core material
core_mat = mcdc.MaterialMG(
    capture=np.array([0.05]),
    scatter=np.array([[0.3]]),
    fission=np.array([0.1]),
    nu_p=np.array([2.5]),
)

# PATTERN: Absorbing shell material
shell_mat = mcdc.MaterialMG(
    capture=np.array([0.4]),
    scatter=np.array([[0.5]]),
)

# PATTERN: Low-interaction outer material
outer_mat = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.08]]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: Concentric spheres — same center, different radii
inner_sphere = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=2.0)
outer_sphere = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=4.0)

# PATTERN: Bounding box for outermost region
x_min = mcdc.Surface.PlaneX(x=-8.0, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=8.0, boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=-8.0, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=8.0, boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=-8.0, boundary_condition="vacuum")
z_max = mcdc.Surface.PlaneZ(z=8.0, boundary_condition="vacuum")
# === PHASE_SETUP END ===

bounding_box = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max

# === PHASE_GEOMETRY START ===
# PATTERN: Nested shells
# Core: inside inner sphere
core_cell = mcdc.Cell(region=-inner_sphere, fill=core_mat)

# Shell: between inner and outer spheres
# +inner_sphere means outside inner, -outer_sphere means inside outer
shell_cell = mcdc.Cell(region=+inner_sphere & -outer_sphere, fill=shell_mat)

# Outer: outside outer sphere, bounded by box
outer_cell = mcdc.Cell(region=+outer_sphere & bounding_box, fill=outer_mat)
# === PHASE_GEOMETRY END ===

# === PHASE_FINALIZE START ===
# =============================================================================
# Source
# =============================================================================

# PATTERN: Point source at center
mcdc.Source(
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: Cell tallies on specific regions
mcdc.TallyCell(cell=core_cell, scores=["flux", "fission"])
mcdc.TallyCell(cell=shell_cell, scores=["flux"])

# PATTERN: 3D mesh tally
mesh = mcdc.MeshUniform(
    x=(-8.0, 2.0, 8),
    y=(-8.0, 2.0, 8),
    z=(-8.0, 2.0, 8),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

mcdc.settings.N_particle = 500
mcdc.settings.N_batch = 2

mcdc.run()
# === PHASE_FINALIZE END ===
