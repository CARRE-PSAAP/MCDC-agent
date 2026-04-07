"""Example: Cylinder Inside a Box (CSG Complement)
Demonstrates: 2D geometry with cylinder surface and complement operation.
Complexity: simple
Distinct from benchmarks: cylinder in moderator box — no hierarchy, just flat CSG complement.
Keywords: cylinder, box, plane, complement, moderator, absorber, reflective, source, tally, pin

PATTERNS DEMONSTRATED:
- CylinderZ surface at center (axis-aligned, infinite in z)
- 2D bounding box with PlaneX/PlaneY
- Region complement (~) for "outside cylinder, inside box"
- MeshUniform 2D tally
- implicit_capture variance reduction technique
"""

import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Strongly absorbing material
absorber = mcdc.MaterialMG(
    capture=np.array([2.0]),
    scatter=np.array([[0.1]]),
)

# PATTERN: Moderating/scattering material
moderator = mcdc.MaterialMG(
    capture=np.array([0.05]),
    scatter=np.array([[0.85]]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: 2D bounding box with vacuum boundaries
x0 = mcdc.Surface.PlaneX(x=-3.0, boundary_condition="vacuum")
x1 = mcdc.Surface.PlaneX(x=3.0, boundary_condition="vacuum")
y0 = mcdc.Surface.PlaneY(y=-3.0, boundary_condition="vacuum")
y1 = mcdc.Surface.PlaneY(y=3.0, boundary_condition="vacuum")

# PATTERN: CylinderZ — center is 2D [x, y], extends infinitely in z
cylinder = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=1.0)

# PATTERN: Define intermediate regions for clarity
inside_box = +x0 & -x1 & +y0 & -y1
inside_cylinder = -cylinder

# PATTERN: Complement (~) — "inside box AND NOT inside cylinder"
mcdc.Cell(region=inside_cylinder, fill=absorber)
mcdc.Cell(region=inside_box & ~inside_cylinder, fill=moderator)

# =============================================================================
# Source
# =============================================================================

# PATTERN: Source placed in a specific sub-region (corner of box)
# NOTE: The source region must be strictly inside the geometry bounds.
# The geometry is bounded by x=[-3, 3] and y=[-3, 3].
# The source is placed at x=[-3, -2] and y=[-3, -2], which is inside the bounds.
mcdc.Source(
    x=[-3.0, -2.0],
    y=[-3.0, -2.0],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: 2D MeshUniform — (start, bin_width, N_bins)
mesh = mcdc.MeshUniform(x=(-3.0, 0.25, 24), y=(-3.0, 0.25, 24))
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

mcdc.settings.N_particle = 200
mcdc.settings.N_batch = 2

# PATTERN: Implicit capture for better statistics in absorbing problems
mcdc.simulation.implicit_capture()

mcdc.run()
