"""Example: 3D Homogeneous Box
Demonstrates: 3D single-material geometry with point source and 3D mesh tally.
Complexity: simple
Distinct from benchmarks: purely 3D box geometry with no internal structures.
Keywords: box, cube, plane, sphere, source, tally, moderator, absorber

PATTERNS DEMONSTRATED:
- 6 plane surfaces for 3D bounding box
- 3D cell region using all 6 planes
- Point source at specific position (not volumetric)
- 3D MeshUniform tally
"""

import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Scattering material with moderate capture
air = mcdc.MaterialMG(
    capture=np.array([0.01]),
    scatter=np.array([[0.05]]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: 3D bounding box with 6 boundary surfaces
x_min = mcdc.Surface.PlaneX(x=-5.0, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=-5.0, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=5.0, boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=-5.0, boundary_condition="vacuum")
z_max = mcdc.Surface.PlaneZ(z=5.0, boundary_condition="vacuum")

# PATTERN: 3D cell using all 6 planes
mcdc.Cell(
    region=+x_min & -x_max & +y_min & -y_max & +z_min & -z_max,
    fill=air,
)

# =============================================================================
# Source
# =============================================================================

# PATTERN: Point source at a specific position (not volumetric)
# NOTE: The source position must be strictly inside the geometry bounds.
# The geometry is bounded by x=[-5, 5], y=[-5, 5], and z=[-5, 5].
# The source is placed at [0.0, 0.0, 0.0], which is inside the bounds.
mcdc.Source(
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: 3D MeshUniform — specify all 3 dimensions
mesh = mcdc.MeshUniform(
    x=(-5.0, 1.0, 10),
    y=(-5.0, 1.0, 10),
    z=(-5.0, 1.0, 10),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

mcdc.settings.N_particle = 200
mcdc.settings.N_batch = 2

mcdc.run()
