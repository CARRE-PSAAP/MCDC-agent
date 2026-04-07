"""Example: Fissile Sphere with Surface Tally
Demonstrates: Spherical geometry, volumetric source, and TallySurface.
Complexity: simple
Distinct from benchmarks: standalone sphere (no cube), surface tally for leakage.
Keywords: sphere, box, cube, plane, fissile, source, tally, current, moderator, absorber

PATTERNS DEMONSTRATED:
- Sphere surface with center and radius
- Cell inside sphere: region = -sphere_surf
- Cell outside sphere (bounded): region = +sphere_surf & bounding_box
- TallySurface for leakage current measurement
- Combined mesh + surface tallies
"""

import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Fissile material — MUST have both fission AND nu_p
fuel = mcdc.MaterialMG(
    capture=np.array([0.1]),
    scatter=np.array([[0.2]]),
    fission=np.array([0.15]),
    nu_p=np.array([2.5]),
)

# PATTERN: Non-fissile moderator — scatter only, no fission
moderator = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.85]]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: Sphere surface — center is 3D, radius is scalar
sphere_surf = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=3.0)

# PATTERN: Bounding box for the region outside the sphere
x_min = mcdc.Surface.PlaneX(x=-6.0, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=6.0, boundary_condition="vacuum")
y_min = mcdc.Surface.PlaneY(y=-6.0, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=6.0, boundary_condition="vacuum")
z_min = mcdc.Surface.PlaneZ(z=-6.0, boundary_condition="vacuum")
z_max = mcdc.Surface.PlaneZ(z=6.0, boundary_condition="vacuum")

bounding_box = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max

# PATTERN: -sphere = inside sphere, +sphere = outside sphere
mcdc.Cell(region=-sphere_surf, fill=fuel)                       # Fuel sphere
mcdc.Cell(region=+sphere_surf & bounding_box, fill=moderator)   # Surrounding moderator

# =============================================================================
# Source
# =============================================================================

# PATTERN: Volumetric source filling entire fuel sphere
# NOTE: The source region must be strictly inside the geometry bounds.
# The geometry is bounded by x=[-6, 6], y=[-6, 6], and z=[-6, 6].
# The source is placed at x=[-3, 3], y=[-3, 3], and z=[-3, 3], which is inside the bounds.
mcdc.Source(
    x=[-3.0, 3.0],
    y=[-3.0, 3.0],
    z=[-3.0, 3.0],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: MeshStructured tally covering entire geometry
mesh = mcdc.MeshStructured(
    x=np.linspace(-6.0, 6.0, 25),
    y=np.linspace(-6.0, 6.0, 25),
    z=np.linspace(-6.0, 6.0, 25),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

# PATTERN: TallySurface for measuring net current through a surface
mcdc.TallySurface(surface=sphere_surf, scores=["net-current"])

mcdc.settings.N_particle = 500
mcdc.settings.N_batch = 2

mcdc.run()
