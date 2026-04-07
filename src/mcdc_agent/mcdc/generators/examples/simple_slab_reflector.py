"""Example: 1D Fissile Slab with Reflector
Demonstrates: Simple 1D geometry with fissile material, vacuum BCs, MeshStructured tally.
Complexity: simple
Distinct from benchmarks: fissile slab + non-fissile reflector, 1D in z-direction.
Keywords: slab, fissile, eigenvalue, reflector, absorber, plane, tally, source, moderator

PATTERNS DEMONSTRATED:
- MaterialMG with fission + nu_p (BOTH required for fissile materials)
- scatter MUST be 2D: np.array([[value]])
- 1D geometry using PlaneZ surfaces
- Volumetric source with z-bounds
- MeshStructured tally with np.linspace
"""

import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Fissile material — MUST have both fission AND nu_p
fuel = mcdc.MaterialMG(
    capture=np.array([0.1]),
    scatter=np.array([[0.25]]),       # PATTERN: scatter MUST be 2D array
    fission=np.array([0.08]),
    nu_p=np.array([2.5]),
)

# PATTERN: Non-fissile material — scatter only, no fission/nu_p
reflector = mcdc.MaterialMG(
    capture=np.array([0.01]),
    scatter=np.array([[0.9]]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: 1D slab with PlaneZ surfaces and vacuum boundaries
z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
z1 = mcdc.Surface.PlaneZ(z=5.0)  # Core-reflector interface (no BC = internal)
z2 = mcdc.Surface.PlaneZ(z=8.0, boundary_condition="vacuum")

# PATTERN: Cells using +surface & -surface for 1D regions
mcdc.Cell(region=+z0 & -z1, fill=fuel)       # Core: z=[0,5]
mcdc.Cell(region=+z1 & -z2, fill=reflector)  # Reflector: z=[5,8]

# =============================================================================
# Source
# =============================================================================

# PATTERN: Volumetric source with z-bounds inside geometry
# NOTE: The source region must be strictly inside the geometry bounds.
# The geometry is bounded by z=[0.0, 8.0].
# The source is placed at z=[0.0, 5.0], which is inside the bounds.
mcdc.Source(z=[0.0, 5.0], isotropic=True, energy_group=0)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: MeshStructured with np.linspace for evenly spaced bins
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 8.0, 41))
mcdc.TallyMesh(mesh=mesh, scores=["flux", "fission"])

# PATTERN: Fixed-source settings (no eigenmode)
mcdc.settings.N_particle = 1000
mcdc.settings.N_batch = 2

mcdc.run()
