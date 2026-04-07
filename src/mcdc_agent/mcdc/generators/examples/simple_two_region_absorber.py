"""Example: Two-Region Absorber Slab
Demonstrates: 1D slab with two materials, beam source, MeshUniform tally.
Complexity: simple
Distinct from benchmarks: different dimensions, beam source instead of isotropic.
Keywords: slab, plane, absorber, moderator, source, tally

PATTERNS DEMONSTRATED:
- MaterialMG with capture-only (pure absorber)
- PlaneX surfaces with mixed boundary conditions
- Beam source using position + direction (NOT isotropic)
- MeshUniform tally syntax: (start, bin_width, N_bins)
"""

import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Pure absorber — only capture, no scatter/fission
thin_absorber = mcdc.MaterialMG(
    capture=np.array([0.5]),
    scatter=np.array([[0.1]]),
)

# PATTERN: Strong absorber — higher capture cross-section
thick_absorber = mcdc.MaterialMG(
    capture=np.array([2.0]),
    scatter=np.array([[0.05]]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: Mixed boundary conditions — reflective left, vacuum right
x0 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")
x3 = mcdc.Surface.PlaneX(x=3.0)  # Internal interface
x8 = mcdc.Surface.PlaneX(x=8.0, boundary_condition="vacuum")

# PATTERN: Simple 1D cells using +surface & -surface
mcdc.Cell(region=+x0 & -x3, fill=thin_absorber)   # Thin absorber region
mcdc.Cell(region=+x3 & -x8, fill=thick_absorber)   # Thick absorber region

# =============================================================================
# Source
# =============================================================================

# PATTERN: Beam source — position + direction, NOT isotropic
# NOTE: The source position must be strictly inside the geometry bounds.
# The geometry is bounded by x=[0.0, 8.0].
# The source is placed at x=[0.01, 0.0, 0.0], which is inside the bounds.
mcdc.Source(
    position=[0.01, 0.0, 0.0],
    direction=[1.0, 0.0, 0.0],
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: MeshUniform syntax — (start, bin_width, N_bins)
mesh = mcdc.MeshUniform(x=(0.0, 0.2, 40))
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

mcdc.settings.N_particle = 500
mcdc.settings.N_batch = 2

mcdc.run()
