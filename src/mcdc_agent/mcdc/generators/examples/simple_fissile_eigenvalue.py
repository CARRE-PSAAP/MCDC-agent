"""Example: 1D Fissile Eigenvalue Problem
Demonstrates: Two fissile regions with k-eigenvalue calculation.
Complexity: simple
Distinct from benchmarks: different region sizes and cross-sections.
Keywords: slab, fissile, eigenvalue, criticality, k_eff, plane, source, tally, absorber

PATTERNS DEMONSTRATED:
- MaterialMG with fission + nu_p (both REQUIRED for fissile materials)
- set_eigenmode for k-eigenvalue calculations
- Volumetric isotropic source spanning entire geometry
- MeshStructured tally with np.linspace
"""

import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Fissile material MUST have both fission AND nu_p
highly_enriched = mcdc.MaterialMG(
    capture=np.array([0.05]),
    scatter=np.array([[0.3]]),
    fission=np.array([0.12]),
    nu_p=np.array([2.8]),
)

# PATTERN: Less reactive fissile material — lower nu_p and fission
low_enriched = mcdc.MaterialMG(
    capture=np.array([0.15]),
    scatter=np.array([[0.25]]),
    fission=np.array([0.06]),
    nu_p=np.array([2.2]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: Vacuum boundaries at both ends for finite slab
x0 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
x4 = mcdc.Surface.PlaneX(x=4.0)  # Interface between regions
x10 = mcdc.Surface.PlaneX(x=10.0, boundary_condition="vacuum")

# Cells
high_enriched_cell = mcdc.Cell(region=+x0 & -x4, fill=highly_enriched)  # Left: highly reactive
low_enriched_cell = mcdc.Cell(region=+x4 & -x10, fill=low_enriched)     # Right: less reactive

# =============================================================================
# Source
# =============================================================================

# PATTERN: Volumetric isotropic source spanning entire geometry
# NOTE: The source region must be strictly inside the geometry bounds.
# The geometry is bounded by x=[0.0, 10.0].
# The source is placed at x=[0.0, 10.0], which is inside the bounds.
mcdc.Source(x=[0.0, 10.0], isotropic=True, energy_group=0)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: MeshStructured with np.linspace for uniform mesh
mesh = mcdc.MeshStructured(x=np.linspace(0.0, 10.0, 51))
mcdc.TallyMesh(mesh=mesh, scores=["flux", "fission"])

# PATTERN: TallyCell for measuring flux and fission in a cell
# NOTE: I am using a variable that was assigned to a cell created earlier
# Do NOT create a new cell to use in the tally
mcdc.TallyCell(cell=high_enriched_cell, scores=["flux", "fission"])

# PATTERN: Eigenmode settings — N_particle + set_eigenmode
mcdc.settings.N_particle = 500
mcdc.settings.set_eigenmode(N_inactive=5, N_active=10)

mcdc.run()
