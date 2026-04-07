"""Example: Simple 2x2 Pin Lattice
Demonstrates: Single-level lattice with pin cell universe and eigenvalue mode.
Complexity: complex
Distinct from benchmarks: 2x2 pin lattice, single assembly, eigenvalue mode.
Keywords: pin, cylinder, lattice, universe, assembly, eigenvalue, criticality, k_eff, fissile, moderator, reflective, tally, source

PATTERNS DEMONSTRATED:
- CylinderZ pin surface centered at origin (local coordinates)
- Pin cells: -cylinder (fuel) and +cylinder (moderator)
- Universe wrapping cells for reuse in lattice
- Lattice x=(start, pitch, N) covers [start, start + pitch*N]
- Boundary surfaces MUST match lattice extent
- set_root_universe with outermost cell
- set_eigenmode for k-eigenvalue calculation
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Fissile fuel — MUST have both fission AND nu_p
fuel = mcdc.MaterialMG(
    capture=np.array([0.10]),
    scatter=np.array([[0.20]]),
    fission=np.array([0.15]),
    nu_p=np.array([2.4]),
)

# PATTERN: Non-fissile moderator
moderator = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.90]]),
)

# =============================================================================
# Pin Cell Universe
# =============================================================================

# PATTERN: Pin geometry in local coordinates (origin at center)
pitch = 1.5
radius = 0.5

# PATTERN: CylinderZ center is 2D [x, y], NOT 3D
pin_surface = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius)
# === PHASE_SETUP END ===

# === PHASE_GEOMETRY START ===
# PATTERN: Pin cells — fuel inside cylinder, moderator outside
fuel_cell = mcdc.Cell(region=-pin_surface, fill=fuel)
mod_cell = mcdc.Cell(region=+pin_surface, fill=moderator)

# PATTERN: Wrap cells into Universe for reuse in lattice
pin = mcdc.Universe(cells=[fuel_cell, mod_cell])

# Moderator-only universe (alternative pin type)
mod_only_cell = mcdc.Cell(region=-pin_surface, fill=moderator)
water = mcdc.Universe(cells=[mod_only_cell, mod_cell])

# =============================================================================
# 2x2 Lattice
# =============================================================================

# PATTERN: Lattice x=(start, pitch, N) covers [start, start + pitch*N]
#   Here: [-1.5, -1.5 + 1.5*2] = [-1.5, 1.5]
lattice = mcdc.Lattice(
    x=[-pitch, pitch, 2],
    y=[-pitch, pitch, 2],
    universes=[
        [pin, pin],   # Bottom row (min Y)
        [pin, pin],   # Top row (max Y)
    ],
)

# =============================================================================
# Root Geometry
# =============================================================================

# PATTERN: Boundary surfaces MUST match lattice extent exactly
x0 = mcdc.Surface.PlaneX(x=-pitch, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=pitch, boundary_condition="reflective")
y0 = mcdc.Surface.PlaneY(y=-pitch, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=pitch, boundary_condition="reflective")

# PATTERN: Assembly cell — region bounded by surfaces, fill=lattice
assembly_region = +x0 & -x1 & +y0 & -y1
assembly_cell = mcdc.Cell(region=assembly_region, fill=lattice)

# PATTERN: ALWAYS call set_root_universe with outermost cell when using hierarchies
mcdc.simulation.set_root_universe(cells=[assembly_cell])
# === PHASE_GEOMETRY END ===

# === PHASE_FINALIZE START ===
# =============================================================================
# Source
# =============================================================================

# PATTERN: Volumetric source matching assembly bounds
mcdc.Source(
    x=[-pitch, pitch],
    y=[-pitch, pitch],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: Mesh aligned with lattice structure
mesh = mcdc.MeshStructured(
    x=np.linspace(-pitch, pitch, 13),
    y=np.linspace(-pitch, pitch, 13),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux", "fission"])

# PATTERN: Eigenmode settings for k-eigenvalue calculations
mcdc.settings.N_particle = 100
mcdc.settings.census_bank_buffer_ratio = 2.0
mcdc.settings.source_bank_buffer_ratio = 2.0
mcdc.settings.set_eigenmode(N_inactive=2, N_active=5)

mcdc.run()
# === PHASE_FINALIZE END ===
