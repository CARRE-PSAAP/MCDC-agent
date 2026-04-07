"""Example: 3x3 Pin Assembly with Universe/Lattice (Flat Hierarchy)
Demonstrates: Single-level lattice of pin cell universes with eigenvalue mode.
Complexity: complex
Distinct from benchmarks: 3x3 pins, reflective assembly boundaries, eigenvalue mode.
Keywords: pin, cylinder, lattice, universe, assembly, eigenvalue, criticality, k_eff, fissile, moderator, reflective, tally, source

PATTERNS DEMONSTRATED:
- CylinderZ pin surface centered at origin (local coordinates!)
- Pin cells: -cylinder (fuel) and +cylinder (moderator)
- Universe wrapping cells for reuse in lattice
- Lattice x=(start, pitch, N) covers [start, start + pitch*N]
- Boundary surfaces MUST match lattice extent exactly
- set_root_universe with assembly_cell only
- Eigenmode with set_eigenmode
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# Fuel material (fissile)
fuel = mcdc.MaterialMG(
    capture=np.array([0.10]),
    scatter=np.array([[0.20]]),
    fission=np.array([0.15]),
    nu_p=np.array([2.4]),
)

# Moderator material (scattering)
moderator = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.90]]),
)

# =============================================================================
# Pin Cell Universe (reusable unit)
# =============================================================================

# Pin geometry parameters
pitch = 1.5   # Pin cell width
radius = 0.5  # Fuel pin radius

# Pin surfaces - CENTERED AT ORIGIN (local coordinates!)
# Each lattice cell places the universe at its center
pin_cylinder = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius)
# === PHASE_SETUP END ===

# Pin cells - must cover the full pitch in local coordinates
# The cylinder divides the cell into fuel (inside) and moderator (outside)
# === PHASE_GEOMETRY START ===
fuel_cell = mcdc.Cell(region=-pin_cylinder, fill=fuel)
mod_cell = mcdc.Cell(region=+pin_cylinder, fill=moderator)

# Pin universe - groups cells into a reusable unit
pin = mcdc.Universe(cells=[fuel_cell, mod_cell])

# =============================================================================
# Assembly Lattice (3x3 pins)
# =============================================================================

N = 3  # 3x3 lattice

# CRITICAL: Lattice x=(start, pitch, count) is NOT (min, max, count)!
# The lattice covers: [start, start + pitch*count]
# For a centered lattice: start = -pitch*N/2
half_width = pitch * N / 2  # = 2.25 for 3x3 with pitch 1.5

assembly_lattice = mcdc.Lattice(
    x=(-half_width, pitch, N),  # Covers x from -2.25 to +2.25
    y=(-half_width, pitch, N),  # Covers y from -2.25 to +2.25
    # Omit z for 2D lattice (infinite in z)
    universes=[
        [pin, pin, pin],  # Row 0 = BOTTOM of lattice (min Y)
        [pin, pin, pin],  # Row 1 = middle
        [pin, pin, pin],  # Row 2 = TOP of lattice (max Y)
    ],
)

# =============================================================================
# Root Geometry (contains the lattice)
# =============================================================================

# Boundary surfaces - MUST match lattice extent exactly!
# Lattice covers [-half_width, +half_width] in both x and y
x0 = mcdc.Surface.PlaneX(x=-half_width, boundary_condition="reflective")
x1 = mcdc.Surface.PlaneX(x=+half_width, boundary_condition="reflective")
y0 = mcdc.Surface.PlaneY(y=-half_width, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=+half_width, boundary_condition="reflective")

# Assembly cell filled with the lattice
assembly_region = +x0 & -x1 & +y0 & -y1
assembly_cell = mcdc.Cell(region=assembly_region, fill=assembly_lattice)

# REQUIRED: Explicitly set root universe
# Without this, MC/DC auto-populates root with ALL cells, causing conflicts
mcdc.simulation.set_root_universe(cells=[assembly_cell])
# === PHASE_GEOMETRY END ===

# =============================================================================
# Source
# =============================================================================

# === PHASE_FINALIZE START ===
# Volumetric source covering the entire assembly
mcdc.Source(
    x=[-half_width, half_width],
    y=[-half_width, half_width],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies
# =============================================================================

# Mesh tally for flux distribution
# Note: mesh boundaries need N+1 points for N bins
# Keep mesh coarse relative to particle count to avoid variance issues
mesh = mcdc.MeshStructured(
    x=np.linspace(-half_width, half_width, N + 1),  # 3 bins in x (one per pin column)
    y=np.linspace(-half_width, half_width, N + 1),  # 3 bins in y (one per pin row)
    # Omit z for 2D mesh (or use z=np.array([zmin, zmax]) for single z-bin)
)
mcdc.TallyMesh(mesh=mesh, scores=["flux", "fission"])

# =============================================================================
# Settings
# =============================================================================

# Particle count should be >> mesh bins to avoid variance issues
mcdc.settings.N_particle = 2000
mcdc.settings.N_batch = 5

# Eigenmode for k-eigenvalue calculation
mcdc.settings.set_eigenmode(N_inactive=2, N_active=5)

# =============================================================================
# Run
# =============================================================================

mcdc.run()
# === PHASE_FINALIZE END ===
