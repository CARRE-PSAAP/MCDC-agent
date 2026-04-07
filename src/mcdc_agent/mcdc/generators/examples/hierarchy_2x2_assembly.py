"""Example: Minimal 2x2 Pin Assembly with Eigenmode
Demonstrates: Single-level lattice hierarchy (pins → assembly).
Complexity: complex
Distinct from benchmarks: only 2x2 pins, one assembly, no core lattice.
Keywords: pin, cylinder, lattice, universe, assembly, eigenvalue, criticality, k_eff, fissile, moderator, source, tally, reflective

PATTERNS DEMONSTRATED:
- Pin cell universe: cylinder + moderator cells → Universe
- Lattice with universe array: x=(start, pitch, N)
- Assembly cell bounded by surfaces matching lattice extent
- set_root_universe with outermost cell
- Eigenmode with set_eigenmode
- Lattice extent = [start, start + pitch * N]
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Fissile fuel — MUST have both fission AND nu_p
fuel_mat = mcdc.MaterialMG(
    capture=np.array([0.08]),
    scatter=np.array([[0.25]]),
    fission=np.array([0.12]),
    nu_p=np.array([2.5]),
)

# PATTERN: Moderator (non-fissile) — scatter-dominated
mod_mat = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.9]]),
)

# =============================================================================
# Level 1: Pin Cell Universe
# =============================================================================

# PATTERN: Pin geometry — cylinder at origin (local coordinates)
pin_pitch = 2.0
pin_radius = 0.8 # radius x 2 = 1.6 < pitch = 2.0

pin_cyl = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=pin_radius)
# === PHASE_SETUP END ===

# === PHASE_GEOMETRY START ===
# PATTERN: Pin cells — inside cylinder = fuel, outside = moderator
fuel_cell = mcdc.Cell(region=-pin_cyl, fill=fuel_mat)
mod_cell = mcdc.Cell(region=+pin_cyl, fill=mod_mat)

# PATTERN: Wrap pin cells into a Universe
pin_universe = mcdc.Universe(cells=[fuel_cell, mod_cell])

# =============================================================================
# Level 2: Assembly Lattice
# =============================================================================

# PATTERN: Calculate assembly dimensions from pitch and pin count
#   assembly_width = pin_pitch * N_pins = 2.0 * 2 = 4.0
#   assembly_half = 2.0
assembly_half = pin_pitch * 2 / 2.0  # = 2.0

# PATTERN: Bounding surfaces MUST match lattice extent exactly
asm_x0 = mcdc.Surface.PlaneX(x=-assembly_half, boundary_condition="reflective")
asm_x1 = mcdc.Surface.PlaneX(x=+assembly_half, boundary_condition="reflective")
asm_y0 = mcdc.Surface.PlaneY(y=-assembly_half, boundary_condition="reflective")
asm_y1 = mcdc.Surface.PlaneY(y=+assembly_half, boundary_condition="reflective")

# PATTERN: Lattice syntax — x=(start, pitch, N_cells)
#   Covers [start, start + pitch * N] = [-2.0, -2.0 + 2.0*2] = [-2.0, 2.0]
assembly_lattice = mcdc.Lattice(
    x=(-assembly_half, pin_pitch, 2),
    y=(-assembly_half, pin_pitch, 2),
    universes=[
        [pin_universe, pin_universe],
        [pin_universe, pin_universe],
    ],
)

# PATTERN: Assembly cell — bounded region filled with lattice
assembly_region = +asm_x0 & -asm_x1 & +asm_y0 & -asm_y1
assembly_cell = mcdc.Cell(region=assembly_region, fill=assembly_lattice)

# PATTERN: set_root_universe with outermost cell
mcdc.simulation.set_root_universe(cells=[assembly_cell])
# === PHASE_GEOMETRY END ===

# === PHASE_FINALIZE START ===
# =============================================================================
# Source
# =============================================================================

# PATTERN: Volumetric source covering assembly bounds
mcdc.Source(
    x=[-assembly_half, assembly_half],
    y=[-assembly_half, assembly_half],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: Mesh aligned with pins for meaningful spatial resolution
mesh = mcdc.MeshStructured(
    x=np.linspace(-assembly_half, assembly_half, 5), # start, stop, num_intervals + 1 (4 total intervals)
    y=np.linspace(-assembly_half, assembly_half, 5),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux", "fission"])

# PATTERN: Eigenmode settings
mcdc.settings.N_particle = 500
mcdc.settings.set_eigenmode(N_inactive=3, N_active=5)

mcdc.run()
# === PHASE_FINALIZE END ===
