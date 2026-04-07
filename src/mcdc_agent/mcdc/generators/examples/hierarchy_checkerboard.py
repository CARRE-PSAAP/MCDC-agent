"""Example: Checkerboard Core with Two Assembly Types
Demonstrates: Multi-level hierarchy with different assembly universes in a core lattice.
Complexity: complex
Distinct from benchmarks: only 2x2 core, 2x2 pins per assembly, two distinct assembly types.
Keywords: pin, cylinder, lattice, universe, assembly, core, checkerboard, eigenvalue, fissile, moderator, absorber, reflective, tally

PATTERNS DEMONSTRATED:
- Multiple pin universe types (fuel vs absorber)
- Assembly lattice using different pin universes
- Multiple assembly universes (fuel assembly vs absorber assembly)
- Core lattice using different assembly universes
- Two-level lattice → cell → universe wrapping
- Dimension calculations: pin_pitch, assembly_width, core_width
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

fuel_mat = mcdc.MaterialMG(
    capture=np.array([0.05]),
    scatter=np.array([[0.25]]),
    fission=np.array([0.15]),
    nu_p=np.array([2.4]),
)

absorber_mat = mcdc.MaterialMG(
    capture=np.array([0.8]),
    scatter=np.array([[0.1]]),
)

mod_mat = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.88]]),
)

# =============================================================================
# Level 1: Pin Cell Universes
# =============================================================================

pin_pitch = 1.5
pin_radius = 0.5

# PATTERN: Shared cylinder surface (local coordinates)
pin_cyl = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=pin_radius)
# === PHASE_SETUP END ===

# === PHASE_GEOMETRY START ===
# PATTERN: Fuel pin universe
fuel_pin_cell = mcdc.Cell(region=-pin_cyl, fill=fuel_mat)
fuel_mod_cell = mcdc.Cell(region=+pin_cyl, fill=mod_mat)
u_fuel_pin = mcdc.Universe(cells=[fuel_pin_cell, fuel_mod_cell])

# PATTERN: Absorber pin universe (different material, same geometry)
abs_pin_cell = mcdc.Cell(region=-pin_cyl, fill=absorber_mat)
abs_mod_cell = mcdc.Cell(region=+pin_cyl, fill=mod_mat)
u_abs_pin = mcdc.Universe(cells=[abs_pin_cell, abs_mod_cell])

# =============================================================================
# Level 2: Assembly Lattices → Cells → Universes
# =============================================================================

# PATTERN: Dimension calculations
#   assembly_width = pin_pitch * N_pins = 1.5 * 2 = 3.0
#   assembly_half = 1.5
assembly_width = pin_pitch * 2  # = 3.0 (times 2 for 2x2 assembly)
assembly_half = assembly_width / 2  # = 1.5

# PATTERN: Assembly bounding surfaces (local to assembly)
asm_x0 = mcdc.Surface.PlaneX(x=-assembly_half)
asm_x1 = mcdc.Surface.PlaneX(x=+assembly_half)
asm_y0 = mcdc.Surface.PlaneY(y=-assembly_half)
asm_y1 = mcdc.Surface.PlaneY(y=+assembly_half)

assembly_region = +asm_x0 & -asm_x1 & +asm_y0 & -asm_y1

# PATTERN: Fuel assembly — 2x2 all fuel pins
fuel_asm_lattice = mcdc.Lattice(
    x=(-assembly_half, pin_pitch, 2),
    y=(-assembly_half, pin_pitch, 2),
    universes=[
        [u_fuel_pin, u_fuel_pin],
        [u_fuel_pin, u_fuel_pin],
    ],
)
fuel_asm_cell = mcdc.Cell(region=assembly_region, fill=fuel_asm_lattice)
u_fuel_assembly = mcdc.Universe(cells=[fuel_asm_cell])

# PATTERN: Absorber assembly — 2x2 all absorber pins
abs_asm_lattice = mcdc.Lattice(
    x=(-assembly_half, pin_pitch, 2),
    y=(-assembly_half, pin_pitch, 2),
    universes=[
        [u_abs_pin, u_abs_pin],
        [u_abs_pin, u_abs_pin],
    ],
)
abs_asm_cell = mcdc.Cell(region=assembly_region, fill=abs_asm_lattice)
u_abs_assembly = mcdc.Universe(cells=[abs_asm_cell])

# =============================================================================
# Level 3: Core Lattice
# =============================================================================

# PATTERN: Core dimension calculations
#   core_width = assembly_width * N_assemblies = 3.0 * 2 = 6.0
#   core_half = 3.0
core_width = assembly_width * 2  # = 6.0
core_half = core_width / 2  # = 3.0

# PATTERN: Core boundary surfaces MUST match core lattice extent
core_x0 = mcdc.Surface.PlaneX(x=-core_half, boundary_condition="reflective")
core_x1 = mcdc.Surface.PlaneX(x=+core_half, boundary_condition="vacuum")
core_y0 = mcdc.Surface.PlaneY(y=-core_half, boundary_condition="reflective")
core_y1 = mcdc.Surface.PlaneY(y=+core_half, boundary_condition="vacuum")

# PATTERN: Core lattice with checkerboard arrangement
#   Lattice x=(start, pitch, N) covers [start, start + pitch * N]
#   Here: [-3.0, -3.0 + 3.0*2] = [-3.0, 3.0] ✓
core_lattice = mcdc.Lattice(
    x=(-core_half, assembly_width, 2),
    y=(-core_half, assembly_width, 2),
    universes=[
        [u_fuel_assembly, u_abs_assembly],   # Bottom row
        [u_abs_assembly, u_fuel_assembly],   # Top row (checkerboard)
    ],
)

# PATTERN: Core cell → set_root_universe
core_region = +core_x0 & -core_x1 & +core_y0 & -core_y1
core_cell = mcdc.Cell(region=core_region, fill=core_lattice)
mcdc.simulation.set_root_universe(cells=[core_cell])
# === PHASE_GEOMETRY END ===

# =============================================================================
# Source
# =============================================================================

# === PHASE_FINALIZE START ===
mcdc.Source(
    x=[-core_half, core_half],
    y=[-core_half, core_half],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: Mesh aligned with assembly grid
mesh = mcdc.MeshStructured(
    x=np.linspace(-core_half, core_half, 5),
    y=np.linspace(-core_half, core_half, 5),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux", "fission"])

mcdc.settings.N_particle = 500
mcdc.settings.set_eigenmode(N_inactive=3, N_active=5)

mcdc.run()
# === PHASE_FINALIZE END ===
