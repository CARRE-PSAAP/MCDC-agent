"""DEEPLY ANNOTATED EXAMPLE: Multi-Level Hierarchy (2D Fuel Assembly Core)
Demonstrates: Three-level lattice hierarchy (pin → assembly → core) with eigenvalue mode.
Complexity: complex
Distinct from benchmarks: 2x2 assemblies each with 3x3 pins, two-level lattice wrapping.
Keywords: pin, cylinder, lattice, universe, assembly, core, eigenvalue, criticality, k_eff, fissile, moderator, reflective, tally, source, translation

This example demonstrates the CORRECT pattern for nested geometries in MCDC.
It creates a simplified reactor core with:
- Level 1: Pin cell universes (fuel pin + moderator)
- Level 2: Assembly lattices → cells → universes  
- Level 3: Core lattice → cell → root

================================================================================
KEY PATTERNS TO FOLLOW:
================================================================================

1. LATTICES CONTAIN UNIVERSES, NOT OTHER LATTICES
   - WRONG:  lattice_core = mcdc.Lattice(..., universes=[[lattice_uo2, lattice_mox]])
   - RIGHT:  lattice_core = mcdc.Lattice(..., universes=[[u_uo2, u_mox]])
   Where u_uo2 is a Universe containing a Cell filled with lattice_uo2.

2. EACH LEVEL WRAPS ITS LATTICE IN Cell → Universe BEFORE USE IN NEXT LEVEL
   Level 2 pattern:
     lattice_uo2 = mcdc.Lattice(...)
     assembly_cell = mcdc.Cell(region=..., fill=lattice_uo2)
     u_uo2 = mcdc.Universe(cells=[assembly_cell])
   Then Level 3 uses u_uo2 in its lattice.

3. BOUNDING SURFACES MUST MATCH LATTICE EXTENT EXACTLY
   For lattice with x=(start, pitch, N):
     - Lattice covers [start, start + pitch*N]
     - Bounding planes must be at exactly these positions

4. USE TRANSLATION WHEN REFLECTIVE BOUNDARIES DON'T ALIGN WITH GEOMETRY CENTER
   If using quarter-core symmetry with reflective at x=0, y=0,
   but lattice is centered at origin, use translation to shift.

================================================================================
"""

import numpy as np
import mcdc

# ==============================================================================
# SETUP: MATERIALS 
# ==============================================================================

# === PHASE_SETUP START ===
# Fuel material (fissile) - simplified 1-group cross sections
fuel = mcdc.MaterialMG(
    capture=np.array([0.10]),
    scatter=np.array([[0.20]]),
    fission=np.array([0.15]),
    nu_p=np.array([2.4]),
)

# Moderator material (high scattering, no fission)
moderator = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.90]]),
)
# === PHASE_SETUP END ===

# ==============================================================================
# GEOMETRY: LEVEL 1 - PIN CELL UNIVERSES 
# ==============================================================================
# Pin cells are the innermost level. Each pin is a universe containing cells.
# These universes will be used as elements in the assembly lattices.

# === PHASE_GEOMETRY START ===
# Pin geometry parameters
pitch = 1.5   # Pin cell pitch (cm) - spacing between pin centers
radius = 0.5  # Fuel pin radius (cm)

# Pin surface - CENTERED AT ORIGIN (local coordinates)
# When placed in a lattice, each cell's origin is at its center
pin_cylinder = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=radius)

# Pin cells - define fuel region (inside cylinder) and moderator (outside)
# Together these cells must cover the entire pin cell volume
fuel_cell = mcdc.Cell(region=-pin_cylinder, fill=fuel)      # Inside cylinder
mod_cell = mcdc.Cell(region=+pin_cylinder, fill=moderator)  # Outside cylinder

# Pin universe - groups the two cells into a reusable unit
# This universe will be placed at each lattice position
pin = mcdc.Universe(cells=[fuel_cell, mod_cell])

# ==============================================================================
# GEOMETRY: LEVEL 2 - ASSEMBLY LATTICES → CELLS → UNIVERSES 
# ==============================================================================
# Each assembly is a lattice of pins. We create the lattice, wrap it in a cell,
# then wrap that cell in a universe so it can be used in the core lattice.

N_pins = 3  # 3x3 pin lattice per assembly

# Calculate assembly dimensions
# CRITICAL: Lattice x=(start, pitch, N) covers [start, start + pitch*N]
assembly_half_width = pitch * N_pins / 2  # = 2.25 cm for 3 pins with 1.5 pitch

# Assembly lattice (3x3 pins)
assembly_lattice = mcdc.Lattice(
    x=(-assembly_half_width, pitch, N_pins),  # Covers [-2.25, +2.25]
    y=(-assembly_half_width, pitch, N_pins),  # Covers [-2.25, +2.25]
    # Omit z for 2D (infinite in z)
    universes=[
        [pin, pin, pin],  # Row 0 = BOTTOM (min Y)
        [pin, pin, pin],  # Row 1 = middle
        [pin, pin, pin],  # Row 2 = TOP (max Y)
    ],
)

# Assembly bounding surfaces - MUST match lattice extent exactly
# These define the region that contains the lattice
asm_x0 = mcdc.Surface.PlaneX(x=-assembly_half_width)  # x = -2.25
asm_x1 = mcdc.Surface.PlaneX(x=+assembly_half_width)  # x = +2.25
asm_y0 = mcdc.Surface.PlaneY(y=-assembly_half_width)  # y = -2.25
asm_y1 = mcdc.Surface.PlaneY(y=+assembly_half_width)  # y = +2.25

# Assembly cell - bounded region filled with the lattice
assembly_region = +asm_x0 & -asm_x1 & +asm_y0 & -asm_y1
assembly_cell = mcdc.Cell(region=assembly_region, fill=assembly_lattice)

# CRITICAL: Wrap the assembly cell in a Universe
# This universe can now be used as an element in the core lattice
u_assembly = mcdc.Universe(cells=[assembly_cell])

# ==============================================================================
# GEOMETRY: LEVEL 3 - CORE LATTICE → CELL → ROOT 
# ==============================================================================
# The core is a 2x2 lattice of assemblies. This is the outermost level.

N_assemblies = 2  # 2x2 assembly lattice

# Calculate core dimensions
# Each assembly is (pitch * N_pins) wide = 4.5 cm
assembly_width = pitch * N_pins  # = 4.5 cm
core_half_width = assembly_width * N_assemblies / 2  # = 4.5 cm

# Core lattice (2x2 assemblies)
# NOTE: universes contains Universe objects (u_assembly), NOT Lattice objects!
core_lattice = mcdc.Lattice(
    x=(-core_half_width, assembly_width, N_assemblies),  # Covers [-4.5, +4.5]
    y=(-core_half_width, assembly_width, N_assemblies),  # Covers [-4.5, +4.5]
    universes=[
        [u_assembly, u_assembly],  # Row 0 = BOTTOM
        [u_assembly, u_assembly],  # Row 1 = TOP
    ],
)

# Core boundary surfaces with boundary conditions
# These are the problem boundaries - must match core lattice extent
core_x0 = mcdc.Surface.PlaneX(x=-core_half_width, boundary_condition="reflective")
core_x1 = mcdc.Surface.PlaneX(x=+core_half_width, boundary_condition="reflective")
core_y0 = mcdc.Surface.PlaneY(y=-core_half_width, boundary_condition="reflective")
core_y1 = mcdc.Surface.PlaneY(y=+core_half_width, boundary_condition="reflective")

# Core cell - the outermost cell containing the core lattice
core_region = +core_x0 & -core_x1 & +core_y0 & -core_y1
core_cell = mcdc.Cell(region=core_region, fill=core_lattice)

# REQUIRED: Explicitly set the root universe
# Without this, MCDC auto-populates root with ALL cells, causing conflicts
mcdc.simulation.set_root_universe(cells=[core_cell])
# === PHASE_GEOMETRY END ===

# ==============================================================================
# FINALIZE: SOURCE
# ==============================================================================

# === PHASE_FINALIZE START ===
# Volumetric source covering the entire core
mcdc.Source(
    x=[-core_half_width, core_half_width],
    y=[-core_half_width, core_half_width],
    isotropic=True,
    energy_group=0,
)

# ==============================================================================
# FINALIZE: TALLIES
# ==============================================================================

# Mesh tally aligned with assembly grid (1 bin per assembly)
mesh = mcdc.MeshStructured(
    x=np.linspace(-core_half_width, core_half_width, N_assemblies + 1),
    y=np.linspace(-core_half_width, core_half_width, N_assemblies + 1),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux", "fission"])

# ==============================================================================
# FINALIZE: SETTINGS
# ==============================================================================

mcdc.settings.N_particle = 1000
mcdc.settings.N_batch = 2

# Eigenmode for k-eigenvalue calculation
mcdc.settings.set_eigenmode(N_inactive=2, N_active=5)

# ==============================================================================
# FINALIZE: RUN
# ==============================================================================

mcdc.run()
# === PHASE_FINALIZE END ===
