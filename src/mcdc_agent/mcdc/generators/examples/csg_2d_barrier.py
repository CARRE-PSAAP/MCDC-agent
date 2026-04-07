"""Example: 2D Multi-Room Shielding with Barrier
Demonstrates: 2D geometry with multiple rooms and absorbing barrier.
Complexity: complex
Distinct from benchmarks: 2 rooms + L-shaped barrier, source in corner, reflective symmetry boundaries.
Keywords: box, plane, barrier, shielding, absorber, moderator, complement, source, tally, reflective, union

PATTERNS DEMONSTRATED:
- 2D geometry using x and y planes only
- Multiple cells partitioning 2D space
- Reflective boundaries for symmetry
- Volumetric source in a sub-region (corner)
- 2D MeshUniform tally
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# =============================================================================
# Materials
# =============================================================================

# PATTERN: Room material — moderate scattering
room_mat = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.15]]),
)

# PATTERN: Barrier material — high absorption
barrier_mat = mcdc.MaterialMG(
    capture=np.array([0.5]),
    scatter=np.array([[0.4]]),
)

# =============================================================================
# Geometry
# =============================================================================

# PATTERN: 2D geometry — reflective on 2 sides (symmetry), vacuum on 2 sides
x0 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")
x4 = mcdc.Surface.PlaneX(x=4.0)   # Room-barrier interface
x5 = mcdc.Surface.PlaneX(x=5.0)   # Barrier-room interface
x10 = mcdc.Surface.PlaneX(x=10.0, boundary_condition="vacuum")

y0 = mcdc.Surface.PlaneY(y=0.0, boundary_condition="reflective")
y5 = mcdc.Surface.PlaneY(y=5.0)   # Room divider
y10 = mcdc.Surface.PlaneY(y=10.0, boundary_condition="vacuum")
# === PHASE_SETUP END ===

# === PHASE_GEOMETRY START ===
# PATTERN: Multiple cells covering entire 2D domain
# Source room (bottom-left)
mcdc.Cell(region=+x0 & -x4 & +y0 & -y5, fill=room_mat)

# Upper-left room
mcdc.Cell(region=+x0 & -x4 & +y5 & -y10, fill=room_mat)

# Barrier (vertical wall)
mcdc.Cell(region=+x4 & -x5 & +y0 & -y10, fill=barrier_mat)

# Right room (bottom)
mcdc.Cell(region=+x5 & -x10 & +y0 & -y5, fill=room_mat)

# Right room (top)
mcdc.Cell(region=+x5 & -x10 & +y5 & -y10, fill=room_mat)
# === PHASE_GEOMETRY END ===

# === PHASE_FINALIZE START ===
# =============================================================================
# Source
# =============================================================================

# PATTERN: Volumetric source in a corner sub-region
mcdc.Source(
    x=[0.0, 1.0],
    y=[0.0, 1.0],
    isotropic=True,
    energy_group=0,
)

# =============================================================================
# Tallies and Settings
# =============================================================================

# PATTERN: 2D mesh tally covering entire geometry
mesh = mcdc.MeshUniform(
    x=(0.0, 0.5, 20), # start, step, num_intervals
    y=(0.0, 0.5, 20),
)
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

mcdc.settings.N_particle = 1000
mcdc.settings.N_batch = 2

mcdc.run()
# === PHASE_FINALIZE END ===
