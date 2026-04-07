"""DEEPLY ANNOTATED EXAMPLE: Complex CSG Without Hierarchy (L-Shaped Streaming Duct)
Demonstrates: Complex CSG with multi-segment union, complement, and void streaming.
Complexity: complex
Distinct from benchmarks: L-shaped void channel (not T-shaped), 3D problem with reflective z-base.
Keywords: box, plane, void, channel, duct, streaming, union, complement, shielding, absorber, source, tally, csg

This example demonstrates complex Constructive Solid Geometry (CSG) operations
WITHOUT using universes or lattices. It models neutron streaming through an
L-shaped void channel carved into a shielding block.

Similar to the Kobayashi benchmark, this shows how to:
- Build complex shapes using UNION, INTERSECTION, and COMPLEMENT
- Create connected void channels using overlapping regions
- Carve voids from solid materials

================================================================================
KEY PATTERNS TO FOLLOW:
================================================================================

1. CONNECTED REGIONS MUST OVERLAP SPATIALLY
   If two void segments should connect, their coordinates must share space.
   - WRONG: segment_v y=[0,3], segment_h y=[3,6] - they only TOUCH at y=3
   - RIGHT: segment_v y=[0,3], segment_h y=[0,6] - they OVERLAP at y=[0,3]
   Particles need actual overlap to flow between regions.

2. USE UNION (|) FOR MULTI-SEGMENT CHANNELS
   void_channel = segment_v_region | segment_h_region
   A particle in EITHER region is in the void channel.

3. USE COMPLEMENT FOR CARVING VOIDS FROM SOLID
   Build the void region first, then:
   shield_region = bounding_box & ~void_channel
   This gives everything inside bounding_box EXCEPT the void channel.

4. BUILD COMPLEX SHAPES FROM BASIC SURFACES, NOT FROM OTHER REGIONS
   The complement operator (~) works reliably on simple regions.
   For complex multi-surface regions, build step by step.

5. AVOID OVERLAPPING PHYSICAL CELLS
   Don't create a "detector cell" that occupies the same space as another cell.
   Use Mesh Tallies instead to measure flux in specific regions.

6. CYLINDERS EXTEND INFINITELY ALONG THEIR AXIS
   If using CylinderZ, you MUST bound it with PlaneZ surfaces on both ends,
   otherwise particles can escape along the infinite cylinder.

================================================================================
GEOMETRY DESCRIPTION:
================================================================================

   Y
   ^
   |     Shielding Block (10 x 10 x 10 cm)
   |     with L-shaped void channel
   |
   |  +---------------------------+  y=10
   |  |                           |
   |  |     SHIELD MATERIAL       |
   |  |                           |
   |  |   +---------+             |  y=6
   |  |   |         |             |
   |  |   |  VOID   |  Horizontal |
   |  |   |  (H)    |  segment    |
   |  +---+---------+-------------+  y=3
   |  |   |         |             |
   |  | V |  VOID   |   SHIELD    |
   |  | O |  (V+H)  |             |
   |  | I |  overlap|             |
   |  | D |         |             |
   |  +---+---------+-------------+  y=0
   |  x=0 x=3       x=10
   +--------------------------------> X

The void channel has two segments:
- Vertical segment (V): x=[0,3], y=[0,10], z=[0,10]  (source at bottom)
- Horizontal segment (H): x=[0,10], y=[0,6], z=[0,3] (exits at right)

CRITICAL: Segments overlap at x=[0,3], y=[0,6], z=[0,3] for connectivity!

================================================================================
"""

# === PHASE_SETUP START ===
import numpy as np
import mcdc

# ==============================================================================
# SETUP: MATERIALS
# ==============================================================================

# Shield material - high scattering, moderate absorption
shield = mcdc.MaterialMG(
    capture=np.array([0.10]),
    scatter=np.array([[0.80]]),
)

# Void material - near-vacuum (very low interaction)
# Using small but non-zero cross sections for numerical stability
void_mat = mcdc.MaterialMG(
    capture=np.array([0.0001]),
    scatter=np.array([[0.001]]),
)

# ==============================================================================
# SETUP: SURFACES - BOUNDARIES
# ==============================================================================
# Problem boundaries with appropriate boundary conditions

# X boundaries
x_min = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
x_max = mcdc.Surface.PlaneX(x=10.0, boundary_condition="vacuum")

# Y boundaries  
y_min = mcdc.Surface.PlaneY(y=0.0, boundary_condition="vacuum")
y_max = mcdc.Surface.PlaneY(y=10.0, boundary_condition="vacuum")

# Z boundaries - reflective on bottom (symmetry), vacuum on top
z_min = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="reflective")
z_max = mcdc.Surface.PlaneZ(z=10.0, boundary_condition="vacuum")

# ==============================================================================
# SETUP: SURFACES - INTERNAL (for void channel segments)
# ==============================================================================
# These surfaces define the internal geometry of the void channel.
# They have NO boundary condition (particles pass through freely).

# Vertical segment boundaries
x_vert = mcdc.Surface.PlaneX(x=3.0)  # Right edge of vertical segment

# Horizontal segment boundaries
y_horiz = mcdc.Surface.PlaneY(y=6.0)  # Top edge of horizontal segment
z_horiz = mcdc.Surface.PlaneZ(z=3.0)  # Top of horizontal segment (in z)
# === PHASE_SETUP END ===

# === PHASE_GEOMETRY START ===
# ==============================================================================
# GEOMETRY: REGIONS (CSG Operations)
# ==============================================================================

# Bounding box region (entire problem domain)
bounding_box = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max

# -----------------------------------------------------------------------------
# VERTICAL SEGMENT: x=[0,3], y=[0,10], z=[0,10]
# -----------------------------------------------------------------------------
# Runs from bottom to top of the problem
# Note: We use the boundary surfaces (x_min, y_min, etc.) as walls

segment_v_region = (
    +x_min & -x_vert &    # x in [0, 3]
    +y_min & -y_max &     # y in [0, 10] (full height)
    +z_min & -z_max       # z in [0, 10] (full depth)
)
# RULE 1: All 3 axes explicitly bounded — x by x_min/x_vert, y by y_min/y_max, z by z_min/z_max

# -----------------------------------------------------------------------------
# HORIZONTAL SEGMENT: x=[0,10], y=[0,6], z=[0,3]
# -----------------------------------------------------------------------------
# Runs from left to right at the bottom portion
# RULE 2: y starts at y_min (NOT y_3) so it OVERLAPS the vertical segment
#   → they share the face x=[0,3], y=[0,6] (2D area), allowing particle flow

segment_h_region = (
    +x_min & -x_max &     # x in [0, 10] (full width) — RULE 1: x axis bounded
    +y_min & -y_horiz &   # y in [0, 6]
    +z_min & -z_horiz     # z in [0, 3]
)

# -----------------------------------------------------------------------------
# COMBINED VOID CHANNEL: UNION of both segments
# -----------------------------------------------------------------------------
# A particle is in the void if it's in EITHER segment
# The segments overlap at x=[0,3], y=[0,6], z=[0,3] ensuring connectivity (RULE 2)

void_channel_region = segment_v_region | segment_h_region

# -----------------------------------------------------------------------------
# SHIELD REGION: Everything in bounding box EXCEPT the void channel
# -----------------------------------------------------------------------------
# Using complement (~) to carve out the void

shield_region = bounding_box & ~void_channel_region

# ==============================================================================
# GEOMETRY: CELLS
# ==============================================================================
# Create cells from regions - each region gets exactly one cell
# IMPORTANT: Cells should NOT overlap spatially

void_cell = mcdc.Cell(region=void_channel_region, fill=void_mat)
shield_cell = mcdc.Cell(region=shield_region, fill=shield)
# PATTERN: For pure CSG geometry (no Universe/Lattice), do NOT call set_root_universe
# === PHASE_GEOMETRY END ===

# === PHASE_FINALIZE START ===
# ==============================================================================
# FINALIZE: SOURCE
# ==============================================================================
# Volumetric source at the bottom of the vertical segment
# Neutrons stream up through the void and scatter in the shield

mcdc.Source(
    x=[0.0, 3.0],     # Inside vertical segment
    y=[0.0, 1.0],     # Bottom portion only
    z=[0.0, 10.0],    # Full z extent
    isotropic=True,
    energy_group=0,
)

# ==============================================================================
# FINALIZE: TALLIES
# ==============================================================================
# Mesh tally to visualize the streaming pattern
# Using moderate resolution to keep computation reasonable

mesh = mcdc.MeshStructured(
    x=np.linspace(0.0, 10.0, 11),   # 10 bins in x
    y=np.linspace(0.0, 10.0, 11),   # 10 bins in y
    z=np.array([0.0, 5.0, 10.0]),   # 2 bins in z (bottom half, top half)
)
mcdc.TallyMesh(mesh=mesh, scores=["flux"])

# ==============================================================================
# FINALIZE: SETTINGS
# ==============================================================================

mcdc.settings.N_particle = 1000
mcdc.settings.N_batch = 2

# Fixed-source mode (no eigenvalue calculation)
# Do NOT use eigenmode or population_control for this problem

# ==============================================================================
# FINALIZE: RUN
# ==============================================================================

mcdc.run()
# === PHASE_FINALIZE END ===
