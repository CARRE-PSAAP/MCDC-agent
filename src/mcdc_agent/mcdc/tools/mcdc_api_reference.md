# MCDC API Reference (Quick Reference for LLM Agent)

## Materials

### MaterialMG (Multi-Group)
```python
mcdc.MaterialMG(
    capture: np.ndarray,         # [G] capture cross-section [/cm]
    scatter: np.ndarray,         # [[G,G]] scatter matrix - MUST BE 2D!
    fission: np.ndarray = None,  # [G] fission cross-section [/cm]
    nu_p: np.ndarray = None,     # [G] prompt fission neutron yield (REQUIRED if fission)
    nu_s: np.ndarray = None,     # [G] scattering multiplication
    chi_p: np.ndarray = None,    # [Gout, Gin] prompt fission spectrum
    speed: np.ndarray = None,    # [G] energy group speed [cm/s]
)
# PHYSICS RULE: For subcritical systems: fission * nu_p < capture + scatter
# DEFAULT: Prefer single-group MaterialMG unless the prompt explicitly asks for multiple energy groups
# Example: fission=0.05, nu_p=2.5, capture=0.15 -> 0.125 < 0.15 ✓
# Example: mat = mcdc.MaterialMG(capture=np.array([0.15]), scatter=np.array([[0.05]]), fission=np.array([0.05]), nu_p=np.array([2.5]))
```

### Material (Continuous Energy)
```python
mcdc.Material(
    name: str = '',
    nuclide_composition: dict[str, float],  # {'U235': 0.02, 'U238': 0.98}
)
# DEFAULT: Prefer MaterialMG unless the prompt explicitly asks for continuous-energy materials,
# named real materials, isotopic compositions, or use of the material lookup tool/service.
#
# When CE is requested:
# 1. Call the material lookup tool/service with the material name (e.g. water, uranium_dioxide, boron_carbide).
# 2. Use the returned `nuclide_composition={...}` values exactly.
# 3. Do NOT invent or guess CE number densities by hand if the tool can provide them.
#
# Example CE snippets returned by the lookup service:
# water = mcdc.Material(
#     nuclide_composition={
#         'H1': 6.700879e-02,
#         'O16': 3.342818e-02,
#     },
# )
#
# uranium_dioxide = mcdc.Material(
#     nuclide_composition={
#         'U235': 1.687584e-04,
#         'U238': 2.325563e-02,
#         'O16': 4.673745e-02,
#     },
# )
```
---

## Surfaces

### PlaneX, PlaneY, PlaneZ
```python
mcdc.Surface.PlaneX(x: float, boundary_condition: str = 'none')
mcdc.Surface.PlaneY(y: float, boundary_condition: str = 'none')
mcdc.Surface.PlaneZ(z: float, boundary_condition: str = 'none')
# Example: wall = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")
# boundary_condition: 'none' | 'vacuum' | 'reflective'
```

### CylinderX, CylinderY, CylinderZ
```python
# IMPORTANT: center is 2D - perpendicular to cylinder axis!
mcdc.Surface.CylinderX(center: [y, z], radius: float, boundary_condition: str = 'none')
mcdc.Surface.CylinderY(center: [x, z], radius: float, boundary_condition: str = 'none')  
mcdc.Surface.CylinderZ(center: [x, y], radius: float, boundary_condition: str = 'none')
# Example: pin = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.4)
# CylinderZ at origin: center=[0.0, 0.0] NOT [0,0,0]
```

### Sphere
```python
mcdc.Surface.Sphere(center: [x, y, z], radius: float, boundary_condition: str = 'none')
# Example: core = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=10.0)
# center is 3D for spheres
```

ALL surface values must be floats, not integers!

---

## Cells and Regions

### Cell
```python
mcdc.Cell(
    region: Region,              # Boolean expression of surfaces
    fill: Material | Universe,   # What fills the cell
    translation: [x, y, z] = [0, 0, 0],  # For universe fills, relative to the center of the universe
    rotation: [rx, ry, rz] = [0, 0, 0],  # Rotation angles (degrees)
)
# Example: my_cell = mcdc.Cell(region=+x0 & -x5, fill=water_mat)
# Example: my_cell = mcdc.Cell(region=+x0 & -x5, fill=assembly_universe, translation=[+10,0,0], rotation=[0,5,0])
```

### Region Operators
```python
inside = -sphere           # Inside surface (negative side)
outside = +sphere          # Outside surface (positive side)
intersection = +s1 & -s2   # AND: must satisfy both
union = region1 | region2  # OR: satisfies either
complement = ~region       # NOT: everything except region

# IMPORTANT: USE '-' for inside and '+' for outside
```

---

## Source

```python
mcdc.Source(
    x: [xmin, xmax] = None,      # Position range
    y: [ymin, ymax] = None,
    z: [zmin, zmax] = None,
    position: [x, y, z] = None,  # Point source (alternative to x,y,z)
    isotropic: bool = True,      # Isotropic emission
    energy_group: int = 0,       # For MG problems only
    time: [tmin, tmax] = 0.0,    # Time range
    probability: float = 1.0,    # Relative probability
)
# Example (Point): mcdc.Source(position=[0.0, 0.0, 0.0], isotropic=True, energy_group=0)
# Example (Volumetric): mcdc.Source(x=[0.0, 5.0], y=[-2.0, 2.0], isotropic=True, energy_group=0)
# TIP: Use small source region unless volumetric source explicitly needed
# IMPORTANT: DO NOT declare source at the boundary of the geometry, it will cause the simulation to fail
# Always ensure the entire source region is inside the geometry
# For CE problems, usually omit energy_group.
```

---

## Tallies

### Meshes
```python
mesh = mcdc.MeshUniform(
    x: (xmin, dx, Nx),  # N cells in x
    y: (ymin, dy, Ny),
    z: (zmin, dz, Nz),
)
# Example: grid = mcdc.MeshUniform(x=(0.0, 0.2, 50), y=(-5.0, 0.2, 50))
# Creates a grid from x = 0 to 10 with 50 cells, and y = -5 to 5 with 25 cells, total 1250 cells

mesh = mcdc.MeshStructured(
    x: [x0, x1, x2, ...],  # Explicit grid points
    y: [y0, y1, y2, ...],
    z: [z0, z1, z2, ...],
)
# Example: grid = mcdc.MeshStructured(x=np.linspace(0, 10, 51))
# Creates a grid from x = 0 to 10 with 51 points, total 50 cells
# RULE: Keep mesh coarse! Total cells = Nx*Ny*Nz < 10000
# N_particle >= Total_Mesh_Cells to avoid variance errors
# Use np.linspace to generate the mesh points
```

### Common Tally Parameters
All tally types support these optional parameters:
```python
scores: list[str] = ['flux'],# Score types (see below)
multipliers: list[str] = [], # e.g., ['energy'] for energy-weighted
mu: list[float] = None,      # Polar cosine bins (angular filtering)
azi: list[float] = None,     # Azimuthal angle bins
energy: list[float] = None,  # Energy bin edges
time: list[float] = None,    # Time bin edges
```

### TallyMesh
```python
mcdc.TallyMesh(
    mesh: MeshUniform | MeshStructured,
    scores: list[str] = ['flux'],
    time: list[float] = None,      # Time bin edges
    energy: list[float] = None,    # Energy bin edges  
)
# Example: mcdc.TallyMesh(mesh=grid, scores=['flux', 'fission'])
# Make sure the mesh is defined before creating the tally
```

### TallyCell
```python
mcdc.TallyCell(
    cell: Cell,           # Single cell - NOT cells= (common mistake!)
    scores: list[str] = ['flux'],
    time: list[float] = None,
    energy: list[float] = None,
)
# Example: mcdc.TallyCell(cell=my_cell, scores=['flux'])
# my_cell is a Cell object defined earlier
```

### TallySurface
```python
mcdc.TallySurface(
    surface: Surface,     # Single surface - NOT surfaces=
    scores: list[str] = ['flux'],
    time: list[float] = None,
    energy: list[float] = None,
)
# Example: mcdc.TallySurface(surface=right_wall, scores=['net-current'])
# right_wall is a Surface object defined earlier
```

### TallyGlobal
```python
mcdc.TallyGlobal(
    scores: list[str] = ['flux'],  # Tally over entire geometry
    time: list[float] = None,      # Time bin edges
    energy: list[float] = None,    # Energy bin edges
)
# Example: mcdc.TallyGlobal(scores=['flux'])
```

## Common Scores for Tallies
- 'flux' - Scalar flux
- 'density' - Particle density
- 'collision' - Collision rate
- 'fission' - Fission rate
- 'capture' - Capture rate
- 'net-current' - Net current (surface tallies)


---

## Universe & Lattice

### Universe
```python
mcdc.Universe(
    cells: list[Cell],
)
# Example: pin_univ = mcdc.Universe([fuel_cell, mod_cell])
# fuel_cell and mod_cell are Cell objects defined earlier

# to set root universe
mcdc.simulation.set_root_universe(
    cells: list[Cell],
)
# Example: mcdc.simulation.set_root_universe([core_cell])
```
# RULE: ALWAYS explicitly set the root universe with mcdc.simulation.set_root_universe()
# to define the top-level cells. Failure to do so causes MC/DC to automatically 
# populate the root universe with ALL defined cells, which creates shadowing 
# and priority issues.

### Lattice
```python
mcdc.Lattice(
    x: (start, pitch, Nx),  # Grid in x: x_min, cell_width, count
    y: (start, pitch, Ny),
    z: (start, pitch, Nz),  # Omit z for 2D lattice (infinite in z)
    universes: list[Universe],  # Matches dims: 1D=[x], 2D=[y][x], 3D=[z][y][x]
)
# Example: core_lat = mcdc.Lattice(x=(-5.0, 1.0, 10), y=(-5.0, 1.0, 10), universes=grid_2d)
# grid_2d is a list of Universe objects defined earlier (can also be defined in-line)
```
# RULE: LOCAL COORDINATES - Each lattice cell is centered at (0,0,0).
# Pin cell surfaces must span -pitch/2 to +pitch/2, NOT 0 to pitch.

# RULE: BOUNDING - Always bound the cell containing the lattice to the 
# EXACT grid dimensions: [start, start + pitch*count] to avoid lost particles.

# RULE: INDEXING - Array indexing follows visual layout:
#   - 2D [y][x]: Row 0 = max Y (top), increases downward
#   - 3D [z][y][x]: z=0 = max Z layer, y=0 = max Y row

# RULE: TRANSLATION - Place lattice via Cell translation parameter:
#   center = [pitch*N/2, pitch*N/2, 0.0]  # Lattice center in world coords
#   mcdc.Cell(region=..., fill=lattice, translation=center)

---

## Settings

```python
mcdc.settings.N_particle = 1000   # Number of particles per batch
mcdc.settings.N_batch = 2         # Number of batches
mcdc.settings.active_bank_buffer = 1000  # For fission problems
```

### Particle Count Guidelines:
- Point/small source: N_particle >= 1000
- Large source with mesh: N_particle >= Total_Mesh_Cells
- Fission problems: may need active_bank_buffer if supercritical
- For small benchmark-style test runs, prefer N_particle around 1000 unless the prompt asks otherwise

### Eigenmode (k-eigenvalue)
```python
mcdc.settings.set_eigenmode(
    N_inactive: int = 0,         # Inactive cycles (source convergence)
    N_active: int = 0,           # Active cycles (for statistics)
    k_init: float = 1.0,         # Initial k guess
    gyration_radius: str = None, # 'all', 'infinite-x', 'only-z', etc.
    save_particle: bool = False, # Save final particle bank
)
# Total cycles = N_inactive + N_active
# Use N_inactive=2 and N_active=4 unless the prompt asks otherwise
```

---

## Techniques (Variance Reduction)

```python
# Implicit capture - absorb weight instead of killing particles
mcdc.simulation.implicit_capture(active: bool = True)

# Weighted emission - control fission neutron weights
mcdc.simulation.weighted_emission(
    active: bool = True,
    weight_target: float = 1.0,  # Target weight for emitted particles
)

# Weight roulette - Russian roulette for low-weight particles
mcdc.simulation.weight_roulette(
    weight_threshold: float = 0.0,  # Kill particles below this weight
    weight_target: float = 1.0,     # Survivors get this weight
)
# RULE: weight_threshold must be < weight_target

# Population control - maintain constant particle population
mcdc.simulation.population_control(active: bool = True)
```
