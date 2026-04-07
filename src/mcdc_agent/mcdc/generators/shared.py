"""Shared generator constants used by the active small-model pipeline."""

ERROR_HINTS = {
    "bank is full": (
        "HINT: A particle bank filled faster than particles could be processed. "
        "For multigroup material values, first check whether the system is too "
        "multiplying: prefer fission * nu_p < capture + scatter unless the prompt "
        "explicitly asks for eigenvalue, critical, or supercritical behavior. "
        "To fix this, reduce fission or nu_p, or increase capture/scatter. "
        "If the problem really is eigenmode or otherwise multiplying, you may have to reduce the multiplication factor to its only "
        "mildly multiplying, then enable mcdc.simulation.population_control(active=True). "
        "If the bank can still spike, increase mcdc.settings.active_bank_buffer. "
        "Do not rely on buffers alone to rescue a strongly supercritical multigroup model."
    ),
    "particle is lost": (
        "HINT: Particles are escaping the geometry. Check: "
        "1) That there are no gaps in the geometry and all particle spawn and stay inside cells."
        "2) Lattice dimensions match the containing cell exactly. "
        "   Lattice x=[min, pitch, N] covers: min to min + pitch*N. "
        "3) Universe array is exactly NxM for N x-cells and M y-cells. "
        "4) All cells within universes cover the full pin cell pitch. "
        "5) No gaps between cells - use +surface and -surface correctly."
    ),
    "blankout fn for report_lost": (
        "HINT: There are gaps in the geometry. "
        "Check that all particles spawn and stay inside a cell. "
        "Ensure the source region is completely covered by cells."
    ),
    "missing 1 required positional argument": (
        "HINT: A required parameter is missing. Check the API reference for "
        "the exact parameters required. For TallyCell, you need: cell=<cell_object>. "
        "For TallyMesh, you need: mesh=<mesh_object>."
    ),
    "unexpected keyword argument": (
        "HINT: You're using a parameter that doesn't exist in MCDC. "
        "Check the API reference for valid parameters. "
        "Common mistakes: 'space', 'particle', 'angle' are NOT valid Source parameters. "
        "Use: x=[], y=[], z=[], isotropic=True, energy_group=0"
    ),
    "'tally'": (
        "CRITICAL: There is NO mcdc.Tally class! Use specific tally types instead:\n"
        "  mcdc.TallyCell(cell=cell_object, scores=['flux'])\n"
        "  mcdc.TallyMesh(mesh=mesh_object, scores=['flux'])\n"
        "  mcdc.TallyGlobal(scores=['flux'])\n"
        "Replace mcdc.Tally(...) with one of these."
    ),
    "has no attribute": (
        "HINT: You're calling a function/class that doesn't exist. "
        "MCDC uses PascalCase: mcdc.Source, mcdc.Cell, mcdc.MaterialMG. "
        "COMMON MISTAKES: mcdc.Tally (use TallyCell/TallyMesh/TallyGlobal), "
        "mcdc.source (use mcdc.Source), mcdc.output (doesn't exist). "
        "There is NO mcdc.tally.add() or mcdc.source.add()."
    ),
    "nu_p or nu_d for fissionable": (
        "HINT: When using fission cross-section, you MUST also provide nu_p (neutrons per fission). "
        "Add: nu_p=np.array([2.5]) alongside the fission parameter."
    ),
    "translation must be": (
        "HINT: Translation must be a numpy array, not a list. "
        "Use: translation=np.array([x, y, z]) instead of translation=[x, y, z]"
    ),
    "bad operand type for unary -": (
        "HINT: REGION OPERATOR ERROR - The unary minus (-) only works on Surface objects, NOT Region objects. "
        "You CANNOT do: -region where region = (+surface1 & -surface2). "
        "Instead, rebuild from base surfaces. Example fix: "
        "WRONG: moderator_cell = mcdc.Cell(region=-diamond_region & -sphere, fill=moderator) "
        "RIGHT: moderator_cell = mcdc.Cell(region=(+cylinder1 | -cylinder2) & -sphere, fill=moderator) "
        "Remember: ~region also doesn't work. You must rebuild from surfaces."
    ),
    "timeout": (
        "HINT: TIMEOUT - likely supercritical reaction or geometry loop. "
        "For invented 1-group fission materials, prefer: fission * nu_p < capture + scatter. "
        "Also check: 1) At least one boundary is 'vacuum' (not all reflective), "
        "2) No geometry gaps causing infinite tracking. "
        "Try reducing fission or nu_p, or increasing capture/scatter."
    ),
    "universes array shape": (
        "HINT: LATTICE ERROR - Universe array dimensions don't match lattice spec. "
        "For mcdc.Lattice(x=[min, pitch, Nx], y=[min, pitch, Ny]), "
        "the universes array must be exactly Ny rows x Nx columns. "
        "Example: x has 3 cells, y has 2 cells -> universes=[[u1,u2,u3],[u4,u5,u6]]"
    ),
    "cannot reshape": (
        "HINT: MESH DIMENSION ERROR - Check MeshStructured/MeshUniform arrays. "
        "Every axis array must have 2+ elements (bin BOUNDARIES, not centers). "
        "WRONG: z=np.array([1.0]) creates 0 bins and fails. "
        "RIGHT: z=np.array([0.0, 2.0]) creates 1 bin, or omit z entirely for 2D mesh. "
        "For 2D mesh tally, either omit z OR provide exactly 2 boundary values."
    ),
}
