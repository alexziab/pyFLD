Troubleshooting & FAQ
=====================

Common issues and solutions.

Solver & Convergence
--------------------

**Q: Solver doesn't converge or takes many iterations**

A: Switch to the ``rescaled ilu`` preconditioner, which often converges faster:

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas], preconditioner='rescaled ilu')

For very large problems (> 10⁸ cells), the ILU preconditioner may be expensive. Try ``jacobi`` instead:

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas], preconditioner='jacobi')

You can also adjust the solver tolerance and/or iteration limit:

.. code-block:: python

    Disk.configure_solver(rtol=1e-4, maxiter=20000)

Lower tolerance (stricter convergence) slows down the solver. Increase ``maxiter`` if needed.

**Q: advance() raises "RuntimeError: ... did not converge"**

A: The linear solver did not reach the requested tolerance within ``maxiter`` iterations (or the direct solver returned non-finite values). In that case ``advance()`` leaves the temperature and radiation field unchanged, so you can catch the error and retry, e.g. with a smaller time step or a different preconditioner:

.. code-block:: python

    try:
        Disk.advance(dt=dt)
    except RuntimeError:
        Disk.advance(dt=dt/10)

**Q: Solution oscillates or is unphysical**

A: Check your boundary conditions. The most common mistake is using the wrong boundary constant names. Use these:

.. code-block:: python

    fld.grid.BOUNDARY_FIXEDVALUE   # fixed-value boundary
    fld.grid.BOUNDARY_REFLECTIVE   # zero-gradient (reflective) boundary

Also verify that your opacities are physically reasonable (not negative or extremely large).

If, despite that, the solution still oscillates, try reducing the timestep (stable doesn't mean accurate!) or using under-relaxation, where the new and old solutions are weighed together. Calling ``advance(dt, uf=0.4)`` gives a 40% weight to the old solution and 60% to the new one, which can help stabilize things.

**Q: Solver runs but temperature doesn't change**

A: Likely causes:

1. **Opacities are zero**: Check that you enrolled the opacity functions correctly. Test with a small constant opacity if unsure.
2. **Timestep is too small**: Try increasing the time step in ``advance()`` to see if the solution evolves.

Boundary Conditions
-------------------

**Q: What are the six boundary indices?**

A: The order is: ``[lower_x1, upper_x1, lower_x2, upper_x2, lower_x3, upper_x3]``. In spherical coordinates, this corresponds to ``[inner_radial, outer_radial, inner_polar, outer_polar, lower_azimuthal, upper_azimuthal]``

In code: ``bounds = [bound_0, bound_1, bound_2, bound_3, bound_4, bound_5]``

**Q: Getting a boundary error when calling advance()**

A: Boundaries must be declared before the first call to ``advance()``:

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas])
    Disk.declare_boundaries(bounds, vals)
    Disk.advance(dt=1*u.yr)

All 6 boundaries must be declared, as ``BOUNDARY_FIXEDVALUE`` or ``BOUNDARY_REFLECTIVE``; ``declare_boundaries`` raises a ``ValueError`` if any of them is ``BOUNDARY_UNDEFINED``. For directions with a single cell, use ``BOUNDARY_REFLECTIVE``.

Opacities & Units
-----------------

**Q: Opacity import crashes or gives weird values**

A: Ensure your opacity file is in the correct format. Use ``fld.tools.parse_optool_file()`` for optool files:

.. code-block:: python

    # Correct usage
    wl, kabs, ksca, g = fld.tools.parse_optool_file('dustkappa_01um.inp')

    # Check shapes and ranges
    print(f"wl shape: {wl.shape}, range: {wl.min():.2e} to {wl.max():.2e}")
    print(f"kabs shape: {kabs.shape}, range: {kabs.min():.2e} to {kabs.max():.2e}")

If values are unexpectedly large or small, double-check the file is not corrupted or in a different format.

**Q: Unit conversion errors when setting density/temperature**

A: Plain arrays are accepted and interpreted in cgs (g/cm^3 for density, K for temperature), but it is safer to be explicit about units, so that a value in the wrong unit raises an error instead of being silently misinterpreted:

.. code-block:: python

    # works, but assumes g/cm^3
    gas.set_density(1e-10 * np.ones(grid.shape))

    # explicit, and recommended
    from astropy import units as u
    gas.set_density(1e-10 * u.g/u.cm**3 * np.ones(grid.shape))

**Q: "UnitTypeError" during advance() with irradiation**

A: An enrolled opacity function likely returns a plain number instead of a quantity with units of cm^2/g, so the optical depth is not dimensionless. Make sure enrolled opacities carry units, e.g. ``Gas.enroll_kappaPstar(lambda f: 1e-10 * u.cm**2/u.g)``.

**Q: Grid coordinates don't have the right units**

A: Length coordinates (x1, x2, x3 in Cartesian; x1 in cylindrical/spherical) are interpreted in au if given as plain arrays; passing them with explicit units is recommended. Angles must be dimensionless in radians:

.. code-block:: python

    # a spherical grid
    ri = fld.grid.get_domain_with_ghosts(0.1, 100, Nx=64) * u.au
    thi = fld.grid.get_domain_with_ghosts(0.5, np.pi-0.5, Nx=32)  # plain radians, no units!
    phii = fld.grid.get_domain_with_ghosts(0, 2*np.pi, Nx=1)  # plain radians
    grid = fld.grid.Grid(ri, thi, phii, geometry='spherical')

Keep the polar range (including ghost cells) within :math:`[0, \pi]`.

**Q: Warning "no cell interface at pi/2" when creating a spherical grid**

A: The midplane falls inside a cell, so vertical integrations (e.g. hydrostatic equilibrium) will not start exactly at the midplane. If the additional warning "no cell center at pi/2 either" appears, the midplane is not sampled by the grid at all, and quantities at ``grid.mid`` are offset from it. Choose the polar range and number of cells such that :math:`\pi/2` is a cell interface, e.g. ``get_domain_with_ghosts(np.pi/2-0.3, np.pi/2, Nx=32)`` for the top half of a disk, or an even number of cells in a range symmetric around :math:`\pi/2` for a full disk.

Memory & Performance
--------------------

**Q: Out of memory with large grids**

A: The ILU preconditioner builds a factorization which can use significant memory. Solutions:

1. Switch to a diagonal preconditioner:

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas], preconditioner='jacobi')

2. Reduce grid resolution (fewer cells)

3. Use a machine with more RAM or run on a cluster

**Q: Simulation runs slowly**

A: Profile your bottleneck:

1. **Solver (most common)**: Try different preconditioners or relax tolerance (``rtol``)
2. **Irradiation calculations**: Install ``numexpr`` for speed: ``pip install numexpr``
3. **Hydrostatic equilibrium**: Called infrequently (only when needed), not every step

For large problems, consider:

- Coarser time steps (larger ``dt``)
- Relaxed convergence criterion (larger ``rtol``, e.g., 1e-6 instead of 1e-8)
- Reusing the preconditioner between steps with ``Disk.advance(dt, recompute_preconditioner=False)``

You can also use the ``debug=True`` option in both ``Gas/Dust`` and ``RadiativeEnvironment`` to print debugging information which can help pinpoint issues.

Temperature Structure
---------------------

**Q: Disk temperature profile doesn't look right**

A: Common issues:

1. **Verify irradiation is on**: If you expect irradiation to dominate but it's not enabled, temperature will be off. Set ``irradiation='gray'`` or ``'freqdep'``.
2. **Incomplete settling to equilibrium**: Run more time steps or for longer simulation time before checking.

Frequency-dependent opacities
------------------------------

**Q: Frequency-dependent opacities causing slow simulations**

A: Use fewer wavelength bins in the star's spectrum. For example, when calling ``optool`` to create your opacity file, use the format ``-l lam_min lam_max N_lam`` to specify fewer wavelength points:

.. code-block:: bash

    optool -radmc -l 0.05 10000 60  # creates opacity file with 60 wavelength points

Initial conditions
------------------

**Q: Hydrostatic equilibrium() gives NaNs**

A: Causes:

1. **Unphysical initial density**: Very steep profiles or negative densities fail. Start with a shallower profile.
2. **Missing parameters**: For dust, you must provide ``rhogas`` and enroll a nonzero ``alpha`` (turbulent diffusion).

**Q: Values did not change after modifying a fluid's density**

A: Derived quantities (``rho_eff``, pressure, opacities, and the totals of the radiative environment) are stored arrays, which are updated by ``refresh()``, ``advance()``, and ``enforce_hydrostatic_equilibrium()``. After modifying ``rho`` directly (e.g. ``gas.rho[:] = ...``), call ``Disk.refresh_fluids()`` (or ``gas.refresh()`` for a single fluid).

**Q: Reading a snapshot gives "all fluids must share the same temperature field"**

A: ``fld.FLD.create_from_snapshot`` reads every ``fluid-*`` folder inside the snapshot directory. If the directory also contains folders from an older snapshot (e.g. ``fluid-0`` next to ``fluid-000``), remove them or write the snapshot to a fresh directory.

Getting Help
------------

If you encounter an issue not listed above:

1. Check the `framework <framework.html>`_ for theoretical background
2. Review complete examples in `examples <examples.html>`_
3. Look at the docstrings in the code: ``help(fld.fluids.Gas)`` in Python
4. Visit the GitHub repository and open an issue
