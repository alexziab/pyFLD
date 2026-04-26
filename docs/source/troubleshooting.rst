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

You can also adjust solver tolerance and iteration limit:

.. code-block:: python

    Disk.configure_solver(rtol=1e-6, maxiter=20000)

Lower tolerance (stricter convergence) slows down the solver. Increase ``maxiter`` if needed.

**Q: Solution oscillates or is unphysical**

A: Check your boundary conditions. The most common mistake is using the wrong boundary constant names. Use these:

.. code-block:: python

    fld.grid.BOUNDARY_FIXEDVALUE   # fixed-value boundary
    fld.grid.BOUNDARY_REFLECTIVE   # zero-gradient (reflective) boundary

Also verify that your opacities are physically reasonable (not negative or extremely large).

**Q: Solver runs but temperature doesn't change**

A: Likely causes:

1. **Opacities are zero**: Check that you enrolled the opacity functions correctly. Test with a small constant opacity if unsure.
2. **Timestep is too small**: Try increasing the time step in ``advance()`` to see if the solution evolves.

Boundary Conditions
-------------------

**Q: What are the six boundary indices?**

A: The order is: ``[inner_radial, outer_radial, inner_polar, outer_polar, inner_azimuthal, outer_azimuthal]``

In code: ``bounds = [bound_0, bound_1, bound_2, bound_3, bound_4, bound_5]``

**Q: Getting boundary error despite calling declare_boundaries()**

A: Make sure you're calling it *before* ``setup()``:

.. code-block:: python

    # WRONG
    Disk.setup()  # This checks for boundaries
    Disk.declare_boundaries(bounds, vals)

    # CORRECT
    Disk.declare_boundaries(bounds, vals)
    Disk.setup()

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

A: Always be explicit about units:

.. code-block:: python

    # WRONG – will fail
    gas.set_density(1e-10 * np.ones(grid.shape))

    # CORRECT
    from astropy import units as u
    gas.set_density(1e-10 * u.g/u.cm**3 * np.ones(grid.shape))

**Q: Grid coordinates don't have the right units**

A: Length coordinates (x1, x2, x3 in Cartesian; x1 in cylindrical/spherical) must have units. Angles must be dimensionless in radians:

.. code-block:: python

    # CORRECT spherical grid
    ri = fld.grid.get_domain_with_ghosts(0.1, 100, Nx=64) * u.au
    thi = fld.grid.get_domain_with_ghosts(0.5, np.pi, Nx=32)  # plain radians, no units!
    phii = fld.grid.get_domain_with_ghosts(0, 2*np.pi, Nx=1)  # plain radians
    grid = fld.grid.Grid(ri, thi, phii, geometry='spherical')

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

.. code-block:: python

    optool -radmc -l 0.05 10000 60  # creates opacity file with 60 wavelength points

Initial conditions
------------------

**Q: Hydrostatic equilibrium() gives NaNs**

A: Causes:

1. **Unphysical initial density**: Very steep profiles or negative densities fail. Start with a shallower profile.
2. **Missing parameters**: For dust, you must provide ``rhogas`` and enroll a nonzero ``alpha`` (turbulent diffusion).

Getting Help
------------

If you encounter an issue not listed above:

1. Check the `framework <framework.html>`_ for theoretical background
2. Review complete examples in `examples <examples.html>`_
3. Look at the docstrings in the code: ``help(fld.fluids.Gas)`` in Python
4. Visit the GitHub repository and open an issue
