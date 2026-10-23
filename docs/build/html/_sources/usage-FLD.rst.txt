4. Radiative environment
========================

So far we have set up the grid and star objects, and one or more fluid objects. We can now set up a radiative environment that combines all of these ingredients to perform radiative transfer calculations. This is done with the ``fld.FLD.RadiativeEnvironment`` class, which is initialized with the following call signature:

.. code-block:: python

    fld.FLD.RadiativeEnvironment(
        fluids,
        verbose=True,
        preconditioner='ilu',
        constant_fluxlimiter=False,
        implicit_viscosity=False,
        name=None,  # optional, used in log messages
        debug=False # optional, prints debugging information
    )

The only mandatory argument (and likely the only one you need to worry about) is the list of fluid objects that you want to include in the radiative transfer calculations. All fluids must share the same grid, star, and temperature field. For example, with what we have set up so far, we can initialize the radiative environment with:

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas])

This is enough to declare a protoplanetary disk that accounts for viscous heating with α viscosity, ray-traced irradiation with gray opacities, and radiative transfer with Rosseland and Planck mean opacities.

Boundary conditions
-------------------

The ``RadiativeEnvironment`` class is slightly different as most of the physics have already been set up in the fluid objects. However, we also need to prescribe boundary conditions for the radiative transfer calculations. This is done with the method:

.. code-block:: python

    Disk.declare_boundaries(bounds, vals)

The method expects two lists, 6 elements each, that specify the type of boundary condition and the corresponding value for each of the 6 boundaries (inner and outer in each of the 3 dimensions). The allowed boundary types are ``fld.grid.BOUNDARY_FIXEDVALUE`` for fixed-value boundaries, and ``fld.grid.BOUNDARY_REFLECTIVE`` for zero-gradient (reflective) boundaries. All 6 boundaries must be declared, including those of directions with a single cell (``BOUNDARY_REFLECTIVE`` is the natural choice there); leaving any boundary as ``fld.grid.BOUNDARY_UNDEFINED`` raises a ``ValueError``. The expected values are the radiation energy density :math:`E_\mathrm{r}` for fixed-value boundaries, and are not needed for reflective boundaries. For example, to set a fixed value boundary condition that corresponds to a radiation temperature of 10 K at the inner radial edge, and reflective boundaries everywhere else, we can do:

.. code-block:: python

    from astropy import units as u

    bounds = [
        fld.grid.BOUNDARY_FIXEDVALUE,  # inner radial
        fld.grid.BOUNDARY_REFLECTIVE,  # outer radial
        fld.grid.BOUNDARY_REFLECTIVE,  # inner polar
        fld.grid.BOUNDARY_REFLECTIVE,  # outer polar
        fld.grid.BOUNDARY_REFLECTIVE,  # inner azimuthal
        fld.grid.BOUNDARY_REFLECTIVE,  # outer azimuthal
    ]

    vals = [
        fld.tools.T_to_Er(10 * u.K), # Erad at inner radial edge for T=10 K
        0, 0, 0, 0, 0  # not needed for reflective boundaries
    ]

    Disk.declare_boundaries(bounds, vals)

Boundaries must be declared before the first call to ``advance()``.

Advancing in time
-----------------

We are now ready to perform radiative transfer calculations with the ``Disk`` object. A typical call to the diffusion solver looks like:

.. code-block:: python

    Disk.advance(dt=1*u.yr)

which advances the radiation field and temperature by a time step of 1 year. The method also accepts:

- ``uf``: an under-relaxation factor between 0 and 1 for the temperature update (default 0, i.e. full update). With ``uf=0.5``, the new temperature is the average of the old one and the solution of the step.
- ``recompute_preconditioner``: whether to rebuild the preconditioner before solving (default ``True``). Building the ILU preconditioner can be expensive, so if the system changes little between steps it can be worth setting this to ``False``. The previous preconditioner is then reused when available, and a new one is computed otherwise (e.g., on the first call, or after changing ``preconditioner``).

Options for the ILU factorization (``drop_tol``, ``fill_factor``) are set with ``configure_solver`` (see below), not passed to ``advance()``.

If the linear solver fails, ``advance()`` raises a ``RuntimeError`` and leaves the temperature and radiation field unchanged, so that a loop can catch the error and retry, e.g. with a smaller time step.

What is updated during a step
-----------------------------

Each call to ``advance()`` proceeds as follows:

1. **Refresh**: every fluid recomputes its effective fraction ``eps``, its effective density ``rho_eff = eps * rho``, its opacities, pressure and internal energy, and (with irradiation) its optical depths, all at the current temperature. The totals of the environment are then updated (see below). This is the same as calling ``Disk.refresh_fluids()``.
2. **Solve**: the heating terms are computed, and the implicit FLD system is built and solved for the new radiation energy density.
3. **Update**: the new temperature is computed and copied to all fluids, which then update their pressure and internal energy.

Fractions, opacities and optical depths are therefore evaluated at the temperature of the *start* of the step: after ``advance()`` returns, ``tmp``, ``Er``, ``prs`` and ``eng`` correspond to the new temperature, while ``eps``, ``rho_eff``, the opacities and the optical depths still correspond to the previous one. They are brought up to date at the start of the next ``advance()``, or on demand with:

.. code-block:: python

    Disk.refresh_fluids() # e.g. before plotting or writing a snapshot

The environment holds the total effective density, pressure, and internal energy of all fluids as arrays (``Disk.rho``, ``Disk.prs``, ``Disk.eng``). These are sums over the fluids, in which each fluid contributes with its own ``eps``, ``mu`` and ``gamma``, and they are updated by ``refresh_fluids()``, at the end of each ``advance()``, and by ``enforce_hydrostatic_equilibrium()``. If you modify a fluid directly in between (e.g. ``gas.rho[:] = ...``), call ``Disk.refresh_fluids()`` afterwards. Note that the heat capacity used by the solver is computed from the total density with ``mu`` and ``gamma`` of the first fluid.

The temperature is shared by all fluids, and is owned by the environment: after initialization it is evolved by ``advance()`` and copied to the fluids, so setting a fluid's temperature directly has no effect on the next step.

Hydrostatic equilibrium
-----------------------

The ``RadiativeEnvironment`` class also includes a method that calls ``enforce_hydrostatic_equilibrium`` for all fluids, taking into account which fluids correspond to gas or dust. It is simply called with:

.. code-block:: python

    Disk.enforce_hydrostatic_equilibrium()

Gas fluids are handled first. Since dust settles with respect to the gas, the dust fluids are then handled using the total effective gas density *after* the gas has been put in equilibrium (unless a ``rhogas`` array is passed explicitly). Fractions and opacities are not recomputed by this call; this happens at the start of the next ``advance()``.

Solver configuration
--------------------

Additional control over the solver can be achieved with the ``Disk.configure_solver`` method:

.. code-block:: python

    Disk.configure_solver(solver='bicgstab', rtol=1e-8, maxiter=10000, guess=True,
                          drop_tol=None, fill_factor=None)

- ``solver``: the name of the iterative sparse matrix solver used. Options are ``bicgstab`` (default), ``gmres``, ``cg``, ``cgs``, ``lgmres``, ``minres``, and ``qmr``, but note that the module has only really been tested with ``bicgstab``. For a direct solver, set ``preconditioner='none'`` instead (see below).
- ``rtol``: the relative tolerance for the iterative solver (default is :math:`10^{-8}`).
- ``maxiter``: the maximum number of iterations for the iterative solver (default is 10000).
- ``guess``: if ``True``, passes the current radiation energy density as an initial guess, which can be useful for speeding up convergence when performing multiple calls to ``advance`` in a time-dependent simulation. Otherwise, the solver starts with a zero-valued initial guess (default is ``True``).
- ``drop_tol``, ``fill_factor``: passed to ``scipy.sparse.linalg.spilu`` for the ``ilu`` and ``rescaled ilu`` preconditioners, and ignored by the others. If ``None`` (default), they depend on the grid: ``drop_tol=0.1, fill_factor=3`` for 3D grids (at least 5 cells in every direction), where they were 2–8× faster than the ``spilu`` defaults in tests, and the ``spilu`` defaults (``drop_tol=1e-4, fill_factor=10``) otherwise. See :doc:`troubleshooting` before changing them: settings that are fast in 3D often fail to converge in 2D. New values take effect when the preconditioner is next rebuilt; with ``advance(recompute_preconditioner=False)`` the old factorization keeps being reused.

Each call to ``configure_solver`` resets all arguments that are not passed to their default values, so pass all the settings you want to change in a single call.

Regarding the ``preconditioner`` argument in the class initializer: the default ``ilu`` option applies an incomplete LU factorization to the sparse matrix, which can significantly speed up convergence for large problems. However, for very large problems, the factorization itself can be computationally expensive and may even fail due to memory constraints; if it fails, ``pyFLD`` falls back to Jacobi preconditioning with a warning. In such cases, it may be better to set the preconditioner to ``jacobi`` (i.e., diagonal scaling) from the start.

For large grids, ``preconditioner='amg'`` uses classical algebraic multigrid (Ruge-Stüben, from the optional package ``pyamg``: ``pip install pyamg`` or ``pip install ".[amg]"``). Its iteration count stays roughly constant as the grid is refined, whereas ILU needs more iterations and a more expensive factorization. On a 2D dusty disk at dt = 1e10 yr, both took the same time at 65k cells, while AMG was 1.4× faster at 262k cells and 1.7× faster at 1M cells, with similar memory use on the larger grids. Each AMG iteration costs more than an ILU iteration, so ILU remains the better choice for small grids. Jacobi needed hundreds to thousands of iterations per step on the same problem.

In addition to the above preconditioners, the class supports "rescaled" variants of the default preconditioners, where the sparse matrix is rescaled by the inverse of its diagonal before applying the preconditioner. This is a typically cheap operation that can speed up convergence for some problems. To use the rescaled variants, simply set ``preconditioner`` to ``rescaled ilu`` or ``rescaled jacobi``. Finally, ``preconditioner='none'`` skips the iterative solver altogether and uses the direct solver ``scipy.sparse.linalg.spsolve``, which is robust but only practical for small problems.


We can now follow this workflow to set up more meaningful problems and run radiative transfer calculations on them. We will go through some examples in the `examples section <examples.html>`_.
