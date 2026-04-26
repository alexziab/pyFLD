4. Radiative environment
========================

So far we have set up the grid and star objects, and one or more fluid objects. We can now set up a radiative environment that combines all of these ingredients to perform radiative transfer calculations. This is done with the ``fld.FLD.RadiativeEnvironment`` class, which is initialized with the following call signature:

.. code-block:: python

    fld.FLD.RadiativeEnvironment(
        fluids,
        verbose=True,
        constant_fluxlimiter=False,
        preconditioner='ilu',
        implicit_viscosity=False,
        debug=False
    )

The only mandatory argument (and likely the only one you need to worry about) is the list of fluid objects that you want to include in the radiative transfer calculations. For example, with what we have set up so far, we can initialize the radiative environment with:

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas])

This is enough to declare a protoplanetary disk that accounts for viscous heating with α viscosity, ray-traced irradiation with gray opacities, and radiative transfer with Rosseland and Planck mean opacities.

The ``RadiativeEnvironment`` class is slightly different as most of the physics have already been set up in the fluid objects. However, we also need to prescribe boundary conditions for the radiative transfer calculations. This is done with the method:

.. code-block:: python

    Disk.declare_boundaries(bounds, vals)

The method expects two lists, 6 elements each, that specify the type of boundary condition and the corresponding value for each of the 6 boundaries (inner and outer in each of the 3 dimensions). The allowed boundary types are ``fld.grid.BOUNDARY_FIXEDVALUE`` for fixed-value boundaries, and ``fld.grid.BOUNDARY_REFLECTIVE`` for zero-gradient (reflective) boundaries. The expected values are the radiation energy density :math:`E_\mathrm{r}` for fixed-value boundaries, and are not needed for reflective boundaries. For example, to set a fixed value boundary condition that corresponds to a radiation temperature of 10 K at the inner radial edge, and reflective boundaries everywhere else, we can do:

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

We are now ready to perform radiative transfer calculations with the ``Disk`` object. A typical call to the diffusion solver looks like:

.. code-block:: python

    Disk.advance(dt=<Quantity 1. yr>)

which advances the radiation field by a time step of 1 year.

The ``RadiativeEnvironment`` class also includes a method that calls ``compute_hydrostatic_equilibrium`` for all fluids and then applies the resulting density profiles, taking into account which fluids correspond to gas or dust. It is simply called with:

.. code-block:: python

    Disk.enforce_hydrostatic_equilibrium()

Additional control over the solver can be achieved with the ``Disk.configure_solver`` method, which accepts keyword arguments that are passed to the underlying solver:

- ``solver``: the name of the sparse matrix solver used. Options are ``bicgstab`` (default), ``spsolve``, ``gmres``, ``cg``, ``cgs``, ``lgmres``, ``minres``, and ``qmr``, but note that the module has only really been tested with ``bicgstab`` and sometimes ``spsolve``.
- ``rtol``: the relative tolerance for the iterative solver (default is :math:`10^{-8}`).
- ``maxiter``: the maximum number of iterations for the iterative solver (default is 10000).
- ``guess``: if ``True``, passes the current radiation energy density as an initial guess, which can be useful for speeding up convergence when performing multiple calls to ``advance`` in a time-dependent simulation. Otherwise, the solver starts with a zero-valued initial guess (default is ``True``).

Regarding the ``preconditioner`` argument in the class initializer: the default ``ilu`` option applies an incomplete LU factorization to the sparse matrix, which can significantly speed up convergence for large problems. However, for very large problems, the factorization itself can be computationally expensive and may even fail due to memory constraints. In such cases, it may be better to set the preconditioner to ``jacobi`` (i.e., diagonal scaling).

In addition to the above preconditioners, the class supports "rescaled" variants of the default preconditioners, where the sparse matrix is rescaled by the inverse of its diagonal before applying the preconditioner. This is a typically cheap operation that can speed up convergence for some problems. To use the rescaled variants, simply set ``preconditioner`` to ``rescaled ilu`` or ``rescaled jacobi``.


We can now follow this workflow to set up more meaningful problems and run radiative transfer calculations on them. We will go through some examples in the `examples section <examples.html>`_.
