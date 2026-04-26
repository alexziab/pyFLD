Basic example: 1D diffusion equation
=====================================

This example demonstrates how to solve a 1D radiative diffusion problem with ``pyFLD``. Assuming constant density :math:`\rho`, opacity :math:`\kappa_\mathrm{R}=\kappa_\mathrm{P}=\kappa`, and flux limiter :math:`\lambda`, the coupled equation system reduces to a simpler form:

.. math:: \rho c_\mathrm{V} \frac{\partial T}{\partial t} = -\kappa\rho c (a_\mathrm{R} T^4 - E_\mathrm{r})
.. math:: \frac{\partial E_\mathrm{r}}{\partial t} - \frac{\lambda c}{\kappa\rho}\frac{\partial^2 E_\mathrm{r}}{\partial x^2} = \kappa\rho c (a_\mathrm{R} T^4 - E_\mathrm{r}),

which, in steady state, further reduces to:

.. math:: D \frac{\partial^2 E_\mathrm{r}}{\partial x^2} = 0,

where :math:`E` is the energy density and :math:`D` is the diffusion coefficient. The solution to this equation is a straight line connecting the boundary values of :math:`E_\mathrm{r}`. This example will show how to set up and solve this problem with ``pyFLD`` and compare the results to the expected solution.

Note that to reach steady state, we will need to run the model for several diffusion timescales. For a box of size :math:`L`, the diffusion timescale is approximately :math:`t_\mathrm{diff} \sim L^2 / D`. In this example, with :math:`L = 1\,\mathrm{cm}`, :math:`\kappa = 1\,\mathrm{cm}^2/\mathrm{g}`, :math:`\rho = 1\,\mathrm{g}/\mathrm{cm}^3`, and :math:`\lambda = 1/3`, the diffusion coefficient is :math:`D = \lambda c / (\kappa \rho) \sim 10^{10}\,\mathrm{cm}^2/\mathrm{s}`, leading to a diffusion timescale of :math:`t_\mathrm{diff} \sim 10^{-10}\,\mathrm{s}`. Here, we will just run the model for an absurdly long time to ensure we reach steady state.

.. code-block:: python

    from astropy import units as u
    import pyFLD as fld
    import numpy as np

    # grid setup
    xi = fld.grid.get_domain_with_ghosts(1, 2, Nx=100) * u.cm
    yi = fld.grid.get_domain_with_ghosts(0, 1, Nx=1) * u.cm
    zi = fld.grid.get_domain_with_ghosts(0, 1, Nx=1) * u.cm
    grid = fld.grid.Grid(xi, yi, zi, geometry='cartesian')

    # fluid setup
    Gas = fld.fluids.Gas(grid=grid)

    E0 = 1 * u.erg / u.cm**3
    rho = 1 * u.g/u.cm**3 * np.ones(grid.shape)
    tmp = fld.tools.Er_to_T(E0) * np.ones(grid.shape)
    Gas.set_density(rho)
    Gas.set_temperature(tmp)
    Gas.enroll_kappaR(lambda f: 1 * u.cm**2 / u.g)
    Gas.enroll_kappaP(lambda f: 1 * u.cm**2 / u.g)
    Gas.setup()

    # radiative environment setup
    Box = fld.FLD.RadiativeEnvironment([Gas], verbose=True, debug=False, constant_fluxlimiter=True)

    # boundary conditions
    val, ref = fld.grid.BOUNDARY_FIXEDVALUE, fld.grid.BOUNDARY_REFLECTIVE
    bounds = [val, val, ref, ref, ref, ref]
    vals   = [E0, 2*E0, 0, 0, 0, 0]
    Box.declare_boundaries(bounds, vals)

    # time integration to steady state
    for _ in range(5): Box.advance(dt=1e10 * u.s)

    # visualization
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(ncols=2, figsize=(8, 3), dpi=120)

    # compute expected solution
    x = grid.x1[grid.act1].to_value('cm')
    xstart, xend = grid.x1[grid.ibeg-1].to('cm'), grid.x1[grid.iend+1].to('cm')
    Estart, Eend = Box.boundary_val[0], Box.boundary_val[1]
    Eexp = Estart + (Eend-Estart)/(xend-xstart) * (x*u.cm - xstart)
    Eexp = Eexp.to_value(E0)

    Er_ = Box.Er[grid.active][0,0,:].to_value(E0) # [z=0, y=0, x=all] — only one active cell in y and z
    ax[0].plot(x, Er_, label='pyFLD')
    ax[0].plot(x, Eexp, label='expectation', ls=':', lw=2)

    ax[1].plot(x, np.gradient(Er_, x, edge_order=2), label='pyFLD')
    ax[1].plot(x, np.gradient(Eexp, x, edge_order=2), label='expectation', ls=':', lw=2)

    ax[0].legend()
    for axis in ax: axis.set_xlabel(r'$x$ [cm]')
    ax[0].set_ylabel(r'$E_\mathrm{rad} / E_0$')
    ax[1].set_ylabel(r'$\mathrm{d}E_\mathrm{rad}/\mathrm{d}x$ [$E_0$/cm]')

    fig.tight_layout()
    plt.show()
