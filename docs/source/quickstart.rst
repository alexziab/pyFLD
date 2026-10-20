Quick Start
===========

Get up and running with **pyFLD** in 5 minutes.

Installation
------------

.. code-block:: bash

    git clone https://github.com/alexziab/pyFLD.git
    cd pyFLD
    pip install .

Your first FLD calculation
---------------------------

Here's a minimal example that solves a 1D diffusion problem:

.. code-block:: python

    import pyFLD as fld
    import numpy as np
    from astropy import units as u

    # 1. Set up a 1D Cartesian grid
    xi = fld.grid.get_domain_with_ghosts(0, 1, Nx=100) * u.cm
    yi = fld.grid.get_domain_with_ghosts(0, 1, Nx=1) * u.cm
    zi = fld.grid.get_domain_with_ghosts(0, 1, Nx=1) * u.cm
    grid = fld.grid.Grid(xi, yi, zi, geometry='cartesian')

    # 2. Create a gas fluid with constant properties
    gas = fld.fluids.Gas(grid=grid)
    gas.set_density(1.0 * u.g/u.cm**3 * np.ones(grid.shape))
    gas.set_temperature(100 * u.K * np.ones(grid.shape))
    gas.enroll_kappaR(lambda f: 10.0 * u.cm**2 / u.g)
    gas.enroll_kappaP(lambda f: 1.0 * u.cm**2 / u.g)
    gas.setup()

    # 3. Create a radiative environment that contains the gas fluid
    box = fld.FLD.RadiativeEnvironment([gas])

    # 4. Set boundary conditions for the radiation field:
    # the left boundary is hot (1000 K), the right boundary is cold (100 K),
    # and the other boundaries are reflective
    E_hot = fld.tools.T_to_Er(1000 * u.K)
    E_cold = fld.tools.T_to_Er(100 * u.K)
    bounds = [fld.grid.BOUNDARY_FIXEDVALUE, fld.grid.BOUNDARY_FIXEDVALUE,
              fld.grid.BOUNDARY_REFLECTIVE, fld.grid.BOUNDARY_REFLECTIVE,
              fld.grid.BOUNDARY_REFLECTIVE, fld.grid.BOUNDARY_REFLECTIVE]
    vals = [E_hot, E_cold, 0, 0, 0, 0]
    box.declare_boundaries(bounds, vals)

    # 5. Advance the radiation field. This is the main loop of the model
    for i in range(50): box.advance(dt=1e7 * u.s)

    # 6. Check the final temperature profile:
    # It should be a curve from T_hot at the left boundary to T_cold at the right boundary.
    # Note 1: the curve can be arbitrarily steep depending on your setup, since we're 
    # transitioning through different radiative regimes in this setup. For a smoother curve,
    # try setting constant_fluxlimiter=True in the RadiativeEnvironment constructor.
    # Note 2: how well you reach the actual boundary values depends on grid resolution,
    # as they're defined at the ghost cells!

    import matplotlib.pyplot as plt
    x = grid.x1[grid.act1].to_value('cm') # the x direction
    T = box.tmp[grid.active].to_value('K') # the temperature in K, in 3D, oriented z,y,x
    plt.plot(x, T[0,0,:]) # only cell in z, y, all cells in x
    plt.xlabel("x [cm]")
    plt.ylabel("Temperature [K]")
    plt.scatter([0,1], [1000, 100], c='k', label='Boundary values')
    plt.legend()
    plt.show()

Key concepts
------------

- **Grid**: Defines the spatial domain and coordinate system (Cartesian, cylindrical, or spherical)
- **Star**: Stellar parameters for irradiation/viscosity-related calculations
- **Fluid**: Gas or dust component with density, temperature, and opacity properties
- **RadiativeEnvironment**: Container that combines grids, fluids, and performs FLD calculations
- **Boundary conditions**: Required before running ``advance()``

Next steps
----------

- See `Usage <usage.html>`_ for detailed walkthroughs of each component
- Browse `Examples <examples.html>`_ for more complex setups
- Check `Best Practices <best-practices.html>`_ for recommendations on solver settings and physics choices
