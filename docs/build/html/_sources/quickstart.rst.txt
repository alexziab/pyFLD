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
    gas.enroll_kappaR(lambda f: 1.0 * u.cm**2 / u.g)
    gas.enroll_kappaP(lambda f: 1.0 * u.cm**2 / u.g)
    gas.setup()

    # 3. Create a radiative environment
    box = fld.FLD.RadiativeEnvironment([gas])

    # 4. Set boundary conditions
    E_hot = fld.tools.T_to_Er(1000 * u.K)
    E_cold = fld.tools.T_to_Er(100 * u.K)
    bounds = [fld.grid.BOUNDARY_FIXEDVALUE, fld.grid.BOUNDARY_FIXEDVALUE,
              fld.grid.BOUNDARY_REFLECTIVE, fld.grid.BOUNDARY_REFLECTIVE,
              fld.grid.BOUNDARY_REFLECTIVE, fld.grid.BOUNDARY_REFLECTIVE]
    vals = [E_hot, E_cold, 0, 0, 0, 0]
    box.declare_boundaries(bounds, vals)

    # 5. Advance the radiation field
    for i in range(10):
        box.advance(dt=1e6 * u.s)
        print(f"Step {i+1}: T_mid = {fld.tools.Er_to_T(box.Er[grid.active][0,0,50]):.1f}")

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
