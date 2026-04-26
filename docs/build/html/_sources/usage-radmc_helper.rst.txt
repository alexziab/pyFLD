6. RADMC-3D integration
=======================

The ``fld.radmc_helper`` module provides a bridge between ``pyFLD`` and the `RADMC-3D <https://www.github.com/dullemond/radmc3d-2.0>`_ radiative transfer code. It allows you to export ``pyFLD`` grids, stars, and data to RADMC-3D format, and import RADMC-3D results back.

Overview
--------

RADMC-3D (Dullemond et al., 2012) is a full radiative transfer solver that computes detailed spectral energy distributions and images using a Monte Carlo approach. ``pyFLD`` provides fast temperature calculations via FLD, but if you need full radiative transfer results, you can:

1. Use ``pyFLD`` to compute initial temperature estimates or even just a density structure
2. Export the model to RADMC-3D format files
3. Run RADMC-3D (``mctherm``/``image``/``sed`` commands)
4. Import results back into Python with ``pyFLD`` (using ``radmc3dPy`` under the hood)

``radmc3dPy`` is a Python package included with RADMC-3D that provides tools for reading RADMC-3D output files. The ``fld.radmc_helper`` module uses it to facilitate the import of RADMC-3D results.

Workflow
--------

.. code-block:: python

    import pyFLD as fld
    import numpy as np
    from astropy import units as u

    # Step 1: Set up a pyFLD model (see examples)
    grid = fld.grid.Grid(...)
    star = fld.star.Star(...)
    dust = fld.fluids.Dust(grid=grid, star=star, ...)
    # ... configure dust, set density/temperature, enroll functions ...
    dust.setup()
    # ... run FLD to get temperature structure ... (optional)

    # Step 2: Create RADMC3D-compatible versions
    rmc_grid = fld.radmc_helper.Grid()
    rmc_grid.setup_from(grid)
    rmc_star = fld.radmc_helper.Star()
    rmc_star.setup_from(star)

    # Step 3: Write model files
    rmc_grid.write() # defaults to 'amr_grid.inp'
    rmc_star.write() # defaults to 'stars.inp'
    fld.radmc_helper.write_wavelength_array(star.wl) # 'wavelength_micron.inp'
    fld.radmc_helper.write_density_array([dust.rho[grid.active]]) # 'dust_density.binp'
    fld.radmc_helper.write_temperature_array(dust.tmp[grid.active]) # 'dust_temperature.bdat' (optional)
    fld.radmc_helper.write_opacity_handle(['mydust']) # 'dustopac.inp', expects 'dustkappa_mydust.inp' for opacities
    fld.radmc_helper.write_control_file() # 'radmc3d.inp'

    # Step 4: Run RADMC-3D from shell
    # radmc3d mctherm

    # Step 5: Read and analyze results (requires radmc3dPy)
    rmc_rho, rmc_tmp = fld.radmc_helper.read_and_format_radmc_data()

Notes
-----

- RADMC3D only supports spherical geometry; Cartesian and cylindrical grids must be converted
- Ghost cells are automatically stripped because RADMC3D uses only active cells
- Wavelengths must match between ``Star`` and opacity data for frequency-dependent calculations
- Requires ``radmc3dPy`` package to read RADMC3D output
- See `best practices <best-practices.html>`_ for recommendations on wavelength resolution

See Also
--------

- `Dullemond et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012ascl.soft02015D/abstract>`_: RADMC-3D code paper
- `RADMC-3D <https://www.github.com/dullemond/radmc3d-2.0>`_