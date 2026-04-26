1. the Grid class
=================

All calculations in ``pyFLD`` are performed on a grid, which is represented by the ``Grid`` class. The class can handle up to 3D grids in Cartesian, cylindrical, or spherical coordinates, and is initialized with 3 arrays representing the cell **interfaces** in each dimension. Its call signature is:

.. code-block:: python

    fld.grid.Grid(x1i, x2i, x3i,
                  nghost=2,
                  geometry='cartesian')

The provided arrays must already include "ghost" cells, which are used to handle boundary conditions. At least 2 ghost cells are required in either side of each dimension, and the number of ghost cells must be the same in all dimensions. To help with this, the package provides a helper function to construct 1D arrays of cell interfaces with ghost cells included:

.. code-block:: python

    xi = fld.grid.get_domain_with_ghosts(xmin, xmax, 
                                         Nx=20,
                                         log=False,
                                         nghost=2)

The above function constructs a 1D array of cell interfaces from ``xmin`` to ``xmax`` with ``Nx`` cells, and then adds ``nghost`` ghost cells on either side while maintaining the same cell spacing, which can be either linear (default) or logarithmic (if ``log=True``). For example, in a 1D Cartesian setup with a domain from 1 to 10 cm and 100 cells, a typical grid initialization would look like:

.. code-block:: python

    import pyFLD as fld
    from astropy import units as u
    
    # Construct the grid
    xi = fld.grid.get_domain_with_ghosts(1, 10, Nx=100) * u.cm
    yi = fld.grid.get_domain_with_ghosts(0, 1, Nx=1) * u.cm
    zi = fld.grid.get_domain_with_ghosts(0, 1, Nx=1) * u.cm
    grid = fld.grid.Grid(xi, yi, zi, geometry='cartesian')

Notice that the returned arrays from ``get_domain_with_ghosts`` are multiplied by the appropriate units before being passed to the ``Grid`` initializer. This is required as it ensures that unit conversions are handled correctly in all subsequent calculations.

The resulting ``Grid`` object contains various attributes that are used in the calculations, such as the cell centers, cell volumes, and cell face areas, all of which are returned as ``astropy.Quantity`` objects with appropriate units. Inspired by the ``PLUTO`` code, the grid direction labels are ``x1``, ``x2``, and ``x3``, with corresponding index labels ``i``, ``j``, and ``k``, stored in C order (i.e., ``i`` is the fastest running index). Some examples:

- ``NX1``: number of cells in the ``x1`` direction (excluding ghost cells)
- ``NX1_TOT``: total number of cells in the ``x1`` direction (including ghost cells)
- ``x1``: 1D array of cell centers in the ``x1`` direction (``NX1_TOT`` elements)
- ``x3i``: 1D array of cell interfaces in the ``x3`` direction (``NX3_TOT+1`` elements)
- ``dx2``: 1D array of cell widths in the ``x2`` direction (``NX2_TOT`` elements)
- ``dV``: 3D array of cell volumes (``NX3_TOT`` x ``NX2_TOT`` x ``NX1_TOT`` elements)

The class also contains some helper slices and indices to help with using the grid arrays, such as:

- ``x1[grid.act1]``: 1D array of active cell centers in the ``x1`` direction (``NX1`` elements)
- ``x2[grid.jbeg]``: center of the first active cell in the ``x2`` direction (``float``)
- ``x3[grid.kend]``: center of the last active cell in the ``x3`` direction (``float``)
- ``dV[grid.active]``: 3D array of active cell volumes (``NX3`` x ``NX2`` x ``NX1`` elements, equivalent to ``dV[grid.kbeg:grid.kend+1, grid.jbeg:grid.jend+1, grid.ibeg:grid.iend+1]``)

You can now hold on to that ``Grid`` object, as it's one of the ingredients for the radiative transfer calculations. The next step is to initialize a `star object <usage-star.html>`_.

Notes
-----

- In curvilinear geometries, angle-related arrays (i.e., *θ*, *φ*) should *not* be defined as ``astropy.Quantity`` objects, but rather as plain floats in radians. Keeping angles dimensionless simplifies many calculations involving surface areas, volumes, etc. ``pyFLD`` will internally strip units from angle-related inputs that are passed as ``astropy.Quantity`` objects during grid initialization.
- Unless you know what you are doing, do not mix slices or indices from different directions (e.g., avoid using ``x1[grid.act2]``, ``dx2[grid.kbeg]``, etc.).
