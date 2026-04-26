Installation
============

This page covers how to install the **pyFLD** package.

Requirements
------------

**Required:**

- Python 3.9 or later
- NumPy
- Astropy
- SciPy

**Optional:**

- `numexpr <https://numexpr.readthedocs.io>`_ — strongly recommended if you intend to use stellar irradiation; speeds up the irradiation-related computations significantly. Install with ``pip install numexpr``.
- `optool <https://github.com/cdominik/optool>`_ — a command-line tool for computing dust opacities. Useful for generating frequency-dependent opacity tables in RADMC3D format, which can be read directly by ``fld.tools.parse_optool_file``.

Installing via git
------------------

You can install the package via GitHub with:

.. code-block:: bash

    git clone https://github.com/alexziab/pyFLD.git
    cd pyFLD
    pip install .

This will make ``pyFLD`` available for import in your Python environment.