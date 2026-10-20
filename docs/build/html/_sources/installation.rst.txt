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

This will make ``pyFLD`` available for import in your Python environment. Note that this installs a copy of the package: after pulling updates (or editing the source), run ``pip install .`` again, otherwise scripts and notebooks outside the repository keep importing the old version. You can check which version is imported with ``print(fld.__file__, fld.__version__)``.