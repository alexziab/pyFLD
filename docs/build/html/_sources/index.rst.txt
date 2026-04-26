.. pyFLD documentation master file, created by
   sphinx-quickstart on Sun Feb  1 20:29:07 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``pyFLD`` documentation
=================================

``pyFLD`` is a Python package for computing the flux-limited diffusion (FLD) approximation to radiative transfer in protoplanetary disks. It is designed as a computationally efficient alternative to full radiative transfer calculations, and is particularly well-suited for determining the temperature structure of protoplanetary disks while remaining capable of handling time-dependent problems.

``pyFLD`` relies heavily on the ``astropy.Quantity`` class for handling physical units as a design choice, allowing for hassle-free unit conversions and ensuring that all calculations are performed with consistent units. This (nearly) eliminates errors due to unit mismatches.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quickstart
   framework
   installation
   usage
   examples
   best-practices
   troubleshooting
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
