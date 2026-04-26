Usage
===========

This set of pages will give you a walkthrough of how to use the various tools available in **pyFLD** before running your first radiative transfer calculations.

Importing the package
---------------------

After `installation <installation.html>`_, you can simply import the package in a Python environment. In the following examples we will assume you have imported the package with the alias ``fld``:

.. code-block:: python

    import pyFLD as fld

You can now access all the tools available in the package, which are organized into submodules each containing one or more classes needed to set up and run a radiative transfer calculation. The sections below walk through the main classes in each submodule. For more detailed examples, see the `examples section <examples.html>`_.

.. toctree::
   :maxdepth: 1
   :caption: walkthrough:

   usage-tools
   usage-grid
   usage-star
   usage-fluid
   usage-FLD
   usage-io
   usage-radmc_helper
