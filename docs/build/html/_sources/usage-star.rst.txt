2. the Star class
=================

As ``pyFLD`` was designed with gaseous and dusty media around stars (e.g., protoplanetary disks), a star object is usually necessary when using the module. The ``fld.star.Star`` class holds stellar properties such as mass, radius, and luminosity, and is initialized with the following call signature:

.. code-block:: python

    fld.star.Star(
        M=<Quantity 1. solMass>,
        L=<Quantity 1. solLum>,
        T=<Quantity 5772. K>,
        wl=None,
    )

Providing the stellar mass ``M``, luminosity ``L``, and effective temperature ``T`` is sufficient to initialize the star. The stellar radius is automatically derived from the Stefan–Boltzmann relation :math:`L = 4\pi R^2 \sigma T^4`, and the gravitational parameter ``GM = G * M`` is also set. These are stored as ``star.R`` and ``star.GM``, respectively.

All arguments accept either plain floats in their natural units (``M`` in solar masses, ``L`` in solar luminosities, ``T`` in Kelvin) or ``astropy.Quantity`` objects with compatible units:

.. code-block:: python

    from astropy import units as u

    star = fld.star.Star(
        M=0.5 * u.M_sun,
        L=0.5 * u.L_sun,
        T=4000 * u.K,
    )

Frequency-dependent spectrum
-----------------------------

The optional ``wl`` argument allows you to specify a wavelength array (in cm by default, but any length unit is accepted) over which the star's spectral luminosity will be sampled. This is required when using frequency-dependent irradiation (``irradiation='freqdep'`` in the fluid setup):

.. code-block:: python

    wl, kabs, ksca, g_asy = fld.tools.parse_optool_file('dustkappa_01um.inp')
    star = fld.star.Star(M=1*u.M_sun, L=1*u.L_sun, T=5772*u.K, wl=wl)

When ``wl`` is provided, the class computes ``star.L_wl``, the spectral luminosity sampled at each wavelength bin such that their sum equals the total luminosity ``L``. The number of bins is stored as ``star.nbins``.

With both the star and grid initialized, we can now move on to creating `fluid objects <usage-fluid.html>`_.
