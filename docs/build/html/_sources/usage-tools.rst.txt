0. Various tools
=================

The ``fld.tools`` module contains utility functions that are not specific to any particular fluid type but are generally useful when working with the package. Highlights include:

**Opacity utilities:**

- ``parse_optool_file(filename)``: Parses a file containing frequency-dependent opacities computed with `optool <https://github.com/cdominik/optool>`_ and returns its contents as ``astropy.Quantity`` objects (wavelengths, absorption opacity, scattering opacity, and asymmetry parameter).
- ``compute_mean_opacities(wl, kabs, ksca, g, T=None, ...)``: Computes Rosseland and Planck mean opacities from wavelength-dependent opacity data by integrating over the Planck function. This function is called automatically under the hood when frequency-dependent opacity data are provided to a fluid object, but it can also be useful on its own.

**Radiation and temperature conversion:**

- ``T_to_Er(T)``: Converts temperature to radiation energy density using :math:`E_\mathrm{r} = a_\mathrm{R} T^4`.
- ``Er_to_T(Er)``: The inverse of the above — converts radiation energy density to temperature.

**Initial/boundary condition helpers:**

- ``get_tau0_Flock_2019(fluid)``: Returns the gray optical depth between the stellar surface and the inner radial boundary of the grid, following the prescription of Flock et al. (2019). Intended to be passed to ``fluid.enroll_tau0``.
- ``get_col0_Flock_2019(fluid)``: Similar, but returns the column density for frequency-dependent irradiation. Intended to be passed to ``fluid.enroll_col0``.
- ``get_fraction_Isella_Natta_2005(fluid)``: Returns a dust sublimation fraction following Isella & Natta (2005), useful for modeling the dust sublimation front near the star. Intended to be passed to ``fluid.enroll_effective_fraction``.
- ``get_alpha_Cecil_Flock_2024(fluid)``: Returns a spatially varying turbulent viscosity parameter :math:`\alpha` profile following Cecil & Flock (2024). Intended to be passed to ``fluid.enroll_alpha``.

**Miscellaneous:**

- ``get_surface_density(grid, rho)``: Integrates density along the polar direction to compute the vertically integrated surface density (spherical geometry only).
- ``toQuantity(arr, unit)``: Converts a plain array or scalar to an ``astropy.Quantity`` with the specified unit, handling the case where the input is already a Quantity gracefully.
