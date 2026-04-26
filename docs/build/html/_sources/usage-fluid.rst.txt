3. Fluids (Gas, Dust)
=====================

After initializing the grid and star, we can declare one or more fluid objects. ``pyFLD`` currently supports two branches of the ``RadiativeFluid`` class, for gas and dust, initialized very similarly with the following call signature:

.. code-block:: python

    fld.fluids.Gas/Dust(
        # inherited from GenericFluid
        grid: fld.grid.Grid = None,
        star: fld.star.Star = None,
        name=None,
        mu=2.353,
        gamma=1.4,
        verbose=True,
        debug=False,

        # inherited from RadiativeFluid
        viscosity=False,
        irradiation=False,
        kabs_wl=None,
        ksca_wl=None,
        g_wl=None,
        build_mean_opacities=True,

        # Gas has no specific parameters

        # specific to Dust branch
        grain_rho=<Quantity 1. g / cm3>,
        grain_size=<Quantity 1. um>
    )

The ``grid`` and ``star`` arguments are basically mandatory. Most other arguments have default values that are sufficient for most use cases, but you can customize them as needed. In particular:

- ``viscosity``: toggle whether this fluid contributes to viscous heating
- ``irradiation``: toggle whether this fluid contributes to the optical depth via ray-tracing and the resulting irradiation heating. When not set to ``False``, can be set to ``gray`` or ``freqdep`` for gray or frequency-dependent opacities, respectively. Using ``freqdep`` requires the additional arguments ``kabs_wl``, ``ksca_wl``, and ``g_wl`` (absorption opacity, scattering opacity, and scattering asymmetry parameter as a function of wavelength, respectively). These 3 arrays must match the wavelength grid of the star's spectrum (i.e., the ``wl`` argument passed to the ``Star`` initializer).
- ``build_mean_opacities``: if ``True``, and a frequency-dependent opacity model is used, the class will automatically compute the Rosseland and Planck mean opacities for use in the diffusion solver. Otherwise, the mean opacities must be provided by the user (see below). It is recommended to leave this as ``True`` for consistency.

The ``Gas`` branch has no specific parameters, but the ``Dust`` branch requires the grain internal density ``grain_rho`` and grain size ``grain_size`` to account for dust settling in the vertical direction.

For example, a simple initialization of a ``Gas`` fluid that includes both viscous and irradiation heating would look like:

.. code-block:: python

    gas = fld.fluids.Gas(
        grid=grid,
        star=star,
        name='basic',
        viscosity=True,
        irradiation='gray',
    )

We now have access to various methods within the ``Gas`` object that we can use to flesh out the properties of the fluid, namely:

- ``gas.set_density(rho)``: set the density array (must be 3D and match the grid dimensions, including ghost cells). Example: ``gas.set_density(1e-10 * u.g / u.cm**3 * np.ones(grid.shape))`` for a uniform density of 10^-10 g/cm^3.
- ``gas.set_temperature(T)``: set the temperature array (same requirements as density)
- ``gas.enroll_alpha(func)``: enrolls a user-defined function that takes the fluid object as its only argument and returns the Shakura–Sunyaev α viscosity parameter (e.g., ``gas.enroll_alpha(lambda f: 0.01)`` for a constant α of 0.01)
- ``gas.enroll_kappaR(func)``, ``gas.enroll_kappaP(func)``, and ``gas.enroll_kappaPstar(func)``: similarly used to enroll user-defined functions for the Rosseland and Planck mean opacities, and the Planck mean opacity at the stellar spectrum, respectively.
- ``gas.enroll_tau0(func)``: enrolls a function that returns the gray optical depth between the stellar surface and the inner radial boundary of the grid. Required when ``irradiation='gray'``. See also ``fld.tools.get_tau0_Flock_2019`` for a ready-made prescription.
- ``gas.enroll_col0(func)``: similar to ``enroll_tau0``, but for frequency-dependent irradiation (``irradiation='freqdep'``). This function should return the column density from the stellar surface to the inner radial boundary of the grid. See also ``fld.tools.get_col0_Flock_2019``.

All enrolled functions receive the fluid object itself as their only argument, which gives access to all current fluid properties. Their getter counterparts (e.g., ``gas.get_kappaR(gas)``, ``gas.get_col0(gas)``) follow the same convention and are called with the fluid passed as the argument. For example, a Planck opacity that depends on temperature would look like:

.. code-block:: python

    Gas = fld.fluids.Gas(...)

    def kappaP_func(f):
        T = f.tmp.to_value('K')
        return 0.02 * u.cm**2 / u.g * T ** 2

    Gas.enroll_kappaP(kappaP_func)

After setting up the properties of the fluid and enrolling any necessary functions, you can set up the class with:

.. code-block:: python

    gas.setup()

Finally, both ``Gas`` and ``Dust`` branches include a method that recomputes the vertical structure of the fluid under the assumption of hydrostatic equilibrium, which is useful when constructing self-consistent initial conditions for protoplanetary disks:

.. code-block:: python

    gas.compute_hydrostatic_equilibrium()
    dust.compute_hydrostatic_equilibrium(rhogas=gas.rho)

Here is where the two branches differ: the ``Gas`` branch accounts for radial and vertical pressure support, while the ``Dust`` branch accounts for settling balanced by vertical turbulent diffusion. The latter requires knowing the Stokes number of the dust, which in turn requires a grain size, grain density, and the background gas density (hence the ``rhogas`` argument). Note that in this case, a ``Dust`` object must include a nonzero turbulent α via ``enroll_alpha``, even if the user sets ``viscosity=False`` (i.e., no viscous heating).

Note that ``compute_hydrostatic_equilibrium`` does not update the density array in place, but rather returns a new array with the updated density structure. The user can simply capture the output as ``gas.rho = ...``, but as these classes are designed to be passed to the radiative transfer solver, this is typically not necessary as the solver will handle these updates internally as needed.

We have now set up all the ingredients for the radiative transfer calculations, which we will cover in the `next section <usage-FLD.html>`_.

Notes
-----

- When using frequency-dependent irradiation, the user must provide the absorption and scattering opacities, and scattering asymmetry parameter, as a function of wavelength, even though only the absorption opacity is used to compute the irradiation heating. In that case, you can simply pass zero-valued arrays for the scattering opacity and asymmetry parameter.
