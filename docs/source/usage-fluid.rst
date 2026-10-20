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

        # Gas has no specific parameters

        # specific to Dust branch
        grain_rho=<Quantity 1. g / cm3>,
        grain_size=<Quantity 1. um>
    )

The ``grid`` and ``star`` arguments are basically mandatory. Most other arguments have default values that are sufficient for most use cases, but you can customize them as needed. In particular:

- ``viscosity``: toggle whether this fluid contributes to viscous heating
- ``irradiation``: toggle whether this fluid contributes to the optical depth via ray-tracing and the resulting irradiation heating. When not set to ``False``, can be set to ``gray`` or ``freqdep`` for gray or frequency-dependent opacities, respectively. Using ``freqdep`` requires the additional arguments ``kabs_wl``, ``ksca_wl``, and ``g_wl`` (absorption opacity, scattering opacity, and scattering asymmetry parameter as a function of wavelength, respectively). These 3 arrays must match the wavelength grid of the star's spectrum (i.e., the ``wl`` argument passed to the ``Star`` initializer).
- ``kabs_wl``, ``ksca_wl``, ``g_wl``: if provided, the Rosseland and Planck mean opacities (and, with irradiation, the Planck mean opacity at the stellar temperature) are computed automatically from them during ``setup()``. Any mean opacity enrolled by the user (see below) takes precedence. If all required mean opacities are enrolled, these arrays are not needed for the diffusion solver.

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

- ``gas.set_density(rho)``: set the density array (must be 3D and match the grid dimensions, including ghost cells). Example: ``gas.set_density(1e-10 * u.g / u.cm**3 * np.ones(grid.shape))`` for a uniform density of 10^-10 g/cm^3. Plain arrays are interpreted in g/cm^3.
- ``gas.set_temperature(T)``: set the temperature array (same requirements as density; plain arrays are interpreted in K)
- ``gas.enroll_alpha(func)``: enrolls a user-defined function that takes the fluid object as its only argument and returns the Shakura–Sunyaev α viscosity parameter (e.g., ``gas.enroll_alpha(lambda f: 0.01)`` for a constant α of 0.01)
- ``gas.enroll_effective_fraction(func)``: enrolls a function that returns the effective mass fraction ``eps`` of the fluid (between 0 and 1), e.g. to model dust sublimation with ``fld.tools.get_fraction_Isella_Natta_2005``. Defaults to 1.
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

Raw and effective density
-------------------------

Each fluid stores two densities:

- ``rho``: the raw density, i.e. what is set with ``set_density``, stored in snapshots, and conserved by hydrostatic equilibrium.
- ``rho_eff = eps * rho``: the effective density, which is what enters the opacities, optical depths, pressure, and internal energy.

``rho_eff`` is stored as an array (it is ``None`` before ``setup()``), and is updated whenever ``eps`` is recomputed (``setup()``, ``refresh()``) or ``rho`` changes through hydrostatic equilibrium. ``refresh()`` recomputes, in order, the effective fraction and ``rho_eff``, the mean opacities, the pressure and internal energy, and (with irradiation) the optical depths, all at the current temperature. If you modify ``rho`` directly after ``setup()``, call ``fluid.refresh()`` (or ``fluid.update_effective_density()`` if ``eps`` should not change) to keep the derived quantities consistent.

Hydrostatic equilibrium
-----------------------

Finally, both ``Gas`` and ``Dust`` branches include a method that recomputes the vertical structure of the fluid under the assumption of hydrostatic equilibrium, which is useful when constructing self-consistent initial conditions for protoplanetary disks:

.. code-block:: python

    gas.enforce_hydrostatic_equilibrium()
    dust.enforce_hydrostatic_equilibrium(rhogas=gas.rho)

Here is where the two branches differ: the ``Gas`` branch accounts for radial and vertical pressure support, while the ``Dust`` branch accounts for settling balanced by vertical turbulent diffusion. The latter requires knowing the Stokes number of the dust, which in turn requires a grain size, grain density, and the background gas density (hence the ``rhogas`` argument). Note that in this case, a ``Dust`` object must include a nonzero turbulent α via ``enroll_alpha``, even if the user sets ``viscosity=False`` (i.e., no viscous heating).

Both branches conserve the raw surface density (computed from ``rho``, independently of ``eps``), and only work in spherical geometry. The grid can cover the top half (:math:`\theta \leq \pi/2`), the bottom half (:math:`\theta \geq \pi/2`), or both; in all cases, :math:`\pi/2` must be a cell interface, since the vertical integration starts from the midplane (see the `grid section <usage-grid.html>`_).

``enforce_hydrostatic_equilibrium`` updates ``rho``, ``rho_eff``, and the pressure and internal energy of the fluid. The underlying ``compute_hydrostatic_equilibrium`` method only returns the new density array, without modifying the fluid; if you use it directly and assign the result to ``rho``, call ``refresh()`` or ``update_effective_density()`` afterwards. When the fluids are passed to a radiative environment, it is typically simpler to call ``Disk.enforce_hydrostatic_equilibrium()``, which handles all fluids at once (see the `next section <usage-FLD.html>`_).

We have now set up all the ingredients for the radiative transfer calculations, which we will cover in the `next section <usage-FLD.html>`_.

Notes
-----

- When using frequency-dependent irradiation, the user must provide the absorption and scattering opacities, and scattering asymmetry parameter, as a function of wavelength, even though only the absorption opacity is used for the attenuation of the stellar radiation and the resulting irradiation heating. In that case, you can simply pass zero-valued arrays for the scattering opacity and asymmetry parameter.
- Enrolled functions are not stored in snapshots, and must be enrolled again after reading a fluid with ``create_from_snapshot``.
