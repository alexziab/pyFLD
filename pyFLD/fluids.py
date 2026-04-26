import numpy as np
from astropy import constants as const, units as u
from . import debug_state as dbg, grid as gr, tools as tb, star as st
from scipy.interpolate import interp1d

class _GenericFluid:
    """
        Base class for fluid solvers, not to be used directly.
        Holds common attributes and methods shared by all fluids, related to
        grid, stellar, and basic fluid properties, and boundary conditions.

        By "basic fluid properties" we refer to:

        - mu : mean molecular weight
        - gamma : adiabatic index
        - cV : specific heat at constant volume
        - rho : density
        - tmp : temperature
        - prs : pressure
        - eng : internal energy density
        - Er : radiation energy density

        Also includes methods to compute:

        - eps : effective density fraction (e.g., in case of phase changes)
        - alpha : viscosity parameter
    """

    rho   = None
    tmp   = None
    eps   = None
    name  = None

    _initialized_density = False
    _initialized_temperature = False
    _initialized = False

    def __init__(self, grid:gr.Grid|None=None, star:st.Star|None=None, name=None,
                 mu=2.353, gamma=1.4, verbose=True, debug=False):

        self._set_debug_name(debug=debug, name=name)
        self._push_region("__init__ (GenericFluid)")

        self.verbose = verbose

        if grid is None:
            raise ValueError("please provide a Grid object")

        self.grid = grid
        self.star = star
        if self.grid.nghost < 2:
            raise ValueError("please provide at least 2 ghost cells in the grid")

        self.mu, self.gamma = mu, gamma
        self.cV = tb.Rgas / self.mu / (self.gamma-1)

        self.boundary_type = [gr.BOUNDARY_UNDEFINED] * 2 * grid.dims
        self.boundary_val  = [None] * 2 * grid.dims

        self._pop_region()

    def _set_debug_name(self, **kwargs):
        """Set debug mode and object name from keyword arguments."""
        self.debug = kwargs.get('debug', False)
        self.name  = kwargs.get('name', None)

    def get_prefix(self):
        """Get formatted prefix for logging messages including class name and object name."""
        return f"{self.__class__.__name__} '{self.name}'"

    def _logging(self, msg):
        """Log debug message if debug mode enabled."""
        prefix = self.get_prefix()
        if self.debug:
            print(dbg.format_debug(prefix, msg, dbg.depth()))

    def _log_warning(self, msg):
        """Format and log warning message."""
        prefix = self.get_prefix()
        dbg.format_warning(prefix, msg, dbg.depth())

    def _push_region(self, name):
        """Enter a named debug region, increment depth, and log entry."""
        self._logging(f"{name} ...")
        dbg.push()
    
    def _pop_region(self):
        """Exit the current debug region and log exit."""
        dbg.pop()
        self._logging(f"... returned.")

    def enforce_boundaries(self, q, powerlaw_extrapolation=True):
        """
        Apply boundary conditions to ghost zones of a field.

        Applies zero-gradient boundary conditions in Cartesian geometry, and
        power-law extrapolation in cylindrical/spherical geometries to maintain
        consistency with the radial structure. Modifies the input array in place.

        Arguments
        ----------
        q : astropy.units.Quantity
            Field to apply boundaries to. Must have shape matching self.grid.shape
            (including ghost cells)
        powerlaw_extrapolation : bool, optional
            Use power-law extrapolation in radial direction for cylindrical/spherical
            geometries (default: True). If False, uses zero-gradient everywhere.
        """

        self._push_region("enforce_boundaries")

        g = self.grid # first apply zero-gradient everywhere
        q[:,:,:g.ibeg]   = 1 * q[:,:,g.ibeg][:,:,None]
        q[:,:,g.iend+1:] = 1 * q[:,:,g.iend][:,:,None]
        q[:,:g.jbeg,:]   = 1 * q[:,g.jbeg,:][:,None,:]
        q[:,g.jend+1:,:] = 1 * q[:,g.jend,:][:,None,:]
        q[:g.kbeg,:,:]   = 1 * q[g.kbeg,:,:][None,:,:]
        q[g.kend+1:,:,:] = 1 * q[g.kend,:,:][None,:,:]

        valid_geometry = g.geometry in gr.aux_cylindrical + gr.aux_spherical
        if powerlaw_extrapolation and valid_geometry:
            i, x = g.ibeg, g.x1
            s    = np.log(q[:,:,i+1]/q[:,:,i]) / np.log(x[i+1]/x[i])
            s    = tb.get_capped(s, log=True) # cap slopes
            q[:, :, :i] = q[:,:,i][:,:,None] * (x[:i]/x[i]) ** s[:,:,None]
            
            i, x = g.iend, g.x1
            s    = np.log(q[:,:,i]/q[:,:,i-1]) / np.log(x[i]/x[i-1])
            s    = tb.get_capped(s, log=True) # cap slopes
            q[:,:, i+1:] = q[:,:,i][:,:,None] * (x[i+1:]/x[i]) ** s[:,:,None]

        self._pop_region()

    def set_density(self, rho):
        """
        Set density field and apply boundary conditions.

        Converts input to appropriate units, applies boundary conditions,
        and marks density as initialized. Must be called before setup().

        Arguments
        ----------
        rho : array-like [g/cm^3] or astropy.units.Quantity
            Density field with shape matching grid (including ghost cells)

        Examples
        --------
        Set uniform density at 1e-10 g/cm^3:

            fluid.set_density(1e-10 * u.g/u.cm**3 * np.ones(fluid.grid.shape))
        """
        self._push_region("set_density")

        self.rho = tb.toQuantity(rho, 'g/cm3')
        self.enforce_boundaries(self.rho)
        self._initialized_density = True

        self._pop_region()

    def set_temperature(self, tmp):
        """
        Set temperature field and apply boundary conditions.

        Converts input to appropriate units, applies boundary conditions,
        and marks temperature as initialized. Must be called before setup().

        Arguments
        ----------
        tmp : array-like [K] or astropy.units.Quantity
            Temperature field with shape matching grid (including ghost cells)

        Examples
        --------
        Set uniform temperature at 100 K:

            fluid.set_temperature(100 * u.K * np.ones(fluid.grid.shape))
        """
        self._push_region("set_temperature")

        self.tmp = tb.toQuantity(tmp, u.K)
        self.enforce_boundaries(self.tmp)
        self._initialized_temperature = True

        self._pop_region()

    def update_pressure_energy(self):
        """
        Update pressure and internal energy from density and temperature.

        Computes pressure and internal energy density using the ideal gas law
        and thermodynamic relations. Called automatically by setup() and after
        temperature updates.
        """
        self._push_region("update_pressure_energy")

        rho = self.eps * self.rho # type: ignore[operator]
        self.prs = rho * self.tmp * tb.Rgas / self.mu
        self.eng = self.prs / (self.gamma-1)

        self._pop_region()

    def get_surface_density(self):
        """
        Compute vertically integrated surface density.

        Integrates density in the polar angle direction to obtain surface
        density as a function of radius and azimuth. Accounts for the effective
        density fraction (eps). Works only in spherical geometry.

        Returns
        -------
        astropy.units.Quantity [g/cm^2]
            Vertically integrated surface density (φ, R)

        Raises
        ------
        NotImplementedError
            If grid geometry is not spherical
        """
        self._push_region("get_surface_density")

        act = self.grid.active
        rho = self.eps * self.rho # type: ignore[operator]
        sig = tb.get_surface_density(self.grid, rho[act])

        self._pop_region()
        return sig

    get_alpha = staticmethod(lambda self: 0.0) # by default
    def enroll_alpha(self, func):
        """
        Register a custom viscosity parameter function.

        Enrolls a function to compute the viscosity parameter alpha, which can
        depend on local fluid properties. The function is called during opacity
        and viscous heating calculations.

        Arguments
        ----------
        func : callable
            Function taking the fluid object as argument and returning an array
            broadcastable to grid shape (scalar, 1D in x1, 2D in x1-x2, or 3D
            including ghost cells)

        Examples
        --------
        Enroll a constant alpha of 1e-3:

            fluid.enroll_alpha(lambda fluid: 1e-3)

        Enroll a temperature-dependent alpha:

            fluid.enroll_alpha(lambda fluid: 1e-2 * (fluid.tmp / 100*u.K)**0.5)
        
            
        See also
        --------
        tools.get_alpha_Cecil_Flock_2024 : Helper function for alpha profile from Cecil & Flock (2024)
        """
        self._push_region("enroll_alpha")
        self.get_alpha = func # viscosity
        self._pop_region()

    get_effective_fraction = staticmethod(lambda self: 1.0) # by default
    def enroll_effective_fraction(self, func):
        """
        Register a custom effective density fraction function.

        Enrolls a function to compute the effective density fraction eps, useful
        for modeling phase changes and sublimation. The function is called during
        setup and can account for temperature-dependent changes in the active
        fluid mass.

        Arguments
        ----------
        func : callable
            Function taking the fluid object as argument and returning an array
            broadcastable to grid shape (scalar, 1D in x1, 2D in x1-x2, or 3D
            including ghost cells)

        Examples
        --------
        Enroll a constant fraction of 0.5:

            fluid.enroll_effective_fraction(lambda fluid: 0.5)

        Enroll dust sublimation following Isella & Natta (2005):

            fluid.enroll_effective_fraction(tools.get_fraction_Isella_Natta_2005)

        See Also
        --------
        tools.get_fraction_Isella_Natta_2005 : Helper function for dust sublimation
        """
        self._push_region("enroll_effective_fraction")
        self.get_effective_fraction = func
        self._pop_region()

    def setup(self):
        """Compute auxiliary quantities"""

        self._push_region("setup (GenericFluid)")

        if not self._initialized_density: raise ValueError(f"{self.get_prefix()}: call 'set_density' first")
        if not self._initialized_temperature: raise ValueError(f"{self.get_prefix()}: call 'set_temperature' first")

        self.compute_fractions()
        self.update_pressure_energy()
        self.Er  = tb.aR * self.tmp ** 4 # type: ignore[operator]

        self._initialized = True

        self._pop_region()

    def compute_fractions(self):
        """
        Compute effective density fraction.

        Calls the enrolled effective fraction function to compute eps at all
        grid points. Used internally during setup and updates. Updates the eps
        attribute in-place.
        """
        self._push_region("compute_fractions")
        self.eps = self.get_effective_fraction(self)
        self._pop_region()

    def compute_hydrostatic_equilibrium(self, rhogas=None, uf=0):
        raise NotImplementedError # to be defined in child classes

    def enforce_hydrostatic_equilibrium(self, rhogas=None, uf=0):
        """
        Update density field assuming hydrostatic equilibrium.

        Recomputes the density profile in vertical (θ) direction to satisfy
        hydrostatic equilibrium. Works only in spherical geometry. For dust fluids,
        accounts for dust settling via the rhogas parameter.

        Arguments
        ----------
        rhogas : array-like [g/cm^3] or astropy.units.Quantity, optional
            Gas density field required for dust settling calculations. Must have
            same shape as grid (including ghost cells). Not needed for gas fluids.
            (default: None)
        uf : float [0-1], optional
            Under-relaxation factor for iterative convergence. uf=0 performs full
            update to hydrostatic equilibrium. uf=0.5 relaxes halfway to equilibrium
            each step. Useful for stability in time-stepping. (default: 0)

        Raises
        ------
        NotImplementedError
            If grid geometry is not spherical
        ValueError
            If rhogas is None for dust fluids or if NaNs appear in result
        """
        self._push_region("enforce_hydrostatic_equilibrium")
        self.rho = self.compute_hydrostatic_equilibrium(rhogas=rhogas, uf=uf)
        self.update_pressure_energy()
        self._pop_region()

class _RadiativeFluid(_GenericFluid):
    """
    Base class for radiative fluids (not for direct use).

    Extends `_GenericFluid` to include radiation and irradiation physics. Manages
    wavelength-dependent and mean opacities, optical depth calculations, and
    radiative heating source terms.

    Arguments (in addition to `_GenericFluid`)
    -----------------------------------------
    viscosity : bool, optional
        Enable viscous heating computations (default: False)
    irradiation : {'gray', 'freqdep'} or False, optional
        Type of irradiation to compute: 'gray' for graybody, 'freqdep' for
        frequency-dependent, or False to disable (default: False)
    kabs_wl : array-like [cm^2/g] or astropy.units.Quantity, optional
        Wavelength-dependent absorption opacity (default: None)
    ksca_wl : array-like [cm^2/g] or astropy.units.Quantity, optional
        Wavelength-dependent scattering opacity (default: None)
    g_wl : array-like, optional
        Wavelength-dependent asymmetry parameter (default: None)

    Attributes
    ----------
    kappaR : astropy.units.Quantity [cm^2/g]
        Rosseland mean opacity
    kappaP : astropy.units.Quantity [cm^2/g]
        Planck mean opacity
    kappaPstar : astropy.units.Quantity [cm^2/g]
        Planck mean opacity at stellar temperature (for irradiation)
    tau0 : ndarray
        Optical depth to star at domain inner boundary
    dtau : ndarray
        Cell optical depth in radial direction
    irradiation : {False, 'gray', 'freqdep'}
        Irradiation mode
    viscosity : bool
        Whether viscous heating is enabled
    """

    userdef_Rosseland_opacity = False
    userdef_Planck_opacity    = False
    userdef_stellar_opacity   = False

    kabs_wl = ksca_wl = g_wl = None

    def __init__(self, viscosity=False, irradiation=False,
                 kabs_wl=None, ksca_wl=None, g_wl=None,
                 **kwargs):
        """
        Initialize a radiative fluid object.

        Validates irradiation and viscosity parameters, assigns wavelength-dependent
        opacities if provided, and calls parent class initialization.
        """
        self._set_debug_name(**kwargs)
        self._push_region("__init__ (RadiativeFluid)")
        super().__init__(**kwargs)

        if irradiation not in ['gray', 'freqdep', False]:
            raise ValueError(f"{self.get_prefix()}: irradiation must be one of 'gray', 'freqdep', or False.")
        if self.star is None and self.verbose:
            if viscosity or irradiation:
                raise ValueError(f"{self.get_prefix()}: No Star object provided; irradiation/heating cannot be used.")
        self.viscosity = viscosity

        if irradiation and self.grid.geometry not in gr.aux_spherical:
            raise NotImplementedError(f"{self.get_prefix()}: Irradiation only works in spherical geometry.")

        self.irradiation = irradiation
        
        # if any is missing, error will be raised later
        have_freqdep_opacities = kabs_wl is not None \
                                or ksca_wl is not None \
                                or g_wl is not None
        if have_freqdep_opacities: # build regardless of irradiation 
            self._assign_freqdep_opacities(kabs_wl, ksca_wl, g_wl)
        
        self._pop_region()

    def _assign_freqdep_opacities(self, kabs_wl, ksca_wl, g_wl):
        """
        Assign wavelength-dependent opacity arrays.

        Validates and stores wavelength-dependent opacities for use in computing
        mean opacities. All arrays must have length matching star.wl.

        Arguments
        ----------
        kabs_wl : array-like [cm^2/g] or astropy.units.Quantity
            Absorption opacities at each wavelength in star.wl
        ksca_wl : array-like [cm^2/g] or astropy.units.Quantity
            Scattering opacities at each wavelength in star.wl
        g_wl : array-like
            Asymmetry parameters at each wavelength in star.wl

        Raises
        ------
        ValueError
            If any array is None or has wrong length
        """
        self._push_region("_assign_freqdep_opacities")

        def check(arr, name):
            if arr is None:
                raise ValueError(f"{self.get_prefix()}: Please provide {name}.")
            if arr.size != self.star.wl.size: # type: ignore[union-attr]
                raise ValueError(f"{self.get_prefix()}: Size of {name} must equal that of Star.wl")
        
        check(kabs_wl, 'absorption opacities (kabs_wl)')
        check(ksca_wl, 'scattering opacities (ksca_wl)')
        check(g_wl,    'asymmetry factors (g_wl)')

        self.kabs_wl = 1 * tb.toQuantity(kabs_wl, u.cm**2/u.g)
        self.ksca_wl = 1 * tb.toQuantity(ksca_wl, u.cm**2/u.g)
        self.g_wl = 1 * g_wl

        self._pop_region()

    def compute_opacities(self):
        """
        Compute mean opacity fields at current temperature.

        Evaluates Rosseland and Planck mean opacities using enrolled functions
        or interpolators built from wavelength-dependent data. If irradiation is
        enabled, also computes Planck mean opacity at stellar temperature. See
        tools.compute_mean_opacities() for automatic computation from spectral
        absorption/scattering opacities.

        Updates kappaR, kappaP, and (if irradiation enabled) kappaPstar attributes.

        See Also
        --------
        enroll_kappaR : Register custom Rosseland opacity
        enroll_kappaP : Register custom Planck opacity
        enroll_kappaPstar : Register stellar Planck opacity
        tools.compute_mean_opacities : Compute means from spectral data
        """
        self._push_region("compute_opacities")

        self.kappaR = self.get_kappaR(self)
        self.kappaP = self.get_kappaP(self)
        if self.irradiation: self.kappaPstar = self.get_kappaPstar(self)

        self._pop_region()

    def build_mean_opacity_interpolators(self, T=None):
        """
        Construct temperature-interpolated mean opacity functions.

        Computes Rosseland and Planck mean opacities via tools.compute_mean_opacities()
        across a temperature range and builds interpolation functions to evaluate them
        at any temperature. Used automatically by setup() when wavelength-dependent
        opacities are provided but custom opacity functions are not enrolled.
        Opacities are extrapolated linearly beyond the computed range.

        Arguments
        ----------
        T : array-like [K] or astropy.units.Quantity, optional
            Temperature array for computing means. If None, uses default range
            determined by tools.compute_mean_opacities(). (default: None)

        Sets
        ----
        T_opac : astropy.units.Quantity [K]
            Temperature points at which opacities were computed
        get_kappaR : callable
            Interpolator for Rosseland mean opacity (unless user-enrolled)
        get_kappaP : callable
            Interpolator for Planck mean opacity (unless user-enrolled)
        get_kappaPstar : callable
            Interpolator for stellar Planck opacity (unless user-enrolled and irradiation enabled)

        See Also
        --------
        tools.compute_mean_opacities : Compute Rosseland/Planck means from spectral opacities
        enroll_kappaR : Register custom Rosseland opacity
        enroll_kappaP : Register custom Planck opacity
        enroll_kappaPstar : Register stellar Planck opacity
        """
        self._push_region("build_mean_opacity_interpolators")

        # copy locally
        wl   = 1 * self.star.wl # type: ignore[operator]
        kabs = 1 * self.kabs_wl # type: ignore[operator]
        ksca = 1 * self.ksca_wl # type: ignore[operator]
        g    = 1 * self.g_wl    # type: ignore[operator]
        T, kR, kP = tb.compute_mean_opacities(wl, kabs, ksca, g, T=T)
        TK = T.to_value('K')
        kwargs = dict(bounds_error=False, fill_value='extrapolate')

        self.T_opac = 1 * T
        if not self.userdef_Rosseland_opacity:
            kappaR_interp = interp1d(TK, kR.to_value('cm2/g'), **kwargs) # type: ignore[assignment]
            def get_kappaR(self):
                return kappaR_interp(self.tmp.to_value('K')) * u.cm**2 / u.g
            self.get_kappaR = get_kappaR
        if not self.userdef_Planck_opacity:
            kappaP_interp = interp1d(TK, kP.to_value('cm2/g'), **kwargs) # type: ignore[assignment]
            def get_kappaP(self):
                return kappaP_interp(self.tmp.to_value('K')) * u.cm**2 / u.g
            self.get_kappaP = get_kappaP
        if self.irradiation:
            kappaP_interp = interp1d(TK, kP.to_value('cm2/g'), **kwargs) # type: ignore[assignment]
            def get_kappaPstar(self):
                return kappaP_interp(self.star.T.to_value('K')) * u.cm**2 / u.g
            if not self.userdef_stellar_opacity:
                self.get_kappaPstar = get_kappaPstar

        self._pop_region()
    
    # must receive the fluid object as an argument
    get_kappaR = staticmethod(lambda self: 0.0 * u.cm**2 / u.g) # by default
    def enroll_kappaR(self, func):
        """
        Register a custom Rosseland mean opacity function.

        Enrolls a function to compute wavelength-integrated Rosseland mean opacity,
        which characterizes radiative diffusion in optically thick regions. If not
        enrolled and wavelength-dependent opacities are provided, will be computed
        via tools.compute_mean_opacities().

        Arguments
        ----------
        func : callable
            Function(fluid) -> array_like [cm^2/g]. Returns Rosseland mean opacity
            broadcastable to grid shape.

        Examples
        --------

        Enroll a constant Rosseland mean opacity of 0.1 cm^2/g:

            fluid.enroll_kappaR(lambda fluid: 0.1 * u.cm**2 / u.g)

        Enroll a temperature-dependent Rosseland mean opacity:

            fluid.enroll_kappaR(lambda fluid: 0.02 * (fluid.tmp / u.K)**2 * u.cm**2 / u.g)

        See Also
        --------
        tools.compute_mean_opacities : Compute Rosseland/Planck means from spectral opacities
        build_mean_opacity_interpolators : Automatic mean opacity from wavelength data
        tools.get_gas_Rosseland_opacity_Freedman_2008 : Helper function for gas opacities
        """
        self._push_region("enroll_kappaR")
        self.userdef_Rosseland_opacity = True
        self.get_kappaR = func # for Rosseland mean opacity
        self._pop_region()

    get_kappaP = staticmethod(lambda self: 0.0 * u.cm**2 / u.g) # by default
    def enroll_kappaP(self, func):
        """
        Register a custom Planck mean opacity function.

        Enrolls a function to compute wavelength-integrated Planck mean opacity,
        which characterizes thermal radiation at the fluid temperature. If not
        enrolled and wavelength-dependent opacities are provided, will be computed
        via tools.compute_mean_opacities().

        Arguments
        ----------
        func : callable
            Function(fluid) -> array_like [cm^2/g]. Returns Planck mean opacity
            at fluid temperature broadcastable to grid shape.

        Examples
        --------

        Enroll a constant Planck mean opacity of 0.1 cm^2/g:

            fluid.enroll_kappaP(lambda fluid: 0.1 * u.cm**2 / u.g)
        
        Enroll a temperature-dependent Planck mean opacity:

            fluid.enroll_kappaP(lambda fluid: 0.02 * (fluid.tmp / u.K)**2 * u.cm**2 / u.g)

        See Also
        --------
        tools.compute_mean_opacities : Compute Rosseland/Planck means from spectral opacities
        build_mean_opacity_interpolators : Automatic mean opacity from wavelength data
        tools.get_gas_Planck_opacity_Freedman_2008 : Helper function for gas opacities
        """
        self._push_region("enroll_kappaP")
        self.userdef_Planck_opacity = True
        self.get_kappaP = func # for Planck mean opacity
        self._pop_region()

    get_kappaPstar = staticmethod(lambda self: 0.0 * u.cm**2 / u.g) # by default
    def enroll_kappaPstar(self, func):
        """
        Register a custom Planck mean opacity at stellar temperature function.

        Enrolls a function to compute Planck mean opacity at the stellar temperature,
        used for frequency-dependent irradiation calculations. Required when
        irradiation='freqdep' and custom opacity functions are used. If not enrolled
        and wavelength-dependent opacities are provided, will be computed via
        tools.compute_mean_opacities().

        Arguments
        ----------
        func : callable
            Function(fluid) -> array_like [cm^2/g]. Returns Planck mean opacity
            at stellar temperature broadcastable to grid shape.

        Notes
        -----
        Only used when irradiation is enabled. For gray irradiation, tau0 and col0
        can be enrolled directly instead.

        Examples
        --------

        Enroll a constant stellar Planck mean opacity of 0.1 cm^2/g:

            fluid.enroll_kappaPstar(lambda fluid: 0.1 * u.cm**2 / u.g)

        See Also
        --------
        tools.compute_mean_opacities : Compute Rosseland/Planck means from spectral opacities
        build_mean_opacity_interpolators : Automatic mean opacity from wavelength data
        tools.get_gas_Planck_opacity_Freedman_2008 : Helper function for gas opacities
        enroll_tau0 : Register optical depth function
        """
        self._push_region("enroll_kappaPstar")
        self.userdef_stellar_opacity = True
        self.get_kappaPstar = func # for stellar opacity
        self._pop_region()

    get_tau0 = staticmethod(lambda self: 0.0) # by default
    def enroll_tau0(self, func):
        """
        Register a custom optical depth function.

        Enrolls a function to compute optical depth τ at the domain's inner radial
        boundary, used for gray irradiation calculations. For frequency-dependent
        irradiation, this is computed automatically from absorption opacities
        (kabs_wl) and column density. See tools.compute_mean_opacities() for
        spectral opacity handling.

        Arguments
        ----------
        func : callable
            Function(fluid) -> array_like. Returns optical depth at inner boundary,
            broadcastable to grid shape (φ, θ).

        Examples
        --------
        Enroll a constant optical depth:

            fluid.enroll_tau0(lambda fluid: 1.0)

        Use the Flock et al. (2019) formula:

            fluid.enroll_tau0(tools.get_tau0_Flock_2019)

        Notes
        -----
        Only used when irradiation='gray'. For frequency-dependent irradiation,
        use kabs_wl instead.

        See Also
        --------
        enroll_col0 : Register column density function
        tools.get_tau0_Flock_2019 : Compute optical depth following Flock et al. (2019)
        tools.compute_mean_opacities : Opacity computations from spectral data
        """
        self._push_region("enroll_tau0")
        self.get_tau0 = func
        self._pop_region()

    get_col0 = staticmethod(lambda self: 0.0 * u.g/u.cm**2) # by default
    def enroll_col0(self, func):
        """
        Register a custom column density function.

        Enrolls a function to compute column density at the domain's inner radial
        boundary, used with optical depth for irradiation calculations. For
        frequency-dependent irradiation, combines with absorption opacities
        (kabs_wl) to compute wavelength-dependent τ₀. See tools.compute_mean_opacities()
        for spectral opacity handling.

        Arguments
        ----------
        func : callable
            Function(fluid) -> array_like [g/cm^2] or astropy.units.Quantity.
            Returns column density at inner boundary, broadcastable to grid shape (φ, θ).

        Examples
        --------
        Enroll a constant column density:

            fluid.enroll_col0(lambda fluid: 100*u.g/u.cm**2)

        Use the Flock et al. (2019) formula:

            fluid.enroll_col0(tools.get_col0_Flock_2019)

        Notes
        -----
        Used with tau0 for gray irradiation, or with kabs_wl for frequency-dependent
        irradiation. Default is zero (transparent inner boundary).

        See Also
        --------
        enroll_tau0 : Register optical depth function
        tools.get_col0_Flock_2019 : Compute column density following Flock et al. (2019)
        tools.compute_mean_opacities : Opacity computations from spectral data
        """
        self._push_region("enroll_col0")
        self.get_col0 = func
        self._pop_region()

    def _check_and_build_opacity_interpolators(self):
        have_fully_userdef_opacities = self.userdef_Rosseland_opacity \
                                        and self.userdef_Planck_opacity
        if self.irradiation:
            have_fully_userdef_opacities = have_fully_userdef_opacities \
                                        and self.userdef_stellar_opacity

        if (not have_fully_userdef_opacities):
            # we can build the Rosseland/Planck means from kappa_wl
            if self.star.wl is None: # type: ignore[union-attr]
                raise ValueError(f"{self.get_prefix()}: Star object must have wl defined to build mean opacity interpolators. Alternatively, enroll custom mean opacity functions for kappaR, kappaP, kappaPstar.")
            if self.kabs_wl is None:
                raise ValueError(f"{self.get_prefix()}: please provide kabs_wl, ksca_wl, g_wl to build mean opacity interpolators. The star must also have wl defined. Alternatively, enroll custom mean opacity functions for kappaR, kappaP, kappaPstar.")
            self.build_mean_opacity_interpolators()

    def setup(self):
        """Compute auxiliary quantities"""
        self._push_region("setup (RadiativeFluid)")

        super().setup()

        self._check_and_build_opacity_interpolators()

        self.compute_opacities()

        if self.irradiation:
            self.compute_tau0()
            self.compute_cell_optical_depth()

        self._pop_region()

    def compute_tau0(self):
        """
        Compute optical depth to inner boundary.

        Evaluates τ₀ at the domain's inner radial edge using either enrolled
        tau0/col0 functions (for gray irradiation) or wavelength-dependent opacities
        (for frequency-dependent irradiation). For 'freqdep' mode, computes
        τ₀(λ) = κ_abs(λ) × col0 for each wavelength bin.

        Updates
        -------
        tau0 : astropy.units.Quantity or ndarray
            Optical depth; shape (φ, θ) for gray or (λ, φ, θ) for freqdep

        See Also
        --------
        enroll_tau0 : Register tau0 function
        enroll_col0 : Register column density function
        tools.compute_mean_opacities : Opacity computations from spectral data
        """
        self._push_region("compute_tau0")

        if self.irradiation == 'freqdep': # shape is nbins, nx3, nx2
            col0 = tb.atleast_nd(self.get_col0(self), d=2)
            self.tau0 = self.kabs_wl[:,None,None] * col0 # type: ignore[operator]
        else: self.tau0 = tb.atleast_nd(self.get_tau0(self), d=2) # nx3, nx2

        self._pop_region()

    def compute_cell_optical_depth(self):
        """
        Compute radial optical depth in each cell.

        Calculates Δτ = ε κ ρ Δr for each cell, where κ is either the Planck mean
        (gray irradiation) or absorption opacity (frequency-dependent irradiation).
        Used in irradiation and ray-tracing calculations. See tools.compute_mean_opacities()
        for opacity computation from spectral data.

        Updates
        -------
        dtau : astropy.units.Quantity
            Cell optical depth; shape (Nx1, Nx2, Nx3) for gray or
            (λ, Nx1, Nx2, Nx3) for frequency-dependent irradiation

        See Also
        --------
        tools.compute_mean_opacities : Opacity computations from spectral data
        enroll_kappaPstar : Register Planck opacity at stellar temperature
        """
        self._push_region("compute_cell_optical_depth")
        if self.irradiation == 'gray': kappa = self.kappaPstar
        else: kappa = self.kabs_wl[:,None,None,None] # type: ignore[operator]
        self.dtau = (self.eps * kappa * self.rho * self.grid.dx1)
        self._pop_region()

    def refresh(self):
        self.compute_opacities()
        self.compute_fractions()
        if self.irradiation:
            self.compute_cell_optical_depth()
            self.compute_tau0()

    def write_snapshot(self, dirc='./', name='fluid', output_objects=True):

        self._push_region("write_snapshot")

        model_dirc = f'{dirc}/{name}/'
        tb.mkdir(model_dirc)

        # skip getters, derivatives, and objects
        skipped_keys = ['get_col0', 'get_tau0', 'get_alpha',
                        'get_kappaP', 'get_kappaR', 'get_kappaPstar',
                        'prs', 'eng', 'Er', 'grid', 'star', 'dtau']

        tb.extract_and_store_params(self, dirc=model_dirc,
                                    skipped_keys=skipped_keys)
        # also dump star and grid within, optionally.
        # not needed if we intend to use these within a
        # RadiativeEnvironment, as the star/grid info is there

        tb.try_writing(self.rho, f'{model_dirc}/rho_cgs.dbl', u.g/u.cm**3)
        tb.try_writing(self.tmp, f'{model_dirc}/tmp_cgs.dbl', u.K)
        tb.try_writing(tb.atleast_nd(self.eps, d=1), f'{model_dirc}/eps.dbl')
        tb.try_writing(tb.atleast_nd(self.kappaR, d=1), f'{model_dirc}/kappaR_cgs.dbl', u.cm**2/u.g)
        tb.try_writing(tb.atleast_nd(self.kappaP, d=1), f'{model_dirc}/kappaP_cgs.dbl', u.cm**2/u.g)
        if self.irradiation:
            tb.try_writing(tb.atleast_nd(self.kappaPstar, d=1), f'{model_dirc}/kappaPstar_cgs.dbl', u.cm**2/u.g)
            tb.try_writing(tb.atleast_nd(self.tau0, d=1), f'{model_dirc}/tau0.dbl', u.Unit(''))
        
        tb.try_writing(self.kabs_wl, f'{model_dirc}/kabs_wl_cgs.dbl', u.cm**2/u.g)
        tb.try_writing(self.ksca_wl, f'{model_dirc}/ksca_wl_cgs.dbl', u.cm**2/u.g)
        tb.try_writing(self.g_wl, f'{model_dirc}/g_wl.dbl')

        if output_objects:
            if self.star is not None:
                self.star.write_snapshot(dirc=model_dirc, name='star')
            self.grid.write_snapshot(dirc=model_dirc, name='grid')

        self._pop_region()


        # tb.try_writing(self.x1i, f'{model_dirc}/x1i_cgs.dbl', units[0])

    def _rebuild_after_read(self):
        # we can't call setup() because that requires getters. But we can
        # recompute fields and interpolators.

        # Here I'll assume that since we're reading from a Fluid object,
        # that object was valid (i.e., its __init__ and super().__init__
        # both ran). So no safety checks.

        self._push_region("_rebuild_after_read")

        # from GenericFluid.setup()
        self.update_pressure_energy()
        self.Er  = tb.aR * self.tmp ** 4 # type: ignore[operator]

        # from RadiativeFluid.setup()
        self._check_and_build_opacity_interpolators()
        if self.irradiation: self.compute_cell_optical_depth()

        self._pop_region()

class Dust(_RadiativeFluid):
    """
    Dust fluid for protoplanetary disk simulations.

    Extends `_RadiativeFluid` to include dust-specific properties such as grain
    size and density. Implements dust settling in hydrostatic equilibrium via
    the Stokes number and dust-to-gas coupling.

    Arguments (in addition to ``_RadiativeFluid``)
    -----------------------------------------------

    grain_rho : float [g/cm^3] or astropy.units.Quantity, optional
        Material density of dust grains (default: 1 g/cm^3)
    grain_size : float [cm] or astropy.units.Quantity, optional
        Grain size assuming monodisperse distribution (default: 1 μm)

    Attributes
    ----------
    grain_rho : astropy.units.Quantity [g/cm^3]
        Material density of dust grains
    grain_size : astropy.units.Quantity [cm]
        Grain size
    isDust : bool
        Identifier flag (True for Dust)
    isGas : bool
        Identifier flag (False for Dust)
    """

    def __init__(self, grain_rho=1*u.g/u.cm**3, grain_size=1*u.um, **kwargs):
        """
        Initialize a dust fluid object.

        Stores grain properties and calls parent class initialization with
        radiative fluid capabilities (opacities, irradiation, viscosity).

        Arguments
        ----------
        grain_rho : float [g/cm^3] or astropy.units.Quantity, optional
            Material density of dust grains (default: 1 g/cm^3)
        grain_size : float [cm] or astropy.units.Quantity, optional
            Grain size assuming monodisperse distribution (default: 1 μm)
        **kwargs
            Additional arguments passed to `_RadiativeFluid.__init__`
            (grid, star, viscosity, irradiation, kabs_wl, ksca_wl, g_wl, etc.)

        Examples
        --------
        Create a dust fluid with custom grain properties and opacity:

            dust = Dust(grain_rho=1.4*u.g/u.cm**3, grain_size=0.1*u.um,
                        grid=grid, star=star, irradiation='gray')
        """
        self._set_debug_name(**kwargs)
        self._push_region("__init__ (Dust)")
        
        super().__init__(**kwargs)
        self.grain_rho  = tb.toQuantity(grain_rho, u.g/u.cm**3)
        self.grain_size = tb.toQuantity(grain_size, u.cm)
    
        self.isDust = True
        self.isGas  = False

        self._pop_region()

    def compute_hydrostatic_equilibrium(self, rhogas=None, uf=0):
        """
        Compute dust density profile under hydrostatic equilibrium with settling.

        Solves for the vertical dust density structure accounting for gravity,
        pressure support, and dust-gas interaction via the Stokes number.
        Dust settles toward the midplane proportional to the Stokes number.
        Uses a second-order predictor-corrector integration scheme.

        Works only in spherical geometry. Enforces correct surface density and
        prevents density from increasing toward the poles (falling matter).

        Arguments
        ----------
        rhogas : array-like [g/cm^3] or astropy.units.Quantity
            Gas density field required to compute dust-gas interaction and settling.
            Must have same shape as grid (including ghost cells). (required)
        uf : float [0-1], optional
            Under-relaxation factor for convergence. (default: 0)

        Returns
        -------
        ndarray [g/cm^3]
            Updated dust density field with boundaries applied

        Raises
        ------
        ValueError
            If rhogas is None, geometry is not spherical, grid structure is invalid,
            or NaNs appear in result
        """

        self._push_region("compute_hydrostatic_equilibrium")

        if rhogas is None:
            raise ValueError("please provide rhogas for dust hydrostatic equilibrium")

        rho_a = self.grain_rho * self.grain_size

        g = self.grid
        if g.geometry not in gr.aux_spherical: raise ValueError("Only works in spherical coordinates")
        first, last = g.x2i[g.act2i][[0, -1]]
        halfdisk = np.isclose(first, np.pi/2) or np.isclose(last, np.pi/2)
        if halfdisk: side = 'top' if np.isclose(last, np.pi/2) else 'bottom'
        else: side = 'both'
        if not halfdisk and not np.any(np.isclose(g.x2i[g.act2i], np.pi/2)):
            raise ValueError("When dual-sided, grid must contain pi/2 as a cell interface within active cells. Alternatively, use a half-disk grid.")

        # below, most quantities only contain active cells
        R  = g.R[g.act2, g.act1]
        z  = g.z[g.act2, g.act1]
        GM = self.star.GM # type: ignore[union-attr]
        OmegaK = np.sqrt(GM/R**3)
        mid = g.mid

        # in case alpha was a single value, we need to broadcast to 3D, then keep active cells
        alpha = (tb.atleast_nd(self.get_alpha(self), d=3) * np.ones(g.shape))[g.active]
        cs2 = (self.prs/self.rho)[g.active] # still fine
        H   = np.sqrt(cs2) / OmegaK
        sig = self.get_surface_density() # only contains active cells!
        sig_gas = tb.get_surface_density(g, rhogas[g.active])
        zH  = (z / H).to_value('') # shape: nx3, nx2, nx1

        uden = u.g/u.cm**3
        shape = g.active_shape
        rho_new = np.zeros(shape) * uden
        # Stokes / alpha parameter
        St_a  = np.sqrt(np.pi/8) * rho_a / (rhogas[g.active] * H) / alpha

        def integrate(start, end, step):
            lneps = np.zeros(shape) # temporary array
            s_ = np.s_[:, mid if step < 0 else mid+1]

            lneps[s_] = np.log(sig/sig_gas * np.sqrt(1 + St_a[s_])) #midplane
            for j in range(start, end, step): # from midplane to pole
                delta_w = zH[:,j+step]**2 - zH[:,j]**2
                # predictor
                S1 = - 0.5 * St_a[:,j]
                new = lneps[:,j] + S1 * delta_w
                new[np.isnan(new)] = tb.tiny_exp # floor NaNs

                # corrector (can be commented out)
                S2 = - 0.5 * St_a[:,j+step]
                new = lneps[:,j] + 0.5 * (S1 + S2) * delta_w
                new[np.isnan(new)] = tb.tiny_exp # floor NaNs

                # finalize
                new = np.minimum(new, lneps[:,j]) # prevent increase
                lneps[:,j+step] = tb.get_capped(new, log=True) # cap values
            return np.exp(lneps) * rhogas[g.active]
    
        if side == 'both' or side == 'top': # top half
            rho_new[:, :mid+1] = integrate(mid, 0, -1)[:, :mid+1]
        if side == 'both' or side == 'bottom': # bottom half
            rho_new[:, mid+1:] = integrate(mid+1, g.NX2-1, 1)[:, mid+1:]

        sig_new = tb.get_surface_density(g, rho_new)
        f = sig/sig_new # enforce correct surface density: f(φ,r)
        rho_new = np.maximum(rho_new * f[:,None], tb.tiny * uden)
        rho_new[np.isnan(rho_new)] = tb.tiny * uden

        rho_updated = np.zeros_like(self.rho)
        rho_updated[g.active] = rho_new * (1-uf) + self.rho[g.active] * uf # type: ignore[assignment]
        self.enforce_boundaries(rho_updated, powerlaw_extrapolation=True) # boundaries

        nans = np.isnan(rho_updated)
        if np.any(nans):
            raise ValueError(f"{self.get_prefix()}: NaNs found in hydrostatic equilibrium density")

        self._pop_region()

        return rho_updated

class Gas(_RadiativeFluid):
    """
    Gas fluid for protoplanetary disk simulations.

    Extends `_RadiativeFluid` with gas-specific methods. Implements gas hydrostatic
    equilibrium with vertical structure determined by pressure support and gravity.

    Attributes
    ----------
    isGas : bool
        Identifier flag (True for Gas)
    isDust : bool
        Identifier flag (False for Gas)
    """

    def __init__(self, **kwargs):
        """
        Initialize a gas fluid object.

        Calls parent class initialization with radiative fluid capabilities
        (opacities, irradiation, viscosity) configured for gas.

        Arguments
        ----------
        **kwargs
            Arguments passed to `_RadiativeFluid.__init__`
            (grid, star, viscosity, irradiation, kabs_wl, ksca_wl, g_wl, mu, gamma, etc.)

        Examples
        --------
        Create a gas fluid with irradiation:

            gas = Gas(grid=grid, star=star, irradiation='gray', mu=2.353)
        """
        self._set_debug_name(**kwargs)
        self._push_region("__init__ (Gas)")

        super().__init__(**kwargs)

        self.isGas  = True
        self.isDust = False

        self._pop_region()

    def compute_hydrostatic_equilibrium(self, rhogas=None, uf=0):
        """
        Compute gas density profile under hydrostatic equilibrium.

        Solves for the vertical gas density structure from pressure gradient balance.
        Uses the sound speed and scale height to determine vertical structure.
        Integrates radial pressure gradients and uses a second-order predictor-corrector
        scheme for accurate vertical profiles.

        Works only in spherical geometry. Enforces correct surface density and prevents
        density from increasing toward poles (prevents material flowing upward).

        Arguments
        ----------
        rhogas : array-like [g/cm^3] or astropy.units.Quantity, optional
            Not used for gas fluids. Included for API compatibility.
            (default: None)
        uf : float [0-1], optional
            Under-relaxation factor for convergence. (default: 0)

        Returns
        -------
        ndarray [g/cm^3]
            Updated gas density field with boundaries applied

        Raises
        ------
        ValueError
            If geometry is not spherical, grid structure is invalid, or NaNs
            appear in result
        """
        self._push_region("compute_hydrostatic_equilibrium")

        g = self.grid
        if g.geometry not in gr.aux_spherical: raise ValueError("Only works in spherical coordinates")
        first, last = g.x2i[g.act2i][[0, -1]]
        halfdisk = np.isclose(first, np.pi/2) or np.isclose(last, np.pi/2)
        if halfdisk: side = 'top' if np.isclose(last, np.pi/2) else 'bottom'
        else: side = 'both'
        if not halfdisk and not np.any(np.isclose(g.x2i[g.act2i], np.pi/2)):
            raise ValueError("When dual-sided, grid must contain pi/2 as a cell interface within active cells. Alternatively, use a half-disk grid.")

        # below, most quantities only contain active cells
        r  = g.x1[g.act1]
        th = g.x2[g.act2]
        R  = g.R[g.act2, g.act1]
        GM = self.star.GM # type: ignore[union-attr]
        OmegaK = np.sqrt(GM/R**3)
        mid = g.mid

        cs2 = (self.prs/self.rho)[g.active]
        H   = np.sqrt(cs2) / OmegaK
        sig = self.get_surface_density() # only contains active cells!

        ulen = u.au
        uden = u.g/u.cm**3
        uvel = u.au/u.yr

        rho_ = self.rho[g.active] # type: ignore[assignment]
        rho_new = np.zeros_like(rho_)
    
        #axis = -1 always returns a radial derivative in cylindrical/spherical
        def d(y, x, axis=-1): return np.gradient(y, x, axis=axis, edge_order=2)
        ln = np.log # note the use of ln below

        lnr         = ln(r/ulen)
        dln_cs2_dth = d(ln(cs2/uvel**2), th, axis=g.THETAAXIS)

        def integrate(start, end, step):
            lnrho = np.zeros(rho_.shape) # temporary array
            s_ = np.s_[:, mid:mid+2] if side == 'both' else np.s_[:, mid]
            lnrho[s_] = ln(sig / np.sqrt(2*np.pi) / H[s_] / uden) # midplane
            for j in range(start, end, step): # from midplane to pole (we slice in θ)
                delta_th = th[j+step]-th[j]
                # predictor
                p = d(lnrho[:,j], lnr)
                vphi2 = d(cs2[:,j], lnr) + cs2[:,j] * p + GM/r
                S1 = vphi2/cs2[:,j]/np.tan(th[j]) - dln_cs2_dth[:,j]
                new = lnrho[:,j] + S1 * delta_th
                new[np.isnan(new)] = tb.tiny_exp # floor NaNs

                # corrector (can be commented out)
                p = d(new, lnr)
                vphi2 = d(cs2[:,j+step], lnr) + cs2[:,j+step] * p + GM/r
                S2 = vphi2/cs2[:,j+step]/np.tan(th[j+step]) - dln_cs2_dth[:,j+step]
                new = lnrho[:,j] + 0.5 * (S1 + S2) * delta_th
                new[np.isnan(new)] = tb.tiny_exp # floor NaNs

                # finalize
                new = np.minimum(new, lnrho[:,j]) # prevent increase
                lnrho[:,j+step] = tb.get_capped(new, log=True) # floor small values
            return np.exp(lnrho) * uden
    
        if side == 'both' or side == 'top': # top half
            rho_new[:, :mid+1] = integrate(mid, 0, -1)[:, :mid+1]
        if side == 'both' or side == 'bottom': # bottom half
            rho_new[:, mid+1:] = integrate(mid+1, g.NX2-1, 1)[:, mid+1:]

        sig_new = tb.get_surface_density(g, rho_new)
        f = sig/sig_new # enforce correct surface density: f(φ,r)
        rho_new = np.maximum(rho_new * f[:,None], tb.tiny * uden)
        rho_new[np.isnan(rho_new)] = tb.tiny * uden

        rho_updated = np.zeros_like(self.rho)
        rho_updated[g.active] = rho_new * (1-uf) + self.rho[g.active] * uf # type: ignore[assignment]
        self.enforce_boundaries(rho_updated) # boundaries

        nans = np.isnan(rho_updated)
        if np.any(nans):
            raise ValueError(f"{self.get_prefix()}: NaNs found in hydrostatic equilibrium density")
        
        self._pop_region()
        return rho_updated

def create_from_snapshot(dirc='./', name='fluid'):
    # fluid objects need also a grid and (optionally) a star object
    model_dirc = f'{dirc}/{name}'
    grid = gr.create_from_snapshot(dirc=model_dirc)
    try: star = st.create_from_snapshot(dirc=model_dirc)
    except FileNotFoundError: star = None # no star object

    # still need to figure out the fluid type. Peek into the param file
    d = tb.read_dictionary(model_dirc)
    if 'isDust' in d:
        if d['isDust']: fluidClass = Dust
        else: fluidClass = Gas
    elif 'isGas' in d:
        if d['isGas']: fluidClass = Gas
        else: fluidClass = Dust
    else:
        raise ValueError(f"Could not determine fluid type from snapshot parameters. Expected 'isDust' or 'isGas' flag.")

    # instantiate the class
    fluid = fluidClass(grid=grid, star=star)
    # __init__ has been run -> star and grid
    # must already exist, so don't read them.
    # Use the create_from_snapshot() function to
    # fully init a fresh object instead.
    tb.read_and_assign_params(fluid, dirc=model_dirc)

    fluid.rho = tb.try_reading(f'{model_dirc}/rho_cgs.dbl', u.g/u.cm**3, shape=grid.shape, force=True)
    fluid.tmp = tb.try_reading(f'{model_dirc}/tmp_cgs.dbl', u.K, shape=grid.shape, force=True)

    def assign_multitype(attr_name, file_name, unit=None, force=True, shape=grid.shape):
        q = tb.try_reading(f'{model_dirc}/{file_name}', unit, force=force)
        try:
            if len(q) > 1: # it's an array, need to reshape
                setattr(fluid, attr_name, q.reshape(shape)) # type: ignore
            else: setattr(fluid, attr_name, q) # it's a single value
        except TypeError: # None has no len(), assign None
            setattr(fluid, attr_name, None)

    assign_multitype('eps', 'eps.dbl')
    assign_multitype('kappaR', 'kappaR_cgs.dbl', u.cm**2/u.g)
    assign_multitype('kappaP', 'kappaP_cgs.dbl', u.cm**2/u.g)
    if fluid.irradiation:
        assign_multitype('kappaPstar', 'kappaPstar_cgs.dbl', u.cm**2/u.g)
        # assign_multitype('tau0', 'tau0.dbl', force=False, shape=)
    
    Nwl = star.nbins # type: ignore[union-attr]
    assign_multitype('kabs_wl', 'kabs_wl_cgs.dbl', u.cm**2/u.g, force=False, shape=Nwl)
    assign_multitype('ksca_wl', 'ksca_wl_cgs.dbl', u.cm**2/u.g, force=False, shape=Nwl)
    assign_multitype('g_wl', 'g_wl.dbl', force=False, shape=Nwl)

    fluid._rebuild_after_read()

    return fluid