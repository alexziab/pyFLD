import numpy as np
from astropy import constants as const, units as u
from . import debug_state as dbg, tools as tb
from scipy.integrate import simpson

class Star:
    """
    Stellar parameters for radiative irradiation calculations.

    Holds stellar mass, luminosity, effective temperature, and optionally
    a frequency-dependent spectrum. Computes stellar radius from the
    Stefan-Boltzmann law.

    Arguments
    ----------
    M : float [Msun] or astropy.units.Quantity, optional
        Stellar mass (default: 1 Msun)
    L : float [Lsun] or astropy.units.Quantity, optional
        Stellar luminosity (default: 1 Lsun)
    T : float [K] or astropy.units.Quantity, optional
        Stellar effective temperature (default: 5772 K)
    wl : array-like [cm] or astropy.units.Quantity, optional
        Wavelengths for spectrum sampling in cm (default: None, no spectrum)

    Attributes
    ----------
    M : astropy.units.Quantity
        Stellar mass in solar masses
    L : astropy.units.Quantity
        Stellar luminosity in solar luminosities
    T : astropy.units.Quantity
        Stellar effective temperature in Kelvin
    R : astropy.units.Quantity
        Stellar radius in solar radii (computed from L and T)
    GM : astropy.units.Quantity
        Gravitational parameter G * M
    wl : astropy.units.Quantity, optional
        Wavelengths for spectrum sampling in cm (set if provided at initialization)
    nbins : int, optional
        Number of wavelength bins (set if wl is provided)
    L_wl : astropy.units.Quantity, optional
        Stellar luminosity at each wavelength such that sum(L_wl) = L (set if wl is provided)
    """

    wl = None
    L_wl = None
    nbins = None

    def __init__(self, M=1.0 * u.M_sun, L=1.0 * u.L_sun, T=5772 * u.K, wl=None):
        """
        Initialize a Star object with mass, luminosity, and temperature.

        If wavelengths are provided, computes the frequency-dependent spectrum
        from the Planck function normalized to the total luminosity.
        """
        self.prefix = "Star"
        self.L = tb.toQuantity(L, u.L_sun)
        self.M = tb.toQuantity(M, u.M_sun)
        self.T = tb.toQuantity(T, u.K)
        self.GM = const.G * self.M

        # compute stellar radius from L and T
        self.R = np.sqrt( self.L / (4*np.pi*const.sigma_sb*self.T**4) ).to('Rsun')

        if wl is not None: # build stellar spectrum
            self.wl = tb.toQuantity(tb.atleast_nd(wl, d=1), u.cm)
            self.nbins = len(self.wl)
            self.get_and_set_spectral_L()
            self._enforce_L_consistency()
        
    def get_wavelength_spacing(self):
        """
        Compute wavelength spacing array for spectrum calculations.

        Assumes wavelengths are logarithmically spaced and computes spacing
        in logarithmic space. Works with any spacing pattern.

        Returns
        -------
        astropy.units.Quantity
            Wavelength spacing array in cm

        Raises
        ------
        ValueError
            If wavelength array wl is not defined for this star
        """
        if self.wl is None:
            raise ValueError("Wavelength array wl is not defined for this star.")
        # wl is likely logarithmically spaced -> take gradient in log space
        log_wl = np.log(self.wl.to_value('cm'))
        dlog_wl = np.gradient(log_wl, edge_order=2)
        dwl = (self.wl * dlog_wl).to('cm')
        return dwl

    def get_spectral_L(self):
        """
        Compute frequency-dependent stellar luminosity from Planck function.

        Calculates spectral luminosity at each wavelength using the Planck
        function and normalizes such that the integrated sum equals the total
        luminosity L.

        Returns
        -------
        astropy.units.Quantity
            Stellar luminosity at each wavelength in solar luminosities
        """
        dwl = self.get_wavelength_spacing()

        dL_dwl = (4*np.pi**2 * self.R**2 * tb.BBflux(T=self.T, wl=self.wl)[0]).to('Lsun/cm')
        L_wl = (dL_dwl * dwl).to('Lsun')
        return L_wl

    def _set_spectral_L(self, L_wl):
        """
        Set the frequency-dependent stellar luminosity array.

        Validates that the provided luminosity array matches the number of
        wavelength bins and stores it as a Quantity in solar luminosities.

        Arguments
        ----------
        L_wl : array-like [Lsun] or astropy.units.Quantity
            Stellar luminosity at each wavelength

        Raises
        ------
        ValueError
            If wavelength array wl is not defined or if L_wl length does not
            match the number of wavelength bins
        """
        if self.wl is None:
            raise ValueError("Wavelength array wl is not defined for this star.")
        if len(L_wl) != self.nbins:
            raise ValueError(f"Length of L_wl ({len(L_wl)}) does not match number of wavelength bins ({self.nbins}).")
        self.L_wl = tb.toQuantity(L_wl, u.L_sun)
    
    def get_and_set_spectral_L(self):
        """
        Compute and store the frequency-dependent spectrum.

        Calculates spectral luminosity from the Planck function and ensures
        consistency with the total luminosity through normalization.
        """
        L_wl = self.get_spectral_L()
        self._set_spectral_L(L_wl)
        self._enforce_L_consistency()

    def _enforce_L_consistency(self):
        """
        Ensure frequency-dependent spectrum integrates to total luminosity.

        Recomputes the total luminosity from the frequency-dependent spectrum
        and updates the L attribute if integration differs from the provided value.
        Issues a warning if recalculation becomes necessary.

        Raises
        ------
        ValueError
            If wavelength array wl is not defined for this star
        """
        if self.wl is None:
            raise ValueError("Wavelength array wl is not defined for this star.")
        L_wl = self.get_spectral_L()
        Ltot = np.sum(L_wl).to('Lsun')
        if not np.isclose(self.L, Ltot, rtol=1e-4):
            dbg.format_warning(prefix=self.prefix, msg=f"Recalculated frequency-dependent stellar luminosity ({Ltot:.16e}) "
                    f"does not match provided L ({self.L:.16e}). L is set to {Ltot:.16e}.", indent=dbg.depth())
            self.L = Ltot

    def write_snapshot(self, dirc='./', name='star'):
        tb.mkdir(dirc)
        model_dirc = f'{dirc}/{name}'
        tb.mkdir(model_dirc)

        skipped_keys = ['wl', 'nbins', 'L_wl']
        tb.extract_and_store_params(self, dirc=model_dirc,
                                    skipped_keys=skipped_keys)
        tb.try_writing(self.wl, f'{model_dirc}/wl_cgs.dbl', 'cm')
        tb.try_writing(self.L_wl, f'{model_dirc}/L_wl_cgs.dbl', 'erg/s')

    def _rebuild_after_read(self):
        if self.wl is not None and self.L_wl is not None:
            assert len(self.wl) == len(self.L_wl)
            self.nbins = len(self.wl)

def create_from_snapshot(dirc='./', name='star'):
    model_dirc = f'{dirc}/{name}/'
    star = Star() # default
    tb.read_and_assign_params(star, dirc=model_dirc)
    star.wl   = tb.try_reading(f'{model_dirc}/wl_cgs.dbl', 'cm')
    star.L_wl = tb.try_reading(f'{model_dirc}/L_wl_cgs.dbl', 'erg/s')
    star._rebuild_after_read()
    return star