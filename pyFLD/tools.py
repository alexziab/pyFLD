import numpy as np
from scipy.integrate import simpson
from astropy import constants as const, units as u
from . import grid as gr, star as st, debug_state as dbg
import pickle as pk
from os import mkdir as os_mkdir
from os.path import exists as os_exists

Rgas = const.k_B / const.u
aR   = 4 * const.sigma_sb / const.c
tiny     = 1e-100
huge     = 1e100
tiny_exp = np.log(tiny) # avoid overflow in exp
huge_exp = np.log(huge) # avoid underflow in exp

def get_capped(arr, side='both', log=False):
    """
    Return a copy of input array with values capped to prevent overflow/underflow.

    Arguments
    ----------
    arr : array-like
        Input array to cap
    side : str, optional
        Which side to cap: 'both', 'min', or 'max' (default: 'both')
    log : bool, optional
        Whether input array is in log space (default: False)

    Returns
    -------
    ndarray
        Copy of input array with values capped

    Raises
    ------
    ValueError
        If 'side' argument is not 'both', 'min', or 'max'
    """
    q = 1 * arr # make a copy
    if side not in ['both', 'min', 'max']:
        raise ValueError("Invalid 'side' argument. Must be 'both', 'min', or 'max'.")
    if log:
        if side in ['both', 'min']: q = np.maximum(q, tiny_exp)
        if side in ['both', 'max']: q = np.minimum(q, huge_exp)
    else:
        if side in ['both', 'min']: q = np.maximum(q, tiny)
        if side in ['both', 'max']: q = np.minimum(q, huge)
    return q

def parse_optool_file(filename):
    """
    Read and parse an optool opacity file.

    Extracts wavelength-dependent absorption opacity, scattering opacity,
    and asymmetry parameter from an optool-formatted file and converts
    to astropy Quantity objects.

    Arguments
    ----------
    filename : str
        Path to the optool file (typically named "dustkappa*.inp")

    Returns
    -------
    wl : astropy.units.Quantity [cm]
        Wavelength sampling points
    kabs : astropy.units.Quantity [cm^2/g]
        Absorption opacity at each wavelength
    ksca : astropy.units.Quantity [cm^2/g]
        Scattering opacity at each wavelength
    g : ndarray
        Asymmetry parameter at each wavelength (Henyey & Greenstein 1941)

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is invalid or cannot be parsed
    """

    # first find the header line and skip its contents
    header = 20
    while True:
        try: data = np.genfromtxt(filename, skip_header=header, unpack=True)
        except ValueError: header += 1
        else: break

    # now apply units
    wl   = data[0] * (1*u.um).to(u.cm)
    kabs = data[1] * u.cm**2 / u.g
    ksca = data[2] * u.cm**2 / u.g
    g    = data[3] # asymmetry parameter

    return wl, kabs, ksca, g

def compute_mean_opacities(wl, kabs, ksca, g, T=None,
                           Tmin=0.1*u.K, Tmax=1e5*u.K, nT=1000):
    """
    Compute Rosseland and Planck mean opacities from wavelength-dependent opacities.

    Integrates absorption and scattering opacities weighted by the Planck and
    inverse-mean Rosseland functions across wavelength to obtain temperature-dependent
    mean opacities.

    Arguments
    ----------
    wl : array-like [cm] or astropy.units.Quantity
        Wavelengths to sample on
    kabs : array-like [cm^2/g] or astropy.units.Quantity
        Absorption opacity at each wavelength
    ksca : array-like [cm^2/g] or astropy.units.Quantity
        Scattering opacity at each wavelength
    g : array-like
        Asymmetry parameter at each wavelength
    T : array-like [K] or astropy.units.Quantity, optional
        Temperature array at which to sample mean opacities. If provided,
        overrides Tmin, Tmax, and nT (default: None)
    Tmin : float [K] or astropy.units.Quantity, optional
        Minimum temperature for mean opacity computation (default: 0.1 K)
    Tmax : float [K] or astropy.units.Quantity, optional
        Maximum temperature for mean opacity computation (default: 1e5 K)
    nT : int, optional
        Number of temperature points to compute (default: 1000)

    Returns
    -------
    T : astropy.units.Quantity [K]
        Array of temperatures at which mean opacities were computed
    kR : astropy.units.Quantity [cm^2/g]
        Rosseland mean opacity at each temperature
    kP : astropy.units.Quantity [cm^2/g]
        Planck mean opacity at each temperature
    """
    
    wl_   = toQuantity(wl, u.cm)
    Tmin_ = toQuantity(Tmin, u.K).to_value('K')
    Tmax_ = toQuantity(Tmax, u.K).to_value('K')
    if T is None: T = np.geomspace(Tmin_, Tmax_, nT) * u.K
    Bnu, Unu = BBflux(T=T[:,None], wl=wl_) # [nT, nwl]

    kabs_cgs = toQuantity(kabs, u.cm**2 / u.g).to_value('cm2/g')
    ksca_cgs = toQuantity(ksca, u.cm**2 / u.g).to_value('cm2/g')
    ktot_cgs = kabs_cgs + (1 - g) * ksca_cgs

    # scipy.simpson requires unitless arrays
    Bnu_cgs  = Bnu.to_value('erg/(cm3*s)')
    Unu_cgs  = Unu.to_value('erg/(cm3*s*K)')
    wl_cgs = wl_.to_value('cm')
    kP = simpson(kabs_cgs * Bnu_cgs, x=wl_cgs, axis=1) \
        / simpson(Bnu_cgs, x=wl_cgs, axis=1) # [nT]
    kP = kP * u.cm**2 / u.g

    kR = simpson(Unu_cgs, x=wl_cgs, axis=1) \
        / simpson(Unu_cgs / ktot_cgs, x=wl_cgs, axis=1) # [nT]
    kR = kR * u.cm**2 / u.g
    return T, kR, kP

def atleast_nd(a, d=3):
    """
    Extend array to at least d dimensions by appending new axes.

    Similar to np.atleast_1d/2d/3d but works for arbitrary dimensions.
    New dimensions are always appended at the end.

    Arguments
    ----------
    a : array-like or scalar
        Input array or scalar
    d : int, optional
        Desired number of dimensions (default: 3)

    Returns
    -------
    ndarray
        Input array reshaped to have at least d dimensions
    """
    arr = np.atleast_1d(a)
    while len(arr.shape) < d: arr = arr.reshape((*arr.shape, 1))
    return arr

def isQuantity(q):
    """
    Check if input is an astropy Quantity.

    Arguments
    ----------
    q : any
        Input object to check

    Returns
    -------
    bool
        True if q is an astropy.units.Quantity, False otherwise
    """
    return isinstance(q, u.Quantity)

def toQuantity(arr, unit):
    """
    Convert array-like input to an astropy Quantity with specified unit.

    If input is already a Quantity, converts to the target unit.
    If input is not a Quantity, multiplies by the specified unit.

    Arguments
    ----------
    arr : array-like or astropy.units.Quantity
        Input array or Quantity
    unit : str or astropy.units.Unit
        Target unit for the result

    Returns
    -------
    astropy.units.Quantity
        Input converted to specified unit

    Raises
    ------
    astropy.units.UnitConversionError
        If arr is a Quantity with incompatible units
    """
    if isQuantity(arr): return arr.to(unit)
    else: return np.array(arr) * u.Unit(unit)

def toArray(q, unit=''):
    """
    Convert astropy Quantity to numpy array (removes units).

    If input is a Quantity, converts to specified unit and returns array.
    If input is not a Quantity, converts to numpy array.

    Arguments
    ----------
    q : array-like or astropy.units.Quantity
        Input Quantity or array
    unit : str or astropy.units.Unit, optional
        Target unit for conversion (default: '')

    Returns
    -------
    ndarray
        Input as numpy array without units

    Raises
    ------
    astropy.units.UnitConversionError
        If input Quantity has incompatible units
    """
    if isQuantity(q): return q.to_value(unit)
    else: return np.array(q)

def T_to_Er(T):
    """
    Convert temperature to radiation energy density.

    Uses the relation Er = aR * T^4 where aR is the radiation constant.

    Arguments
    ----------
    T : float [K] or astropy.units.Quantity
        Temperature

    Returns
    -------
    astropy.units.Quantity [erg/cm^3]
        Radiation energy density
    """
    aR = 4 * const.sigma_sb / const.c
    return aR * T.to('K') ** 4

def Er_to_T(Er):
    """
    Convert radiation energy density to temperature.

    Inverts the relation Er = aR * T^4 to solve for T.

    Arguments
    ----------
    Er : float [erg/cm^3] or astropy.units.Quantity
        Radiation energy density

    Returns
    -------
    astropy.units.Quantity [K]
        Temperature
    """
    aR = 4 * const.sigma_sb / const.c
    return (Er / aR).to('K4') ** 0.25

def BBflux(T=5780*u.K, wl=1e-4*u.cm):
    """
    Compute black body flux spectrum and its temperature derivative.

    Calculates the Planck function and its derivative with respect to
    temperature using the standard black body formula:

    .. code-block:: text

        f = exp(h*c/(λkT))
        B(λ) = 2hc²/λ⁵ / (f - 1)
        U(λ) = 2hc²/λ⁵ * (h*c/(λkT²)) * f / (f - 1)²

    Arguments
    ----------
    T : float [K] or astropy.units.Quantity, optional
        Temperature of the black body (default: 5780 K, solar temperature)
    wl : float [cm] or astropy.units.Quantity, optional
        Wavelengths to sample (default: 1e-4 cm)

    Returns
    -------
    B : astropy.units.Quantity [erg/(cm^3 s)]
        Black body flux at requested wavelengths
    U : astropy.units.Quantity [erg/(cm^3 s K)]
        Derivative of black body flux with respect to temperature
    """

    tmp = toQuantity(T, u.K)
    lam = toQuantity(wl, u.cm)

    h = const.h
    c = const.c
    kB= const.k_B

    expr = np.minimum(h*c/(lam*kB*tmp), huge_exp)

    constant = 2*h*c*c/lam**5
    B = constant / np.expm1(expr)

    constant_U = constant * h*c/(lam*kB*tmp*tmp)
    U = constant_U * np.exp(expr) / (np.expm1(expr))**2
    return B.to('erg/(cm3*s)'), U.to('erg/(cm3*s*K)')

def is_halfdisk(grid:gr.Grid):
    """
    Determine if grid covers only one hemisphere in spherical geometry.

    Checks if the grid's polar angle range is [0, π/2] or [π/2, π],
    indicating a half-disk domain.

    Arguments
    ----------
    grid : pyFLD.grid.Grid
        Computational grid object

    Returns
    -------
    bool
        True if grid covers one hemisphere, False otherwise

    Raises
    ------
    NotImplementedError
        If grid geometry is not spherical
    """
    if grid.geometry not in gr.aux_spherical:
        raise NotImplementedError("Spherical geometry only")
    first, last = grid.x2i[grid.act2i][[0, -1]]
    return np.isclose(first, np.pi/2) or np.isclose(last, np.pi/2)

def get_surface_density(grid:gr.Grid, rho):
    """
    Compute vertically integrated surface density.

    Integrates volume density in the polar angle direction to obtain
    surface density as a function of radius and azimuth.

    Arguments
    ----------
    grid : pyFLD.grid.Grid
        Computational grid object (spherical geometry only)
    rho : astropy.units.Quantity [g/cm^3]
        Volume density field with shape matching grid (including ghost cells)

    Returns
    -------
    astropy.units.Quantity [g/cm^2]
        Vertically integrated surface density (φ, R)

    Raises
    ------
    NotImplementedError
        If grid geometry is not spherical
    """
    if grid.geometry not in gr.aux_spherical:
        raise NotImplementedError("Spherical geometry only.")
    sig_factor = 2 if is_halfdisk(grid) else 1

    dz = grid.dz[grid.act2, grid.act1]
    sig = np.sum(rho*dz, axis=grid.THETAAXIS) * sig_factor
    return sig.to('g/cm2')

# np.vectorize is too slow
def flux_limiter_kley(Rrad):
    """
    Compute radiation flux limiter using the Kley (1989) formula.

    The flux limiter is a function of the normalized radiation gradient R
    that transitions between the free-streaming limit (large R) and
    diffusion limit (small R).

    Arguments
    ----------
    Rrad : float or array-like
        Normalized radiation gradient (dimensionless)

    Returns
    -------
    float or ndarray
        Flux limiter value(s) at requested R
    """
    try:
        len(Rrad) # check if array-like
        R = np.array(Rrad, dtype=float)
        lim = np.empty_like(R)
        mask = R <= 2
        lim[mask] = 2.0/(3 + np.sqrt(9 + 10*R[mask]**2))
        lim[~mask] = 10.0/(10*R[~mask] + 9 + np.sqrt(81 + 180*R[~mask]))
        return lim

    except TypeError: # default to scalar case
        if Rrad <= 2: return 2.0/(3 + np.sqrt(9 + 10*Rrad**2))
        return 10.0/(10*Rrad + 9 + np.sqrt(81 + 180*Rrad))

# some handy functions that might be used frequently
# with the RadiativeFluid classes (Dust, Gas)

def get_alpha_Cecil_Flock_2024(fluid):
    """
    Compute effective alpha viscosity parameter following Cecil & Flock (2024).

    Calculates a temperature-dependent alpha viscosity parameter that transitions
    smoothly from 1e-3 at low temperatures to 0.1 at high temperatures,
    with a transition centered around 900 K and a width of 25 K.

    Arguments
    ----------
    fluid : GenericFluid or higher class
        Fluid object with temperature field

    Returns
    -------
    ndarray
        Effective alpha viscosity parameter across the domain
    """
    T = fluid.tmp.to_value('K')
    return 1e-3 + (0.1-1e-3) * 0.5 * (1 + np.tanh((T-900)/25))

def get_tau0_Flock_2019(fluid):
    """
    Compute optical depth to inner boundary following Flock et al. (2019).

    Calculates the optical depth at the inner edge of the domain using the
    relation τ = ρ * ε * κ_P,* * (R - 3*R_*), where κ_P,* is the Planck
    mean opacity at stellar temperature.

    Arguments
    ----------
    fluid : RadiativeFluid or higher class
        Fluid object with density, effective fraction, and opacity fields

    Returns
    -------
    ndarray
        Optical depth at innermost part of domain (φ, θ, ibeg)
    """
    g, s = fluid.grid, fluid.star
    return (fluid.rho * fluid.eps * fluid.kappaPstar * (g.R-3*s.R))[:, :, g.ibeg]

def get_col0_Flock_2019(fluid):
    """
    Compute fluid column density at inner boundary following Flock et al. (2019).

    Calculates the column density at the inner edge of the domain using the
    relation Σ = ρ * ε * (R - 3*R_*).

    Arguments
    ----------
    fluid : GenericFluid or higher class
        Fluid object with density and effective fraction fields

    Returns
    -------
    ndarray
        Fluid column at innermost part of domain (φ, θ, ibeg)
    """
    g, s = fluid.grid, fluid.star
    return (fluid.rho * fluid.eps * (g.R-3*s.R))[:, :, g.ibeg]

def get_fraction_Isella_Natta_2005(fluid):
    """
    Compute dust sublimation fraction following Isella & Natta (2005).

    Calculates the effective dust fraction accounting for sublimation at
    high temperatures using a smooth transition function. The sublimation
    temperature depends on dust density via T_subl = 2000 * ρ^0.0195 K.

    Arguments
    ----------
    fluid : GenericFluid or higher class
        Fluid object with temperature and density fields. Ideally Dust.

    Returns
    -------
    ndarray
        Dust sublimation fraction (0 to 1) across the domain
    """
    T_subl = 2000.0 * fluid.rho.to_value('g/cm3') ** 0.0195
    T      = fluid.tmp.to_value('K')
    fmin   = 1e-10
    f      = fmin + (1-fmin) * 0.5 * (1 + np.tanh(-(T-T_subl)/100))
    return f

def get_gas_Planck_opacity_Freedman_2008(fluid):
    """
    Compute gas Planck mean opacity following Freedman et al. (2008).

    Simple temperature-dependent parameterization of gas opacity based on
    fitting functions from Freedman et al. (2008).

    Arguments
    ----------
    fluid : GenericFluid or higher class
        Fluid object with temperature field. Ideally Gas.

    Returns
    -------
    astropy.units.Quantity [cm^2/g]
        Planck mean opacity at each location in the domain
    """
    T  = fluid.tmp.to_value('K')
    kP = 1e-3 + (1-1e-3) * 0.5 * (1 + np.tanh((T-100)/20))
    return kP * u.cm**2 / u.g
    
def get_gas_Rosseland_opacity_Freedman_2008(fluid):
    """
    Compute gas Rosseland mean opacity following Freedman et al. (2008).

    Density-dependent parameterization of gas opacity based on fitting
    functions from Freedman et al. (2008).

    Arguments
    ----------
    fluid : GenericFluid or higher class
        Fluid object with density field. Ideally Gas.

    Returns
    -------
    astropy.units.Quantity [cm^2/g]
        Rosseland mean opacity at each location in the domain
    """
    r  = fluid.rho.to_value('g/cm3')
    kR = 2e1 * r ** 0.8 + 1e-6
    return kR * u.cm**2 / u.g

def get_zero(fluid):
    """
    Return a zero value with appropriate shape and units for a fluid.

    Helper function for enrolling custom functions that compute fluid
    properties (e.g., opacity, viscosity) with proper dimensionality.

    Arguments
    ----------
    fluid : RadiativeFluid
        Fluid object (used for context)

    Returns
    -------
    float
        Zero value (shape and units handled by broadcasting)
    """
    return 0.0 #np.zeros(fluid.grid.shape, dtype=float)

def get_one(fluid):
    """
    Return a unit value with appropriate shape and units for a fluid.

    Helper function for enrolling custom functions that compute fluid
    properties (e.g., opacity, viscosity) with proper dimensionality.

    Arguments
    ----------
    fluid : RadiativeFluid
        Fluid object (used for context)

    Returns
    -------
    float
        Unit value (shape and units handled by broadcasting)
    """
    return 1.0 #np.ones(fluid.grid.shape, dtype=float)

def have_same_class_contents(cls1, cls2):
    """
    Compare instance attributes and methods of two class objects.

    Checks if two class instances have identical instance-level attributes
    and methods, excluding built-in methods. Useful for validating that
    objects in a collection have consistent state.

    Arguments
    ----------
    cls1 : object
        First object to compare
    cls2 : object
        Second object to compare

    Returns
    -------
    bool
        True if both objects have identical instance contents, False otherwise
    """
    # only instance attributes/methods, ignore built-ins
    keys1 = set(k for k in cls1.__dict__ if not k.startswith('__'))
    keys2 = set(k for k in cls2.__dict__ if not k.startswith('__'))
    if keys1 != keys2:
        return False
    for k in keys1:
        if np.any(cls1.__dict__[k] != cls2.__dict__[k]):
            return False
    return True

def get_Wien_wavelength(T):
    """
    Compute peak wavelength of black body emission (Wien's displacement law).

    Arguments
    ----------
    T : float [K] or astropy.units.Quantity
        Temperature

    Returns
    -------
    astropy.units.Quantity [um]
        Peak wavelength at which black body emission is maximum
    """
    return (const.b_wien / toQuantity(T, u.K)).to('um')

def get_Wien_temperature(wl):
    """
    Compute temperature from peak wavelength (Wien's displacement law).

    Inverts Wien's displacement law to determine the black body temperature
    corresponding to a given peak emission wavelength.

    Arguments
    ----------
    wl : float [cm] or astropy.units.Quantity
        Peak emission wavelength

    Returns
    -------
    astropy.units.Quantity [K]
        Black body temperature
    """
    return (const.b_wien / toQuantity(wl, u.cm)).to('K')

def mkdir(name):
    if not os_exists(name): os_mkdir(name)

def get_parameters(obj, skipped_keys=[]):

    keys = []
    vals = []

    for key, val in obj.__dict__.items():
        if key in skipped_keys: continue

        else:
            keys.append(key)
            vals.append(val)

    return keys, vals

def extract_dictionary(obj, skipped_keys=[]):
    keys, vals = get_parameters(obj, skipped_keys=skipped_keys)
    params = {}
    for key, val in zip(keys, vals): params[key] = val
    return params

def write_dictionary(d, dirc='./'):
    with open(f'{dirc}/parameters.pk', 'wb') as f: pk.dump(d, f)

def read_dictionary(dirc='./'):
    with open(f'{dirc}/parameters.pk', 'rb') as f: params = pk.load(f)
    return params

def assign_dictionary(obj, d):
    for key, val in d.items(): setattr(obj, key, val)

def extract_and_store_params(obj, dirc='./', skipped_keys=[]):
    params = extract_dictionary(obj, skipped_keys=skipped_keys)
    write_dictionary(params, dirc=dirc)

def read_and_assign_params(obj, dirc='./'):
    params = read_dictionary(dirc)
    assign_dictionary(obj, params)

def try_reading(filename, unit=None, shape=None, force=False):
    q = None

    if os_exists(filename):
        q = np.fromfile(filename)
        if shape is not None: q = q.reshape(shape)
        if unit is not None: q = q * u.Unit(unit)
    else:
        if force:
            raise ValueError(f"Required file {filename} not found.")

    return q

def try_writing(q, filename, unit=None):
    if q is None: return
    if isQuantity(q) and unit is None:
        raise ValueError("a unit should be provided. Hint:", q.unit)

    if unit != None: q.to_value(unit).tofile(filename)
    else: q.tofile(filename)