"""
Radiative transfer helper functions for RADMC3D integration.

A collection of tools to streamline setting up and analyzing RADMC3D models
when used alongside ``radmc3dPy``. Provides utilities for managing stars, grids,
opacities, and density/temperature fields in a standardized format.

Typical Workflow
^^^^^^^^^^^^^^^^
1. Set up a star and grid using :class:`radmc_helper.Star` and :class:`radmc_helper.Grid`.
2. Define a volume density field or use helper functions with surface density
   and temperature profiles.
3. Generate dust opacities using OpTool or equivalent.
4. Use ``write_*()`` functions to output RADMC3D-format files.
5. Run RADMC3D (mctherm/image) to compute radiative transfer.
6. Read results with ``radmc3dPy.analyze.readData()`` and process with
   helper functions.

See Also
--------
radmc_helper.Star : RADMC3D-compatible star class (extends star.Star)
radmc_helper.Grid : RADMC3D-compatible grid class (extends grid.Grid)
write_wavelength_array : Write wavelength grid
write_density_array : Write dust density
write_temperature_array : Write dust temperature
read_and_format_radmc_data : Import RADMC3D results
"""

import numpy as np
from astropy import constants as const, units as u
import copy
from . import debug_state as dbg, grid as gr, tools as tb, star as st
from os import chdir, getcwd

class Star(st.Star):
    """
    RADMC3D-compatible star class with file I/O support.

    Extends :class:`star.Star` to add RADMC3D file I/O capabilities. Reads/writes
    RADMC3D-format star files (stars.inp) with spectral information and stellar
    parameters (mass, radius, temperature, luminosity).

    Parameters
    ----------
    p : list of 3 floats, optional
        Position of the star in Cartesian coordinates [cm].
        Default is ``[0, 0, 0]``. Currently not used in output.

    dirc : str, optional
        Directory where the star file will be written.
        Default is ``'./'``.

    flnm : str, optional
        Filename of the RADMC3D star file.
        Default is ``'stars.inp'``.

    ``**super_kwargs``
        Additional keyword arguments passed to the parent :class:`star.Star`
        class (e.g., M, L, T, wl).

    Attributes
    ----------
    dirc : str
        Output directory for RADMC3D files.
    flnm : str
        Full path to the star file.
    p : list of 3 floats
        Star position [cm] (x, y, z coordinates).

    Examples
    ---------

    >>> rmc_star = radmc_helper.Star()
    >>> rmc_star.read_from_file('stars.inp')

    >>> rmc_star = radmc_helper.Star()
    >>> rmc_star.setup_from(<a star.Star object>)

    See Also
    --------
    star.Star : Base star class with stellar parameters
    write : Output star parameters to RADMC3D stars.inp file
    read_from_file : Load star parameters from RADMC3D stars.inp file
    """

    def __init__(self, p=[0,0,0], dirc='./', flnm='stars.inp', **super_kwargs):
        super().__init__(**super_kwargs)

        self.dirc = dirc
        self.flnm = f'{dirc}/{flnm}'
        self.p = p # ! add astropy quantity support

    def write(self, flnm=None):
        """
        Write star parameters to RADMC3D stars.inp file.

        Outputs stellar mass, radius, temperature, position, and spectral
        information in RADMC3D format. Currently supports single-star systems.

        Parameters
        ----------
        flnm : str, optional
            Output filename. If None, uses the filename from initialization.
            Default is None.

        Returns
        -------
        None
            Writes to file as a side effect.

        Notes
        -----
        - Format: RADMC3D format version 2
        - Temperature is written as negative value in file
        - Wavelengths are stored for each spectral element
        - Only one star per file is currently supported

        Examples
        --------
        >>> rmc_star = Star(M=1*u.Msun, T=5772*u.K, R=1*u.Rsun, wl=np.logspace(-1, 4, 100)*u.um)
        >>> rmc_star.write('stars.inp')
        """

        wl = self.wl.to_value('um')

        mstar = self.M.to_value('g')
        rstar = self.R.to_value('cm')
        tstar = self.T.to_value('K')
        p     = self.p

        filename = flnm if flnm is not None else self.flnm
        with open(filename, 'w') as f:
            f.write('2\n')
            f.write(f'1 {wl.size}\n\n')
            f.write(f'{rstar:.18e} {mstar:.18e} {p[0]:.18e} {p[1]:.18e} {p[2]:.18e}\n\n')
            for v in wl: f.write(f'{v:.18e}\n')
            f.write(f'\n{-tstar:.18e}\n')

    def setup_from(self, star: st.Star):
        """
        Initialize this star from another Star object.

        Copies all attributes (mass, radius, temperature, luminosity,
        wavelengths, etc.) from the source star to this instance.

        Parameters
        ----------
        star : star.Star
            Source star object to copy parameters from.

        Returns
        -------
        None
            Modifies instance in place.

        Examples
        --------
        >>> source_star = star.Star(wl=np.logspace(-1, 4, 100)*u.um)
        >>> radmc_star = radmc_helper.Star()
        >>> radmc_star.setup_from(source_star)
        """

        for attr in star.__dict__:
            setattr(self, attr, getattr(star, attr))

    def read_from_file(self, filename=None):
        """
        Read star parameters from RADMC3D stars.inp file.

        Parses a RADMC3D format star file and initializes stellar parameters
        (mass, radius, temperature, luminosity, spectral data).

        Parameters
        ----------
        filename : str, optional
            Input filename. If None, uses the filename from initialization.
            Default is None.

        Returns
        -------
        None
            Initializes parent class via super().__init__() with extracted
            parameters.

        Raises
        ------
        ValueError
            If file format version is not 2.
            If number of stars is not 1.

        Notes
        -----
        - Supports RADMC3D format version 2
        - Only single-star files are supported
        - Temperature is stored as negative in file, converted to positive
        - Wavelengths converted from microns to astropy units
        - Luminosity computed from Stefan-Boltzmann law

        Examples
        --------
        >>> rmc_star = radmc_helper.Star()
        >>> rmc_star.read_from_file('stars.inp')
        """

        flnm = filename if filename is not None else self.flnm
        with open(flnm, 'r') as f:
            if int(f.readline()) != 2: raise ValueError('iformat should be 2')
            nstars, Npts = map(int, f.readline().split())
            if nstars != 1: raise ValueError('Only one star is supported at the moment.')
            f.readline() # skip
            rstar, mstar, x, y, z = map(float, f.readline().split())
            wl = np.fromfile(f, sep='\n', count=Npts, dtype=np.float64)
            tstar = -float(f.readline())
        
        R = rstar * u.cm
        T = tstar * u.K
        M = mstar * u.g
        L = 4 * np.pi * R ** 2 * const.sigma_sb * T ** 4

        super().__init__(M=M, L=L, T=T, wl=wl*u.um)
    
    def update_wavelengths(self, wl):
        """
        Update wavelength grid and recompute spectral luminosity.

        Replaces the wavelength array and recalculates luminosity in each
        wavelength bin based on stellar spectrum.

        Parameters
        ----------
        wl : array_like [cm] or astropy.units.Quantity
            New wavelength array. Can be astropy quantity or numpy array
            (interpreted as cm if plain array).

        Returns
        -------
        None
            Modifies instance attributes (wl, nbins, spectral_L) in place.

        Notes
        -----
        - Wavelengths are stored in cm internally
        - Spectral luminosity recomputed assuming Planck spectrum
        - Uses methods get_and_set_spectral_L() and _enforce_L_consistency()
        - Total luminosity should remain consistent

        Examples
        --------
        >>> rmc_star = radmc_helper.Star(wl=np.logspace(-1, 4, 50)*u.um)
        >>> new_wl = np.logspace(-1, 4, 200)*u.um  # Finer grid
        >>> rmc_star.update_wavelengths(new_wl)
        """

        self.wl = tb.toQuantity(wl, u.cm)
        self.nbins = len(self.wl)
        self.get_and_set_spectral_L()
        self._enforce_L_consistency()

class Grid(gr.Grid):
    """
    RADMC3D-compatible grid class with file I/O support.

    Extends :class:`grid.Grid` to add RADMC3D file I/O capabilities. Reads/writes
    RADMC3D-format grid files (amr_grid.inp) with spherical coordinates (radial,
    polar, azimuthal dimensions) and no AMR structure (regular grid only).

    Parameters
    ----------
    dirc : str, optional
        Directory where the grid file will be written.
        Default is ``'./'``.

    flnm : str, optional
        Filename of the RADMC3D grid file.
        Default is ``'amr_grid.inp'``.

    ``**super_kwargs``
        Additional keyword arguments passed to the parent :class:`grid.Grid`
        class (e.g., x1i, x2i, x3i, geometry).

    Attributes
    ----------
    dirc : str
        Output directory for RADMC3D files.
    flnm : str
        Full path to the grid file.

    Examples
    --------
    Create a spherical grid and export to RADMC3D:

    >>> ri = np.logspace(0, 3, 100)*u.au   # 100 radial cells
    >>> thi = np.linspace(0, np.pi, 50)     # 50 polar cells
    >>> rmc_grid = radmc_helper.Grid(x1i=ri, x2i=thi, geometry='spherical')
    >>> rmc_grid.write('amr_grid.inp')

    Load from an existing file:

    >>> rmc_grid = radmc_helper.Grid() # will raise a warning
    >>> rmc_grid.read_from_file('amr_grid.inp')
    >>> print(f"Nr={rmc_grid.NX1}, Ntheta={rmc_grid.NX2}")

    See Also
    --------
    grid.Grid : Base grid class
    write : Output grid parameters to RADMC3D amr_grid.inp file
    read_from_file : Load grid parameters from RADMC3D amr_grid.inp file
    """

    def __init__(self, dirc='./', flnm='amr_grid.inp', **super_kwargs):

        self.dirc = dirc
        self.flnm = f'{dirc}/{flnm}'
        try: super().__init__(**super_kwargs)
        except TypeError: # if super_kwargs is empty, leave an empty class
            dbg.format_warning(msg='No grid parameters provided. Created an empty Grid object. Use setup_from() or read_from_file() to set up the grid.')

    def setup_from(self, grid: gr.Grid):
        """
        Initialize this grid from another Grid object.

        Copies all attributes (coordinates, dimensions, geometry, etc.)
        from the source grid to this instance.

        Parameters
        ----------
        grid : grid.Grid
            Source grid object to copy parameters from.

        Returns
        -------
        None
            Modifies instance in place.

        Examples
        --------
        >>> source_grid = gr.Grid(x1i=ri, x2i=thi, x3i=phii, geometry='spherical')
        >>> rmc_grid = radmc_helper.Grid()
        >>> rmc_grid.setup_from(source_grid)
        """

        for attr in grid.__dict__:
            setattr(self, attr, getattr(grid, attr))

    def write(self, flnm=None):
        """
        Write grid parameters to RADMC3D amr_grid.inp file.

        Outputs spherical coordinate grid (r, theta, phi) in RADMC3D format.
        Automatically handles 2D (r-theta) and 3D (r-theta-phi) grids.

        Parameters
        ----------
        flnm : str, optional
            Output filename. If None, uses the filename from initialization.
            Default is None.

        Returns
        -------
        None
            Writes to file as a side effect.

        Notes
        -----
        - Format: RADMC3D format version 1, regular grid (no AMR)
        - Coordinate system: spherical (100)
        - Automatically detects if phi dimension is used
        - Grid is assumed to use only active cells (no ghost cells in output)
        - Radial grid converted from cm to cm for output

        Examples
        --------
        >>> ri = np.logspace(0, 3, 100)*u.au
        >>> thi = np.linspace(0, np.pi, 50)
        >>> phii = np.linspace(0, 2*np.pi, 10)
        >>> rmc_grid = radmc_helper.Grid(x1i=ri, x2i=thi, x3i=phii, geometry='spherical')
        >>> rmc_grid.write('amr_grid.inp')
        """

        ri_cm = self.x1i[self.act1i].to_value('cm')
        thi   = 1 * self.x2i[self.act2i]
        phii  = 1 * self.x3i[self.act3i]
        have_phi = len(phii) > 2

        filename = flnm if flnm is not None else self.flnm
        with open(filename, 'w') as f:
            f.write('1\n')     # iformat
            f.write('0\n')     # AMR grid style  (0=regular grid, no AMR)
            f.write('100\n')   # Coordinate system: spherical
            f.write('0\n')     # gridinfo
            f.write('1 1 %d\n'%have_phi) # Include r,theta,phi coordinates
            f.write(f'{self.NX1} {self.NX2} {self.NX3}\n') # Size of grid
            for v in ri_cm: f.write(f'{v:.18e}\n')
            for v in thi:   f.write(f'{v:.18e}\n')
            for v in phii:  f.write(f'{v:.18e}\n')

    def read_from_file(self, filename=None):
        """
        Read grid parameters from RADMC3D amr_grid.inp file.

        Parses a RADMC3D format grid file and initializes grid parameters
        (radial, polar, azimuthal coordinates).

        Parameters
        ----------
        filename : str, optional
            Input filename. If None, uses the filename from initialization.
            Default is None.

        Returns
        -------
        None
            Initializes parent class via super().__init__() with extracted
            coordinate arrays.

        Raises
        ------
        ValueError
            If file format version is not 1.
            If grid style is not 0 (regular grid).
            If coordinate system is not 100 (spherical).

        Notes
        -----
        - Supports RADMC3D format version 1 only
        - Only regular grids supported (AMR style = 0)
        - Ghost cells are added automatically (2 cells, logarithmic in r)
        - Output geometry is set to 'spherical'

        Examples
        --------
        >>> rmc_grid = radmc_helper.Grid()
        >>> rmc_grid.read_from_file('amr_grid.inp')
        >>> print(f"Radial cells: {rmc_grid.NX1}")
        >>> print(f"Grid range: {rmc_grid.x1c[0]:.2f} to {rmc_grid.x1c[-1]:.2f}")
        >>> print(f"Geometry: {rmc_grid.geometry}")
        """

        flnm = filename if filename is not None else self.flnm
        with open(flnm, 'r') as f:
            if int(f.readline()) != 1: raise ValueError('iformat should be 1')
            if int(f.readline()) != 0: raise ValueError('grid style should be 0')
            if int(f.readline()) != 100: raise ValueError('coord_sys should be 100')
            f.readline() # gridinfo, skip
            f.readline() # r,theta,phi coordinates, skip
            Nr, Ntheta, Nphi = map(int, f.readline().split())
            ri = np.fromfile(f, sep='\n', count=Nr+1, dtype=np.float64)
            thi = np.fromfile(f, sep='\n', count=Ntheta+1, dtype=np.float64)
            phii = np.fromfile(f, sep='\n', count=Nphi+1, dtype=np.float64)
        
        ri = gr.add_ghost_cells(ri*u.cm, 2, log=True)
        thi = gr.add_ghost_cells(thi, 2)
        phii = gr.add_ghost_cells(phii, 2)

        super().__init__(x1i=ri, x2i=thi, x3i=phii, geometry='spherical')

def get_wavelength_array(wl_min=0.1*u.um, wl_max=1*u.cm, Npts=100, refine_10um=True, Tstar=5772*u.K):
    """
    Generate a logarithmically-spaced wavelength array for radiative transfer.

    Creates a wavelength grid suitable for RADMC3D simulations, with optional
    refinement around the 10 um silicate feature and validation of stellar
    peak coverage.

    Parameters
    ----------
    wl_min : float or astropy.units.Quantity, optional
        Minimum wavelength [um]. Default is 0.1 um.
    wl_max : float or astropy.units.Quantity, optional
        Maximum wavelength [um]. Default is 1 cm.
    Npts : int, optional
        Number of wavelength points (per segment if refine_10um=True).
        Default is 100.
    refine_10um : bool, optional
        If True, refines grid around 10 um silicate feature using 2*Npts
        points in the range [7, 25] um. Default is True.
    Tstar : float or astropy.units.Quantity, optional
        Stellar temperature [K] used to estimate emission peak.
        Default is 5772 K (Sun).

    Returns
    -------
    wl : astropy.units.Quantity
        Wavelength array [um], logarithmically spaced.

    Warns
    -----
    UserWarning
        If wl_min is greater than 1/4 of the stellar emission peak wavelength,
        indicating insufficient resolution of stellar spectrum.

    Notes
    -----
    - If refine_10um=True, grid has 3 segments:
        - [wl_min, 7 um]: Npts//2 points
        - [7 um, 25 um]: 2*Npts points (refined around 10 um feature)
        - [25 um, wl_max]: Npts//2 points
    - Stellar emission peak computed using Wien displacement law: lambda_peak = 2898 um*K / T
    - Useful for dust continuum radiative transfer with silicate absorption

    Examples
    --------
    >>> wl = get_wavelength_array(wl_min=0.1*u.um, wl_max=10*u.mm, Npts=50)
    >>> print(f"Total points: {len(wl)}")
    >>> # Output: Total points: 175 (50//2 + 2*50 + 50//2)

    >>> # Without refinement (simple log grid)
    >>> wl_simple = get_wavelength_array(Npts=200, refine_10um=False)
    >>> print(f"Total points: {len(wl_simple)}")
    >>> # Output: Total points: 200
    """

    wl_min = tb.toQuantity(wl_min, u.um)
    wl_max = tb.toQuantity(wl_max, u.um)
    Tstar  = tb.toQuantity(Tstar,  u.K)

    wl_peak = 2898 * u.um * u.K / Tstar
    if wl_min > wl_peak / 4:
        dbg.format_warning(msg=f"Warning: stellar emission peaks at {wl_peak.to('um'):.2f}, and minimum wavelength is {wl_min.to('um'):.2f}. Consider lowering wl_min.", indent=dbg.depth())

    if not refine_10um: return np.geomspace(wl_min.to_value('um'), wl_max.to_value('um'), Npts)

    wl_min_um  = wl_min.to_value('um')
    wl_max_um  = wl_max.to_value('um')
    wl_ref_min_um = 7.0 # um
    wl_ref_max_um = 25.0 # um
    
    arr1 = np.geomspace(wl_min_um, wl_ref_min_um, Npts//2, endpoint=False)
    arr2 = np.geomspace(wl_ref_min_um, wl_ref_max_um, 2*Npts, endpoint=False)
    arr3 = np.geomspace(wl_ref_max_um, wl_max_um, Npts//2, endpoint=True)
    return np.concatenate([arr1, arr2, arr3]) * u.um

def write_wavelength_array(wl, dirc='./', flnm='wavelength_micron.inp'):
    """
    Write wavelength array to RADMC3D format file.

    Outputs a wavelength grid to wavelength_micron.inp in RADMC3D format.

    Parameters
    ----------
    wl : array_like [um] or astropy.units.Quantity
        Wavelength array. Can be astropy quantity or numpy array
        (interpreted as um if plain array).
    dirc : str, optional
        Output directory. Default is ``'./'``.
    flnm : str, optional
        Output filename. Default is ``'wavelength_micron.inp'``.

    Returns
    -------
    None
        Writes to file as a side effect.

    Notes
    -----
    - First line: number of wavelength points
    - Following lines: wavelength values in um, one per line
    - Values written in scientific notation with 18 decimal places
    - Compatible with RADMC3D wavelength_micron.inp format

    Examples
    --------
    >>> wl = np.logspace(-1, 4, 100)*u.um
    >>> write_wavelength_array(wl, dirc='./', flnm='wavelength_micron.inp')
    >>> # File format:
    >>> # 100
    >>> # 1.000000000000000e-01
    >>> # ...
    """

    wl = tb.toQuantity(wl, u.um).to_value('um')

    with open(f'{dirc}/{flnm}', 'w') as f:
        f.write(f'{wl.size}\n')
        for v in wl: f.write(f'{v:.18e}\n')

def read_wavelength_array(dirc='./', flnm='wavelength_micron.inp'):
    """
    Read wavelength array from RADMC3D format file.

    Parses a wavelength_micron.inp file and returns the wavelength array.

    Parameters
    ----------
    dirc : str, optional
        Input directory. Default is ``'./'``.
    flnm : str, optional
        Input filename. Default is ``'wavelength_micron.inp'``.

    Returns
    -------
    wl : astropy.units.Quantity
        Wavelength array [um].

    Notes
    -----
    - First line: number of wavelength points (Npts)
    - Following Npts lines: wavelength values in um
    - Expects space or newline-separated format

    Examples
    --------
    >>> wl = read_wavelength_array(dirc='./', flnm='wavelength_micron.inp')
    >>> print(f"Wavelength range: {wl[0]:.2e} to {wl[-1]:.2e}")
    >>> print(f"Number of points: {len(wl)}")
    """
    with open(f'{dirc}/{flnm}', 'r') as f:
        Npts = int(f.readline())
        wl = np.fromfile(f, sep='\n', count=Npts, dtype=np.float64)
    return wl * u.um

def write_density_array(rho=[], dirc='./', flnm='dust_density.binp'):
    """
    Write dust density array(s) to RADMC3D binary format file.

    Outputs volume density for one or more dust species to dust_density.binp
    in RADMC3D binary format.

    Parameters
    ----------
    rho : list of array_like [g/cm3] or astropy.units.Quantity
        List of dust density arrays (one per species). Each array can be
        astropy quantity or numpy array (interpreted as g/cm3 if plain array).
        Default is empty list.
    dirc : str, optional
        Output directory. Default is ``'./'``.
    flnm : str, optional
        Output filename. Default is ``'dust_density.binp'``.

    Returns
    -------
    None
        Writes to binary file as a side effect.

    Notes
    -----
    - Binary format: C-style (little-endian double precision)
    - Header format: [iformat=1, precision=8, Ncells, Nr_species]
    - All density arrays must have same shape (Ncells)
    - Single species: pass as [rho], not rho
    - Units converted to g/cm3 before writing

    Examples
    --------
    >>> # Single dust species
    >>> rho = 1e-11 * u.g/u.cm**3 * np.ones((100, 50, 30))
    >>> write_density_array([rho], dirc='./', flnm='dust_density.binp')

    >>> # Multiple species (same spatial grid)
    >>> rho1 = 1e-11 * u.g/u.cm**3 * np.ones((100, 50, 30))
    >>> rho2 = 5e-12 * u.g/u.cm**3 * np.ones((100, 50, 30))
    >>> write_density_array([rho1, rho2], dirc='./')
    """
    nr_species = len(rho)

    with open(f'{dirc}/{flnm}', 'w') as f:
        # correspond to Format=1, precision=double, Ncells, DustSpcs
        hdr = np.array([1, 8, len(rho[0].flatten()), nr_species], dtype=int)
        hdr.tofile(f)
        for d in rho: tb.toQuantity(d, u.g/u.cm**3).to_value('g/cm3').tofile(f)

def write_temperature_array(tmp=[], dirc='./', flnm='dust_temperature.bdat'):
    """
    Write dust temperature array(s) to RADMC3D binary format file.

    Outputs dust temperature for one or more dust species to dust_temperature.bdat
    in RADMC3D binary format. Compatible with RADMC3D mctherm output.

    Parameters
    ----------
    tmp : list of array_like [K] or astropy.units.Quantity
        List of dust temperature arrays (one per species). Each array can be
        astropy quantity or numpy array (interpreted as K if plain array).
        Default is empty list.
    dirc : str, optional
        Output directory. Default is ``'./'``.
    flnm : str, optional
        Output filename. Default is ``'dust_temperature.bdat'``.

    Returns
    -------
    None
        Writes to binary file as a side effect.

    Notes
    -----
    - Binary format: C-style (little-endian double precision)
    - Header format: [iformat=1, precision=8, Ncells, Nr_species]
    - All temperature arrays must have same shape (Ncells)
    - Single species: pass as [tmp], not tmp
    - Units converted to K before writing

    Examples
    --------
    >>> # Single dust species at thermal equilibrium
    >>> T_dust = 100 * u.K * np.ones((100, 50, 30))
    >>> write_temperature_array([T_dust], dirc='./', flnm='dust_temperature.bdat')

    >>> # Multiple species (with different temperatures)
    >>> T_small = 150 * u.K * np.ones((100, 50, 30))
    >>> T_large = 80 * u.K * np.ones((100, 50, 30))
    >>> write_temperature_array([T_small, T_large], dirc='./')
    """
    nr_species = len(tmp)

    with open(f'{dirc}/{flnm}', 'w') as f:
        # correspond to Format=1, precision=double, Ncells, DustSpcs
        hdr = np.array([1, 8, len(tmp[0].flatten()), nr_species], dtype=int)
        hdr.tofile(f)
        for d in tmp: tb.toQuantity(d, u.K).to_value('K').tofile(f)

def write_opacity_handle(names=[], dirc='./', flnm='dustopac.inp'):
    """
    Write dust opacity file handles to RADMC3D format file.

    Outputs opacity file references to dustopac.inp in RADMC3D format.
    References OpTool-generated or custom ``dustkappa_<name>.inp`` files.

    Parameters
    ----------
    names : list of str
        Opacity file handles (names after ``dustkappa_`` prefix).
        Example: ['silicate', 'graphite'] -> ``dustkappa_silicate.inp``, ``dustkappa_graphite.inp``
        Default is empty list.
    dirc : str, optional
        Output directory. Default is ``'./'``.
    flnm : str, optional
        Output filename. Default is ``'dustopac.inp'``.

    Returns
    -------
    None
        Writes to file as a side effect.

    Notes
    -----
    - Format: RADMC3D dustopac.inp version 2
    - Each opacity assumes thermal grain (no aligned grains)
    - File names expanded to full ``dustkappa_<name>.inp`` format
    - Compatible with RADMC3D and OpTool

    Examples
    --------
    >>> names = ['1um', '10um', '100um']
    >>> write_opacity_handle(names, dirc='./', flnm='dustopac.inp')
    >>> # Creates references to:
    >>> # dustkappa_1um.inp
    >>> # dustkappa_10um.inp
    >>> # dustkappa_100um.inp
    """
    Nspec = len(names)

    with open(f'{dirc}/{flnm}', 'w') as f:
        f.write('2               Format number of this file\n')
        f.write(f'{Nspec}               Nr of dust species\n')
        f.write('============================================================================\n')
        for name in names:
            f.write('1               Way in which this dust species is read\n')
            f.write('0               0=Thermal grain\n')
            f.write(f'{name}            Extension of name of dustkappa_<name>.inp file\n')
            f.write('----------------------------------------------------------------------------\n')

def write_control_file(dirc='./', flnm='radmc3d.inp', extra=True, **kwargs):
    """
    Write RADMC3D control file with simulation parameters.

    Generates a radmc3d.inp configuration file with specified parameters.
    Includes sensible defaults for common simulations if extra=True.

    Parameters
    ----------
    dirc : str, optional
        Output directory. Default is ``'./'``.
    flnm : str, optional
        Output filename. Default is ``'radmc3d.inp'``.
    extra : bool, optional
        If True, adds helpful default parameters for dust continuum
        radiative transfer. Default is True.
    **kwargs
        Custom parameters to write as key=value pairs. Overrides defaults
        if extra=True.

    Returns
    -------
    None
        Writes to file as a side effect.

    Notes
    -----
    Default parameters (if extra=True):
        - incl_dust=1: Include dust continuum
        - incl_lines=0: Exclude line radiative transfer
        - nphot=1e9: Number of photon packages (mctherm/image)
        - nphot_scat=1e8: Photon packages for scattering
        - nphot_spec=1e8: Packages for spectrum
        - nphot_mono=1e8: Packages for monochromatic
        - rto_style=3: Use Monte Carlo (Bjorkman & Wood)
        - istar_sphere=1: Treat star as sphere
        - setthreads=4: OpenMP threads
        - tgas_eq_tdust=1: Gas temp = dust temp
        - modified_random_walk=1: Use modified random walk for absorption
        - scattering_mode_max=3: Maximum scattering order
        - iranfreqmode=1: Frequency-dependent mode
        - mc_scat_maxtauabs=5.0: Maximum optical depth for scattering

    Examples
    --------
    >>> # Default parameters (good for most simulations)
    >>> write_control_file(dirc='./')

    >>> # Custom parameters
    >>> write_control_file(dirc='./', nphot=int(1e10), setthreads=8)

    >>> # No extra defaults but custom parameters
    >>> write_control_file(dirc='./', extra=False, nphot=int(1e8))
    """
    my_kwargs = copy.deepcopy(kwargs)

    def check_or_set(key, default):
        if key not in my_kwargs: my_kwargs[key] = default

    check_or_set('incl_dust', 1)
    check_or_set('incl_lines', 0)
    check_or_set('nphot', int(1e9))
    check_or_set('nphot_scat', int(1e8))
    check_or_set('nphot_spec', int(1e8))
    check_or_set('nphot_mono', int(1e8))
    check_or_set('rto_style', 3)
    if extra:
        check_or_set('istar_sphere', 1)
        check_or_set('setthreads', 4)
        check_or_set('tgas_eq_tdust', 1)
        check_or_set('modified_random_walk', 1)
        check_or_set('scattering_mode_max', 3)
        check_or_set('iranfreqmode', 1)
        check_or_set('mc_scat_maxtauabs', 5.0)

    # keys that should be written as floats
    keys_float = ['mc_scat_maxtauabs']

    with open(f'{dirc}/{flnm}','w') as inp:
        for key, value in my_kwargs.items():
            if key in keys_float: inp.write(f'{key} = {value:g}\n')
            else: inp.write(f'{key} = {value:d}\n')

def write_flat_opacities(wl, kabs, dirc='./', flnm='dustkappa_flat.inp'):
    """
    Write dust opacity file in RADMC3D format.

    Outputs a wavelength-dependent dust opacity to dustkappa_*.inp in
    RADMC3D format. Includes absorption and scattering opacities (scattering
    set to zero, phase function albedo to zero).

    .. note::

       RADMC3D expects wavelengths in microns in the output file. This function
       interprets plain arrays as cm by default (following pyFLD convention),
       but properly handles astropy quantities with any length unit.

    All three calls produce output in microns but interpret input differently:

    - ``wl = np.array([1, 2, 10])`` in dubious units
    - ``write_flat_opacities(wl)`` interprets ``wl`` as cm
    - ``write_flat_opacities(wl*u.um)`` uses ``wl`` as is (in um)
    - ``write_flat_opacities(wl*u.au)`` converts from au to um

    Parameters
    ----------
    wl : array_like [cm] or astropy.units.Quantity
        Wavelength array. Can be astropy quantity with any length unit, or
        numpy array (interpreted as cm if plain array). Output will be in um.
    kabs : array_like [cm2/g] or astropy.units.Quantity
        Absorption opacity (mass absorption coefficient). Can be astropy
        quantity or numpy array (interpreted as cm2/g if plain array).
        Must have same shape as wl. Scattering and albedo set to zero.
    dirc : str, optional
        Output directory. Default is ``'./'``.
    flnm : str, optional
        Output filename. Default is ``'dustkappa_flat.inp'``.

    Returns
    -------
    None
        Writes to file as a side effect.

    Notes
    -----
    - Format: RADMC3D opacity file format version 3
    - Each line: wavelength [um], k_abs [cm2/g], k_scat [cm2/g], phase_albedo
    - Scattering opacity set to 0 (absorption-only)
    - Phase function albedo set to 0
    - Compatible with OpTool and RADMC3D

    Examples
    --------
    >>> # Wavelength-dependent absorption (e.g., silicate)
    >>> wl = np.logspace(-1, 4, 100)*u.um
    >>> kabs = 1e-3 * u.cm**2/u.g * np.exp(-((wl.to_value('um') - 10)/5)**2)
    >>> write_flat_opacities(wl, kabs, dirc='./', flnm='dustkappa_silicate.inp')

    >>> # Constant opacity
    >>> kabs_const = np.ones_like(wl) * 1e-2 * u.cm**2/u.g
    >>> write_flat_opacities(wl, kabs_const, dirc='./')
    """

    wl_um = tb.toQuantity(wl, u.cm).to_value('um')
    kabs_cgs = tb.toQuantity(kabs, u.cm**2/u.g).to_value('cm2/g')

    with open(f'{dirc}/{flnm}', 'w') as f:
        f.write('3\n') # iformat
        f.write(f'{wl_um.size}\n')
        for w in wl_um:
            f.write(f'{w:.18e} {kabs_cgs:.18e} {0.0:.18e} {0.0:.18e}\n')

def read_and_format_radmc_data(dirc='./'):
    """
    Read and format RADMC3D output files using radmc3dPy.

    Parses RADMC3D mctherm output (density, temperature, grid) and returns
    formatted astropy quantities with proper dimensional units.

    Parameters
    ----------
    dirc : str, optional
        Directory containing RADMC3D output files. Default is ``'./'``.

    Returns
    -------
    rho : astropy.units.Quantity
        Dust volume density array with shape (Nr_species, Nr, Ntheta, Nphi)
        Units: [g/cm3].
    tmp : astropy.units.Quantity
        Dust temperature array with shape (Nr_species, Nr, Ntheta, Nphi)
        Units: [K].

    Requires
    --------
    radmc3dPy : External package for RADMC3D data I/O
        Must be installed and importable.

    Raises
    ------
    ImportError
        If radmc3dPy is not available.

    Notes
    -----
    - Expected files in directory: amr_grid.inp, dust_density.binp, dust_temperature.bdat
    - Array axis order transposed to (Nr_species, Nr, Ntheta, Nphi)
    - Current working directory temporarily changed during file read, then restored
    - Requires successful RADMC3D mctherm run

    Examples
    --------
    >>> # After RADMC3D mctherm run
    >>> rho, tmp = read_and_format_radmc_data(dirc='./radmc_output/')
    >>> print(f"Density shape: {rho.shape}")  # (Nr_species, Nr, Ntheta, Nphi)
    >>> print(f"Temperature range: {tmp.min():.1f} to {tmp.max():.1f}")
    >>> print(f"Density units: {rho.unit}")
    """
    from radmc3dPy import analyze

    # save current working directory and change to target directory
    cwd = getcwd()
    chdir(dirc)

    try:
        data = analyze.readData(ddens=True, dtemp=True)
        rho = data.rhodust.transpose(3,2,1,0) * u.g/u.cm**3
        tmp = data.dusttemp.transpose(3,2,1,0) * u.K
    except Exception as e:
        raise RuntimeError(f"Error reading RADMC3D data: {e}")
    finally: # change back to original working directory
        chdir(cwd)

    return rho, tmp
