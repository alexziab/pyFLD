import numpy as np
from astropy import units as u

# Geometry options
aux_cartesian = ['cart', 'cartesian', 'xyz']
aux_spherical = ['spherical', 'sph', 'rtheta', 'rthetaphi']
aux_cylindrical = ['polar', 'cylindrical', 'cyl', 'rphi', 'rphiz']

# Boundary condition types
BOUNDARY_UNDEFINED  = -1
BOUNDARY_PERIODIC   = 0 # ! currently not implemented
BOUNDARY_REFLECTIVE = 1
BOUNDARY_FIXEDVALUE = 2

def add_ghost_cells(arr, nghost=2, log=False):
    """
    Add ghost cells to a 1D array of cell interfaces.

    Ghost cells are appended on both sides of the array and spaced according to
    the spacing of the outermost cells in the original array.

    Arguments
    ----------
    arr : ndarray
        1D array of cell interfaces
    nghost : int, optional
        Number of ghost cells to add on either side (default: 2)
    log : bool, optional
        Use logarithmic spacing for ghost cells (default: False)

    Returns
    -------
    ndarray
        1D array of cell interfaces with ghost cells added (length Nx+1 + 2*nghost)
    """

    func = np.geomspace if log else np.linspace
    if log:
        for n in range(nghost):
            xl, xr = arr[0], arr[1]
            f = (xr-xl) / (xl+xr)
            xm1 = (1-f)/(1+f) * xl

            xl, xr = arr[-2], arr[-1]
            f = (xr-xl) / (xl+xr)
            xp1 = (1+f)/(1-f) * xr

            arr = np.concatenate(([xm1], arr, [xp1]))
    else:
        for n in range(nghost):
            dx = arr[1] - arr[0] # uniform grid spacing
            arr = np.concatenate(([arr[0]-dx], arr, [arr[-1]+dx]))

    return arr

def get_domain_with_ghosts(xmin, xmax, Nx=20, log=False, nghost=2):
    """
    Create a 1D array of cell interfaces with ghost cells.

    Generates Nx+1 cell interfaces from xmin to xmax with optional logarithmic
    spacing and appends ghost cells on both sides.

    Arguments
    ----------
    xmin : float
        Minimum value of the domain (left edge of first cell)
    xmax : float
        Maximum value of the domain (right edge of last cell)
    Nx : int, optional
        Number of cells in the domain (default: 20)
    log : bool, optional
        Use logarithmic spacing (default: False)
    nghost : int, optional
        Number of ghost cells to append on either side (default: 2)

    Returns
    -------
    ndarray
        1D array of cell interfaces including ghost cells
    """

    func = np.geomspace if log else np.linspace
    xi = func(xmin, xmax, Nx+1) # set up the main grid
    xi = add_ghost_cells(xi, nghost=nghost, log=log)
    return xi

class Grid:
    """
    3D computational grid for the FLD solver.

    Supports Cartesian, cylindrical, and spherical geometries with automatic
    handling of coordinates and metric factors.
    Arguments with length units (e.g., x1i) can be provided as
    astropy quantities or as plain arrays in au.
    Angles (x2i for cylindrical/spherical, x3i for spherical) can be provided
    as astropy quantities or as plain arrays in radians.
    The class will internally convert all length-related inputs to 
    astropy quantities, and strip units from angle-related inputs
    if they are astropy quantities. This makes it easier to work with
    the grid class internally while still allowing flexible input formats.

    Arguments
    ----------
    x1i : array_like or astropy.units.Quantity
        1D array of cell interfaces in direction 1.
        - For Cartesian: x-coordinate [au].
        - For cylindrical: radial distance [au].
        - For spherical: radial distance [au].
    x2i : array_like or astropy.units.Quantity
        1D array of cell interfaces in direction 2.
        For Cartesian: y-coordinate [au].
        For cylindrical: azimuthal angle (radians).
        For spherical: polar angle (radians).
    x3i : array_like or astropy.units.Quantity
        1D array of cell interfaces in direction 3.
        For Cartesian: z-coordinate [au].
        For cylindrical: z-coordinate [au].
        For spherical: azimuthal angle (radians).
    nghost : int, optional
        Number of ghost cells in each direction (default: 2)
    geometry : str, optional
        Grid geometry type: 'cartesian', 'cylindrical', or 'spherical' (default: 'cartesian')

    Attributes
    ----------
    x1, x2, x3 : astropy.units.Quantity (ndarray)
        Cell center coordinates in each direction
    dx1, dx2, dx3 : astropy.units.Quantity (ndarray)
        Cell widths in each direction
    A1, A2, A3 : astropy.units.Quantity (ndarray)
        3D arrays of cell face areas in each direction
    dV : astropy.units.Quantity (ndarray)
        3D array of cell volumes
    h1, h2, h3 : astropy.units.Quantity (ndarray) or float
        Metric scale factors in each direction (equal to 1 for Cartesian)
    R, z : astropy.units.Quantity (ndarray)
        2D arrays of cylindrical coordinates (spherical geometry only)
    NX1, NX2, NX3 : int
        Number of active cells in each direction
    NX1_TOT, NX2_TOT, NX3_TOT : int
        Total number of cells (including ghost cells) in each direction
    ibeg, jbeg, kbeg : int
        Starting indices of active cells in each direction
    iend, jend, kend : int
        Ending indices of active cells in each direction
    act1, act2, act3 : slice
        Slice objects for active cells in each direction
    active : slice
        Slice object for all active cells in 3D
    """

    def __init__(self, x1i, x2i, x3i, nghost=2, geometry='cartesian'):
        """
        Initialize the 3D computational grid.

        Validates geometry type and constructs grid metrics including cell centers,
        widths, volumes, face areas, and index mappings.
        """
        self.x1i, self.x2i, self.x3i = x1i, x2i, x3i
        self.dims = 3
        self.nghost = nghost

        self.geometry = geometry
        if   geometry in aux_cartesian: pass
        elif geometry in aux_cylindrical: pass
        elif geometry in aux_spherical: pass
        else: raise ValueError(f"Unrecognized geometry: {geometry}")

        self._assess_geometry()
        self._build_grid()
        self._set_indices()
        self._build_helpers()

    def _assess_geometry(self):
        """
        Convert interface arrays to appropriate units based on geometry.

        Ensures all coordinates have proper units:
        distances in au, angles are unitless in radians.
        """
        from . import tools as tb # avoid circular import

        if self.geometry in aux_cartesian:
            self.x1i = tb.toQuantity(self.x1i, u.au)
            self.x2i = tb.toQuantity(self.x2i, u.au)
            self.x3i = tb.toQuantity(self.x3i, u.au)
        
        if self.geometry in aux_cylindrical:
            self.x1i = tb.toQuantity(self.x1i, u.au)
            self.x2i = tb.toArray(self.x2i, u.rad)
            self.x3i = tb.toQuantity(self.x3i, u.au)
        
        if self.geometry in aux_spherical:
            self.x1i = tb.toQuantity(self.x1i, u.au)
            self.x2i = tb.toArray(self.x2i, u.rad)
            self.x3i = tb.toArray(self.x3i, u.rad)

    def _build_helpers(self):
        """
        Compute cylindrical coordinate mappings (spherical geometry only).

        Calculates R and z coordinates and associated azimuthal differentials
        for spherical grids to facilitate cylindrical coordinate access.
        """
        if self.geometry not in aux_spherical: return
        self.R = self.x1 * np.sin(self.x2[:,None])
        self.z = self.x1 * np.cos(self.x2[:,None])
        self.dz = self.R * self.dx2[:,None]

    def _set_indices(self):
        """
        Compute index bounds and slice objects for active and ghost cells.

        Creates index arrays and slice objects accounting for ghost cells to enable
        efficient and uniform array indexing across all directions.
        """
        nghost = self.nghost

        self.ibeg, self.iend = nghost, nghost + self.NX1 - 1
        self.jbeg, self.jend = nghost, nghost + self.NX2 - 1
        self.kbeg, self.kend = nghost, nghost + self.NX3 - 1

        self.act1  = np.s_[self.ibeg : self.iend+1]
        self.act2  = np.s_[self.jbeg : self.jend+1]
        self.act3  = np.s_[self.kbeg : self.kend+1]
        self.act1i = np.s_[self.ibeg : self.iend+2]
        self.act2i = np.s_[self.jbeg : self.jend+2]
        self.act3i = np.s_[self.kbeg : self.kend+2]

        self.active = np.s_[self.act3, self.act2, self.act1]

        if self.geometry in aux_spherical:
            self.PHIAXIS, self.THETAAXIS, self.RAXIS = 0, 1, 2
            self.mid  = np.argmin(np.abs(self.x2[self.act2] - np.pi/2))
            self.midi = np.argmin(np.abs(self.x2i[self.act2i] - np.pi/2))

    def _build_grid(self):
        """
        Construct grid metrics: cell volumes, face areas, and scale factors.

        Computes cell centers, widths, volumes, and face areas for all supported
        geometries. Metric scale factors (h1, h2, h3) are computed to account for
        coordinate curvature in cylindrical and spherical geometries.
        """

        i2D, i3D = np.s_[:,None], np.s_[:,None,None]
        self.i2D = i2D
        self.i3D = i3D
        uLen = self.x1i.unit
        nghost = self.nghost

        self.x1l, self.x1r = self.x1i[:-1], self.x1i[1:]
        self.x2l, self.x2r = self.x2i[:-1], self.x2i[1:]
        self.x3l, self.x3r = self.x3i[:-1], self.x3i[1:]
        
        self.NX1, self.NX1_TOT = self.x1i.size - 1 - 2*nghost, self.x1i.size - 1
        self.NX2, self.NX2_TOT = self.x2i.size - 1 - 2*nghost, self.x2i.size - 1
        self.NX3, self.NX3_TOT = self.x3i.size - 1 - 2*nghost, self.x3i.size - 1

        self.x1, self.dx1 = 0.5*(self.x1l + self.x1r), self.x1r - self.x1l
        self.x2, self.dx2 = 0.5*(self.x2l + self.x2r), self.x2r - self.x2l
        self.x3, self.dx3 = 0.5*(self.x3l + self.x3r), self.x3r - self.x3l

        dV = np.zeros((self.NX3_TOT, self.NX2_TOT, self.NX1_TOT),   dtype=float) * uLen ** 3
        A1 = np.zeros((self.NX3_TOT, self.NX2_TOT, self.NX1_TOT+1), dtype=float) * uLen ** 2
        A2 = np.zeros((self.NX3_TOT, self.NX2_TOT+1, self.NX1_TOT), dtype=float) * uLen ** 2
        A3 = np.zeros((self.NX3_TOT+1, self.NX2_TOT, self.NX1_TOT), dtype=float) * uLen ** 2

        if self.geometry in aux_cartesian: # dV = dx * dy * dz
            h1, h2, h3 = 1, 1, 1
            A1[:] = self.dx2[i2D] * self.dx3[i3D] # dy * dz
            A2[:] = self.dx1 * self.dx3[i3D]      # dx * dz
            A3[:] = self.dx1 * self.dx2[i2D]      # dx * dy
            dV[:] = self.dx1 * self.dx2[i2D] * self.dx3[i3D] # dx * dy * dz

        elif self.geometry in aux_cylindrical: # dV = R * dR * dφ * dz
            h1, h2, h3 = 1, 1/self.x1, 1
            A1[:] = self.x1i * self.dx2[i2D] * self.dx3[i3D]  # Rl * dφ * dz
            A2[:] = self.dx1 * self.dx3[i3D]                  # dR * dz
            A3[:] = self.x1 * self.dx1 * self.dx2[i2D]        # R * dR * dφ
            dV[:] = self.x1 * self.dx1 * self.dx2[i2D] * self.dx3[i3D] # R * dR * dφ * dz

        elif self.geometry in aux_spherical: # dV = r**2*sinθ * dr * dθ * dφ
            h1, h2, h3 = 1, 1/self.x1, 1/(self.x1 * np.sin(self.x2[i2D]))
            dVr = (self.x1r**3-self.x1l**3) / 3
            dmu = np.abs(np.cos(self.x2l) - np.cos(self.x2r))
            A1[:] = self.x1i**2 * dmu[i2D] * self.dx3[i3D]                     # rl**2 * dμ * dφ
            A2[:] = self.x1 * self.dx1 * np.sin(self.x2i[i2D]) * self.dx3[i3D] # rsinθ * dr * dφ
            A3[:] = self.x1 * self.dx1 * self.dx2[i2D]                         # r * dr * dθ
            dV[:] = dVr * dmu[i2D] * self.dx3[i3D] # r**2 * dr * dμ * dφ
        
        else: raise ValueError(f"Unrecognized geometry: {self.geometry}")

        self.dV = dV
        self.A1, self.A2, self.A3 = A1, A2, A3
        self.h1, self.h2, self.h3 = h1, h2, h3

        self.shape = (self.NX3_TOT, self.NX2_TOT, self.NX1_TOT)
        self.active_shape = (self.NX3, self.NX2, self.NX1)

    def write_snapshot(self, dirc='./', name='grid'):
        from . import tools as tb # avoid circular import
        model_dirc = f'{dirc}/{name}/'
        tb.mkdir(dirc)
        tb.mkdir(model_dirc)

        # I really just need to store nghost and geometry...
        skipped_keys = [k for k, v in self.__dict__.items()]
        for key in ['nghost', 'geometry']:
            skipped_keys.pop(skipped_keys.index(key))

        tb.extract_and_store_params(self, dirc=model_dirc,
                                    skipped_keys=skipped_keys)
        if self.geometry in aux_cartesian:   units = ['cm', 'cm', 'cm']
        elif self.geometry in aux_cylindrical: units = ['cm', None, 'cm']
        elif self.geometry in aux_spherical:   units = ['cm', None, None]
        else: raise ValueError("Unknown geometry", self.geometry)

        tb.try_writing(self.x1i, f'{model_dirc}/x1i_cgs.dbl', units[0])
        tb.try_writing(self.x2i, f'{model_dirc}/x2i_cgs.dbl', units[1])
        tb.try_writing(self.x3i, f'{model_dirc}/x3i_cgs.dbl', units[2])

    def _rebuild_after_read(self):
        self.dims = 3 # legacy; keep for now

        self._assess_geometry()
        self._build_grid()
        self._set_indices()
        self._build_helpers()

def create_from_snapshot(dirc='./', name='grid'):
    from . import tools as tb # avoid circular import

    model_dirc = f'{dirc}/{name}/'

    a = get_domain_with_ghosts(0, 1, Nx=1) * u.cm # dummy array
    grid = Grid(a, a, a) # default, fast

    tb.read_and_assign_params(grid, dirc=model_dirc)

    # already loaded geometry
    if grid.geometry in aux_cartesian:     units = ['cm', 'cm', 'cm']
    elif grid.geometry in aux_cylindrical: units = ['cm', None, 'cm']
    elif grid.geometry in aux_spherical:   units = ['cm', None, None]
    else: raise ValueError(f"Unknown geometry: {grid.geometry}")

    grid.x1i = tb.try_reading(f'{model_dirc}/x1i_cgs.dbl', units[0], force=True)
    grid.x2i = tb.try_reading(f'{model_dirc}/x2i_cgs.dbl', units[1], force=True)
    grid.x3i = tb.try_reading(f'{model_dirc}/x3i_cgs.dbl', units[2], force=True)

    grid._rebuild_after_read()
    return grid