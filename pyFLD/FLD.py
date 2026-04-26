import numpy as np
import scipy.sparse as sp
from astropy import constants as const, units as u
import copy
from . import grid as gr, tools as tb, fluids as fl
import glob

try:
    import numexpr as ne
    _have_numexpr = True
except ImportError:
    _have_numexpr = False
    print("if you intend to use irradiation, it is recommended to install numexpr:")
    print("pip install numexpr")

class RadiativeEnvironment(fl._GenericFluid):
    """
    Radiation hydrodynamics solver using Flux-Limited Diffusion (FLD).

    Collects multiple ``fluids._RadiativeFluid`` objects (``Gas`` and ``Dust``)
    and solves the coupled radiation-hydrodynamics equations using FLD
    approximation. Handles irradiation, viscous heating, and
    frequency-dependent opacities. Assumes all fluids share the same
    grid, star, and temperature field.

    Arguments
    ---------
    fluids : list of ``_RadiativeFluid``
        List of ``Dust`` and/or ``Gas`` fluid objects to solve
    verbose : bool, optional
        Enable verbose output (default: True)
    constant_fluxlimiter : bool, optional
        Use constant flux limiter (lambda=1/3) instead of Kley formula (default: False)
    preconditioner : str, optional
        Preconditioning method for iterative solver: 'ilu', 'jacobi', 'rescaled ilu', 'rescaled jacobi', or False (default: 'ilu')
    implicit_viscosity : bool, optional
        Treat viscosity implicitly, experimental (default: False)
    ``**kwargs``
        Arguments passed to ``fluids._GenericFluid`` (debug, name)

    Attributes
    ----------
    E0 : astropy.units.Quantity [erg/cm^3]
        Reference radiation energy density
    Qirr : astropy.units.Quantity [erg/(s cm^3)]
        Irradiation heating term
    Qvisc : astropy.units.Quantity [erg/(s cm^3)]
        Viscous heating term
    irradiation : bool
        Any fluid has irradiation enabled
    viscosity : bool
        Any fluid has viscosity enabled
    M : ndarray
        Radiation matrix (7 diagonals for 3D)
    B : astropy.units.Quantity [erg/cm^3]
        Right-hand side vector
    solve_x1, solve_x2, solve_x3 : bool
        Whether to solve in each direction
    """

    E0 = tb.T_to_Er(10*u.K).to('erg/cm3')
    Qirr  = None
    Qvisc = None
    irradiation = False
    viscosity   = False
    _scaled_arrays = False
    iters = -1

    _initialized_boundaries = False

    def __init__(self, fluids, verbose=True,
                 preconditioner='ilu', constant_fluxlimiter=False,
                 implicit_viscosity=False, **kwargs):
        self._set_debug_name(**kwargs)
        self._push_region("__init__ (RadiativeEnvironment)")

        self.fluids  = fluids

        self.check_grids()
        self.check_stars()
        self.check_temperatures()
        self.check_source_terms()

        self.grid    = copy.deepcopy(fluids[0].grid)
        if self.irradiation: self.compute_irradiation_geometry()

        self.star    = copy.deepcopy(fluids[0].star)
        self.mu      = 1 * fluids[0].mu
        self.gamma   = 1 * fluids[0].gamma
        self.cV  = tb.Rgas / self.mu / (self.gamma-1)

        self.implicit_viscosity = implicit_viscosity
        self.verbose = verbose
        self.constant_fluxlimiter = constant_fluxlimiter
        precondition_options = ['ilu', 'jacobi', 'rescaled ilu', 'rescaled jacobi', False]
        if preconditioner not in precondition_options:
            raise ValueError(f"Invalid preconditioner. Please specify any of: {precondition_options}")
        self.preconditioner = preconditioner
        self._precondition = bool(self.preconditioner) # keep a boolean internally
        if preconditioner.find('rescaled') >= 0 and verbose:
            self._log_warning(f'{preconditioner} preconditioning is experimental.')

        if self.implicit_viscosity:
            self._log_warning("Implicit viscosity is experimental and may be unstable/buggy. Use with caution.")
        self.configure_solver()
        self.prepare_grid()
        self.setup()

        self._pop_region()

    def configure_solver(self, solver='bicgstab', rtol=1e-8, maxiter=10000, guess=True):
        """Configure iterative linear solver parameters."""
        self.solver_settings = dict(solver=solver, rtol=rtol, maxiter=maxiter, guess=guess)

    def compute_irradiation_geometry(self):
        """Precompute geometric factors for irradiation heating term."""
        self._push_region("compute_irradiation_geometry")
        g = self.grid
        self.geom = g.A1[:,:,:-1] / g.dV / (4*np.pi * g.x1l**2)
        self._pop_region()

    def get_surface_density(self):
        """
        Compute vertically integrated surface density.

        Integrates total density in polar angle direction to obtain surface
        density as function of radius and azimuth. Works only in spherical geometry.

        Returns
        -------
        astropy.units.Quantity [g/cm^2]
            Vertically integrated surface density (phi, R)
        """
        self._push_region("get_surface_density")
        sigma = tb.get_surface_density(self.grid, self.rho[self.grid.active])
        self._pop_region()
        return sigma

    def update_pressure_energy(self):
        """
        Update pressure and internal energy from density and temperature.

        Computes pressure using ideal gas law and internal energy density
        using adiabatic relations. Total density already includes effective
        fractions (eps) from all fluids.
        """
        self._push_region("update_pressure_energy")

        self.prs = self.rho * self.tmp * tb.Rgas / self.mu
        self.eng = self.prs / (self.gamma-1)

        self._pop_region()

    def setup(self):
        """
        Initialize all fluids and compute auxiliary quantities.

        Ensures all fluids are initialized, computes total density (summing
        effective fractions), initializes heating terms (Qirr, Qvisc), and
        prepares for FLD solving. Shared temperature and grid from fluids[0].
        """
        self._push_region("setup (RadiativeEnvironment)")
        for f in self.fluids:
            if not f._initialized:
                raise ValueError(f"{self.get_prefix()}: fluid '{f.name}' is not fully initialized")

        uden = u.g / u.cm**3 # in case of future cgs update
        self.rho = np.zeros(self.grid.shape) * uden
        self.tmp = 1 * self.fluids[0].tmp
        self.Er  = tb.aR * self.tmp ** 4

        for f in self.fluids: self.rho += f.rho * f.eps
        
        self.update_pressure_energy()
        self.Qirr = np.zeros(self.grid.shape) * u.Unit('erg/(s cm3)')
        self.Qvisc = np.zeros(self.grid.shape) * u.Unit('erg/(s cm3)')

        self._pop_region()

    def check_grids(self):
        """Verify all fluids share the same grid object."""
        self._push_region("check_grids")
        g0 = self.fluids[0].grid
        for f in self.fluids[1:]:
            if not tb.have_same_class_contents(g0, f.grid):
                raise ValueError(f"{self.get_prefix()}: all fluids must share the same grid")

        self._pop_region()
    
    def check_stars(self):
        """Verify all fluids share the same star object."""
        self._push_region("check_stars")
        s0 = self.fluids[0].star
        for f in self.fluids[1:]:
            if not tb.have_same_class_contents(s0, f.star):
                raise ValueError(f"{self.get_prefix()}: all fluids must share the same star")
        self._pop_region()

    def check_temperatures(self):
        """Verify all fluids share the same temperature field."""
        self._push_region("check_temperatures")
        t0 = self.fluids[0].tmp
        for f in self.fluids[1:]:
            if not np.allclose(f.tmp, t0):
                raise ValueError(f"{self.get_prefix()}: all fluids must share the same temperature field")
        self._pop_region()

    def check_source_terms(self):
        """Determine which heating source terms are active across all fluids."""
        self._push_region("check_source_terms")
        for f in self.fluids:
            if f.irradiation: self.irradiation = True
            if f.viscosity:   self.viscosity   = True
        self._pop_region()

    def declare_boundaries(self, bound, val):
        """
        Set boundary conditions for radiation field (FLD).

        Specifies boundary type and value for each domain boundary.
        Arrays must have length 2*dims (6 for 3D): rmin, rmax, thetamin,
        thetamax, phimin, phimax.

        Arguments
        ----------
        bound : array-like, length 2*dims
            Boundary types: FIXEDVALUE, PERIODIC, REFLECTIVE, UNDEFINED
        val : array-like, length 2*dims
            Boundary values for FIXEDVALUE boundaries [erg/cm^3] or Quantity

        Raises
        ------
        ValueError
            If arrays have wrong length or contain unrecognized types
        """
        self._push_region("declare_boundaries")

        nbounds = 2 * self.grid.dims
        if len(bound) < nbounds or len(val) < nbounds:
            raise ValueError(f"please provide at least {nbounds} boundaries and values")

        recognized_boundary_values = [
            gr.BOUNDARY_FIXEDVALUE, # gr.BOUNDARY_PERIODIC,
            gr.BOUNDARY_REFLECTIVE, gr.BOUNDARY_UNDEFINED
        ]

        for b in bound:
            if b not in recognized_boundary_values:
                raise ValueError(f"Boundary type {b} is not recognized")

        self.boundary_type = [b for b in bound] # copy
        def check(v):
            if tb.isQuantity(v): return v.to(self.E0)
            elif v == 0.0: return 0.0 * self.E0
            else: raise ValueError(f"Boundary value {v} must be a Quantity [erg/cm3^] or 0.0")
        self.boundary_val  = [check(v) for v in val]
        self._initialized_boundaries = True

        self._pop_region()

    def prepare_grid(self):
        """
        Initialize matrix structure and determine solution dimensions.

        Sets up sparse matrix for FLD equations and determines which directions
        have resolution > 1 (for diffusion terms). Matrix has 7 diagonals for
        3D (center + 6 off-diagonals).
        """
        self._push_region("prepare_grid")
        
        g = self.grid
        self.Mactive = np.s_[:, g.act3, g.act2, g.act1]

        self.M = np.zeros((1+2*g.dims, g.NX3_TOT, g.NX2_TOT, g.NX1_TOT))

        self.solve_x1 = g.NX1 > 1
        self.solve_x2 = g.NX2 > 1
        self.solve_x3 = g.NX3 > 1

        self._pop_region()

    def get_flux_limiter(self, R):
        """Compute flux limiter from normalized gradient using Kley (1989) formula."""
        return tb.flux_limiter_kley(R)

    def _build_matrix(self, dt=1*u.yr):
        """
        Build implicit FLD system matrix and right-hand side vector.

        Constructs the discretized FLD equations: (1 + Δt ∇·D∇) E_rad = E_old + ...
        Includes diffusion with flux-limited coefficients, radiation-matter coupling,
        and source terms (irradiation, viscous heating). Updates M (matrix) and B (RHS).

        Arguments
        ----------
        dt : float [s] or astropy.units.Quantity, optional
            Timestep size (default: 1 yr)
        """
        self._push_region("_build_matrix")

        g   = self.grid
        dV  = g.dV
        h1, h2, h3 = g.h1, g.h2, g.h3
        A1, A2, A3 = g.A1, g.A2, g.A3
        dx1_ = g.dx1
        dx2_ = g.dx2[g.i2D]
        dx3_ = g.dx3[g.i3D]

        # combined inverse mean free paths
        rho_kR = 1e-300 * np.ones(self.rho.shape) / u.au
        rho_kP = 1e-300 * np.ones(self.rho.shape) / u.au

        for f in self.fluids:
            rho_kR += f.eps * f.kappaR * f.rho
            rho_kP += f.eps * f.kappaP * f.rho
        
        DEx1 = DEx2 = DEx3 = 0 # technically 0 * self.E0 / u.au or something
        if self.solve_x1: DEx1 = np.gradient(self.Er, g.x1, edge_order=2, axis=2) * h1
        if self.solve_x2: DEx2 = np.gradient(self.Er, g.x2, edge_order=2, axis=1) * h2
        if self.solve_x3: DEx3 = np.gradient(self.Er, g.x3, edge_order=2, axis=0) * h3
        R = np.sqrt(DEx1**2 + DEx2**2 + DEx3**2) / (self.Er * rho_kR)
        if self.constant_fluxlimiter: lam = 1.0 / 3.0
        else: lam = self.get_flux_limiter(R.to_value('')) # need to call .to_value() here
        D = lam * const.c / rho_kR

        # the shift argument is flipped so that "-1" means "previous cell"
        def shift(a, shift, axis=0): return np.roll(a, shift=-shift, axis=axis)
        
        eng = self.rho * self.cV * self.tmp

        Y = rho_kP * const.c * dt
        X = tb.aR * self.tmp ** 4 / eng * Y
        Z = 1 + 4*X 
        fac = eng + 4 * (self.Qirr + self.Qvisc) * dt # type: ignore[operator]
        if self.implicit_viscosity:
            Z -= self.Qvisc * dt / eng
            fac -= self.Qvisc * dt # remove part of the contribution
        self.B = self.Er + X/Z * fac

        # compile matrix terms. astropy automatically converts to dimensionless
        # important: the "-=" here is essential for unit correct conversion
        self.M[:] = 0

        if self.solve_x1:
            Dbar_m1 = 2 * dx1_ / (dx1_/D + shift(dx1_,-1)/shift(D,-1,2))
            Dbar_p1 = 2 * dx1_ / (dx1_/D + shift(dx1_,+1)/shift(D,+1,2))
            self.M[1] -= Dbar_m1 * h1 / dx1_ * dt/dV * A1[:,:,:-1]
            self.M[2] -= Dbar_p1 * h1 / dx1_ * dt/dV * A1[:,:,+1:]
        if self.solve_x2:
            Dbar_m2 = 2 * dx2_ / (dx2_/D + shift(dx2_,-1)/shift(D,-1,1))
            Dbar_p2 = 2 * dx2_ / (dx2_/D + shift(dx2_,+1)/shift(D,+1,1))
            self.M[3] -= Dbar_m2 * h2 / dx2_ * dt/dV * A2[:,:-1,:]
            self.M[4] -= Dbar_p2 * h2 / dx2_ * dt/dV * A2[:,+1:,:]
        if self.solve_x3:
            Dbar_m3 = 2 * dx3_ / (dx3_/D + shift(dx3_,-1)/shift(D,-1,0))
            Dbar_p3 = 2 * dx3_ / (dx3_/D + shift(dx3_,+1)/shift(D,+1,0))
            self.M[5] -= Dbar_m3 * h3 / dx3_ * dt/dV * A3[:-1,:,:]
            self.M[6] -= Dbar_p3 * h3 / dx3_ * dt/dV * A3[+1:,:,:]

        self.M[0] = 1 + Y*(Z-4*X)/Z - np.sum(self.M[1:], axis=0)

        if np.any(np.isnan(self.M)) or np.any(np.isnan(self.B)):
            raise ValueError(f"{self.get_prefix()}: NaN encountered in matrix construction")

        self.X, self.Y, self.Z = X, Y, Z

        self.enforce_matrix_boundaries()
        self._pop_region()
        
    def enforce_matrix_boundaries(self):
        """Apply boundary conditions to FLD matrix and RHS vector."""

        self._push_region("enforce_matrix_boundaries")
        boundary_type = self.boundary_type
        boundary_val  = self.boundary_val
        g = self.grid

        for i in range(2*g.dims):
            idx = i + 1
            slicer = {0:np.s_[:,:,g.ibeg], 1:np.s_[:,:,g.iend],
                      2:np.s_[:,g.jbeg,:], 3:np.s_[:,g.jend,:],
                      4:np.s_[g.kbeg,:,:], 5:np.s_[g.kend,:,:]
                     }[i]
        
            if boundary_type[i] == gr.BOUNDARY_FIXEDVALUE:
                v = 1 * boundary_val[i]
                self.B[slicer] -= v * self.M[idx][slicer]
                self.M[idx][slicer] = 0
                
            elif boundary_type[i] == gr.BOUNDARY_REFLECTIVE:
                self.M[0][slicer] += self.M[idx][slicer]
                self.M[idx][slicer] = 0
        
        self._pop_region()

    def _compute_preconditioner(self):
        """Compute diagonal preconditioner from matrix diagonal."""
        self.P = np.sqrt(np.abs(self.M[0]))

    def _scale_arrays(self, E=None):
        """
        Scale matrix and RHS by preconditioner for improved solver conditioning.

        Rescales M and B such that matrix diagonal is normalized to +/-1.
        Optional radiation energy E is also scaled if provided for use as guess.

        Arguments
        ----------
        E : astropy.units.Quantity [erg/cm^3], optional
            Radiation energy array to scale in-place (default: None)
        """
        self._push_region("_scale_arrays")

        if self._scaled_arrays:
            raise ValueError("Arrays have already been scaled. This shouldn't happen.")

        if E is not None: E *= self.P
        self.B /= self.P # scale RHS
        self.M[0] /= self.P ** 2 # scale diagonal to +/- 1

        def shift(a, shift, axis=0): return np.roll(a, shift=-shift, axis=axis)
        self.M[1] /= self.P * shift(self.P, -1, 2)
        self.M[2] /= self.P * shift(self.P, +1, 2)
        self.M[3] /= self.P * shift(self.P, -1, 1)
        self.M[4] /= self.P * shift(self.P, +1, 1)
        self.M[5] /= self.P * shift(self.P, -1, 0)
        self.M[6] /= self.P * shift(self.P, +1, 0)

        self._scaled_arrays = True
        self._pop_region()

    def _unscale_arrays(self, E=None):
        """
        Reverse scaling from _scale_arrays().

        Rescales matrix, RHS, and optional radiation energy array back to
        original units and magnitudes.

        Arguments
        ----------
        E : astropy.units.Quantity [erg/cm^3], optional
            Radiation energy array to unscale in-place (default: None)
        """
        self._push_region("_unscale_arrays")
        if not self._scaled_arrays:
            raise ValueError("Arrays have already been un-scaled. This shouldn't happen.")

        if E is None: raise ValueError("Erad array must be provided")
        E /= self.P # un-scale solved array in place
        self.B *= self.P # un-scale RHS
        self.M[0] *= self.P ** 2 # reset diagonal

        def shift(a, shift, axis=0): return np.roll(a, shift=-shift, axis=axis)
        self.M[1] *= self.P * shift(self.P, -1, 2)
        self.M[2] *= self.P * shift(self.P, +1, 2)
        self.M[3] *= self.P * shift(self.P, -1, 1)
        self.M[4] *= self.P * shift(self.P, +1, 1)
        self.M[5] *= self.P * shift(self.P, -1, 0)
        self.M[6] *= self.P * shift(self.P, +1, 0)

        self._scaled_arrays = False
        self._pop_region()

    def _solve_system(self):
        """
        Solve the linear FLD system using iterative or direct solver.

        Constructs sparse matrix from M array and solves Ax=b using BiCGStab
        (or other iterative solver with preconditioning) or direct solver.
        Handles rescaled vs non-rescaled arrays and provides solver statistics.

        Returns
        -------
        Enew : astropy.units.Quantity [erg/cm^3]
            Updated radiation energy density field (includes ghost cells)

        Sets
        ----
        iters : int or None
            Number of iterations for convergence (None for direct solver)
        """

        self._push_region("_solve_system")
        g = self.grid
        act = g.active

        if self._precondition and self.preconditioner.find('rescaled') >= 0:
            self._compute_preconditioner()
            self._scale_arrays(E=self.Er)

        M = self.M[self.Mactive]
        diagonals = [M[0].flatten()]
        offsets   = [0]
        if self.solve_x1: 
            n = 1
            diagonals = [M[1].flatten()[n:]] + diagonals + [M[2].flatten()[:-n]]
            offsets   = [-n] + offsets + [n]
        if self.solve_x2:
            n = g.NX1
            diagonals = [M[3].flatten()[n:]] + diagonals + [M[4].flatten()[:-n]]
            offsets   = [-n] + offsets + [n]
        if self.solve_x3:
            n = g.NX1 * g.NX2
            diagonals = [M[5].flatten()[n:]] + diagonals + [M[6].flatten()[:-n]]
            offsets   = [-n] + offsets + [n]

        Enew = np.zeros_like(self.B) # includes unit info
        Bact_ = self.B[act].to_value(self.E0)

        A = sp.diags(diagonals, offsets, format='csc') # type: ignore[attr-defined]
        
        if self._precondition:
            if self.preconditioner in ['ilu', 'rescaled ilu']:
                try:
                    self._logging("Attempting ILU preconditioning...")
                    ilu = sp.linalg.spilu(A) # Sparse ILU decomposition
                    Mat = sp.linalg.LinearOperator(A.shape, ilu.solve)
                    self._logging("ILU preconditioning successful.")
                except:
                    self._log_warning("ILU failed, falling back to Jacobi")
                    self._logging("Using Jacobi preconditioning...")
                    D_inv = 1.0 / A.diagonal()
                    Mat = sp.linalg.LinearOperator(A.shape, lambda x: D_inv * x)
                    self._logging("Jacobi preconditioning successful.")
            elif self.preconditioner in ['jacobi', 'rescaled jacobi']:
                self._logging("Using Jacobi preconditioning...")
                D_inv = 1.0 / A.diagonal()
                Mat = sp.linalg.LinearOperator(A.shape, lambda x: D_inv * x)
                self._logging("Jacobi preconditioning successful.")
            else: raise ValueError(f"Unknown preconditioner type: {self.preconditioner}")

            iters = 0
            def cb(xk): # count iterations
                nonlocal iters
                iters += 1
            
            solver = str(self.solver_settings['solver'])
            rtol   = self.solver_settings['rtol']
            guess  = self.solver_settings['guess']
            maxiter = self.solver_settings['maxiter']

            self._logging(f"Solving with {solver}...")
            kwargs = dict(M=Mat, rtol=rtol, maxiter=maxiter, callback=cb)
            if guess: # provide Erad as initial guess
                x0 = self.Er[act].to_value(self.E0).flatten()
                kwargs['x0'] = x0

            solve_func = {'bicgstab': sp.linalg.bicgstab, 'gmres': sp.linalg.gmres,
                            'cg': sp.linalg.cg, 'cgs': sp.linalg.cgs, 'lgmres': sp.linalg.lgmres,
                            'minres': sp.linalg.minres, 'qmr': sp.linalg.qmr}[solver]
            output, exit_code = solve_func(A, Bact_.flatten(), **kwargs)
            self.iters = iters
            self._logging(f'{solver} finished in {iters} iterations with exit code {exit_code}.')

            Enew[act] = output.reshape(Bact_.shape) * self.E0

            if self.preconditioner.find('rescaled') >= 0:
                self._unscale_arrays(E=Enew)

            if exit_code != 0:
                self._log_warning(f'{solver} exited with code {exit_code}. Stopping.')
                self._pop_region()
                return self.Er
            else:
                if self.verbose: print(f'{self.get_prefix()}: Converged in {iters} iterations.')
        else:
            self._logging("Solving with spsolve...")
            Enew[act] = sp.linalg.spsolve(A, Bact_.flatten()).reshape(Bact_.shape) * self.E0
            self._logging("spsolve finished.")
            self.iters = -1 # indicate direct solve

        self._pop_region()
        return Enew

    def update_temperature(self, Enew, dt, uf=0):
        """
        Update temperature field from new radiation energy density.

        Solves the coupled radiation-matter energy balance equation to obtain
        temperature. Includes source terms from irradiation and (optionally)
        implicit viscosity. Applies temperature floor at 3 K to prevent crashes.

        Arguments
        ----------
        Enew : astropy.units.Quantity [erg/cm^3]
            New radiation energy density from FLD solve
        dt : astropy.units.Quantity [s]
            Timestep size
        uf : float [0-1], optional
            Under-relaxation factor (default: 0, full update)

        Updates
        -------
        tmp : astropy.units.Quantity [K]
            Updated temperature field (with under-relaxation)
        Er : astropy.units.Quantity [erg/cm^3]
            Updated radiation energy density
        prs, eng : astropy.units.Quantity
            Updated pressure and internal energy from new temperature
        """
        self._push_region("update_temperature")

        self.cV = tb.Rgas / self.mu / (self.gamma-1)

        self.Er = 1 * Enew
        rho_cV = self.rho * self.cV
        X, Y = self.X, self.Y
        
        num  = self.tmp*(1 + 3*X) + Y*self.Er / rho_cV
        if self.implicit_viscosity: num += dt * self.Qirr / rho_cV
        else: num += dt * (self.Qvisc + self.Qirr) / rho_cV # type: ignore[operator]

        tmp_new = num / self.Z
        self.tmp = tmp_new * (1-uf) + self.tmp * uf
        self.tmp = np.maximum(self.tmp, 3*u.K)
        self.update_pressure_energy()

        self._pop_region()

    def update_fluids(self):
        """
        Synchronize all fluids with updated temperature and radiation energy.

        Copies temperature and radiation energy from RadiativeEnvironment to
        each fluid and recomputes pressure and internal energy.
        """
        self._push_region("update_fluids")
        for f in self.fluids:
            f.tmp = 1 * self.tmp
            f.update_pressure_energy()
            f.Er  = 1 * self.Er
        self._pop_region()

    def compute_Qvisc(self):
        """
        Compute viscous heating source term.

        Integrates viscous dissipation across all fluids with viscosity enabled:
        Q_visc = (9/4) alpha sqrt(gamma)^2 P Omega_K. Only works in spherical geometry.

        Updates
        -------
        Qvisc : astropy.units.Quantity [erg/(s cm^3)]
            Total viscous heating rate

        Raises
        ------
        NotImplementedError
            If grid geometry is not spherical
        ValueError
            If star object is not available
        """

        self._push_region("compute_Qvisc")

        if self.grid.geometry in gr.aux_spherical: R = 1 * self.grid.R
        elif self.grid.geometry in gr.aux_cylindrical: R = 1 * self.grid.x1
        else: raise NotImplementedError("Viscous heating is only implemented for spherical and cylindrical geometries.")

        if self.star is None:
            raise ValueError(f"{self.get_prefix()}: star object is required for viscous heating computation")

        OmegaK = np.sqrt(self.star.GM/R**3)

        Qvisc = np.zeros(self.rho.shape) * u.Unit('erg/(s cm3)')

        for f in self.fluids:
            if not f.viscosity: continue
            
            pre = 9./4. * f.get_alpha(f) * np.sqrt(f.gamma)
            F = pre * f.prs * OmegaK
            Qvisc += F # already expects units of erg/(s cm^3)

        self.Qvisc = Qvisc

        self._pop_region()

    def compute_Qirr(self):
        """
        Compute irradiation heating source term via ray tracing.

        Performs ray-tracing from star inward through domain to compute
        radiation flux at each cell accounting for local opacity. Supports both
        gray and frequency-dependent irradiation. Computes Q_irr = F x geom_factor
        where F is attenuated stellar flux.

        Accumulates dtau and tau0 across all fluids with irradiation enabled.
        Ray tracing accounts for cumulative optical depth from inner boundary.

        Updates
        -------
        Qirr : astropy.units.Quantity [erg/(s cm^3)]
            Total irradiation heating rate
        tau : astropy.units.Quantity
            Optical depth field used in calculation

        Raises
        ------
        ValueError
            If star object is not available
        """

        self._push_region("compute_Qirr")

        if self.star is None:
            raise ValueError(f"{self.get_prefix()}: star object is required for irradiation computation")

        nx3, nx2, nx1 = self.rho.shape
        
        freqdep = False
        nbins = 1
        for f in self.fluids:
            if f.irradiation == 'freqdep':
                nbins = self.star.nbins
                freqdep = True
                break

        dtau = np.zeros((nbins, nx3, nx2, nx1), dtype=float)
        tau0 = np.zeros((nbins, nx3, nx2), dtype=float)

        # sum up dtau and tau0 across fluids
        irr_fluids = [f for f in self.fluids if f.irradiation]
        for f in irr_fluids:
            dtau += f.dtau
            tau0 += f.tau0

        # now do the ray-tracing. RAXIS is always the last in spherical.
        RAXIS = -1
        ibeg = self.grid.ibeg
        tau = np.zeros_like(dtau)
            
        tau[:,:,:,ibeg:] = np.cumsum(dtau[:,:,:,ibeg:], axis=RAXIS)
        # shift so that tau at ibeg is 0, then add tau0
        tau = np.roll(tau, axis=RAXIS, shift=1) + tau0[:,:,:,None]

        # numexpr is ~35% faster here, and this is a bottleneck, so use it if available
        if freqdep:
            L = self.star.L_wl[:,None,None,None].to_value('Lsun')
            if _have_numexpr: F_wl = ne.evaluate("L * exp(-tau) * (-expm1(-dtau))") # type: ignore[operator]
            else: F_wl = L * np.exp(-tau) * -np.expm1(-dtau)
            F    = F_wl.sum(0) * u.Lsun
        else: # note that there's a 1-element-long axis in front since nbins=1
            L = self.star.L.to_value('Lsun')
            if _have_numexpr: F_wl = ne.evaluate("L * exp(-tau) * (-expm1(-dtau))") # type: ignore[operator]
            else: F_wl = L * np.exp(-tau) * -np.expm1(-dtau)
            F = F_wl[0] * u.Lsun

        self.tau = tau
        self.Qirr = F * self.geom # already expects units of erg/(s cm^3)

        self._pop_region()
    
    def advance(self, dt=1*u.yr, uf=0):
        """
        Advance radiation-hydrodynamics by one timestep.

        Complete time stepping procedure: update opacities and effective fractions
        for all fluids, compute source terms (irradiation, viscous heating),
        build and solve FLD system, update temperature and synchronize fluids.
        This is the main function to call in time-stepping loops.

        Arguments
        ----------
        dt : astropy.units.Quantity [s], optional
            Timestep size (default: 1 yr)
        uf : float [0-1], optional
            Under-relaxation factor for temperature update (default: 0)

        Raises
        ------
        ValueError
            If any fluid is not initialized
        """

        self._push_region("advance")
        if not self._initialized_boundaries:
            raise ValueError(f"{self.get_prefix()}: boundaries must be declared before setup. Call 'declare_boundaries' first.")

        for f in self.fluids: f.refresh()
        
        if self.irradiation: self.compute_Qirr()
        if self.viscosity: self.compute_Qvisc()
        self._build_matrix(dt=dt)

        Enew = self._solve_system()
        self.enforce_boundaries(Enew)
        self.update_temperature(Enew, dt, uf=uf)
        self.update_fluids()

        self._pop_region()

    def enforce_hydrostatic_equilibrium(self, rhogas=None, uf=0):
        """
        Enforce hydrostatic equilibrium for all fluids.

        Solves for vertical structure of all fluids under gravity with pressure
        and viscous support (dust also settles). Recomputes total density from
        equilibrium profiles and updates pressure/energy fields.

        Arguments
        ----------
        rhogas : astropy.units.Quantity [g/cm^3], optional
            Gas density field for dust settling. If None, computed from Gas fluids
            (default: None)
        uf : float [0-1], optional
            Under-relaxation factor for convergence (default: 0, full update)
        """
        self._push_region("enforce_hydrostatic_equilibrium")
        uden = u.g/u.cm**3
        if rhogas is None:
            rhogas = np.zeros(self.grid.shape) * uden
            for f in self.fluids:
                if f.isGas: rhogas += f.rho * f.eps
        self.rho *= 0
        for f in self.fluids:
            f.enforce_hydrostatic_equilibrium(rhogas=rhogas, uf=uf)
            self.rho += f.rho * f.eps # f.rho was updated above
        
        self.update_pressure_energy()
        self._pop_region()

    def refresh_fluids(self):
        """Call refresh() on all fluids to update opacities and effective fractions.
        Useful to call before visualizing or dumping snapshots."""
        self._push_region("refresh_fluids")
        for f in self.fluids: f.refresh()
        self._pop_region()

    def write_snapshot(self, dirc='./', name='REnv'):

        self._push_region("write_snapshot")

        model_dirc = f'{dirc}/{name}/'
        tb.mkdir(dirc)
        tb.mkdir(model_dirc)

        skipped_keys = ['M', 'X', 'Y', 'Z', 'B', 'P', # matrix
                'eng', 'prs', # derivative quantities
                'geom', 'fluids', 'grid', 'star', 'Mactive', # objects and misc
                'solve_x1', 'solve_x2', 'solve_x3', '_precondition', # flags
                'irradiation']

        tb.extract_and_store_params(self, dirc=model_dirc,
                                    skipped_keys=skipped_keys)
        
        # maybe we can avoid writing grid and star
        self.grid.write_snapshot(dirc=model_dirc, name='grid')
        self.star.write_snapshot(dirc=model_dirc, name='star')
        for i, f in enumerate(self.fluids):
            f.write_snapshot(dirc=model_dirc, name=f'fluid-{i}')

        self._pop_region()

    def _rebuild_after_read(self):

        self._push_region("_rebuild_after_read")

        for f in self.fluids:
            if not f._initialized:
                raise ValueError(f"{self.get_prefix()}: fluid '{f.name}' is not fully initialized")

        self.update_pressure_energy()

        self._pop_region()



def create_from_snapshot(dirc='./', name='REnv'):
    # bruteforce the entire thing for now
    # I'll assume that everything starting with "fluid" is part of the fluids

    model_dirc = f'{dirc}/{name}/'

    dircs = glob.glob(f'{model_dirc}/fluid-*')
    names = sorted([d.strip().split('/')[-1] for d in dircs])

    fluids = []
    for name in names:
        fluids.append(fl.create_from_snapshot(dirc=model_dirc, name=name))
    
    REnv = RadiativeEnvironment(fluids=fluids)
    tb.read_and_assign_params(REnv, dirc=model_dirc)
    REnv._rebuild_after_read()

    return REnv
