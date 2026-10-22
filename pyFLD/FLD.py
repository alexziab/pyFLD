import numpy as np
import scipy.sparse as sp
from astropy import constants as const, units as u
import copy
from . import grid as gr, tools as tb, fluids as fl
import glob
import os

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
        Preconditioning method for iterative solver: 'ilu', 'jacobi', 'rescaled ilu', 'rescaled jacobi', or 'none' (default: 'ilu')
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
    Mat = None # linear operator for preconditioner
    _Mat_kind = None # preconditioner type that Mat was built for
    iters = -1

    _initialized_boundaries = False

    def __init__(self, fluids, verbose=False,
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
        precondition_options = ['ilu', 'jacobi', 'rescaled ilu', 'rescaled jacobi', 'none']
        if preconditioner not in precondition_options:
            raise ValueError(f"Invalid preconditioner. Please specify any of: {precondition_options}")
        self.preconditioner = preconditioner
        self._precondition = False if self.preconditioner == 'none' else True # keep a boolean internally
        if preconditioner.find('rescaled') >= 0 and verbose:
            self._log_warning(f'{preconditioner} preconditioning is experimental.')

        if self.implicit_viscosity:
            self._log_warning("Implicit viscosity is experimental and may be unstable/buggy. Use with caution.")
            self._log_warning("Note that implicit viscosity fails when Δt > e/Q_visc.")
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
        sigma = tb.get_surface_density(self.grid,
                                    self.rho[self.grid.active]) # type: ignore[operator]
        self._pop_region()
        return sigma

    def refresh_from_fluids(self):
        """
        Recompute the total density, pressure, and internal energy from the fluids.

        Sums the stored rho_eff, prs, and eng of all fluids, so each fluid
        contributes with its own eps, mu, and gamma. All fluids share the same
        temperature. Called by setup(), refresh_fluids(), update_fluids(),
        enforce_hydrostatic_equilibrium(), and _rebuild_after_read().
        Does not refresh the fluids themselves; see refresh_fluids() for that.
        """
        self._push_region("refresh_from_fluids")

        self.rho *= 0
        self.prs *= 0
        self.eng *= 0

        for f in self.fluids:
            self.rho += f.rho_eff
            self.prs += f.prs
            self.eng += f.eng

        self._pop_region()

    def update_pressure_energy(self):
        """Overrides the fluid version, which does not apply here; equivalent to refresh_from_fluids()."""
        self.refresh_from_fluids()

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

        # tmp and Er are initialized here only; afterwards they are evolved by
        # advance() and copied to the fluids, so they must not be reset elsewhere
        self.tmp = 1 * self.fluids[0].tmp
        self.Er  = tb.aR * self.tmp ** 4

        self.rho = np.zeros(self.grid.shape) * u.Unit('g/cm3')
        self.prs = np.zeros(self.grid.shape) * u.Unit('erg/cm3')
        self.eng = np.zeros(self.grid.shape) * u.Unit('erg/cm3')

        self.refresh_from_fluids()
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
            Boundary types: FIXEDVALUE or REFLECTIVE. Every boundary must be
            declared, including directions with a single cell (REFLECTIVE is
            the natural choice there).
        val : array-like, length 2*dims
            Boundary values for FIXEDVALUE boundaries [erg/cm^3] or Quantity

        Raises
        ------
        ValueError
            If arrays have wrong length, contain unrecognized types, or
            leave any boundary UNDEFINED
        """
        self._push_region("declare_boundaries")

        nbounds = 2 * self.grid.dims
        if len(bound) < nbounds or len(val) < nbounds:
            raise ValueError(f"please provide at least {nbounds} boundaries and values")

        recognized_boundary_values = [
            gr.BOUNDARY_FIXEDVALUE, # gr.BOUNDARY_PERIODIC,
            gr.BOUNDARY_REFLECTIVE,
        ]

        for i, b in enumerate(bound):
            if b == gr.BOUNDARY_UNDEFINED:
                raise ValueError(f"Boundary {i} is UNDEFINED; please declare all {nbounds} boundaries")
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
            rho_kR += f.kappaR * f.rho_eff
            rho_kP += f.kappaP * f.rho_eff
        
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
            ubar_m1 = 2 / (dx1_/(h1*D) + shift(dx1_,-1)/shift(h1*D,-1,2))
            ubar_p1 = 2 / (dx1_/(h1*D) + shift(dx1_,+1)/shift(h1*D,+1,2))
            self.M[1] -= ubar_m1 * dt/dV * A1[:,:,:-1]
            self.M[2] -= ubar_p1 * dt/dV * A1[:,:,+1:]
        if self.solve_x2:
            ubar_m2 = 2 / (dx2_/(h2*D) + shift(dx2_,-1)/shift(h2*D,-1,1))
            ubar_p2 = 2 / (dx2_/(h2*D) + shift(dx2_,+1)/shift(h2*D,+1,1))
            self.M[3] -= ubar_m2 * dt/dV * A2[:,:-1,:]
            self.M[4] -= ubar_p2 * dt/dV * A2[:,+1:,:]
        if self.solve_x3:
            ubar_m3 = 2 / (dx3_/(h3*D) + shift(dx3_,-1)/shift(h3*D,-1,0))
            ubar_p3 = 2 / (dx3_/(h3*D) + shift(dx3_,+1)/shift(h3*D,+1,0))
            self.M[5] -= ubar_m3 * dt/dV * A3[:-1,:,:]
            self.M[6] -= ubar_p3 * dt/dV * A3[+1:,:,:]

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

    def _solve_system(self, recompute_preconditioner=True, **ilu_kwargs):
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

        Raises
        ------
        RuntimeError
            If the iterative solver does not converge, or the direct solver
            returns non-finite values. No state (Er, tmp) is modified.
        """

        self._push_region("_solve_system")
        g = self.grid
        act = g.active

        # only M and B are rescaled; self.Er is never modified here
        rescaled = self._precondition and 'rescaled' in self.preconditioner
        if rescaled:
            self._compute_preconditioner()
            self._scale_arrays()

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
            # reuse the previous preconditioner only if it exists and is of the requested type
            reusable = self.Mat is not None and self._Mat_kind == self.preconditioner
            if recompute_preconditioner or not reusable:
                if not recompute_preconditioner:
                    self._logging("No reusable preconditioner available, computing a new one.")
                if self.preconditioner in ['ilu', 'rescaled ilu']:
                    try:
                        self._logging("Attempting ILU preconditioning...")
                        ilu = sp.linalg.spilu(A, **ilu_kwargs) # Sparse ILU decomposition
                        self.Mat = sp.linalg.LinearOperator(A.shape, ilu.solve)
                        self._logging("ILU preconditioning successful.")
                    except (RuntimeError, MemoryError):
                        self._log_warning("ILU failed, falling back to Jacobi")
                        self._logging("Using Jacobi preconditioning...")
                        D_inv = 1.0 / A.diagonal()
                        self.Mat = sp.linalg.LinearOperator(A.shape, lambda x: D_inv * x)
                        self._logging("Jacobi preconditioning successful.")
                elif self.preconditioner in ['jacobi', 'rescaled jacobi']:
                    self._logging("Using Jacobi preconditioning...")
                    D_inv = 1.0 / A.diagonal()
                    self.Mat = sp.linalg.LinearOperator(A.shape, lambda x: D_inv * x)
                    self._logging("Jacobi preconditioning successful.")
                else: raise ValueError(f"Unknown preconditioner type: {self.preconditioner}")
                self._Mat_kind = self.preconditioner


            iters = 0
            def cb(xk): # count iterations
                nonlocal iters
                iters += 1
            
            solver = str(self.solver_settings['solver'])
            rtol   = self.solver_settings['rtol']
            guess  = self.solver_settings['guess']
            maxiter = self.solver_settings['maxiter']

            self._logging(f"Solving with {solver}...")
            kwargs = dict(M=self.Mat, rtol=rtol, maxiter=maxiter, callback=cb)
            if guess: # provide Erad as initial guess
                x0 = self.Er[act].to_value(self.E0)
                if rescaled: x0 = x0 * self.P[act] # guess must live in the rescaled system
                kwargs['x0'] = x0.flatten()

            solve_func = {'bicgstab': sp.linalg.bicgstab, 'gmres': sp.linalg.gmres,
                            'cg': sp.linalg.cg, 'cgs': sp.linalg.cgs, 'lgmres': sp.linalg.lgmres,
                            'minres': sp.linalg.minres, 'qmr': sp.linalg.qmr}[solver]
            output, exit_code = solve_func(A, Bact_.flatten(), **kwargs)
            self.iters = iters
            self._logging(f'{solver} finished in {iters} iterations with exit code {exit_code}.')

            Enew[act] = output.reshape(Bact_.shape) * self.E0

            if rescaled:
                self._unscale_arrays(E=Enew)

            if exit_code != 0:
                self._pop_region()
                raise RuntimeError(f"{self.get_prefix()}: {solver} did not converge "
                                   f"(exit code {exit_code} after {iters} iterations). "
                                   f"Try a smaller dt, a different preconditioner, or adjust "
                                   f"rtol/maxiter via configure_solver().")
            if self.verbose: print(f'{self.get_prefix()}: Converged in {iters} iterations.')
        else:
            self._logging("Solving with spsolve...")
            Enew[act] = sp.linalg.spsolve(A, Bact_.flatten()).reshape(Bact_.shape) * self.E0
            self._logging("spsolve finished.")
            self.iters = -1 # indicate direct solve
            if not np.all(np.isfinite(Enew[act])):
                self._pop_region()
                raise RuntimeError(f"{self.get_prefix()}: direct solver returned non-finite values "
                                   f"(singular matrix?).")

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

        prs and eng are updated afterwards in update_fluids(), once the
        fluids have received the new temperature.
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

        self._pop_region()

    def update_fluids(self):
        """
        Synchronize all fluids with updated temperature and radiation energy.

        Copies temperature and radiation energy from RadiativeEnvironment to
        each fluid and recomputes pressure and internal energy. Fractions,
        rho_eff, and opacities are not recomputed here; that happens at the
        start of the next advance() (or via refresh_fluids()).
        """
        self._push_region("update_fluids")
        for f in self.fluids:
            f.tmp = 1 * self.tmp
            f.update_pressure_energy()
            f.Er  = 1 * self.Er
        self.refresh_from_fluids()
        self._pop_region()

    def compute_Qvisc(self):
        """
        Compute viscous heating source term.

        Integrates viscous dissipation across all fluids with viscosity enabled:
        Q_visc = (9/4) alpha sqrt(gamma) P Omega_K. Works in cylindrical/spherical geometry.

        Updates
        -------
        Qvisc : astropy.units.Quantity [erg/(s cm^3)]
            Total viscous heating rate

        Raises
        ------
        NotImplementedError
            If grid geometry is not spherical or cylindrical
        ValueError
            If star object is not available

        Returns
        -------
        new_dt : astropy.units.Quantity [s]
            Suggested new timestep based on viscous heating: dt < e/Q_visc
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

        new_dt = np.min((self.eng / self.Qvisc)[self.grid.active]).to('s')
        self._pop_region()
        return new_dt

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
        self.Qirr = (F * self.geom).to('erg/(s cm3)') # F [Lsun] * geom [1/au^3]

        self._pop_region()
    
    def advance(self, dt=1*u.yr, uf=0, recompute_preconditioner=True, **ilu_kwargs):
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
        recompute_preconditioner : bool, optional
            Whether to recompute preconditioner before solving (default: True)
            It might be worth setting this to False if the system
            doesn't change much between steps to save time, especially when
            using ILU as it can be expensive to compute. If False, the previous
            preconditioner is reused when available, and a new one is computed
            otherwise (e.g., on the first call or after changing ``preconditioner``).
        ilu_kwargs : dict, optional
            Additional keyword arguments to pass to spilu for ILU preconditioning (default: {})
            Options include 'drop_tol' for controlling fill-in and 'fill_factor' for maximum nonzeros.

        Raises
        ------
        ValueError
            If any fluid is not initialized
        RuntimeError
            If the linear solver fails; tmp and Er are then left unchanged
        """

        self._push_region("advance")
        if not self._initialized_boundaries:
            raise ValueError(f"{self.get_prefix()}: boundaries must be declared before advance(). Call 'declare_boundaries' first.")

        self.refresh_fluids()
        
        if self.irradiation: self.compute_Qirr()
        if self.viscosity:
            dt_visc = self.compute_Qvisc()
            if self.implicit_viscosity and dt_visc < dt:
                raise ValueError(f"{self.get_prefix()}: implicit viscosity requires dt < {dt_visc.to('yr'):.3e} to avoid negative energy update")
        self._build_matrix(dt=dt)

        Enew = self._solve_system(recompute_preconditioner=recompute_preconditioner, **ilu_kwargs)
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
            Gas density field for dust settling. If None, computed from the Gas
            fluids after they have been put in equilibrium (default: None)
        uf : float [0-1], optional
            Under-relaxation factor for convergence (default: 0, full update)
        """
        self._push_region("enforce_hydrostatic_equilibrium")
        # gas first: its equilibrium is self-contained, while dust settles
        # on top of the gas density, so dust must see the updated gas
        gases = [f for f in self.fluids if f.isGas]
        dusts = [f for f in self.fluids if not f.isGas]
        for f in gases: f.enforce_hydrostatic_equilibrium(uf=uf)
        if rhogas is None:
            rhogas = np.zeros(self.grid.shape) * u.g/u.cm**3
            for f in gases: rhogas += f.rho_eff
        for f in dusts: f.enforce_hydrostatic_equilibrium(rhogas=rhogas, uf=uf)

        self.refresh_from_fluids()
        self._pop_region()

    def refresh_fluids(self):
        """
        Call refresh() on all fluids, then update the totals.

        Recomputes effective fractions, rho_eff, opacities, pressure/energy, and
        optical depths at the current temperature. Called at the start of
        advance(); also useful before visualizing or dumping snapshots.
        """
        self._push_region("refresh_fluids")
        for f in self.fluids: f.refresh()
        self.refresh_from_fluids()
        self._pop_region()

    def write_snapshot(self, dirc='./', name='REnv'):

        self._push_region("write_snapshot")

        model_dirc = f'{dirc}/{name}/'
        tb.mkdir(dirc)
        tb.mkdir(model_dirc)

        skipped_keys = ['M', 'X', 'Y', 'Z', 'B', 'P', 'Mat', '_Mat_kind', # matrix
                'eng', 'prs', # derivative quantities
                'geom', 'fluids', 'grid', 'star', 'Mactive', # objects and misc
                'solve_x1', 'solve_x2', 'solve_x3', '_precondition', # flags
                'irradiation']

        # record which fluid folders belong to this snapshot, so that
        # create_from_snapshot ignores leftovers from older snapshots
        self._fluid_dirs = [f'fluid-{i:03d}' for i in range(len(self.fluids))]
        stale = sorted(set(os.path.basename(os.path.normpath(d))
                           for d in glob.glob(f'{model_dirc}/fluid-*')) - set(self._fluid_dirs))
        if stale:
            self._log_warning(f"{model_dirc} also contains {stale} from an older snapshot; "
                              "they will be ignored when reading")

        tb.extract_and_store_params(self, dirc=model_dirc,
                                    skipped_keys=skipped_keys)

        # maybe we can avoid writing grid and star
        self.grid.write_snapshot(dirc=model_dirc, name='grid')
        if self.star is not None:
            self.star.write_snapshot(dirc=model_dirc, name='star')
        for fluid_dir, f in zip(self._fluid_dirs, self.fluids):
            f.write_snapshot(dirc=model_dirc, name=fluid_dir)

        self._pop_region()

    def _rebuild_after_read(self):

        self._push_region("_rebuild_after_read")

        for f in self.fluids:
            if not f._initialized:
                raise ValueError(f"{self.get_prefix()}: fluid '{f.name}' is not fully initialized")

        self.refresh_from_fluids()

        self._pop_region()



def create_from_snapshot(dirc='./', name='REnv'):
    model_dirc = f'{dirc}/{name}/'

    params = tb.read_dictionary(model_dirc)
    if '_fluid_dirs' in params: # the snapshot lists its own fluid folders
        fluid_dirs = params['_fluid_dirs']
    else: # older snapshots: discover the folders, in numeric order
        found = [os.path.basename(os.path.normpath(d)) for d in glob.glob(f'{model_dirc}/fluid-*')]
        index = [int(n.split('-', 1)[1]) for n in found]
        if len(set(index)) != len(index): # e.g. fluid-0 next to fluid-000
            raise ValueError(f"{model_dirc}: ambiguous fluid folders {sorted(found)}; "
                             "remove the ones from older snapshots")
        fluid_dirs = [n for _, n in sorted(zip(index, found))]

    fluids = [fl.create_from_snapshot(dirc=model_dirc, name=d) for d in fluid_dirs]

    REnv = RadiativeEnvironment(fluids=fluids)
    tb.read_and_assign_params(REnv, dirc=model_dirc)
    REnv._rebuild_after_read()

    return REnv
