# make PyLance ignore this file
# test_pyFLD.py: # type: ignore

import pytest
import pyFLD as fld
import numpy as np
import copy
from astropy import constants as const, units as u
import os

test_dirc = 'tests/.tmp/'
if not os.path.exists(test_dirc): os.makedirs(test_dirc)

# to be run with:
# pytest --cov pyFLD  --cov-report html

def get_opacity_arrays(): #wl, kabs, ksca, g_asy
    return fld.tools.parse_optool_file('examples/opacities/dustkappa_0.1um.inp')

def get_cartesian_grid(nghost=2, ncells_offdim=1):
    xi   = fld.grid.get_domain_with_ghosts(1, 2, Nx=100) * u.cm
    yi   = fld.grid.get_domain_with_ghosts(0, 1, Nx=ncells_offdim) * u.cm
    zi   = fld.grid.get_domain_with_ghosts(0, 1, Nx=ncells_offdim) * u.cm
    grid = fld.grid.Grid(xi, yi, zi, geometry='cartesian', nghost=nghost)
    return grid

def get_cylindrical_grid(nghost=2):
    Ri   = fld.grid.get_domain_with_ghosts(0, 1, Nx=128) * u.au
    phii = fld.grid.get_domain_with_ghosts(0, 2*np.pi, Nx=1)
    zi   = fld.grid.get_domain_with_ghosts(0, 1, Nx=1) * u.au
    grid = fld.grid.Grid(Ri, phii, zi, geometry='cylindrical', nghost=nghost)
    return grid

def get_spherical_grid(nghost=2, halfdisk=True):
    ri   = fld.grid.get_domain_with_ghosts(0.1, 10, Nx=80, log=True) * u.au
    if halfdisk:
        thi  = fld.grid.get_domain_with_ghosts(np.pi/2-0.35, np.pi/2, Nx=32)
    else:
        thi  = fld.grid.get_domain_with_ghosts(np.pi/2-0.35, np.pi/2+0.35, Nx=48)
    phii = fld.grid.get_domain_with_ghosts(0, 2*np.pi, Nx=2)
    grid = fld.grid.Grid(ri, thi, phii, geometry='spherical', nghost=nghost)
    return grid

def get_basic_star(L=1, wl=None): return fld.star.Star(L=L, wl=wl)

def build_basic_fluid(fluid_type='gas', halfdisk=True, L=1, wl=None, **kwargs):
    star = get_basic_star(L=L, wl=wl)
    grid = get_spherical_grid(halfdisk=halfdisk)
    if fluid_type == 'gas':
        Fluid = fld.fluids.Gas(grid=grid, star=star, **kwargs)
    elif fluid_type == 'dust':
        Fluid = fld.fluids.Dust(grid=grid, star=star, **kwargs)
    else:
        raise ValueError("Unknown fluid type")
    Fluid.set_density(1e-10 * u.g/u.cm**3 * np.ones(grid.shape))
    Fluid.set_temperature(100 * u.K * np.ones(grid.shape))
    Fluid.enroll_kappaP(lambda f: 1.0 * u.cm**2/u.g)
    Fluid.enroll_kappaR(lambda f: 1.0 * u.cm**2/u.g)
    Fluid.enroll_kappaPstar(lambda f: 1.0 * u.cm**2/u.g)
    return Fluid

def test_fail_multifluid_stars():
    gas1 = build_basic_fluid(L=1)
    gas2 = build_basic_fluid(L=2)
    with pytest.raises(ValueError):
        fld.FLD.RadiativeEnvironment([gas1, gas2], name='disk')

    gas3 = build_basic_fluid(L=1)
    gas3.star.new_attr = 1 # different star attributes
    with pytest.raises(ValueError):
        fld.FLD.RadiativeEnvironment([gas1, gas3], name='disk')

def test_fail_multifluid_grids():
    gas1 = build_basic_fluid(halfdisk=True) # default grid
    gas2 = build_basic_fluid(halfdisk=False) # different grid
    with pytest.raises(ValueError):
        fld.FLD.RadiativeEnvironment([gas1, gas2], name='disk')

def test_fail_multifluid_temperature():
    gas1 = build_basic_fluid()
    gas2 = build_basic_fluid()
    gas2.tmp *= 2 # different temperature array
    with pytest.raises(ValueError):
        fld.FLD.RadiativeEnvironment([gas1, gas2], name='disk')

def test_cylindrical_grid():
    grid = get_cylindrical_grid()
    assert grid.geometry == 'cylindrical'
    assert grid.shape == (5, 5, 132) # includes ghost cells

def test_fail_grid_geometry():
    with pytest.raises(ValueError): 
        fld.grid.Grid([0,1], [0,1], [0,1], geometry='unknown')

def test_fail_no_grid():
    star = get_basic_star()
    with pytest.raises(ValueError): Gas = fld.fluids.Gas(star=star)

def test_fail_bad_grid():
    grid = get_spherical_grid(nghost=1) # too few ghost cells
    with pytest.raises(ValueError): Gas = fld.fluids.Gas(grid=grid)

def test_fail_no_star():
    grid = get_spherical_grid()
    with pytest.raises(ValueError): Gas = fld.fluids.Gas(grid=grid, irradiation='gray')

def test_alert_star_too_low_resolution():
    L = 1 * u.L_sun
    # alert as the integral will be inaccurate
    star = get_basic_star(L=L, wl=np.array([0.1, 0.5, 1]) * u.um) # too few wavelength points

def test_fail_irradiation_options():
    star = get_basic_star()
    grid = get_spherical_grid()
    with pytest.raises(ValueError):
        Gas = fld.fluids.Gas(grid=grid, star=star, irradiation=True)
    with pytest.raises(ValueError):
        Gas = fld.fluids.Gas(grid=grid, star=star, irradiation='grey')
    with pytest.raises(ValueError):
        Gas = fld.fluids.Gas(grid=grid, star=star, irradiation=None)

def test_fail_irradiation_geometry():
    star = get_basic_star()
    grid = get_cartesian_grid()
    with pytest.raises(NotImplementedError):
        fld.fluids.Gas(grid=grid, star=star, irradiation='gray')

def test_fail_halfdisk_geometry():
    with pytest.raises(NotImplementedError):
        fld.tools.is_halfdisk(get_cartesian_grid())

def test_halfdisk_geometry():
    assert fld.tools.is_halfdisk(get_spherical_grid(halfdisk=True))
    assert not fld.tools.is_halfdisk(get_spherical_grid(halfdisk=False))

def test_fail_viscosity_geometry():
    star = get_basic_star()
    grid = get_cartesian_grid()
    Gas = fld.fluids.Gas(grid=grid, star=star, viscosity=True)
    Gas.set_density(1e-10 * u.g/u.cm**3 * np.ones(grid.shape))
    Gas.set_temperature(100 * u.K * np.ones(grid.shape))
    Gas.enroll_kappaP(lambda f: 1e-2 * u.cm**2/u.g)
    Gas.enroll_kappaR(lambda f: 1e-2 * u.cm**2/u.g)
    Gas.setup()
    Disk = fld.FLD.RadiativeEnvironment([Gas], name='disk')
    with pytest.raises(NotImplementedError):
        Disk.compute_Qvisc()

def test_fail_viscosity_no_star():
    star = get_basic_star()
    grid = get_cartesian_grid()
    with pytest.raises(ValueError): # no star
        fld.fluids.Gas(grid=grid, viscosity=True)
    
def test_fail_bad_wavelength_arrays():
    wl, kabs, ksca, g_asy = get_opacity_arrays()
    star = get_basic_star(wl=wl[:-1]) # mismatch in wl length
    grid = get_spherical_grid()
    with pytest.raises(ValueError):
        fld.fluids.Dust(grid=grid, star=star,
                        kabs_wl=kabs, ksca_wl=ksca, g_wl=g_asy)

def test_fail_missing_kappa_wl():
    wl, kabs, ksca, g_asy = get_opacity_arrays()
    star = get_basic_star(wl=wl)
    grid = get_spherical_grid()
    with pytest.raises(ValueError):
        fld.fluids.Dust(grid=grid, star=star, # skip ksca
                        kabs_wl=kabs, g_wl=g_asy)

def test_fail_preconditioner():
    Gas = build_basic_fluid()
    Gas.setup()
    with pytest.raises(ValueError):
        fld.FLD.RadiativeEnvironment([Gas], preconditioner='unknown')

def test_fail_uninitialized_fluid():
    Gas = build_basic_fluid() # no setup()
    with pytest.raises(ValueError):
        fld.FLD.RadiativeEnvironment([Gas])

def test_fail_hydrostatic_GenericFluid():
    grid = get_spherical_grid()
    F = fld.fluids._GenericFluid(grid=grid, name='test')
    with pytest.raises(NotImplementedError):
        F.compute_hydrostatic_equilibrium()

def test_convert_T_to_Er():
    T = 10 * u.K
    assert np.isclose(T, fld.tools.Er_to_T(fld.tools.T_to_Er(T)), rtol=1e-6)

def prep_test_radiative_line(precondition=True,
                             preconditioner='ilu',
                             trigger_nan=False,
                             trigger_wrong_boundary_length=False,
                             trigger_wrong_boundary_type=False):
    grid = get_cartesian_grid(ncells_offdim=2)
    Gas  = fld.fluids.Gas(grid=grid, debug=True)
    E0 = 1 * u.erg / u.cm**3
    Gas.set_density(1 * u.g/u.cm**3 * np.ones(grid.shape))
    Gas.set_temperature(fld.tools.Er_to_T(E0) * np.ones(grid.shape))
    Gas.enroll_kappaR(lambda f: 1 * u.cm**2 / u.g)
    Gas.enroll_kappaP(lambda f: 1 * u.cm**2 / u.g)
    Gas.setup()
    Box = fld.FLD.RadiativeEnvironment([Gas], verbose=True, name='box',
                                       constant_fluxlimiter=True,
                                       precondition=precondition,
                                       preconditioner=preconditioner)
    val, ref = fld.grid.BOUNDARY_FIXEDVALUE, fld.grid.BOUNDARY_REFLECTIVE
    bounds = [val, val, ref, ref, ref, ref]
    vals   = [E0, 2*E0, 0, 0, 0, 0]
    if trigger_wrong_boundary_length: Box.declare_boundaries(bounds[:-1], vals)
    elif trigger_wrong_boundary_type: Box.declare_boundaries(['a']*6, vals)
    else: Box.declare_boundaries(bounds, vals)
    Box.configure_solver(rtol=1e-10)
    for _ in range(5): Box.advance(dt=1e10 * u.s)

    # compare profile to analytical solution
    x = grid.x1[grid.act1].to_value('cm')
    def fmt(qty3D): return 1 * qty3D[grid.active][0,0] # take the 0th (only) element in z and y

    # pyFLD output
    Er_ = fmt(Box.Er).to_value(E0)

    # analytical solution: linear profile between the two boundaries
    xstart, xend = grid.x1[grid.ibeg-1].to('cm'), grid.x1[grid.iend+1].to('cm')
    Estart, Eend = Box.boundary_val[0], Box.boundary_val[1]
    Eexp = Estart + (Eend-Estart)/(xend-xstart) * (x*u.cm - xstart)
    Eexp = Eexp.to_value(E0)

    assert np.allclose(Er_, Eexp, rtol=1e-8)

def test_radiative_line(): prep_test_radiative_line(precondition=True)
def test_radiative_line_unpreconditioned(): prep_test_radiative_line(precondition=False)

def test_radiative_line_failures():
    with pytest.raises(ValueError):
        prep_test_radiative_line(trigger_wrong_boundary_length=True)
    with pytest.raises(ValueError):
        prep_test_radiative_line(trigger_wrong_boundary_type=True)

def test_radiative_line_preconditioner_alert():
    # still works, just throws a warning about the rescaling
    prep_test_radiative_line(preconditioner='rescaled jacobi')

def test_flux_limiter_kley():
    print(fld.tools.flux_limiter_kley(0))
    assert np.isclose(fld.tools.flux_limiter_kley(0), 1./3, rtol=1e-16)

def test_get_zero(): assert fld.tools.get_zero(None) == 0.0
def test_get_one(): assert fld.tools.get_one(None) == 1.0

def test_get_capped():
    arr = np.array([1e-308, 1e308])
    with pytest.raises(ValueError):
        fld.tools.get_capped(arr, side=None)
    assert np.isclose(np.max(fld.tools.get_capped(arr, side='max')), fld.tools.huge, rtol=1e-16)
    assert np.isclose(np.min(fld.tools.get_capped(arr, side='min')), fld.tools.tiny, rtol=1e-16)

def test_compute_mean_opacities():
    wl, kabs, ksca, g_asy = get_opacity_arrays()
    # for a flat spectrum, the means should equal the input
    kabs[:] = kabs[0]
    ksca[:] = ksca[0]
    g_asy[:] = 0.5
    T, kR, kP = fld.tools.compute_mean_opacities(wl, kabs, ksca, g_asy, T=np.linspace(100, 200, 10)*u.K)
    assert np.allclose(kP, kabs[0], rtol=1e-16)
    assert np.allclose(kR, (kabs + ksca*(1-g_asy))[0], rtol=1e-16)

def test_build_mean_opacity_interpolators():
    wl, kabs, ksca, g_asy = get_opacity_arrays()
    # for a flat spectrum, the means should equal the input
    kabs[:] = kabs[0]
    ksca[:] = ksca[0]
    g_asy[:] = 0.5
    star = get_basic_star(L=1, wl=wl)
    grid = get_spherical_grid()
    Dust = fld.fluids.Dust(grid=grid, star=star, irradiation='freqdep', kabs_wl=kabs, ksca_wl=ksca, g_wl=g_asy)
    Dust.set_density(1e-10 * u.g/u.cm**3 * np.ones(grid.shape))
    Dust.set_temperature(100 * u.K * np.ones(grid.shape))
    Dust.setup()

def test_enrolls():
    Gas = build_basic_fluid()
    
    kappaR = 1e-2 * u.cm**2/u.g
    Gas.enroll_kappaR(lambda f: kappaR)
    assert np.isclose(Gas.get_kappaR(None), kappaR, rtol=1e-16)

    kappaP = 1e0 * u.cm**2/u.g
    Gas.enroll_kappaP(lambda f: kappaP)
    assert np.isclose(Gas.get_kappaP(None), kappaP, rtol=1e-16)

    kappaPstar = 1e-1 * u.cm**2/u.g
    Gas.enroll_kappaPstar(lambda f: kappaPstar)
    assert np.isclose(Gas.get_kappaPstar(None), kappaPstar, rtol=1e-16)

    alpha = 1e-3
    Gas.enroll_alpha(lambda f: alpha)
    assert np.isclose(Gas.get_alpha(None), alpha, rtol=1e-16)

    tau0 = 0.1
    Gas.enroll_tau0(lambda f: tau0)
    assert np.isclose(Gas.get_tau0(None), tau0, rtol=1e-16)

    col0 = 1e-2 * u.g/u.cm**2
    Gas.enroll_col0(lambda f: col0)
    assert np.isclose(Gas.get_col0(None), col0, rtol=1e-16)

    frac0 = 0.5
    Gas.enroll_effective_fraction(lambda f: frac0)
    assert np.isclose(Gas.get_effective_fraction(None), frac0, rtol=1e-16)

def test_fail_surface_density_geometry():
    grid = get_cartesian_grid()
    with pytest.raises(NotImplementedError):
        fld.tools.get_surface_density(grid, 1e-10 * u.g/u.cm**3)

def prep_test_hydrostatic_equilibrium(approx=False):
    Gas = build_basic_fluid('gas', halfdisk=True)
    g = Gas.grid
    p, q, h, sigma0 = -1.5, -1, 0.05, 100 * u.g/u.cm**2
    T0 = 627.6448542468106 * u.K * (h/0.05) ** 2
    R = g.R.to('au')
    H = h * R
    rho0 = sigma0 / (np.sqrt(2*np.pi) * h*const.au)
    if approx: rho = rho0 * R.to_value('au') ** p * np.exp(-0.5*(g.z/H)**2)
    else: rho = rho0 * R.to_value('au') ** p * np.exp(-(1-R/g.x1)/h**2)
    rho = rho * np.ones(g.shape) # make it 3D
    Gas.set_density(rho)
    Gas.set_temperature(T0 * R.to_value('au') ** q * np.ones(g.shape))
    Gas.setup()
    rho_new = Gas.compute_hydrostatic_equilibrium()
    # check that Gas itself was updated
    def fmt(q): return q[g.active].to_value('g/cm3')
    if approx: # it should change significantly
        assert not np.allclose(np.log10(fmt(Gas.rho)), np.log10(fmt(rho_new)), rtol=1e-3)
    else: # it should be very close to the original
        assert np.allclose(np.log10(fmt(Gas.rho)), np.log10(fmt(rho_new)), rtol=1e-3)

def test_hydrostatic_equilibrium():
    prep_test_hydrostatic_equilibrium(approx=True)
    prep_test_hydrostatic_equilibrium(approx=False)

def prep_test_compute_surface_density_gas(halfdisk=True):
    Gas = build_basic_fluid('gas', halfdisk=halfdisk)
    g = Gas.grid
    p, q, h, sigma0 = -1.5, -1, 0.05, 100 * u.g/u.cm**2
    R = g.R.to('au')
    H = h * R
    rho0 = sigma0 / (np.sqrt(2*np.pi) * h*const.au)
    rho = rho0 * R.to_value('au') ** p * np.exp(-(1-R/g.x1)/h**2)
    Gas.set_density(rho * np.ones(g.shape))
    Gas.setup()
    Disk = fld.FLD.RadiativeEnvironment([Gas], name='disk')
    Gas.enforce_hydrostatic_equilibrium() # update rho in Gas
    Disk.enforce_hydrostatic_equilibrium() # update rho in Gas

    def fmt(q): return q[g.active].to_value('g/cm3')
    # check that Gas itself was updated
    assert not np.allclose(np.log10(fmt(Gas.rho)), np.log10(fmt(rho * np.ones(g.shape))), rtol=1e-16)
    assert np.allclose(np.log10(fmt(Gas.rho)), np.log10(fmt(Disk.rho)), rtol=1e-16)

    Sigma   = Gas.get_surface_density()[0] # first index in phi
    Sigma_disk = Disk.get_surface_density()[0] # one fluid -> same

    assert np.allclose(Sigma, Sigma_disk, rtol=1e-16)
    Sigma_exp = sigma0 * g.x1[g.act1].to_value('au') ** (p+0.5*(q+3))
    assert np.allclose(Sigma, Sigma_exp, rtol=1e-3)

def test_compute_surface_density_gas():
    prep_test_compute_surface_density_gas(halfdisk=True)
    prep_test_compute_surface_density_gas(halfdisk=False)

def prep_test_compute_surface_density_dust(halfdisk=True):
    Dust = build_basic_fluid('dust', halfdisk=halfdisk)
    Dust.grain_rho *= 0 # no settling
    g = Dust.grid
    p, q, h, sigma0 = -1.5, -1, 0.05, 100 * u.g/u.cm**2
    R = g.R.to('au')
    H = h * R
    rho0 = sigma0 / (np.sqrt(2*np.pi) * h*const.au)
    rho = rho0 * R.to_value('au') ** p * np.exp(-(1-R/g.x1)/h**2)
    Dust.set_density(rho * np.ones(g.shape))
    Dust.enroll_alpha(lambda f: 1e-3)
    Dust.setup()

    with pytest.raises(ValueError):
        Dust.enforce_hydrostatic_equilibrium(rhogas=None) # missing rhogas

    Dust.enforce_hydrostatic_equilibrium(rhogas=Dust.rho) # update rho
    Sigma   = Dust.get_surface_density()[0] # first index in phi
    Sigma_exp = sigma0 * g.x1[g.act1].to_value('au') ** (p+0.5*(q+3))
    assert np.allclose(Sigma, Sigma_exp, rtol=1e-3)

def test_compute_surface_density_dust():
    prep_test_compute_surface_density_dust(halfdisk=True)
    prep_test_compute_surface_density_dust(halfdisk=False)

def test_dtau():
    Gas = build_basic_fluid(irradiation='gray')
    rho, kappaP = 1e-1 * u.g/u.cm**3, 1e-2 * u.cm**2/u.g
    Gas.enroll_kappaR(lambda f: kappaP)
    Gas.enroll_kappaP(lambda f: kappaP)
    Gas.enroll_kappaPstar(lambda f: kappaP)
    Gas.setup()

    Gas.rho[:] = rho
    Gas.compute_cell_optical_depth()

    idx = Gas.grid.ibeg
    cell = np.s_[Gas.grid.kbeg, Gas.grid.jbeg, idx]

    assert Gas.dtau[cell].to('') == (rho * kappaP * Gas.grid.dx1[idx]).to('')
    
# Test that a full setup involving everything can be run, for coverage purposes. Not testing the results here.

def prep_test_optically_thin_viscous_disk(alpha=1e-3, implicit_viscosity=False, freqdep=False):
    # follows Chrenko+2026. Only tested for this alpha with this scheme/resolution,
    # otherwise needs a more careful timestepping strategy (see paper)
    ri   = fld.grid.get_domain_with_ghosts(0.2, 4, Nx=64, log=True) * u.au
    thi  = fld.grid.get_domain_with_ghosts(np.pi/2-0.2, np.pi/2, Nx=32)
    phii = fld.grid.get_domain_with_ghosts(0, 2*np.pi, Nx=1)
    grid = fld.grid.Grid(ri, thi, phii, geometry='spherical')
    wl, kabs, ksca, g_asy = get_opacity_arrays()
    star = fld.star.Star(M=1*u.M_sun, L=1*u.L_sun, T=5770*u.K, wl=wl)

    irr = 'gray' if not freqdep else 'freqdep'
    kabs_wl = ksca_wl = g_wl = None
    kappa = 1e-6 * u.cm**2 / u.g
    if freqdep:
        kabs[:] = kappa
        ksca[:] = kappa
        g_asy[:] = 0.0
        kabs_wl, ksca_wl, g_wl = kabs, ksca, g_asy
    Gas  = fld.fluids.Gas(grid=grid, star=star, irradiation=irr, viscosity=True,
                          kabs_wl=kabs_wl, ksca_wl=ksca_wl, g_wl=g_wl)

    s, q, h0 = -0.5, -1, 0.025
    p = s - 0.5 * (q+3)
    R  = grid.R.to('au')
    z  = grid.z.to('au')
    sig_ref = 100 * u.g / u.cm ** 2
    rho_ref = sig_ref / h0 / u.au / np.sqrt(2*np.pi)
    rho = rho_ref * (R/u.au) ** p * np.exp(-0.5 * (z/(h0*R))**2)

    Gas.set_density(np.ones(grid.shape) * rho)
    Gas.set_temperature(np.ones(grid.shape) * 50 * u.K)
    
    Gas.enroll_kappaR(lambda f:  kappa)
    Gas.enroll_kappaP(lambda f: kappa)
    Gas.enroll_kappaPstar(lambda f:  kappa)
    Gas.enroll_alpha (lambda f: alpha)
    Gas.enroll_tau0(lambda f: 0.1)
    Gas.setup()
    Disk = fld.FLD.RadiativeEnvironment([Gas], verbose=False, implicit_viscosity=implicit_viscosity)

    ref, val = fld.grid.BOUNDARY_REFLECTIVE, fld.grid.BOUNDARY_FIXEDVALUE
    bounds = [val, val, val, ref, ref, ref]
    E0     = fld.tools.T_to_Er(10*u.K)
    vals   = [E0, fld.tools.T_to_Er(50*u.K), E0, 0, 0, 0]
    Disk.declare_boundaries(bounds, vals)

    for _ in range(50): Disk.advance(dt=1e10 * u.yr)

    # compute the steady state solution
    def T_theo_complex(alpha):
        r = grid.x1[grid.act1].to_value('au')

        GM, gamma = star.GM, Gas.gamma
        Rgas, mu, sigma = fld.tools.Rgas, Gas.mu, const.sigma_sb
        kappa = Gas.kappaP
        b = 9/16 * Rgas * alpha * np.sqrt(gamma) * np.sqrt(GM)/(mu*kappa*sigma) * u.au**-1.5 / u.K**3
        d = 1/4 * star.L / (4 * np.pi * u.au**2 * sigma * u.K**4) * np.exp(-0.1)
        b, d = b.to_value(''), d.to_value('')
        # this is now a polynomial of form x^4 - px - q = 0, with x = T/K, r=R/au, p = b / r^1.5, q = d / r^2
        p, q = b * r ** -1.5, d * r ** -2
        qplus  = p**2/2 + np.sqrt(p**4/4 + 64/27*q**3)
        qminus = p**2/2 - np.sqrt(p**4/4 + 64/27*q**3)
        z = np.sign(qplus) * np.abs(qplus) ** (1/3) + np.sign(qminus) * np.abs(qminus) ** (1/3)
        m = np.sqrt(z)
        with np.errstate(invalid='ignore'): # if inviscid, m -> 0: ignore b term, recover irradiation solution
            y = np.where(m<1e-15, q ** 0.25, 0.5 * (m + np.sqrt(-z + 2*p/m)))

        return y
    
    T_theo = T_theo_complex(alpha)
    T_num = Gas.tmp[grid.active][0,grid.mid].to_value('K')

    assert np.allclose(T_num, T_theo, rtol=1e-3)

def test_optically_thin_viscous_disk():
    prep_test_optically_thin_viscous_disk(alpha=1e-2)
    prep_test_optically_thin_viscous_disk(alpha=1e-2, freqdep=True)
    # prep_test_optically_thin_viscous_disk(alpha=1e-2, implicit_viscosity=True)

def dont_test_pyFLD_freqdep():
    # === opacities ===
    wl, kabs, ksca, g_asy = get_opacity_arrays()

    # === grid and star ===
    star = get_basic_star(wl=wl)
    grid = get_spherical_grid()

    # === fluids ===
    Gas = build_basic_fluid('gas', irradiation='gray', wl=wl, viscosity=True)
    Dust = build_basic_fluid('dust', irradiation='freqdep', wl=wl,
                        kabs_wl=kabs, ksca_wl=ksca, g_wl=g_asy,
                        grain_rho=2.08*u.g/u.cm**3, grain_size=0.1*u.um)

    # enroll various tool functions to verify they are properly called
    Gas.enroll_kappaR(fld.tools.get_gas_Rosseland_opacity_Freedman_2008)
    Gas.enroll_kappaP(fld.tools.get_gas_Planck_opacity_Freedman_2008)
    Gas.enroll_kappaPstar(fld.tools.get_gas_Planck_opacity_Freedman_2008)
    Gas.enroll_effective_fraction(fld.tools.get_one)
    Gas.enroll_alpha(fld.tools.get_zero)
    Gas.enroll_tau0(fld.tools.get_tau0_Flock_2019)
    Dust.enroll_col0(fld.tools.get_col0_Flock_2019)
    Dust.enroll_alpha(lambda f: 1e-1)
    Dust.enroll_effective_fraction(fld.tools.get_fraction_Isella_Natta_2005)

    Gas.setup()
    Dust.setup()

    Disk = fld.FLD.RadiativeEnvironment([Gas, Dust], debug=True)

    bounds, vals = [fld.grid.BOUNDARY_FIXEDVALUE] * 6, [fld.tools.T_to_Er(10*u.K)] * 6
    bounds[3] = fld.grid.BOUNDARY_REFLECTIVE # the value doesn't matter
    with pytest.raises(ValueError):
        Disk.declare_boundaries(bounds[:-1], vals[:-1]) # too few boundaries
    with pytest.raises(ValueError):
        bounds_ = copy.deepcopy(bounds)
        bounds_[-1] = 'unknown'
        Disk.declare_boundaries(bounds_, vals) # bad option
    Disk.declare_boundaries(bounds, vals)

    Disk.advance(dt=1e0 * u.yr)

    # check some other options
    Disk.preconditioner = 'rescaled jacobi'
    Disk.implicit_viscosity = True
    Disk.advance(dt=1e0 * u.yr, guess=True)
    Disk.precondition = False
    Disk.advance(dt=1e0 * u.yr)
    
    Disk.enforce_hydrostatic_equilibrium()
    Disk.get_surface_density()

    Gas.enroll_alpha(lambda f: np.nan)
    with pytest.raises(ValueError):
        Disk.advance(dt=1e0 * u.yr)

def test_radmc_helper_star():
    wl = np.array([0.1, 0.5, 1]) * u.um
    star = get_basic_star(wl=wl) # should warn about low resolution
    rmc_star = fld.radmc_helper.Star()
    rmc_star.setup_from(star)
    assert np.isclose(rmc_star.L, star.L, rtol=1e-16)
    assert np.allclose(rmc_star.wl, star.wl, rtol=1e-16)

def test_radmc_helper_star_special_wl():
    wl = np.array([0.1, 0.5, 1]) * u.um
    star = get_basic_star(wl=wl) # should warn about low resolution
    rmc_star = fld.radmc_helper.Star()
    wl = fld.radmc_helper.get_wavelength_array(Tstar=star.T)
    rmc_star.setup_from(star)
    rmc_star.update_wavelengths(wl)
    assert np.isclose(1*u.L_sun, rmc_star.L, rtol=1e-5)

def test_radmc_helper_star_IO():
    wl = np.array([0.1, 0.5, 1]) * u.um
    star = get_basic_star(wl=wl) # should warn about low resolution
    rmc_star = fld.radmc_helper.Star(dirc=test_dirc)
    rmc_star.setup_from(star)
    rmc_star.write()

    rmc_star2 = fld.radmc_helper.Star(dirc=test_dirc)
    rmc_star2.read_from_file() # should warn about low resolution again
    assert np.isclose(rmc_star2.L, star.L, rtol=1e-16)
    assert np.allclose(rmc_star2.wl, star.wl, rtol=1e-16)

def test_radmc_helper_grid():
    grid = get_spherical_grid()
    rmc_grid = fld.radmc_helper.Grid() # warns about empty args
    rmc_grid.setup_from(grid)
    assert np.allclose(rmc_grid.x1, grid.x1, rtol=1e-16)
    assert np.allclose(rmc_grid.x2, grid.x2, rtol=1e-16)
    assert np.allclose(rmc_grid.x3, grid.x3, rtol=1e-16)

def test_radmc_helper_grid_IO():
    grid = get_spherical_grid()
    rmc_grid = fld.radmc_helper.Grid(dirc=test_dirc) # warns about empty args
    rmc_grid.setup_from(grid)
    rmc_grid.write()

    rmc_grid2 = fld.radmc_helper.Grid(dirc=test_dirc) # same warning
    rmc_grid2.read_from_file()
    assert np.allclose(rmc_grid2.x1, grid.x1, rtol=1e-16)
    assert np.allclose(rmc_grid2.x2, grid.x2, rtol=1e-16)
    assert np.allclose(rmc_grid2.x3, grid.x3, rtol=1e-16)

def test_poor_wavelength_limits():
    T = 10000 * u.K
    wl_peak = fld.tools.get_Wien_wavelength(T)
    # should warn about poor limits, but still return something
    fld.radmc_helper.get_wavelength_array(wl_min=wl_peak, Tstar=T)

def test_Wien_peak():
    T = 10000 * u.K
    wl_peak = fld.tools.get_Wien_wavelength(T)
    T_peak  = fld.tools.get_Wien_temperature(wl_peak)
    assert np.isclose(T.to_value('K'), T_peak.to_value('K'), rtol=1e-16)

def test_radmc_helper_wavelength_IO():
    wl = np.array([0.1, 0.5, 1]) * u.um
    fld.radmc_helper.write_wavelength_array(wl, dirc=test_dirc)
    wl2 = fld.radmc_helper.read_wavelength_array(dirc=test_dirc)
    assert np.allclose(wl, wl2, rtol=1e-16)

def test_radmc_helper_general_IO():
    fld.radmc_helper.write_control_file(dirc=test_dirc, dummy=2)
    with open(f'{test_dirc}/radmc3d.inp', 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'dummy' in line:
            assert int(line.split()[-1]) == 2
            break
    else: raise AssertionError("dummy parameter not found in control file")

def test_radmc_helper_recover_data():
    wl = fld.radmc_helper.get_wavelength_array(Tstar=5772*u.K)
    rmc_star = fld.radmc_helper.Star(dirc=test_dirc)
    rmc_star.setup_from(get_basic_star(wl=wl))
    rmc_grid = fld.radmc_helper.Grid(dirc=test_dirc) # empty args
    rmc_grid.setup_from(get_spherical_grid())
    rho = np.random.rand(*rmc_grid.active_shape) * u.g/u.cm**3
    tmp = np.random.rand(*rmc_grid.active_shape) * u.K
    
    fld.radmc_helper.write_density_array(rho=[rho], dirc=test_dirc)
    fld.radmc_helper.write_temperature_array(tmp=[tmp], dirc=test_dirc)
    rmc_star.write()
    rmc_grid.write()
    fld.radmc_helper.write_wavelength_array(wl, dirc=test_dirc)
    rho2, tmp2 = fld.radmc_helper.read_and_format_radmc_data(test_dirc)
    assert np.allclose(rho, rho2[0], rtol=1e-16) # 0 since 1 fluid
    assert np.allclose(tmp, tmp2[0], rtol=1e-16) # same

def test_radmc_helper_write_flat_opacities():
    wl = np.array([0.1, 0.5, 1]) * u.um
    kabs = 1e-2 * u.cm**2/u.g
    fld.radmc_helper.write_flat_opacities(wl=wl, kabs=kabs, dirc=test_dirc)
    with open(f'{test_dirc}/dustkappa_flat.inp', 'r') as f:
        lines = f.readlines()
    assert lines[0].strip() == '3' # format number
    assert int(lines[1].strip()) == len(wl)

    kappa = kabs.to_value('cm2/g')
    def fmt(line, idx): return float(line.split()[idx])
    for i, line in enumerate(lines[2:]):
        wl_, kabs_, ksca_, g_asy = fmt(line, 0), fmt(line, 1), fmt(line, 2), fmt(line, 3)
        assert np.isclose(wl_, wl[i].to_value('um'), rtol=1e-16)
        assert np.isclose(kabs_, kappa, rtol=1e-16)
        assert np.isclose(ksca_, 0, rtol=1e-16)
        assert np.isclose(g_asy, 0, rtol=1e-16)

def test_radmc_helper_write_opacity_handle():
    names = ['name1', 'name2']
    fld.radmc_helper.write_opacity_handle(names=names, dirc=test_dirc)
    with open(f'{test_dirc}/dustopac.inp', 'r') as f:
        lines = f.readlines()
    assert lines[0].strip().split()[0] == '2' # format number
    assert int(lines[1].strip().split()[0]) == len(names)
    found_names = [False] * len(names)

    def check_name(line):
        for i, name in enumerate(names):
            if name in line:
                found_names[i] = True
                return

    for i, line in enumerate(lines[2:]):
        check_name(line)
    assert all(found_names)

def test_tau0_flock():
    rho = 1e-10 * u.g/u.cm**3
    kappa = 1e-2 * u.cm**2/u.g
    Gas = build_basic_fluid(irradiation='gray')
    Gas.enroll_kappaR(lambda f: kappa)
    Gas.enroll_kappaP(lambda f: kappa)
    Gas.enroll_kappaPstar(lambda f: kappa)
    Gas.set_density(rho * np.ones(Gas.grid.shape))
    Gas.set_temperature(100 * u.K * np.ones(Gas.grid.shape))
    Gas.enroll_tau0(fld.tools.get_tau0_Flock_2019)
    Gas.setup()
    Gas.compute_tau0()

    tau0 = 1 * Gas.tau0
    g = Gas.grid
    tau0_exp = rho * kappa * g.R[:,g.ibeg]
    assert np.allclose(tau0[0], tau0_exp, rtol=1e-16)

def test_col0_flock():
    rho = 1e-10 * u.g/u.cm**3
    wl = np.array([0.1, 0.5, 1]) * u.um
    kabs = 1e-3 * u.cm**2/u.g * np.ones(len(wl))
    ksca = np.zeros(len(wl)) * u.cm**2/u.g
    g_asy = np.zeros(len(wl))
    grid = get_spherical_grid()
    star = get_basic_star(wl=wl) # low resolution
    Gas = fld.fluids.Gas(grid=grid, star=star, irradiation='freqdep', 
                        kabs_wl=kabs, ksca_wl=ksca, g_wl=g_asy)
    Gas.set_density(rho * np.ones(Gas.grid.shape))
    Gas.set_temperature(10 * u.K * np.ones(Gas.grid.shape))
    Gas.enroll_col0(fld.tools.get_col0_Flock_2019)
    Gas.setup()
    Gas.compute_tau0()

    tau0 = 1 * Gas.tau0
    g = Gas.grid
    s = Gas.star
    tau0_exp = (rho * kabs[0] * (g.R[:,g.ibeg]-3*s.R)).to('')
    assert np.allclose(tau0[0,0], tau0_exp, rtol=1e-16)

def test_misc_enrollables():
    class DummyFluid: pass
    f = DummyFluid()
    uopac = u.cm**2/u.g
    f.rho = 1e-10 * u.g/u.cm**3
    f.tmp = 1e-100 * u.K
    frac_low = fld.tools.get_fraction_Isella_Natta_2005(f)
    opac_low = fld.tools.get_gas_Planck_opacity_Freedman_2008(f)
    assert np.isclose(frac_low, 1, rtol=1e-6)
    # won't reach 1e-3 due to the wide transition width
    assert opac_low < 1.1e-3 * uopac

    f.tmp = 1e10 * u.K
    frac_high = fld.tools.get_fraction_Isella_Natta_2005(f)
    opac_high = fld.tools.get_gas_Planck_opacity_Freedman_2008(f)
    assert np.isclose(frac_high, 1e-10, rtol=1e-6)
    assert np.isclose(opac_high, 1.0 * uopac, rtol=1e-6)
    
    opacR = fld.tools.get_gas_Rosseland_opacity_Freedman_2008(f)
    opac_exp = 2e1 * f.rho.to_value('g/cm3') ** 0.8 + 1e-6
    assert np.isclose(opacR, opac_exp * uopac, rtol=1e-6)

def test_fail_stellar_wavelengths():
    star = get_basic_star()
    with pytest.raises(ValueError):
        star.get_wavelength_spacing()
    
    with pytest.raises(ValueError):
        star._set_spectral_L(None)

    with pytest.raises(ValueError):
        star._enforce_L_consistency()

def test_fail_stellar_spectrum():
    wl = np.array([0.1, 0.5, 1]) * u.um
    star = get_basic_star(wl=wl) # low resolution
    star.nbins = len(wl) + 1 # mismatch
    L_wl = star.get_spectral_L()
    with pytest.raises(ValueError):
        star._set_spectral_L(L_wl)

def test_star_IO():
    wl = np.array([0.1, 0.5, 1]) * u.um
    star = get_basic_star(wl=wl)
    dirc = test_dirc
    star.write_snapshot(dirc=dirc)

    snew = fld.star.create_from_snapshot(dirc=dirc)
    assert np.isclose(snew.L, star.L, rtol=1e-16)
    assert np.allclose(snew.wl, star.wl, rtol=1e-16)

def test_star_IO_no_wl():
    star = get_basic_star()
    dirc = test_dirc
    star.write_snapshot(dirc=dirc)

    snew = fld.star.create_from_snapshot(dirc=dirc)
    assert np.isclose(snew.L, star.L, rtol=1e-16)

def test_grid_IO():
    grid = get_spherical_grid()
    dirc = test_dirc
    grid.write_snapshot(dirc=dirc)

    gnew = fld.grid.create_from_snapshot(dirc=dirc)
    assert np.allclose(gnew.x1, grid.x1, rtol=1e-16)
    assert np.allclose(gnew.x2, grid.x2, rtol=1e-16)
    assert np.allclose(gnew.x3, grid.x3, rtol=1e-16)

def test_fluid_IO():
    Gas = build_basic_fluid()
    Gas.setup()
    dirc = test_dirc
    Gas.write_snapshot(dirc=dirc, name='gas')

    Gnew = fld.fluids.create_from_snapshot(dirc=dirc, name='gas')
    assert np.allclose(Gnew.rho, Gas.rho, rtol=1e-16)
    assert np.allclose(Gnew.tmp, Gas.tmp, rtol=1e-16)
    assert np.isclose(Gnew.mu, Gas.mu, rtol=1e-16)


def test_freqdep_fluid_IO():
    wl, kabs, ksca, g_asy = get_opacity_arrays()
    # for a flat spectrum, the means should equal the input
    kabs[:] = kabs[0]
    ksca[:] = ksca[0]
    g_asy[:] = 0.5
    star = get_basic_star(L=1, wl=wl)
    grid = get_spherical_grid()
    Dust = fld.fluids.Dust(grid=grid, star=star, irradiation='freqdep', kabs_wl=kabs, ksca_wl=ksca, g_wl=g_asy)
    Dust.set_density(1e-10 * u.g/u.cm**3 * np.ones(grid.shape))
    Dust.set_temperature(100 * u.K * np.ones(grid.shape))
    Dust.setup()

    dirc = test_dirc
    Dust.write_snapshot(dirc=dirc, name='dust')

    Dnew = fld.fluids.create_from_snapshot(dirc=dirc, name='dust')
    assert np.allclose(Dnew.rho, Dust.rho, rtol=1e-16)
    assert np.allclose(Dnew.tmp, Dust.tmp, rtol=1e-16)
    assert np.isclose(Dnew.mu, Dust.mu, rtol=1e-16)
    assert np.allclose(Dnew.kabs_wl, Dust.kabs_wl, rtol=1e-16)
    assert np.allclose(Dnew.ksca_wl, Dust.ksca_wl, rtol=1e-16)
    assert np.allclose(Dnew.g_wl, Dust.g_wl, rtol=1e-16)

def test_REnv_IO():
    Gas = build_basic_fluid()
    Gas.setup()
    Renv = fld.FLD.RadiativeEnvironment([Gas], name='REnv')
    dirc = test_dirc
    Renv.write_snapshot(dirc=dirc)

    Rnew = fld.FLD.create_from_snapshot(dirc=dirc, name='REnv')
    assert len(Rnew.fluids) == 1
    Gnew = Rnew.fluids[0]
    assert np.allclose(Gnew.rho, Gas.rho, rtol=1e-16)
    assert np.allclose(Gnew.tmp, Gas.tmp, rtol=1e-16)
    assert np.isclose(Gnew.mu, Gas.mu, rtol=1e-16)

def test_fail_scale_arrays():
    Gas = build_basic_fluid()
    Gas.setup()
    Disk = fld.FLD.RadiativeEnvironment([Gas], name='disk',
                                        preconditioner='rescaled ilu')
    Disk._scaled_arrays = True # force an error
    with pytest.raises(ValueError):
        Disk.advance(dt=1e0 * u.yr)

def test_fail_NaN_in_matrix():
    Gas = build_basic_fluid()
    Gas.setup()
    Disk = fld.FLD.RadiativeEnvironment([Gas], name='disk',
                                        preconditioner='rescaled ilu')
    g = Gas.grid
    Disk.rho[g.kbeg,g.jbeg,g.ibeg] = np.nan # force a NaN in the matrix
    with pytest.raises(ValueError):
        Disk._build_matrix(dt=1*u.yr)
