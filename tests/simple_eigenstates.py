import numpy as np
from astropy import constants as const, units as u

def get_setup_cgs(k0=1, rho_ref=1e-12, output=False):
    Rgas = const.k_B / const.u
    gamma = 1.4
    mu = 2.353
    R0 = 1 * u.au
    rho0 = rho_ref * u.g/u.cm**3
    T0 = 30*u.K
    v0 = np.sqrt(Rgas / mu * T0)
    L = 1 * u.au
    A = 1e-4
    cs = v0
    c = const.c
    k = 2*np.pi / L
    l = 1.0 / 3.0

    rho0   = rho_ref * u.g/u.cm**3
    prs0   = cs**2 * rho0
    tmp0   = mu / Rgas * prs0/rho0
    aR     = 4 * const.sigma_sb / const.c
    E0 = aR * tmp0 ** 4
    e0 = prs0 / (gamma-1)

    chi = (e0 / E0).to_value('')

    kappaR = k0 * u.cm**2/u.g
    kappaP = k0 * u.cm**2/u.g

    omega_R = kappaR * rho0 * c
    omega_P = kappaP * rho0 * c
    omega_c = np.sqrt(l) * k * c

    omega0 = (v0 / L).to('1/yr') # inverse time unit
    A2 = (chi * omega_R).to_value(omega0)
    A1 = (-(4+chi) * omega_P * omega_R - chi*omega_c**2).to_value(omega0**2)
    A0 = (4*omega_c**2 * omega_P).to_value(omega0**3)

    poly = np.array([A2, A1, A0])
    roots = np.roots(poly)

    for root in roots:
        if root.imag > 1e-5: break

    omega = root * omega0
    dT = 1 # this is T' / T0 -> dimensionless, arbitrary
    dE = ((4*omega_P - chi*omega)/omega_P).to_value('') * dT # this is Er' / E0

    if output:
        print(f'log opacity {np.log10(k0)} --- root: {root.real} + {root.imag} i')

        print(f'KAPPA {k0:.16f}')
        print(f'dTR {dT.real:.16f}')
        print(f'dTI {dT.imag:.16f}')
        print(f'dErR {dE.real:.16f}')
        print(f'dErI {dE.imag:.16f}')
        print('')

    return omega, [dT, dE]