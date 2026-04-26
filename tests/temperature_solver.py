#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# User inputs
# ============================================================
# run_dir = "alpha1em1"          # folder to read from
alpha = 1e-1                  # alpha for this run
kappaP = 1e-6                 # cm^2/g (Planck mean opacity used in Qirr/Qrad)
snapshots = [_ for _ in range(0,21,2)] # snapshots to compare (nplot indices)

# Star / gas parameters
T_star = 5770.0               # K
R_star_Rsun = 1.0             # R_sun
M_star_Msun = 1.0             # match your PLUTO Mstar input parameter
tau_min = 0.1                 # tau_star(r_min)

mu = 2.3
gamma_ad = 1.43

# Code units (from your PLUTO setup)
UNIT_LENGTH_AU = 1.0          # UNIT_LENGTH = CONST_au
UNIT_DENSITY_CGS = 1e-10      # UNIT_DENSITY = 1e-10 g/cm^3

# If you want OmegaK based on cylindrical radius R = r*sin(theta) (like your C code)
USE_CYLINDRICAL_RADIUS = True

# ODE solver settings
USE_SCIPY_IF_AVAILABLE = True
rtol = 1e-6
atol = 1e-8

# Output
# out_pdf = "T_ode_vs_pluto.pdf"

# ============================================================
# Constants (Astropy if available, else fallback)
# ============================================================
from astropy import constants as const, units as u
G_CGS = const.G.cgs.value
SIGMA_SB_CGS = const.sigma_sb.cgs.value
RSUN_CGS = const.R_sun.cgs.value
MSUN_CGS = const.M_sun.cgs.value
AU_CGS = const.au.cgs.value
kB_CGS = const.k_B.cgs.value
mp_CGS = const.m_p.cgs.value

# Derived stellar constants
R_star = R_star_Rsun * RSUN_CGS
M_star = M_star_Msun * MSUN_CGS
L_star = 4.0 * np.pi * R_star**2 * SIGMA_SB_CGS * T_star**4  # erg/s

# Gas constants
Rgas_over_mu = kB_CGS / (mu * mp_CGS)             # erg g^-1 K^-1
c_v = Rgas_over_mu / (gamma_ad - 1.0)             # erg g^-1 K^-1
unit_v2 = G_CGS * (M_star_Msun * MSUN_CGS) / AU_CGS
g_idealGasConst = unit_v2 / Rgas_over_mu

# Code time unit (seconds): UNIT_TIME = AU / sqrt(G*M/AU)
UNIT_LENGTH_CGS = UNIT_LENGTH_AU * AU_CGS
UNIT_V2 = G_CGS * M_star / UNIT_LENGTH_CGS
UNIT_V = np.sqrt(UNIT_V2)
UNIT_TIME = UNIT_LENGTH_CGS / UNIT_V              # s

# ============================================================
# Helpers
# ============================================================
def load_midplane(snap):
    g = snap.grid
    Gas = snap.fluids[0]
    prs = Gas.prs[g.active][0]
    rho = Gas.rho[g.active][0]

    unit_length = UNIT_LENGTH_AU * u.au
    unit_rho = UNIT_DENSITY_CGS * u.g/u.cm**3
    unit_prs = UNIT_DENSITY_CGS * UNIT_V ** 2 * u.g/u.cm**3 * (u.cm/u.s)**2
    unit_time = UNIT_TIME * u.s
    def fmt(v): return v[g.active][0, g.mid]
    
    T_num = fmt(Gas.tmp).to_value('K')
    r_au = g.x1[g.act1].to_value(unit_length)
    theta0 = g.x2[g.act2][g.mid]
    dx1 = g.dx1[g.act1].to_value(unit_length)
    rho_code = fmt(Gas.rho).to_value(unit_rho)
    prs_code = fmt(Gas.prs).to_value(unit_prs)
    t_code = snap.time.to_value(unit_time)
    return g.mid, r_au, theta0, dx1, rho_code, prs_code, T_num, t_code

def compute_tau_midplane(rho_code, dx1_code):
    rho_cgs = rho_code * UNIT_DENSITY_CGS
    dx1 = np.asarray(dx1_code, dtype=float)
    if dx1.size == 1:
        dx1 = np.full_like(rho_cgs, float(dx1))
    dr_cgs = dx1 * UNIT_LENGTH_CGS

    tau = np.empty_like(rho_cgs)
    tau[0] = tau_min
    if len(tau) > 1:
        tau[1:] = tau_min + np.cumsum(rho_cgs[:-1] * kappaP * dr_cgs[:-1])
    return tau

def build_coeffs(r_au, theta0, tau):
    # Radius for irradiation and OmegaK
    if USE_CYLINDRICAL_RADIUS:
        R_au = r_au * np.sin(theta0)
    else:
        R_au = r_au

    R_cgs = R_au * AU_CGS
    flux = L_star / (4.0 * np.pi * R_cgs**2)            # erg s^-1 cm^-2
    OmegaK = np.sqrt(G_CGS * M_star / R_cgs**3)         # 1/s

    # ODE: dT/dt_code = A + B*T - C*T^4  (coeffs are arrays except C scalar)
    # Derived from: rho c_v dT/dt_phys = Qirr + Qvisc + Qrad, with:
    # Qirr = kappaP*rho*flux*exp(-tau)
    # Qvisc = (9/4)*alpha*P*OmegaK = (9/4)*alpha*rho*(Rgas/mu)*T*OmegaK
    # Qrad  = -4*kappaP*rho*sigmaSB*T^4
    # Density cancels.
    A = (UNIT_TIME / c_v) * (kappaP * flux * np.exp(-tau))

    # B = (UNIT_TIME / c_v) * ((9.0 / 4.0) * alpha * Rgas_over_mu * OmegaK)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # DAVID: CHANGE TO THIS FOR QVISC USING GAMMA_AD!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    B = (UNIT_TIME / c_v) * ((9.0 / 4.0) * alpha * np.sqrt(gamma_ad) * Rgas_over_mu * OmegaK)

    C = (UNIT_TIME / c_v) * (4.0 * kappaP * SIGMA_SB_CGS)
    return A, B, C, R_au

def rhs_autonomous(T, A, B, C):
    # keep it safe if solver probes negative values
    Tclip = np.maximum(T, 0.0)
    return A + B * Tclip - C * Tclip**4

def integrate_temperature(T0, t_eval, A, B, C):
    t0 = float(t_eval[0])
    tmax = float(t_eval[-1])

    if USE_SCIPY_IF_AVAILABLE:
        try:
            from scipy.integrate import solve_ivp

            def fun(t, y):
                return rhs_autonomous(y, A, B, C)

            sol = solve_ivp(
                fun,
                (t0, tmax),
                T0,
                t_eval=t_eval,
                method="BDF",
                rtol=rtol,
                atol=atol,
            )
            if not sol.success:
                print("[WARN] solve_ivp failed:", sol.message)
            return sol.y.T  # shape (len(t_eval), nx1)
        except Exception as e:
            print("[WARN] SciPy not available or failed, falling back to RK4:", e)

    # Fallback: simple RK4 stepping between requested times
    nx = T0.size
    out = np.zeros((len(t_eval), nx), dtype=float)
    out[0, :] = T0.copy()

    for k in range(1, len(t_eval)):
        t_start = t_eval[k - 1]
        t_end = t_eval[k]
        dt_total = t_end - t_start

        # heuristic substepping
        nsub = max(20, int(np.ceil(abs(dt_total) / 0.02)))  # 0.02 code-time per substep
        dt = dt_total / nsub

        T = out[k - 1, :].copy()
        for _ in range(nsub):
            k1 = rhs_autonomous(T, A, B, C)
            k2 = rhs_autonomous(T + 0.5 * dt * k1, A, B, C)
            k3 = rhs_autonomous(T + 0.5 * dt * k2, A, B, C)
            k4 = rhs_autonomous(T + dt * k3, A, B, C)
            T = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            T = np.maximum(T, 0.0)
        out[k, :] = T

    return out

# ============================================================
# Load snapshots, set up ODE, integrate, compare
# ============================================================

if __name__ == '__main__':
    snapshots = sorted(list(dict.fromkeys(snapshots)))
    Ds = []
    times_code = []
    Tnum_list = []
    
    for nplot in snapshots:
        print(nplot)
        times_code.append(nplot)
        Tnum_list.append(T_num)
    
    times_code = np.array(times_code, dtype=float)
    Tnum_list = np.array(Tnum_list, dtype=float)  # shape (nsnap, nx1)
    
    # Use the first snapshot as initial condition for the ODE
    D0 = Ds[0]
    _, r_au, theta0, dx1, rho_code, _, T0_num, t0_code = load_midplane(D0)
    
    # Compute tau profile once (frozen coefficients)
    tau = compute_tau_midplane(rho_code, dx1)
    
    # Build ODE coefficients
    A, B, C, R_au_used = build_coeffs(r_au, theta0, tau)
    
    # Integrate to all snapshot times (code units)
    T_model = integrate_temperature(T0_num, times_code, A, B, C)  # shape (nsnap, nx1)
    
    # Error metrics (excluding non-finite)
    print("Snapshot comparison (midplane):")
    for k, nplot in enumerate(snapshots):
        Tn = Tnum_list[k]
        Tm = T_model[k]
        mask = np.isfinite(Tn) & np.isfinite(Tm) & (Tn > 0)
        rel_l2 = np.sqrt(np.mean(((Tm[mask] - Tn[mask]) / Tn[mask])**2)) if np.any(mask) else np.nan
        print(f"  nplot={nplot:4d}  t={times_code[k]/(2*np.pi):8.3f} yr   rel_L2={rel_l2:.3e}")
    
    # ============================================================
    # Plot comparison
    # ============================================================
    plt.rcParams.update({
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.2))
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(snapshots)))
    
    # label only a few snapshots to reduce clutter
    label_idx = set([0, len(snapshots) - 1])
    if len(snapshots) > 3:
        label_idx.update([len(snapshots)//2])
    
    for k, (nplot, c) in enumerate(zip(snapshots, colors)):
        t_yr = times_code[k] / (2.0 * np.pi)
        lab = rf"$t={int(np.rint(t_yr))}\,\mathrm{{yr}}$" if k in label_idx else None
    
        ax.semilogx(r_au, Tnum_list[k], color=c, marker="o", markersize=5,
                    linestyle="None", markerfacecolor="none", markeredgewidth=0.9, label=lab)
        ax.semilogx(r_au, T_model[k], color=c, lw=1.2)
    
    ax.set_xlabel(r"$r\;(\mathrm{au})$")
    ax.set_ylabel(r"$T\;(\mathrm{K})$")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.set_title(rf"$\alpha={alpha}$")
    
    style_handles = [
        Line2D([0], [0], color="k", lw=1.6, label=r"Exact (solid)"),
        Line2D([0], [0], color="k", marker="o", markersize=5,
                    linestyle="None", markerfacecolor="none", markeredgewidth=0.9, label=r"PLUTO (markers)"),
    ]
    
    handles, leglabels = ax.get_legend_handles_labels()
    ax.legend(handles=style_handles + handles, frameon=False, fontsize=9)
    
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_pdf}")
