Advanced example: protoplanetary disk
=====================================

This example shows how to set up a protoplanetary disk with viscous heating, frequency-dependent ray-traced irradiation, and radiative transfer with Rosseland and Planck mean opacities. Several similar examples are included in the ``examples`` folder of the repository, which you can run and modify as needed.

To simplify the computation of frequency-dependent opacities, we will use the `optool <https://github.com/cdominik/optool>`_ package. After running:

.. code-block:: bash

    optool -radmc 01um -a 0.1

we have a file ``dustkappa_01um.inp`` containing absorption and scattering opacities and the scattering asymmetry parameter as a function of wavelength for 0.1 μm grains. We will use this to set up frequency-dependent irradiation in our example.

.. code-block:: python
    
    from astropy import units as u
    import pyFLD as fld
    import numpy as np

    wl, kabs, ksca, g_asy = fld.tools.parse_optool_file('dustkappa_01um.inp')

    # star object
    star = fld.star.Star(M=1*u.M_sun, L=1*u.L_sun, T=5772*u.K, wl=wl)

    # grid object
    ri   = fld.grid.get_domain_with_ghosts(0.1, 10, Nx=128, log=True) * u.au
    thi  = fld.grid.get_domain_with_ghosts(np.pi/2-0.35, np.pi/2, Nx=256) # half disk in polar direction
    phii = fld.grid.get_domain_with_ghosts(0, 2*np.pi, Nx=1) # axisymmetric grid
    grid = fld.grid.Grid(ri, thi, phii, geometry='spherical')

    # fluid objects
    Gas = fld.fluids.Gas(grid=grid, star=star, irradiation='gray', viscosity=True)
    Dust = fld.fluids.Dust(grid=grid, star=star, irradiation='freqdep',
                        kabs_wl=kabs, ksca_wl=ksca, g_wl=g_asy,
                    grain_rho=1*u.g/u.cm**3, grain_size=0.1*u.um)

    # set up fluids: density has a radial power law and Gaussian vertical profile, temperature is a radial power law, and alpha viscosity is constant
    R_ = grid.R.to_value('au')
    z_ = grid.z.to_value('au')
    H_ = 0.03 * R_ ** 1.25
    rho = 3e-10 * u.g/u.cm**3 * R_ ** -1.5 * np.exp(-0.5*(z_/H_)**2)
    T = 150 * u.K * R_ ** -0.5

    # gas properties
    Gas.set_density(rho * np.ones(grid.shape))
    Gas.set_temperature(T * np.ones(grid.shape))
    Gas.enroll_alpha(lambda f: 1e-4) # w/ viscous heating
    Gas.enroll_tau0(fld.tools.get_tau0_Flock_2019)
    Gas.enroll_kappaR(lambda f: 1.0 * u.cm**2/u.g)
    Gas.enroll_kappaP(lambda f: 1.0 * u.cm**2/u.g)
    Gas.enroll_kappaPstar(lambda f: 1.0 * u.cm**2/u.g)

    # dust properties
    Dust.set_density(rho * 1e-3 * np.ones(grid.shape))
    Dust.set_temperature(T * np.ones(grid.shape))
    Dust.enroll_alpha(lambda f: 1e-2) # w/o viscous heating
    Dust.enroll_col0(fld.tools.get_col0_Flock_2019)
    # no need to enroll kappaR/kappaP/kappaPstar for the dust as they will be automatically computed from the provided frequency-dependent opacity model

    Gas.setup()
    Dust.setup()

    Disk = fld.FLD.RadiativeEnvironment([Gas, Dust], name='disk')

    # 10 K in all boundaries
    bounds = [fld.grid.BOUNDARY_FIXEDVALUE] * 6
    vals   = [fld.tools.T_to_Er(10*u.K)] * 6
    bounds[3] = fld.grid.BOUNDARY_REFLECTIVE # reflective midplane
    Disk.declare_boundaries(bounds, vals)

    # now run model for a very large time step to get close to radiative equilibrium. This will take several steps.

    for _ in range(20): Disk.advance(dt=1e10*u.yr)

    # you can now plot the results, for example the midplane temperature profile:

    import matplotlib.pyplot as plt

    r = grid.x1[grid.act1].to_value('au')
    T_mid = Disk.tmp[grid.active][0,-1,:].to_value('K') # [phi=0, theta=-1 (midplane), r=all]
    plt.plot(r, T_mid, label='pyFLD')
    plt.plot(r, 145 * r ** -0.49, label='Power law fit')
    plt.xlabel('Radius (au)')
    plt.ylabel('Midplane Temperature (K)')
    plt.title('Midplane Temperature Profile')
    plt.loglog()
    plt.legend()
    plt.show()
    plt.close()


    # you can also opt to enforce hydrostatic equilibrium every few steps for a more self-consistent solution
    Disk.configure_solver(rtol=1e-4) # for this example
    Disk.verbose = False
    for i in range(5):
        for _ in range(20): Disk.advance(dt=1e10*u.yr)
        Disk.enforce_hydrostatic_equilibrium()
        print(f'finished supercycle {i+1}')

    # and then plot the results again
    r = grid.x1[grid.act1].to_value('au')
    T_mid = Disk.tmp[grid.active][0,-1,:].to_value('K') # [phi=0, theta=-1 (midplane), r=all]
    plt.plot(r, T_mid, label='pyFLD')
    plt.plot(r, 145 * r ** -0.49, label='Power law fit')
    plt.xlabel('Radius (au)')
    plt.ylabel('Midplane Temperature (K)')
    plt.title('Midplane Temperature Profile')
    plt.loglog()
    plt.legend()
    plt.show()
