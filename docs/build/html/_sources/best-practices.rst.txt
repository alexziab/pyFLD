Best Practices
==============

Recommendations for getting the most out of **pyFLD**.

Choosing the right grid resolution
-----------------------------------

**Rule of thumb:** Start coarse, refine until results converge.

- **Radial resolution**: 64–256 cells for protoplanetary disks. More for steep features (e.g., snow lines, opacity gaps).
- **Vertical resolution**: 32–128 cells. More if you care about disk structure near the midplane or the photosphere.
- **Azimuthal resolution**: 1 cell for axisymmetric; 64–512 for typical non-axisymmetric structures.

Always use logarithmic spacing in radius (and when appropriate, in other directions):

.. code-block:: python

    ri = fld.grid.get_domain_with_ghosts(0.1, 1000, Nx=128, log=True) * u.au

Gray vs. frequency-dependent opacities
--------------------------------------

**Use gray opacities when:**

- You want quick estimates
- Dust composition is uniform or mean properties suffice
- Computational cost is critical
- Wavelength-dependent properties don't strongly affect temperature

**Use frequency-dependent opacities when:**

- You want accurate temperature structure, especially for dust
- Dust opacity has strong features (e.g., silicate, ice, graphite absorption)
- Irradiation wavelength dependence matters for your problem

Wavelength grid for frequency-dependent opacities
--------------------------------------------------

**Resolution matters:**

- Too few bins (~10): Mean opacities poorly estimated; misses absorption features
- Sweet spot (~50–100): Good accuracy and reasonable speed
- Too many (~1000): Slow, diminishing returns

Solver configuration
--------------------

**Default settings** (usually fine):

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas], preconditioner='ilu')
    Disk.configure_solver(solver='bicgstab', rtol=1e-8, maxiter=10000)

**For faster convergence** (at cost of potentially lower accuracy):

.. code-block:: python

    Disk.configure_solver(rtol=1e-4)

**For very large problems** (:math:`> 10^8` cells):

.. code-block:: python

    Disk = fld.FLD.RadiativeEnvironment([gas], preconditioner='rescaled jacobi') # or 'rescaled ilu' if memory allows
    Disk.configure_solver(rtol=1e-6)

**To speed up multiple ``advance()`` calls** (e.g., time evolution):

.. code-block:: python

    Disk.configure_solver(guess=True)  # Use previous solution as initial guess
    for _ in range(100):
        Disk.advance(dt=1e5*u.yr)

Boundary conditions
-------------------

**For disk models (spherical geometry):**

.. code-block:: python

    # Typical: fixed T at all boundaries (radiative equilibrium)
    T_bc = 10 * u.K  # Outer boundary temperature
    bounds = [fld.grid.BOUNDARY_FIXEDVALUE] * 6
    vals = [fld.tools.T_to_Er(T_bc)] * 6
    
    # Keep midplane reflective if computing its vertical structure from irradiation
    bounds[3] = fld.grid.BOUNDARY_REFLECTIVE # the value is ignored when reflective
    
    Disk.declare_boundaries(bounds, vals)

Time stepping
-------------

**Choosing time step dt:**

FLD is implicit, so large time steps are stable (unlike explicit methods). However:

- **Too small**: Wastes computation time
- **Too large**: May not resolve fast transients; convergence slower

Start with:

.. code-block:: python

    dt = (L**2 / D)  # Characteristic diffusion time
    # where L is domain size, D = lambda*c/(kappa*rho) is diffusion coefficient

For a disk at 1 AU with typical parameters, try ``dt``:math:`\sim 10^6-10^7\,\mathrm{yr}` and adjust.

Opacity functions
-----------------

**Constant opacity** (simplest for testing):

.. code-block:: python

    Gas.enroll_kappaR(lambda f: 0.1 * u.cm**2/u.g)
    Gas.enroll_kappaP(lambda f: 0.1 * u.cm**2/u.g)

**Temperature-dependent opacity** (more realistic):

.. code-block:: python

    def opacity_P(f):
        T = f.tmp.to_value('K')
        # Simplified power law: kappa propto T^2 (example)
        return 1e-4 * u.cm**2/u.g * (T / 100)**2
        
    Gas.enroll_kappaP(opacity_P)

**Automatic from frequency-dependent data** (recommended):

When using frequency-dependent opacities with ``build_mean_opacities=True`` (default), Rosseland and Planck means are computed automatically. No need to enroll them.

Viscosity
---------

**When to include:**

- Standard α-viscosity in protoplanetary disks
- Studying viscous heating effects
- Modeling accretion

**Simple constant α:**

.. code-block:: python

    Gas.enroll_alpha(lambda f: 1e-4)  # Shakura-Sunyaev alpha

**variable α** (e.g., α(T) from Cecil & Flock 2024):

.. code-block:: python

    Gas.enroll_alpha(fld.tools.get_alpha_Cecil_Flock_2024)

**Dust without viscous heating** (but with turbulent diffusion):

.. code-block:: python

    Dust = fld.fluids.Dust(..., viscosity=False)
    Dust.enroll_alpha(lambda f: 1e-3)  # For settling, not heating

Irradiation
-----------

**Initial setup:**

.. code-block:: python

    # Gray irradiation: must enroll tau0 function
    Gas.enroll_tau0(fld.tools.get_tau0_Flock_2019)
    
    # Frequency-dependent: must enroll col0 function
    Dust.enroll_col0(fld.tools.get_col0_Flock_2019)

**Check irradiation is enabled:**

.. code-block:: python

    print(Gas.irradiation)  # Should print 'gray' or 'freqdep', not False

**Avoid common mistakes:**

- Do not forget to enroll ``tau0`` (gray) or ``col0`` (freqdep)
- Ensure wavelength arrays match between Star and Dust
- For frequency-dependent, provide absorption opacity (scattering can be zeros if not needed)

Hydrostatic equilibrium
-----------------------

**When to use:**

For self-consistent disk structure (gas pressure support, dust settling), especially with multiple species.

**Best practice workflow:**

.. code-block:: python

    # Initial setup
    Disk = fld.FLD.RadiativeEnvironment([Gas, Dust])
    Disk.declare_boundaries(bounds, vals)
    
    # Solve to target temperature profile (here, to equilibrium)
    for _ in range(50):
        Disk.advance(dt=1e8*u.yr)
    
    # Update density to satisfy hydrostatic equilibrium
    Disk.enforce_hydrostatic_equilibrium()
    
    # Continue with refined calculation (T will adjust)
    for _ in range(100):
        Disk.advance(dt=1e8*u.yr)
    
    # Update HSE periodically if needed
    for cycle in range(5):
        for _ in range(20):
            Disk.advance(dt=1e8*u.yr)
        Disk.enforce_hydrostatic_equilibrium()

Validation & testing
--------------------

**Analytic benchmarks:**

Test against known solutions before tackling complex models:

Example: **1D diffusion** — Linear temperature profile (see `example-basic`)

**Convergence studies:**

Run with different resolutions and check results converge:

.. code-block:: python

    for Nx in [32, 64, 128, 256]:
        grid = setup_grid(Nx=Nx)
        Disk = setup_model(grid)
        Disk.advance(dt=1e8*u.year)
        T_mid = Disk.tmp[...]
        print(f"Nx={Nx}: T_mid(r=10au) = {T_mid[...]} K")

Performance monitoring
----------------------

**Check solver performance:**

.. code-block:: python

    Disk.verbose = True  # Print solver iterations
    for i in range(10):
        Disk.advance(dt=1e6*u.yr)
        print(f"Step {i}: iters={Disk.iters}")

**Profile the code** using the ``debug`` flag to identify bottlenecks:

.. code-block:: python

    Gas = fld.fluids.Gas(..., debug=True)
    Disk = fld.FLD.RadiativeEnvironment([Gas], debug=True)
    Disk.advance(dt=1e6*u.yr) # will print progress info
