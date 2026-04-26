Introduction
=====================================

``pyFLD`` solves the coupled equations of radiative diffusion and thermal energy evolution including external source terms:

.. math:: \rho c_\mathrm{V} \frac{\partial T}{\partial t} = -\kappa_\mathrm{P}\rho c (a_\mathrm{R} T^4 - E_\mathrm{r}) + Q_\mathrm{ext}
.. math:: \frac{\partial E_\mathrm{r}}{\partial t} - \nabla \cdot\left[\frac{\lambda c}{\kappa_\mathrm{R}\rho} \nabla E_\mathrm{r}\right] = \kappa_\mathrm{P}\rho c (a_\mathrm{R} T^4 - E_\mathrm{r}),

where :math:`\rho` is the density, :math:`c_\mathrm{V}` is the specific heat capacity at constant volume, :math:`T` is the temperature, :math:`E_\mathrm{r}` is the radiation energy density, :math:`\kappa_\mathrm{R}` and :math:`\kappa_\mathrm{P}` are the Rosseland and Planck mean opacities, :math:`c` is the speed of light, :math:`a_\mathrm{R}` is the radiation constant, :math:`\lambda` is the flux limiter, and :math:`Q_\mathrm{ext}` represents any external source terms.

Regarding source terms, ``pyFLD`` includes a treatment of ray-traced irradiation from a central star in the context of protoplanetary disks, which is a common application of the package:

.. math:: Q_\mathrm{irr} = -\nabla \cdot \left(F_\mathrm{irr} \hat{r} \right), \qquad F_\mathrm{irr}(r) = \frac{L_\star}{4\pi r^2} e^{-\tau(r)},

where :math:`L_\star` is the stellar luminosity, :math:`r` is the distance from the star, :math:`\tau(r)` is the optical depth along the ray from the star to the point of interest, and :math:`\hat{\mathbf{r}}` is the unit vector pointing from the star to that point. The optical depth is calculated by integrating along radial rays from the star:

.. math:: \tau(\mathbf{r}) = \int_{R_\mathrm{\star}}^r \kappa_\mathrm{P}(\mathbf{r}', T_\star) \rho(\mathbf{r}') dr' = \tau_0 + \int_{r_\mathrm{min}}^r \kappa_\mathrm{P}(\mathbf{r}', T_\star) \rho(\mathbf{r}') dr',

where :math:`R_\mathrm{\star}` is the stellar radius and :math:`\tau_0` is an optional initial optical depth that can be specified at the inner boundary of the model. If frequency-dependent opacities are provided instead, the optical depth is calculated per wavelength :math:`\lambda` as:

.. math:: \tau_\lambda(\mathbf{r}) = \kappa_\lambda\left(\sigma_0 + \sigma\right), \qquad \sigma(\mathbf{r}) = \int_{r_\mathrm{min}}^r \rho(\mathbf{r}') dr',

where :math:`\sigma` is the column density along the ray and :math:`\sigma_0` is an optional initial column density between the stellar surface and the inner radial boundary of the domain.

Viscous heating is explicitly modeled as

.. math:: Q_\mathrm{visc} = \frac{9}{4} \nu \rho \Omega^2,

where :math:`\nu=\alpha c_\mathrm{s} H = \alpha\sqrt{\gamma}c_\mathrm{s,iso}^2/\Omega_\mathrm{K}` is the kinematic viscosity with :math:`\alpha` being the dimensionless viscosity parameter following Shakura & Sunyaev (1973), :math:`c_\mathrm{s}` the sound speed, :math:`H` the pressure scale height, :math:`\gamma` the adiabatic index, :math:`c_\mathrm{s,iso}` the isothermal sound speed, and :math:`\Omega_\mathrm{K}` the local Keplerian angular velocity.

Both stellar irradiation and viscous heating are currently only implemented for spherical geometries.

Implementation
===============

``pyFLD`` is fully implemented in Python, using an implicit finite volume method to solve the coupled equations for :math:`T` and :math:`E_\mathrm{r}`. The implicit nature of the method allows for much larger time steps and better stability compared to explicit methods, which is particularly beneficial for stiff problems like radiative diffusion. The implementation largely follows the approach described in Kolb et al. (2013), with several modifications to enhance performance and flexibility. A more detailed description can be found `here <https://alexziab.github.io/resources/fld-handbook/>`_.