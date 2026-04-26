# pyFLD
`pyFLD` is a Python module for performing flux-limited diffusion (FLD) radiative transfer calculations in astrophysical fluid simulations. It is designed to be modular and extensible, allowing users to define custom opacity models, viscosity prescriptions, and other physical parameters.

<!-- table of contents with hyperlinks: -->
- [Installation](#installation)
     - [1. Downloading the module](#1-downloading-the-module)
     - [2. Installing `pyFLD`](#2-installing-pyFLD)
- [Typical Workflow](#typical-workflow)
- [File Outputs](#file-outputs)
- [Examples](#examples)
- [Requirements](#requirements)
- [Related Publication](#related-publication)
- [References](#references)

### Installation

To install, simply clone the repository:

```bash
git clone https://github.com/alexziab/pyFLD.git
cd pyFLD
```

and install the package with pip from the root directory:

```bash
pip install .
```

Done correctly, the file structure should look like this:

```pyFLD/
├── pyFLD/
│   ├── __init__.py
│   ├── pyFLD.c
│   └── ...
├── pyproject.toml
├── README.md
├── docs/
│   └── ...
├── examples/
│   └── ...
├── tests/
│   └── ...
└── ...
```

### Typical Workflow

1. **Create a Grid object**:
     Define a coordinate system and its extents for up to 3 spatial dimensions.
2. **Create a Star object** (if needed):
     Define stellar parameters such as radius and temperature. A Star is required for irradiation and viscous heating.
3. **Create Gas or Dust species**:
     Instantiate each fluid, provide density and temperature fields, and prescribe opacity and viscosity models.
4. **Define a Radiative Environment**:
     Combine the Grid, Star, and fluid species into a RadiativeEnvironment object that encapsulates all necessary information for radiative transfer calculations.
5. **Solve Radiative Transfer**:
     Use the RadiativeEnvironment to perform radiative transfer calculations, re-compute densities and temperatures, and iterate as needed.

### File Outputs

`pyFLD` provides a simple input/output system for storing and loading the state of most objects in the package, including grids, stars, fluids, and radiative environments, using a rather simplistic combination of `pickle` and binary files in a hierarchical manner (i.e., a RadiativeEnvironment snapshot contains its star, grid, and fluids, each containing their own copy of the star and grid, etc.). This allows loading individual objects within the file tree.

The snapshot system has been tested very lightly. At present, it is not recommended to rely on it with the intent to restart a simulation at a later time, as it may not be robust to changes in the package. However, it can be useful for short-term storage of quantities that are expensive to compute such as density and temperature. Feedback on the snapshot system is very welcome, and it will likely be improved in the future.

### Examples

See notebooks in the `examples/` folder for step-by-step demonstration of setups with increasing complexity.

_note: the example notebooks require `matplotlib` for plotting._

### Requirements

- Python `>=3.9`, Numpy, Scipy, Astropy

### Related Publication

For scientific details and methodology, see the related publication:  
(TO BE ADDED) 
_Ondřej Chrenko, ..., Alexandros Ziampras, ..._

### References

- Sphinx documentation can be found in the `docs/build/` folder, also hosted [here](https://alexziab.github.io/resources/pyFLD/readthedocs/).
- For additional details, see docstrings/comments in the `pyFLD/` folder.
- See publication for further context and usage recommendations.
