5. Basic IO
=================

``pyFLD`` provides a simple input/output (IO) system for storing and loading the state of most objects in the package, including grids, stars, fluids, and radiative environments. This can be useful for saving the output of an expensive calculation, or for running multiple models in parallel.

The IO system is not that sophisticated; it uses a combination of ``pickle`` and binary files to store the relevant information for each object. Realistically, you will want to simply dump the state of a ``RadiativeEnvironment`` object as follows:

.. code-block:: python

    Env = fld.FLD.RadiativeEnvironment(...)
    Env.write_snapshot(
            dirc='path/to/snapshot_dir',
            name='snapshot_name'
        )

This will create a hierarchy of files in the specified directory, containing the information for the grid, star, and fluids enrolled in the radiative environment. To load the snapshot back into a new ``RadiativeEnvironment`` object, simply call:

.. code-block:: python

    Env2 = fld.FLD.create_from_snapshot(
                    dirc='path/to/snapshot_dir',
                    name='snapshot_name'
                )

This will read the relevant files and create a new ``RadiativeEnvironment`` object with the same properties as the original one.

Since the snapshot system outputs a hierarchy of files, it also outputs the grid and star information for each fluid in its own subdirectory. This means you can also load individual fluids using ``fld.fluids.create_from_snapshot``, or the grid and star information using ``fld.grid.create_from_snapshot`` and ``fld.star.create_from_snapshot``, respectively.

Notes
-----

- The snapshot system has been tested very lightly. At present, it is not recommended to rely on it with the intent to restart a simulation at a later time, as it may not be robust to changes in the package. However, it can be useful for short-term storage of quantities that are expensive to compute such as density and temperature. Feedback on the snapshot system is very welcome, and it will likely be improved in the future.