# Trivial Dynamics in Schrodinger Equation (TDSE)

This code was developed at JILA in Boulder, CO by @Joel-Venzke during his PHD, and contributed to by Cory Goldsmith. It solves the Time Dependent Schroedinger Equation for ultrafast laser pulses interacting with matter.

# Dependencies

HDF5 with c++ and mpi

PETSC with complex

Boost MPI

Compiler with GCC 4.9 or greater (C++11 support)

# INSTALL DEPENDENCIES

You will need hdf5 with c++ and mpi support (see system specifics below). 

Then install petsc with complex support (see system specifics below).

## Mac

Here is the brew command to install on mac 

Note: It looks like the brew install script has dropped support for hdf5 with all of the options we need. However, you can run `brew edit hdf5` and change the install script to include 
~~~~
if build.without?("mpi")
  args << "--enable-cxx"
else
  args << "--enable-cxx"
  args << "--enable-cxx"
end
~~~~

Then run 

`brew install hdf5 --c++11 --with-fortran --with-mpi --with-threadsafe --with-unsupported`

Then install petsc with complex support. Here is the brew command:

`brew install petsc --with-complex`

Then install slepc with complex support. Here is the brew command:

`brew install slepc --with-complex`

Now you will need to build the TDSE code in the INSTALL TDSE section. 

## Summit CU

Make sure you set PETSC_DIR and SLEPC_DIR in your `~/.bashrc` or equivalent file once they are installed before running the next install script. HDF5 must be installed first, followed by PETSC, followed by SLEPC. 

Use the `hdf5_build.summitcu`, `petsc_build.summitcu`, and `slepc_build.summitcu` scripts to compile on summit. Make appropriate changes to the paths and modules. Running these scripts can take a while depending on the system

HDF5 must be installed from the local install directory. Both PETSC and SLEPC are installed from their respective repositories.

# INSTALL TDSE

Make sure both the PETSC_DIR and SLEPC_DIR variables set in you `~/.bashrc` or equivalent file. Make sure you "source" the file as well.

First you will need to make a bin directory in the root directory of the TDSE repository. 

Then go into the src directory and copy `Makefile.${SYSTEM}` to `Makefile`. Open the Make file and adjust any paths to the various dependencies. 

Then back to the TDSE root directory and copy a `build.${SYSTEM}` file that is similar to your system to `build`. Now you will need to update any modules that are loaded.

Finally run `.\build` and the code will compile if everything is installed correctly.

If there is crazy behavior with the code, run a `./clean` and `./build`

# USAGE
The code can be used by using the `TDSE` binary. Here is an example run command using 4 processors.

`mpiexec -n 4 ${TDSE_DIR}/bin/TDSE`

When using the Power method, it is best to utilize a direct solver since the matrix is nearly singular. To achive this, add the following arguments on the end of the run command. This requires the SuperLU Distribute code 

`-eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist`

the resulting command will look like 

`mpiexec -n 4 ${TDSE_DIR}/bin/TDSE -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist`

When using SLEPC for eigen states, it is nice to use the argument 

`-eps_monitor`

to see how your eigen state calculation is coming along.

If you want to try out various solvers for propagation use

`-prop_ksp_type ${KSP_TYPE}`

For more information on trying out different solvers, preconditions, and other optimizations, checkout PETSC's user manual.

# Development 

we use the following clang_format 

`{
    "BasedOnStyle": "Google",
    "AlignConsecutiveAssignments": true,
    "BreakBeforeBraces": "Allman",
    "SpacesInAngles": true
}`

# NOTES
anaconda may cause issues with install
