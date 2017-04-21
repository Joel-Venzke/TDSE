# Trivial Dynamics in Schrodinger Equation (TDSE)

This code was developed by @Joel-Venzke during his PHD. It solves the Time Dependent Schroedinger Equation.

# Dependencies

HDF5 with c++ and mpi

PETSC with complex

need clang 8.0.0
sudo xcode-select --switch /Library/Developer/CommandLineTools

# INSTALL
You will need hdf5 with c++ and mpi support (see system specifics below). 

Then install petsc with complex support (see system specifics below).

Then go into the src directory and copy `Makefile.${SYSTEM}` to `Makefile` and update the paths

Then back to the root directory and copy a `build.${SYSTEM}` file that is similar to your system to `build`. 

Finally make the needed changes to paths and run `.\build`.

## Mac

Here is the brew command to install on mac 

Note: you need to do a brew edit to make the cxx if statement: `if build.with?("cxx") && (build.without?("mpi") || build.with?("unsupported"))`

`brew install hdf5 --c++11 --with-fortran --with-mpi --with-threadsafe --with-unsupported`

Then install petsc with complex support. Here is the brew command:

`brew install petsc --with-complex`

## Summit CU

Use the `hdf5_build.summitcu` and `petsc_build.summitcu` scripts to compile on summit. Make appropriate changes to the paths


# USAGE
The code can be used by using the `TDSE` binary. Here is an example run command using 4 processors.

`${TDSE_DIR}/bin/mpiexec -n 4 ../../bin/TDSE  -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist`

# Development 

we use the following clang_format 

`{
    "BasedOnStyle": "Google",
    "AlignConsecutiveAssignments": true,
    "BreakBeforeBraces": "Allman"
}`

# NOTES
anaconda may cause issues with install
