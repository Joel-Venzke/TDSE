# Trivial Dynamics in Schrodinger Equation (TDSE)

This code was developed by @Joel-Venzke during his PHD. It solves the Time Dependent Schroedinger Equation for ultrafast laser pulses interacting with matter.

# Dependencies

HDF5 with c++ and mpi

PETSC with complex

Compiler with GCC 4.9 or greater

C++11 support

# INSTALL

You will need hdf5 with c++ and mpi support (see system specifics below). 

Then install petsc with complex support (see system specifics below).

Then go into the src directory and copy `Makefile.${SYSTEM}` to `Makefile` and update the paths

Then back to the root directory and copy a `build.${SYSTEM}` file that is similar to your system to `build`. 

Finally make the needed changes to paths and run `.\build`.

## Mac

Need clang 8.0.0 if running on mac here is the command to run once you installed the 10.12 command line tools

`sudo xcode-select --switch /Library/Developer/CommandLineTools`

then

`clang -v` 

should tell you the version is 8.0.0

Here is the brew command to install on mac 

Note: you need to do a brew edit to make the cxx if statement: `if build.with?("cxx") && (build.without?("mpi") || build.with?("unsupported"))`

`brew install hdf5 --c++11 --with-fortran --with-mpi --with-threadsafe --with-unsupported`

Then install petsc with complex support. Here is the brew command:

`brew install petsc --with-complex`

## Summit CU

Use the `hdf5_build.summitcu` and `petsc_build.summitcu` scripts to compile on summit. Make appropriate changes to the paths


# USAGE
The code can be used by using the `TDSE` binary. Here is an example run command using 4 processors.

`mpiexec -n 4 ${TDSE_DIR}/bin/TDSE  -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist`

# Development 

we use the following clang_format 

`{
    "BasedOnStyle": "Google",
    "AlignConsecutiveAssignments": true,
    "BreakBeforeBraces": "Allman"
}`

# NOTES
anaconda may cause issues with install
