# Time Dependent Schrodinger Equation (TDSE) Solver

This code was developed at JILA in Boulder, CO by @Joel-Venzke during his PHD, and contributed to by Cory Goldsmith. It solves the Time Dependent Schroedinger Equation for ultrafast laser pulses interacting with matter.

# Dependencies

HDF5 with c++ and mpi

PETSC with complex

Boost MPI

Compiler with GCC 4.9 or greater (C++11 support)

GNU Scientific Library (GSL)

# INSTALL 

See docs/developer_notes.pdf for latest install method

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
