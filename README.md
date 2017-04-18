# TDSE
This code was developed by @Joel-Venzke during his PHD. It solves the Time Dependent Schroedinger Equation.

# Dependencies

HDF5 with c++ and mpi

PETSC with complex

# INSTALL
You will need hdf5 with c++ and mpi support. Here is the brew command to install on mac

brew install hdf5 --c++11 --with-fortran --with-mpi --with-threadsafe --with-unsupported

Then install petsc with complex support. Here is the brew command:

brew install petsc --with-complex

Copy a `build.${SYSTEM}` file that is similar to your system to `build`. Then make the needed changes to paths and run `.\build`.


#USAGE
The code can be used by using `TDSE` in the ${TDSE_DIR}/bin/
mpiexec -n 4 ../../bin/TDSE  -eigen_ksp_type preonly -eigen_pc_type lu -eigen_pc_factor_mat_solver_package superlu_dist -log_view -options_left


#NOTES
anaconda may cause issues with install
