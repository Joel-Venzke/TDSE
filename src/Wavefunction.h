#pragma once
#include <iostream>
#include "Parameters.h"
#include <complex>
#include "HDF5Wrapper.h"

#define dcomp std::complex<double>

class Wavefunction {
private:
    int    num_dims;    // number of dimensions
    double *dim_size;   // sizes of each dimension in a.u.
    double *delta_x;    // step sizes of each dimension in a.u.
    int    *num_x;      // number of grid points in each dimension
    double **x_value;   // location of grid point in each dimension
    bool   psi_12_alloc // true if psi_1 and psi_2 are allocated
    dcomp  *psi_1;      // wavefunction for electron 1
    dcomp  *psi_2;      // wavefunction for electron 2
    dcomp  *psi;        // wavefunction for 2 electron system
public:
    // Constructor
    Wavefunction(HDF5Wrapper& data_file, Parameters& p);

    // destructor
    ~Wavefunction();

    void checkpoint(HDF5Wrapper& data_file);

    void end_run(std::string str);
    void end_run(std::string str, int exit_val);

};
