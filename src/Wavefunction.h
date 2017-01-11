#pragma once
#include <iostream>
#include "Parameters.h"
#include <complex>
#include "HDF5Wrapper.h"

#define dcomp std::complex<double>

class Wavefunction {
private:
    double** x_value;
    dcomp* test; 
public:
    // Constructor
    Wavefunction(HDF5Wrapper& data_file, Parameters& p);

    // destructor
    ~Wavefunction();

    void checkpoint(HDF5Wrapper& data_file);
};
