#include <iostream>
#include "Wavefunction.h"
#include <complex>

#define dcomp std::complex<double>

Wavefunction::Wavefunction(HDF5Wrapper& data_file, Parameters & p) {
    std::cout << "Creating Wavefunction";
    test = new dcomp[3];
    test[0] = dcomp(1,0);
    test[1] = dcomp(1.000234,1);
    test[2] = test[0]+test[1];
    std::cout << "\n";
    std::cout << test[0] << "\n";
    std::cout << test[1] << "\n";
    std::cout << test[2] << "\n";

    checkpoint(data_file);

    std::cout << "Wavefunction created";
}

void Wavefunction::checkpoint(HDF5Wrapper& data_file) {
    std::cout << "checkpointing Wavefunction\n";
    data_file.write_object(test, 3, "/Wavefunction/test");
}

// destructor
Wavefunction::~Wavefunction(){
    std::cout << "Deleting Wavefunction\n";
}
