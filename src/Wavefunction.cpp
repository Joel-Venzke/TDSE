#include <iostream>
#include "Wavefunction.h"
#include <complex>

#define dcomp std::complex<double>

// prints error message, kills code and returns -1
void Wavefunction::end_run(std::string str) {
    std::cout << "\n\nERROR: " << str << "\n" << std::flush;
    exit(-1);
}

// prints error message, kills code and returns exit_val
void Wavefunction::end_run(std::string str, int exit_val) {
    std::cout << "\n\nERROR: " << str << "\n";
    exit(exit_val);
}

Wavefunction::Wavefunction(HDF5Wrapper& data_file, Parameters & p) {
    std::cout << "Creating Wavefunction\n";

    // initialize values
    psi_12_alloc = false;
    num_dims     = p.get_num_dims();
    dim_size     = p.get_dim_size();
    delta_x      = p.get_delta_x();

    // validation
    if (num_dims>1) {
        end_run("Only 1D is currently supported");
    }

    checkpoint(data_file);

    std::cout << "Wavefunction created\n";
}

void Wavefunction::checkpoint(HDF5Wrapper& data_file) {
    std::cout << "Checkpointing Wavefunction\n";
    // data_file.write_object(test, 3, "/Wavefunction/test");
}

// destructor
Wavefunction::~Wavefunction(){
    std::cout << "Deleting Wavefunction\n";
}
