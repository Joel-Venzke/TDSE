#pragma once
#include "Parameters.h"
#include "Wavefunction.h"
#include "HDF5Wrapper.h"
#include "Hamiltonian.h"
#include "Pulse.h"
#include <iostream>

class Simulation {
private:
    Hamiltonian  *hamiltonian;
    Wavefunction *wavefunction;
    Pulse        *pulse;
    HDF5Wrapper  *file;
    Eigen::SparseMatrix<dcomp>* cur_hamlitonian;
    Eigen::VectorXcd  *psi;
    double       *time;
    int          time_length;
public:
    // Constructor
    Simulation(Hamiltonian &h, Wavefunction &w, Pulse &pulse_in,
        HDF5Wrapper& data_file, Parameters &p);

    void propagate();
};