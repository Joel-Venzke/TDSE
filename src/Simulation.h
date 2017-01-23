#pragma once
#include "Parameters.h"
#include "Wavefunction.h"
#include "HDF5Wrapper.h"
#include "Hamiltonian.h"
#include "Pulse.h"
#include <iostream>

class Simulation {
private:
    Hamiltonian                *hamiltonian;
    Wavefunction               *wavefunction;
    Pulse                      *pulse;
    Parameters                 *parameters;
    HDF5Wrapper                *file;
    Eigen::SparseMatrix<dcomp> *idenity;
    Eigen::VectorXcd           *psi;
    double                     *time;
    int                        time_length;

    bool check_convergance(
        Eigen::VectorXcd &psi_1,
        Eigen::VectorXcd &psi_2,
        double tol);
    void create_idenity();
public:
    // Constructor
    Simulation(Hamiltonian &h, Wavefunction &w, Pulse &pulse_in,
        HDF5Wrapper& data_file, Parameters &p);

    void imag_time_prop(int num_states);
    void propagate();

    void modified_gram_schmidt(std::vector<Eigen::VectorXcd> &states);
};