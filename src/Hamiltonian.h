#pragma once
#include "Parameters.h"
#include "Wavefunction.h"
#include "HDF5Wrapper.h"
#include "Pulse.h"
#include <iostream>
#include <Eigen/Sparse>

class Hamiltonian {
private:
    int num_dims;
    int *num_x;
    int num_psi;
    int num_psi_12;
    double z;            // atomic number
    double alpha;        // soft core atomic
    double beta;         // soft core ee
    double beta2;         // soft core ee
    double *delta_x;
    double **x_value;
    Eigen::SparseMatrix<dcomp>* time_independent;
    Eigen::SparseMatrix<dcomp>* time_dependent;
    Eigen::SparseMatrix<dcomp>* total_hamlitonian;

    void create_time_independent();
    void create_time_dependent();
    void create_total_hamlitonian();
public:
    // Constructor
    Hamiltonian(Wavefunction &w, Pulse &pulse, HDF5Wrapper& data_file,
        Parameters &p);

    Eigen::SparseMatrix<dcomp>* get_time_hamiltonian(double time);

    ~Hamiltonian();
};