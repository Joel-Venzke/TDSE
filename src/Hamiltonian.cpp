#include <iostream>
#include "Hamiltonian.h"
#include <vector>
#include <Eigen/Sparse>
#include <math.h>

#define dcomp std::complex<double>

Hamiltonian::Hamiltonian(Wavefunction &w, Pulse &pulse,
    HDF5Wrapper& data_file, Parameters &p) {
    std::cout << "Creating Hamiltonian\n";
    num_dims   = p.get_num_dims();
    num_x      = w.get_num_x();
    num_psi    = w.get_num_psi();
    num_psi_12 = w.get_num_psi_12();
    delta_x    = w.get_delta_x();
    x_value    = w.get_x_value();
    z          = 2.0;
    alpha      = 0.01;
    beta       = 0.05;
    beta2      = beta*beta;

    // total_hamlitonian = new Eigen::SparseMatrix<dcomp>(num_psi,num_psi);
    // set up time independent
    create_time_independent();
    create_time_dependent();
    create_total_hamlitonian();

    std::cout << "Hamiltonian Created\n";
}

void Hamiltonian::create_time_independent(){
    double dx2  = delta_x[0]*delta_x[0];     // dx squared
    double diff; // distance between x_1 and x_2
    dcomp off_diagonal(1.0/(2.0*dx2),0.0);  // off diagonal terms
    dcomp diagonal(0.0,0.0);                // diagonal terms
    time_independent = new Eigen::SparseMatrix<dcomp>(num_psi,num_psi);
    int idx_1; // index for psi_1
    int idx_2; // index for psi_2

    // reserve right amount of memory to save storage
    time_independent->reserve(Eigen::VectorXi::Constant(num_psi,5));

    std::cout << dx2;
    for (int i=0; i<num_psi; i++) {
        if (i-num_x[0]>=0 and i-num_x[0]<num_psi) {
            time_independent->insert(i-num_x[0],i) = off_diagonal;
        }
        if (i-1>=0 and i-1<num_psi) {
            time_independent->insert(i-1,i) = off_diagonal;
        }
        if (i>=0 and i<num_psi) {
            idx_1    = i/num_psi_12;
            idx_2    = i%num_psi_12;
            diff     = x_value[0][idx_1]-x_value[0][idx_2];

            // kinetic term
            diagonal =  dcomp(-2.0/dx2,0.0);
            // nuclei electron 1
            diagonal -= dcomp(z/(x_value[0][idx_1]+alpha),0.0);
            // nuclei electron 2
            diagonal -= dcomp(z/(x_value[0][idx_2]+alpha),0.0);
            // e-e correlation
            diagonal += dcomp(1/sqrt(diff*diff+beta2),0.0);

            time_independent->insert(i,i) = diagonal;

            std::cout << x_value[0][idx_1] << " " << x_value[0][idx_2];
            std::cout << " " << diagonal << "\n";
        }
        if (i+1>=0 and i+1<num_psi) {
            time_independent->insert(i+1,i) = off_diagonal;
        }
        if (i+num_x[0]>=0 and i+num_x[0]<num_psi) {
            time_independent->insert(i+num_x[0],i) = off_diagonal;
        }
    }

    // reduce to correct size
    time_independent->makeCompressed();

    for (int k=0; k<time_independent->outerSize(); ++k) {
        for (Eigen::SparseMatrix<dcomp>::InnerIterator it(*time_independent,k); it; ++it)
        {
            std::cout << it.row() << " " << it.col() << " ";
            std::cout << it.value() << "\n";
        }
    }
}

void Hamiltonian::create_time_dependent(){
    time_dependent = new Eigen::SparseMatrix<dcomp>(num_psi,num_psi);
    time_dependent->reserve(Eigen::VectorXi::Constant(num_psi,5));
    for (int i=0; i<num_psi; i++) {
        if (i-num_x[0]>=0 and i-num_x[0]<num_psi) {
            time_dependent->insert(i-num_x[0],i) = dcomp(-1.0,0.0);
        }
        if (i-1>=0 and i-1<num_psi) {
            time_dependent->insert(i-1,i) = dcomp(-1.0,0.0);
        }
        if (i>=0 and i<num_psi) {
            time_dependent->insert(i,i) = dcomp(2.0,0.0);
        }
        if (i+1>=0 and i+1<num_psi) {
            time_dependent->insert(i+1,i) = dcomp(-1.0,0.0);
        }
        if (i+num_x[0]>=0 and i+num_x[0]<num_psi) {
            time_dependent->insert(i+num_x[0],i) = dcomp(-1.0,0.0);
        }
    }
    time_dependent->makeCompressed();
}

void Hamiltonian::create_total_hamlitonian(){
    total_hamlitonian = new Eigen::SparseMatrix<dcomp>(num_psi,num_psi);
    total_hamlitonian->reserve(Eigen::VectorXi::Constant(num_psi,5));
    for (int i=0; i<num_psi; i++) {
        if (i-num_x[0]>=0 and i-num_x[0]<num_psi) {
            total_hamlitonian->insert(i-num_x[0],i) = dcomp(-1.0,0.0);
        }
        if (i-1>=0 and i-1<num_psi) {
            total_hamlitonian->insert(i-1,i) = dcomp(-1.0,0.0);
        }
        if (i>=0 and i<num_psi) {
            total_hamlitonian->insert(i,i) = dcomp(2.0,0.0);
        }
        if (i+1>=0 and i+1<num_psi) {
            total_hamlitonian->insert(i+1,i) = dcomp(-1.0,0.0);
        }
        if (i+num_x[0]>=0 and i+num_x[0]<num_psi) {
            total_hamlitonian->insert(i+num_x[0],i) = dcomp(-1.0,0.0);
        }
    }
    total_hamlitonian->makeCompressed();
}

Eigen::SparseMatrix<dcomp>* Hamiltonian::get_time_hamiltonian(double time) {
    total_hamlitonian[0] = time*time_dependent[0];
    total_hamlitonian[0] = total_hamlitonian[0]+time_independent[0];
    total_hamlitonian->makeCompressed();
    return total_hamlitonian;
}


Hamiltonian::~Hamiltonian() {
    std::cout << "Deleting Hamiltonian\n";
    delete time_independent;
    delete time_dependent;
    delete total_hamlitonian;
}