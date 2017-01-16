#include <iostream>
#include "Hamiltonian.h"
#include <vector>
#include <Eigen/Sparse>

#define dcomp std::complex<double>

Hamiltonian::Hamiltonian(Wavefunction &w, Pulse &pulse,
    HDF5Wrapper& data_file, Parameters &p) {
    std::cout << "Creating Hamiltonian\n";
    num_dims = p.get_num_dims();
    num_x    = w.get_num_x();
    num_psi  = w.get_num_psi();

    // total_hamlitonian = new Eigen::SparseMatrix<dcomp>(num_psi,num_psi);
    // set up time independent
    create_time_independent();
    create_time_dependent();
    create_total_hamlitonian();

    std::cout << "Hamiltonian Created\n";
}

void Hamiltonian::create_time_independent(){
    time_independent = new Eigen::SparseMatrix<dcomp>(num_psi,num_psi);
    time_independent->reserve(Eigen::VectorXi::Constant(num_psi,5));
    for (int i=0; i<num_psi; i++) {
        if (i-num_x[0]>=0 and i-num_x[0]<num_psi) {
            time_independent->insert(i-num_x[0],i) = dcomp(-1.0,0.0);
        }
        if (i-1>=0 and i-1<num_psi) {
            time_independent->insert(i-1,i) = dcomp(-1.0,0.0);
        }
        if (i>=0 and i<num_psi) {
            time_independent->insert(i,i) = dcomp(2.0,0.0);
        }
        if (i+1>=0 and i+1<num_psi) {
            time_independent->insert(i+1,i) = dcomp(-1.0,0.0);
        }
        if (i+num_x[0]>=0 and i+num_x[0]<num_psi) {
            time_independent->insert(i+num_x[0],i) = dcomp(-1.0,0.0);
        }
    }
    time_independent->makeCompressed();
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
    return total_hamlitonian;
}


Hamiltonian::~Hamiltonian() {
    std::cout << "Deleting Hamiltonian\n";
    delete time_independent;
    delete time_dependent;
    delete total_hamlitonian;
}