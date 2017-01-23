#include <iostream>
#include "Simulation.h"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

Simulation::Simulation(Hamiltonian &h, Wavefunction &w,
    Pulse &pulse_in, HDF5Wrapper& data_file, Parameters &p) {
    std::cout << "Creating Simulation\n";
    hamiltonian  = &h;
    wavefunction = &w;
    pulse        = &pulse_in;
    file         = &data_file;
    parameters   = &p;

    create_idenity();

    std::cout << "Simulation Created\n";
}

void Simulation::propagate() {
    psi = wavefunction->get_psi();
    time = pulse->get_time();
    time_length = pulse->get_max_pulse_length();
    Eigen::SparseLU<Eigen::SparseMatrix<dcomp>> solver;
    for (int i=1; i<20; i++) {
        wavefunction->checkpoint(*file,i,time[i]);
    }
}

void Simulation::imag_time_prop(int num_states) {
    std::cout << "Calculating the lowest "<< num_states;
    std::cout <<" eigenvectors\n" << std::flush;
    // if we are converged
    bool converged           = false;
    // write index for checkpoints
    int i                    = 1;
    int start                = 0;
    // how often do we write data
    int write_frequency      = 100;
    // pointer to actual psi in wavefunction object
    psi                      = wavefunction->get_psi();
    // keeps track of last psi
    Eigen::VectorXcd psi_old = *psi;
    // vector of currently converged states
    std::vector<Eigen::VectorXcd> states;
    // time step
    double dt                = parameters->get_delta_t();
    // factor = i*(-i*dx/2)
    dcomp  factor            = dcomp(dt/2,0.0);
    // solver for Ax=b
    Eigen::SparseLU<Eigen::SparseMatrix<dcomp>> solver;
    // time independent Hamiltonian
    Eigen::SparseMatrix<dcomp>* h      = hamiltonian->get_time_independent();
    // left matrix in Ax=Cb it would be A
    Eigen::SparseMatrix<dcomp> left    = *h;
    // right matrix in Ax=Cb it would be C
    Eigen::SparseMatrix<dcomp> right   = left;
    // file for converged states
    HDF5Wrapper states_file(parameters->get_target()+".h5");

    // left side of crank-nicolson
    left    = (idenity[0]+factor*left);
    left.makeCompressed();
    // right side of crank-nicolson
    right   = (idenity[0]-factor*right);
    right.makeCompressed();

    // do this outside the loop since left never changes
    solver.compute(left);

    for (int iter=0; iter<num_states; iter++) {
        while (!converged) {
            psi_old = psi[0];
            psi[0] = solver.solve(right*psi_old);
            modified_gram_schmidt(states);
            wavefunction->normalize();
            if (i%write_frequency==0) {
                converged = check_convergance(psi[0],psi_old,1e-10);
                wavefunction->checkpoint(*file,i/write_frequency,
                    i/write_frequency);
            }
            i++;
        }
        start = i;
        // make sure all states are orthonormal for mgs
        states.push_back(psi[0]/psi->norm());
        wavefunction->checkpoint_psi(states_file,
            "/States",iter);
        wavefunction->reset_psi();
        converged = false;
    }
}

void Simulation::create_idenity(){
    int num_psi = wavefunction->get_num_psi();
    idenity = new Eigen::SparseMatrix<dcomp>(num_psi,num_psi);
    idenity->reserve(Eigen::VectorXi::Constant(num_psi,1));
    for (int i=0; i<num_psi; i++) {
        idenity->insert(i,i) = dcomp(1.0,0.0);
    }
    idenity->makeCompressed();
}

bool Simulation::check_convergance(
    Eigen::VectorXcd &psi_1,
    Eigen::VectorXcd &psi_2,
    double tol) {
    Eigen::VectorXcd diff = psi_1-psi_2;
    double error = diff.norm();
    std::cout<< "Error: " << error << "\n" << std::flush;
    return error<tol;
}

// for details see:
// https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-fall-2010/lecture-notes/MIT18_335JF10_lec10a_hand.pdf
// Assumes all states are orthornormal and applies modified
// gram-schmit to psi
void Simulation::modified_gram_schmidt(
    std::vector<Eigen::VectorXcd> &states) {
    int size    = states.size();
    for (int i=0; i<size; i++){
        psi[0] = psi[0]-psi->conjugate().dot(states[i])*states[i];
    }
}
