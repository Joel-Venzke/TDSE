#include <iostream>
#include "Simulation.h"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <time.h>

Simulation::Simulation(Hamiltonian &h, Wavefunction &w,
    Pulse &pulse_in, HDF5Wrapper& data_file, Parameters &p) {
    std::cout << "Creating Simulation\n";
    hamiltonian  = &h;
    wavefunction = &w;
    pulse        = &pulse_in;
    file         = &data_file;
    parameters   = &p;
    time         = pulse_in.get_time();
    time_length  = pulse_in.get_max_pulse_length();

    create_idenity();

    std::cout << "Simulation Created\n";
}

void Simulation::propagate() {
    std::cout << "\nPropagating in time";
    // how often do we write data
    int write_frequency      = parameters->get_write_frequency();
    // pointer to actual psi in wavefunction object
    psi                      = wavefunction->get_psi();
    // time step
    double dt                = parameters->get_delta_t();
    // factor = i*(-i*dx/2)
    dcomp  factor            = dcomp(0.0,1.0)*dcomp(dt/2,0.0);
    // solver for Ax=b
    Eigen::SparseLU<Eigen::SparseMatrix<dcomp>> solver;
    // time independent Hamiltonian
    Eigen::SparseMatrix<dcomp>* h =
        hamiltonian->get_time_independent();
    // left matrix in Ax=Cb it would be A
    Eigen::SparseMatrix<dcomp> left    = *h;
    // right matrix in Ax=Cb it would be C
    Eigen::SparseMatrix<dcomp> right   = left;

    std::cout << "Total writes: " << time_length/write_frequency;
    std::cout << "\nSetting up solver\n" << std::flush;
    solver.analyzePattern(left);

    std::cout << "Starting propagation\n" << std::flush;
    for (int i=1; i<time_length; i++) {
        h = hamiltonian->get_total_hamiltonian(0);
        left    = (idenity[0]+factor*h[0]);
        left.makeCompressed();
        right   = (idenity[0]-factor*h[0]);
        right.makeCompressed();
        solver.factorize(left);

        psi[0] = solver.solve(right*psi[0]);

        wavefunction->gobble_psi();

        // only checkpoint so often
        if (i%write_frequency==0) {
            std::cout << "On step: " << i << " of " << time_length;
            std::cout << "\nNorm: " << wavefunction->norm() << "\n";
            // write a checkpoint
            wavefunction->checkpoint(*file, time[i]);
        }
    }
    wavefunction->checkpoint(*file, time[time_length-1]);

}

void Simulation::imag_time_prop(int num_states) {
    std::cout << "\nCalculating the lowest "<< num_states;
    std::cout <<" eigenvectors using ITP\n" << std::flush;

    clock_t t;
    // if we are converged
    bool converged           = false;
    // write index for checkpoints
    int i                    = 1;
    // how often do we write data
    int write_frequency      = parameters->get_write_frequency();
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
    Eigen::SparseMatrix<dcomp>* h =
        hamiltonian->get_time_independent();
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

    // loop over number of states wanted
    for (int iter=0; iter<num_states; iter++) {
        modified_gram_schmidt(states);
        t = clock();
        while (!converged) {
            // copy old state for convergence
            psi_old = psi[0];
            psi[0] = solver.solve(right*psi_old);

            // used to get higher states
            modified_gram_schmidt(states);

            // normalize with respect to space (include dt)
            wavefunction->gobble_psi();
            wavefunction->normalize();

            // only checkpoint so often
            if (i%write_frequency==0) {
                // check convergence criteria
                converged = check_convergance(psi[0],psi_old,
                    parameters->get_tol());
                // save this psi to ${target}.h5
                std::cout << "Energy: ";
                std::cout << wavefunction->get_energy(h) << "\n";
                // write a checkpoint
                // wavefunction->checkpoint(*file,i/write_frequency);
            }
            // increment counter
            i++;
        }
	std::cout << "Time: " << ((float)clock() - t)/CLOCKS_PER_SEC << "\n";
        // make sure all states are orthonormal for mgs
        states.push_back(psi[0]/psi->norm());
        // save this psi to ${target}.h5
        wavefunction->checkpoint_psi(states_file,
            "/States",iter);
        // new Gaussian guess
        wavefunction->reset_psi();
        // reset for next state
        converged = false;
        std::cout << "\n";
    }
    psi[0] = states[states.size()-1];
    wavefunction->normalize();
}

void Simulation::power_method(int num_states) {
    std::cout << "\nCalculating the lowest "<< num_states;
    std::cout <<" eigenvectors using power method\n" << std::flush;
    clock_t t;
    // if we are converged
    bool converged           = false;
    bool gram_schmit         = false;
    // write index for checkpoints
    int i                    = 1;
    // how often do we write data
    int write_frequency      = parameters->get_write_frequency();
    // pointer to actual psi in wavefunction object
    psi                      = wavefunction->get_psi();
    // keeps track of last psi
    Eigen::VectorXcd psi_old = *psi;
    // vector of currently converged states
    std::vector<Eigen::VectorXcd> states;
    // energy guesses
    double* state_energy     = parameters->get_state_energy();
    // solver for Ax=b
    Eigen::SparseLU<Eigen::SparseMatrix<dcomp>> solver;
    // time independent Hamiltonian
    Eigen::SparseMatrix<dcomp>* h =
        hamiltonian->get_time_independent();
    // left matrix in Ax=Cb it would be A
    Eigen::SparseMatrix<dcomp> left    = *h;
    // file for converged states
    HDF5Wrapper states_file(parameters->get_target()+".h5");

    // loop over number of states wanted
    for (int iter=0; iter<num_states; iter++) {
        // left side of power method
        left    = (h[0]-(idenity[0]*state_energy[iter]));
        std::cout << state_energy[iter] << "\n";
        left.makeCompressed();

        // do this outside the loop since left never changes
        solver.compute(left);
        if (iter>0 and
            std::abs(state_energy[iter]-state_energy[iter-1])<1e-10) {
            std::cout << "Starting Gram Schmit\n";
            gram_schmit = true;
            // This is needed because the Power method converges
            // extremely quickly
            modified_gram_schmidt(states);
        }

        t = clock();
        // loop until error is small enough
        while (!converged) {
            // copy old state for convergence
            psi_old = psi[0];
            psi[0] = solver.solve(psi_old);

            // used to get higher states
            if (gram_schmit) modified_gram_schmidt(states);

            // normalize with respect to space (include dt)
            wavefunction->gobble_psi();
            wavefunction->normalize();
            // psi[0] = psi[0]/psi->norm();

            // only checkpoint so often
            if (i%write_frequency==0) {
                // check convergence criteria
                converged = check_convergance(psi[0],psi_old,
                    parameters->get_tol());
                // save this psi to ${target}.h5
                std::cout << "Energy: ";
                std::cout << wavefunction->get_energy(h) << "\n";
                // write a checkpoint
                // wavefunction->checkpoint(*file,i/write_frequency);
            }
            // increment counter
            i++;
        }
        std::cout << "Time: " << ((float)clock() - t)/CLOCKS_PER_SEC << "\n";
        // make sure all states are orthonormal for mgs
        states.push_back(psi[0]/psi->norm());
        // save this psi to ${target}.h5
        checkpoint_state(states_file,iter);
        // new Gaussian guess
        wavefunction->reset_psi();
        // reset for next state
        converged = false;
        std::cout << "\n";
    }
    psi[0] = states[states.size()-1];
    wavefunction->normalize();
}

// creates an identity matrix to use
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
    Eigen::SparseMatrix<dcomp>* h = hamiltonian->
        get_time_independent();
    Eigen::VectorXcd diff = psi_1-psi_2;
    double wave_error = diff.norm();
    diff = psi_1+psi_2;
    double wave_error_2 = diff.norm();
    if (wave_error_2<wave_error) wave_error = wave_error_2;
    dcomp energy_error = psi_1.dot(h[0]*psi_1)/psi_1.squaredNorm();
    energy_error -= psi_2.dot(h[0]*psi_2)/psi_2.squaredNorm();
    std::cout << "Wavefunction Error: " << wave_error;
    std::cout << "\nEnergy Error: " << energy_error.real();
    std::cout << "\nTolerance: " << tol << "\n" << std::flush;
    return wave_error<tol and std::abs(energy_error.real())<tol;
}

// for details see:
// https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-fall-2010/lecture-notes/MIT18_335JF10_lec10a_hand.pdf
// Assumes all states are orthornormal and applies modified
// gram-schmit to psi
void Simulation::modified_gram_schmidt(
    std::vector<Eigen::VectorXcd> &states) {
    int size    = states.size();
    for (int i=0; i<size; i++){
        psi[0] = psi[0]-psi->dot(states[i])*states[i];
    }
}

void Simulation::checkpoint_state(HDF5Wrapper& data_file,
    int write_idx) {
    wavefunction->normalize();
    wavefunction->checkpoint_psi(data_file,
        "/States",write_idx);
    data_file.write_object(
        wavefunction->get_energy(hamiltonian->get_time_independent()),
        "/Energy", "Energy of the corresponding state", write_idx);
}
