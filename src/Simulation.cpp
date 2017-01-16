#include <iostream>
#include "Simulation.h"

Simulation::Simulation(Hamiltonian &h, Wavefunction &w,
    Pulse &pulse_in, HDF5Wrapper& data_file, Parameters &p) {
    std::cout << "Creating Simulation\n";
    hamiltonian  = &h;
    wavefunction = &w;
    pulse        = &pulse_in;
    file         = &data_file;
}

void Simulation::propagate() {
    psi = wavefunction->get_psi();
    time = pulse->get_time();
    time_length = pulse->get_max_pulse_length();
    Eigen::SparseLU<Eigen::SparseMatrix<dcomp>> solver;
    for (int i=1; i<20; i++) {
        cur_hamlitonian = hamiltonian->get_time_hamiltonian(time[i]);
        solver.compute(*cur_hamlitonian);
        psi[0] = solver.solve(*psi);
        psi->normalize();
        wavefunction->checkpoint(*file,i,time[i]);
    }
}