#include "config.h"
#include <iostream>
#include "Hamiltonian.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "Wavefunction.h"
#include "HDF5Wrapper.h"
#include <Eigen/Core>
#include <petscsnes.h>

int main(int argc, char** argv) {
    // initialize all of the classes
    // PetscInitialize(&argc,&argv,(char*)0,"Testing a program!");
    Parameters parameters("input.json");
    HDF5Wrapper data_file(parameters);
    Pulse pulse(data_file, parameters);
    Wavefunction wavefunction(data_file,parameters);
    Hamiltonian hamiltonian(wavefunction,pulse,data_file,parameters);
    Simulation s(hamiltonian,wavefunction,pulse,data_file,parameters);

    // get ground states
    switch (parameters.get_state_solver_idx()) {
        case 0: // File
            break;
        case 1: // ITP
            s.imag_time_prop(parameters.get_num_states());
            break;
        case 2: // Power
            s.power_method(parameters.get_num_states());
            break;
    }

    if (parameters.get_propagate()==1) {
        s.propagate();
    }

    // PetscFinalize();
    return 0;
}
