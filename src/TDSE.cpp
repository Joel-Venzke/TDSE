#include "config.h"
#include <iostream>
#include "Hamiltonian.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "Wavefunction.h"
#include "HDF5Wrapper.h"

int main() {
    // initialize all of the classes
    Parameters parameters("input.json");
    HDF5Wrapper data_file(parameters);
    Pulse pulse(data_file, parameters);
    Wavefunction wavefunction(data_file,parameters);
    Hamiltonian hamiltonian(wavefunction,pulse,data_file,parameters);
    Simulation s(hamiltonian,wavefunction,pulse,data_file,parameters);

    s.imag_time_prop(4);

    return 0;
}
