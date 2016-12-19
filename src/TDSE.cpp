#include "config.h"
#include <iostream>
#include "Hamiltonian.h"
#include "Observables.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "Wavefuction.h"
#include "HDF5Wrapper.h"

int main() {
	Parameters parameters("input.json");
    HDF5Wrapper data_file(parameters);
    Pulse pulse(data_file, parameters);
    Hamiltonian h;
    Observables o;
    Simulation s;
    Wavefuction w;


    int num_pulses;
    int num_dims;
    double  *dim_size;  
	double  *delta_x;   
	int     *size_of_x;
    std::string *pulse_shape;
    double      *cycles_on;
    double      *cycles_plateau;
    double      *cycles_off;
    double      *cep;
    double      *energy;
    double      *e_max;
    double      **grid;
    std::string str;

    std::cout << "\n\ndelta t: " << parameters.get_delta_t() << "\n";
	num_dims   = parameters.get_num_dims();
    std::cout << "num dims: " << num_dims << "\n";
    dim_size   = parameters.get_dim_size();
    delta_x    = parameters.get_delta_x();
    size_of_x  = parameters.get_size_of_x();
    grid       = new double*[size_of_x[0]];


    std::cout << "restart: " << parameters.get_restart() << "\n";
    std::cout << "target: " << parameters.get_target() << "\n";
    num_pulses = parameters.get_num_pulses();
    pulse_shape = parameters.get_pulse_shape();
    cycles_on = parameters.get_cycles_on();
    cycles_plateau = parameters.get_cycles_plateau();
    cycles_off = parameters.get_cycles_off();
    cep = parameters.get_cep();
    energy = parameters.get_energy();
    e_max = parameters.get_e_max();
    for (int i = 0; i < num_pulses; ++i)
    {
        std::cout << "Pulse[" << i << "]: ";
        std::cout << pulse_shape[i] << " ";
        std::cout << cycles_on[i] << " ";
        std::cout << cycles_plateau[i] << " ";
        std::cout << cycles_off[i] << " ";
        std::cout << cep[i] << " ";
        std::cout << energy[i] << " ";
        std::cout << e_max[i] << "\n";
    }

    return 0;
}
