#include "config.h"
#include <iostream>
#include "Hamiltonian.h"
#include "Observables.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "Wavefuction.h"


int main() {
	int num_pulses;
	std::string *pulse_shape;
	double      *cycles_on;
	double      *cycles_plateau;
	double      *cycles_off;
	double      *cep;
	double      *energy;
	double      *e_max;

    Hamiltonian h;
    Observables o;
    Parameters p("input.json");
    std::cout << p.get_delta_t() << "\n";
    std::cout << p.get_restart() << "\n";
    std::cout << p.get_target() << "\n";
    num_pulses = p.get_num_pulses();
    pulse_shape = p.get_pulse_shape();
	cycles_on = p.get_cycles_on();
	cycles_plateau = p.get_cycles_plateau();
	cycles_off = p.get_cycles_off();
	cep = p.get_cep();
	energy = p.get_energy();
	e_max = p.get_e_max();
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
    Pulse pulse;
    Simulation s;
    Wavefuction w;

    return 0;
}
