#pragma once
#include <iostream>

class Parameters {
private:
	// numeric data
	double      delta_t;

	// simulation behavior
	// restart mode
	// 0 no restart
	// 1 restart from file
	int         restart; 
	std::string target; // type of target

	// pulse data
	int         num_pulses; // number of pulses
	// pulse shape {"sin2","linear"}
	std::string *pulse_shape; 
	double      *cycles_on; // ramp on cycles
	double      *cycles_plateau; // plateau cycles
	double      *cycles_off; // ramp off cycles
	double      *cep; // carrier envelope phase
	double      *energy; // photon energy
	double      *e_max; // max amplitude

public:
	// Constructors
	Parameters(std::string file_name);

	// setters
	void set_delta_t(double dt);

	// getters
	double       get_delta_t();
	int          get_restart();
	std::string  get_target();
	int          get_num_pulses();
	std::string* get_pulse_shape();
	double*      get_cycles_on();
	double*      get_cycles_plateau();
	double*      get_cycles_off();
	double*      get_cep();
	double*      get_energy();
	double*      get_e_max();
};