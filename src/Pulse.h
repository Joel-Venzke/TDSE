#pragma once
#include <iostream>
#include "Parameters.h"
#include "HDF5Wrapper.h"

class Pulse {
private:
	const double pi = 3.1415926535897;
	int          num_pulses;       // number of pulses
	double       delta_t;          // time step
	int          max_pulse_length; // length of longest pulse;
	int          *pulse_shape_idx; // index of pulse shape
	double       *cycles_on;       // ramp on cycles
	double       *cycles_plateau;  // plateau cycles
	double       *cycles_off;      // ramp off cycles
	double       *cycles_delay;    // cycles till it starts
	double       *cycles_total;    // cycles in pulse
	double       *cep;             // carrier envelope phase
	double       *energy;          // photon energy
	double       *e_max;           // max amplitude
	int          *pulse_length;    // length of each pulse
	double       *time;
	double       **pulse_value;    // pulse value
	double       **pulse_envelope; // envelope function of pulse

	void initialize_pulse(int i);
	void initialize_pulse();
	void initialize_time();
public:
	// Constructor
	Pulse(HDF5Wrapper& data_file, Parameters& p);

	~Pulse();

	void checkpoint(HDF5Wrapper& data_file);
};