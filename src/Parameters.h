#pragma once
#include <iostream>
#include <netcdfcpp.h>

class Parameters {
private:
	// numeric data
    int     num_dims;   // number of dimensions
	double  *dim_size;  // size of nth dimension in a.u.
	double  *delta_x;   // size of grid step
	int     *size_of_x; // number of grid points in nth dimension
	double  delta_t;    // size of time step

	// simulation behavior
	// restart mode 
	// 0 no restart, 1 restart from file
	int         restart; 
	std::string target; // type of target {"He"}

	// pulse data
	int         num_pulses;      // number of pulses
	std::string *pulse_shape;    // pulse shape {"sin2","linear"}
	double      *cycles_on;      // ramp on cycles
	double      *cycles_plateau; // plateau cycles
	double      *cycles_off;     // ramp off cycles
	double      *cep;            // carrier envelope phase
	double      *energy;         // photon energy
	double      *e_max;          // max amplitude

	// files
	NcFile *data_file;
	NcDim** grid_dims;
	NcDim*  num_grid_dims;

public:
	// Constructors
	Parameters(std::string file_name);

	void checkpoint();
	void write_header(NcFile * nc_data_file);

	// getters
	int          get_num_dims();
	double*      get_dim_size();
	double*      get_delta_x();
	int*         get_size_of_x();
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