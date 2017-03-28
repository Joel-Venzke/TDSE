#pragma once
#include <iostream>
#include "Parameters.h"
#include "HDF5Wrapper.h"

class Pulse {
private:
    const double pi = 3.1415926535897;
    int          rank;
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
    double       *field_max;       // max amplitude
    double       *time;            // stores the time at each point
    double       **pulse_value;    // pulse value
    double       **pulse_envelope; // envelope function of pulse
    double       *a_field;         // total vector potential
    // true if the individual pulses and envelopes are allocated
    bool         pulse_alloc;

    // private to avoid unneeded allocation calls and to protect the
    // developer form accessing garbage arrays
    void initialize_pulse(int i);
    void initialize_pulse();
    void initialize_time();
    void deallocate_pulses();
    void initialize_a_field();
public:
    // Constructor
    Pulse(HDF5Wrapper& data_file, Parameters& p);

    // Destructor
    ~Pulse();

    // write out data
    void checkpoint(HDF5Wrapper& data_file);

    // accessors methods
    double* get_a_field();
    double* get_time();
    int     get_max_pulse_length();
};