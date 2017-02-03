#pragma once
#include <iostream>

class Parameters {
private:
    // numeric data
    int     num_dims;   // number of dimensions
    double  *dim_size;  // size of nth dimension in a.u.
    double  *delta_x;   // size of grid step
    double  delta_t;    // size of time step

    // simulation behavior
    // restart mode
    // 0 no restart, 1 restart from file
    int         restart;
    std::string target;         // type of target {"He"}
    int         target_idx;     // index of target
    double      z;              // atomic number
    double      alpha;          // atomic soft core
    double      beta;           // e-e soft core
    int         write_frequency;// how many steps between checkpoints
    double      gobbler;        // percent (1=100% and .9=90%) gobbler turns on at
    double      sigma;          // std of wave function guess
    int         num_states;     // number of ground states
    double      tol;            // tolerance in error
    std::string state_solver;   // name of the solver used to get states
    int         state_solver_idx; // index of state solver
    double      *state_energy;  // theoretical eigenvalues for each state

    // pulse data
    int         num_pulses;       // number of pulses
    std::string *pulse_shape;     // pulse shape {"sin2","linear"}
    int         *pulse_shape_idx; // index of pulse shape
    double      *cycles_on;       // ramp on cycles
    double      *cycles_plateau;  // plateau cycles
    double      *cycles_off;      // ramp off cycles
    double      *cycles_delay;    // delay in number of cycles
    double      *cep;             // carrier envelope phase
    double      *energy;          // photon energy
    double      *e_max;           // max amplitude

public:
    // Constructors
    Parameters(std::string file_name);
    ~Parameters();

    void checkpoint();
    void validate();

    void end_run(std::string str);
    void end_run(std::string str, int exit_val);

    // getters
    int          get_num_dims();
    double*      get_dim_size();
    double*      get_delta_x();
    double       get_delta_t();

    int          get_restart();
    std::string  get_target();
    int          get_target_idx();
    double       get_z();
    double       get_alpha();
    double       get_beta();
    int          get_write_frequency();
    double       get_gobbler();
    double       get_sigma();
    int          get_num_states();
    double*      get_state_energy();
    double       get_tol();
    int          get_state_solver_idx();
    std::string  get_state_solver();

    int          get_num_pulses();
    std::string* get_pulse_shape();
    int*         get_pulse_shape_idx();
    double*      get_cycles_on();
    double*      get_cycles_plateau();
    double*      get_cycles_off();
    double*      get_cep();
    double*      get_energy();
    double*      get_e_max();
    double*      get_cycles_delay();
};