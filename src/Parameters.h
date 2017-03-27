#pragma once
#include <iostream>
#include <petsc.h>

class Parameters {
private:
    // numeric data
    PetscInt  num_dims;   // number of dimensions
    double    *dim_size;  // size of nth dimension in a.u.
    double    *delta_x;   // size of grid step
    PetscReal delta_t;    // size of time step

    // simulation behavior
    // restart mode
    // 0 no restart, 1 restart from file
    PetscInt    restart;
    std::string target;           // type of target {"He"}
    PetscInt    target_idx;       // index of target
    PetscReal   z;                // atomic number
    PetscReal   alpha;            // atomic soft core
    PetscReal   beta;             // e-e soft core
    PetscInt    write_frequency;  // how many steps between checkpoints
    PetscReal   gobbler;          // percent (1=100% and .9=90%) gobbler turns on at
    PetscReal   sigma;            // std of wave function guess
    PetscInt    num_states;       // number of ground states
    PetscReal   tol;              // tolerance in error
    std::string state_solver;     // name of the solver used to get states
    PetscInt    state_solver_idx; // index of state solver
    double      *state_energy;    // theoretical eigenvalues for each state

    PetscInt    propagate;        // 0: no propagation 1: propagation

    // pulse data
    PetscInt    num_pulses;       // number of pulses
    std::string *pulse_shape;     // pulse shape {"sin2","linear"}
    PetscInt    *pulse_shape_idx; // index of pulse shape
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
    PetscInt     get_num_dims();
    double*      get_dim_size();
    double*      get_delta_x();
    PetscReal    get_delta_t();

    PetscInt     get_restart();
    std::string  get_target();
    PetscInt     get_target_idx();
    PetscReal    get_z();
    PetscReal    get_alpha();
    PetscReal    get_beta();
    PetscInt     get_write_frequency();
    PetscReal    get_gobbler();
    PetscReal    get_sigma();
    PetscInt     get_num_states();
    double*      get_state_energy();
    PetscReal    get_tol();
    PetscInt     get_state_solver_idx();
    std::string  get_state_solver();

    PetscInt     get_propagate();

    PetscInt     get_num_pulses();
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