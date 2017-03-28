#pragma once
#include <iostream>
#include <petsc.h>
#include <fstream>
#include <streambuf>
#include "json.cpp"
#include <cstdlib>
#include "ViewWrapper.h"

/* for convenience */
using json = nlohmann::json;

class Parameters {
private:
    PetscInt  ierr;
    PetscInt  rank;

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
    double      *field_max;       // max amplitude

public:
    // Constructors
    Parameters(ViewWrapper& data_file, std::string file_name);
    ~Parameters();

    void Checkpoint(ViewWrapper& data_file);
    void Validate();

    void EndRun(std::string str);
    void EndRun(std::string str, int exit_val);

    // getters
    PetscInt     GetNumDims();
    double*      GetDimSize();
    double*      GetDeltaX();
    PetscReal    GetDeltaT();

    PetscInt     GetRestart();
    std::string  GetTarget();
    PetscInt     GetTargetIdx();
    PetscReal    GetZ();
    PetscReal    GetAlpha();
    PetscInt     GetWriteFrequency();
    PetscReal    GetGobbler();
    PetscReal    GetSigma();
    PetscInt     GetNumStates();
    double*      GetStateEnergy();
    PetscReal    GetTol();
    PetscInt     GetStateSolverIdx();
    std::string  GetStateSolver();

    PetscInt     GetPropagate();

    PetscInt     GetNumPulses();
    std::string* GetPulseShape();
    int*         GetPulseShapeIdx();
    double*      GetCyclesOn();
    double*      GetCyclesPlateau();
    double*      GetCyclesOff();
    double*      GetCep();
    double*      GetEnergy();
    double*      GetFieldMax();
    double*      GetCyclesDelay();
};