#pragma once
#include <cstdlib>
#include <fstream>
#include <memory>
#include <streambuf>
#include "Utils.h"
#include "json.hpp"

/* for convenience */
using json = nlohmann::json;

class Parameters : protected Utils
{
 private:
  /* numeric data */
  int num_dims;      /* number of dimensions */
  int num_electrons; /* number of dimensions */
  double delta_t;    /* size of time step */

  /* simulation behavior */
  /* restart mode */
  /* 0 no restart, 1 restart from file */
  int restart;
  std::string target;              /* type of target {"He"} */
  int target_idx;                  /* index of target */
  int num_nuclei;                  /* number of nuclei in potential */
  double** location;               /* location of each nuclei */
  double alpha;                    /* atomic soft core */
  int write_frequency_propagation; /* how many steps between checkpoints during
                                      propagation */
  int write_frequency_eigin_state; /* how many steps between checkpoints during
                                      eigen state calculations */
  double gobbler; /* percent (1=100% and .9=90%) gobbler turns on at */
  double sigma;   /* std of wave function guess */
  int num_states; /* number of ground states */
  double tol;     /* tolerance in error */
  std::string state_solver; /* name of the solver used to get states */
  int state_solver_idx;     /* index of state solver */

  int propagate; /* 0: no propagation 1: propagation */

  /* pulse data */
  int num_pulses; /* number of pulses */
  std::string polarization;
  int polarization_idx;

  void Setup(std::string file_name);

 public:
  std::unique_ptr<double[]> dim_size;  /* size of nth dimension in a.u. */
  std::unique_ptr<double[]> delta_x;   /* size of grid step */
  std::unique_ptr<double[]> delta_x_2; /* size of grid step squared */
  std::unique_ptr<double[]> z;         /* atomic number for each nuclei */
  std::unique_ptr<double[]>
      state_energy; /* theoretical eigenvalues for each state */
  std::unique_ptr<std::string[]>
      pulse_shape; /* pulse shape {"sin2","linear"} */
  std::unique_ptr<double[]> polarization_vector; /* ramp on cycles */
  std::unique_ptr<int[]> pulse_shape_idx;        /* index of pulse shape */
  std::unique_ptr<double[]> cycles_on;           /* ramp on cycles */
  std::unique_ptr<double[]> cycles_plateau;      /* plateau cycles */
  std::unique_ptr<double[]> cycles_off;          /* ramp off cycles */
  std::unique_ptr<double[]> cycles_delay;        /* delay in number of cycles */
  std::unique_ptr<double[]> cep;                 /* carrier envelope phase */
  std::unique_ptr<double[]> energy;              /* photon energy */
  std::unique_ptr<double[]> field_max;           /* max amplitude */

  /* Constructors */
  Parameters(std::string file_name);
  ~Parameters();

  void Validate();

  /* getters */
  int GetNumDims();
  int GetNumElectrons();
  double GetDeltaT();

  int GetRestart();
  std::string GetTarget();
  int GetTargetIdx();
  int GetNumNuclei();
  double** GetLocation();
  double GetAlpha();
  int GetWriteFrequencyPropagation();
  int GetWriteFrequencyEigenState();
  double GetGobbler();
  double GetSigma();
  int GetNumStates();
  double GetTol();
  int GetStateSolverIdx();
  std::string GetStateSolver();

  int GetPropagate();

  int GetNumPulses();
  int GetPolarizationIdx();
};