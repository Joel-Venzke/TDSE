#pragma once
#include <cstdlib>
#include <fstream>
#include <memory>
#include <streambuf>
#include "Utils.h"
#include "json.hpp"

using json = nlohmann::json;

class Parameters : protected Utils
{
 private:
  /* numeric data */
  PetscInt num_dims;               ///< number of dimensions
  PetscInt num_electrons;          ///< number of dimensions
  double delta_t;                  ///< size of time step
  std::string coordinate_system;   ///< Cartesian or Cylindrical
  PetscInt coordinate_system_idx;  ///< Cartesian:0, Cylindrical:1

  PetscInt restart;     ///< simulation behavior restart mode (0 no restart, 1
                        /// restart) from file */
  std::string target;   ///< type of target {"He"}
  PetscInt target_idx;  ///< index of target
  PetscInt num_nuclei;  ///< number of nuclei in potential
  double** location;    ///< location of each nuclei
  double** a;           ///< SAE a for each nuclei (coefficient of exponential)
  double** b;           ///< SAE b for each nuclei (in exponential)
  double alpha;         ///< atomic soft core
  PetscInt write_frequency_checkpoint;   ///< how many steps between checkpoints
                                         /// during propagation
  PetscInt write_frequency_observables;  ///< how many steps between observable
                                         /// measurements during propagation
  PetscInt write_frequency_eigin_state;  ///< how many steps between checkpoints
                                         /// during eigen state calculations

  double gobbler;        ///< percent (1=100% and .9=90%) gobbler turns on at
  PetscInt order;        ///< What order finite differences (2,4,6,8,...)
  double sigma;          ///< std of wave function guess
  PetscInt num_states;   ///< number of ground states
  PetscInt start_state;  ///< number of ground states
  double tol;            ///< tolerance in error
  std::string state_solver;   ///< name of the solver used to get states
  PetscInt state_solver_idx;  ///< index of state solver

  PetscInt propagate;       ///< 0: no propagation 1: propagation
  PetscInt free_propagate;  ///< How many free propagation steps (-1 means till
                            /// norm is converged)

  /* pulse data */
  PetscInt num_pulses;           ///< number of pulses
  double** polarization_vector;  ///< polarization vector for each pulse
  double** poynting_vector;      ///< poynting vector of the field

  void Setup(std::string file_name);

 public:
  std::unique_ptr< double[] > dim_size;     ///< size of nth dimension in a.u.
  std::unique_ptr< double[] > delta_x_min;  ///< size of minimum grid step
  std::unique_ptr< double[] >
      delta_x_min_end;  ///< point in space grid step starts to increase
  std::unique_ptr< double[] > delta_x_max;  ///< largest grid step
  std::unique_ptr< double[] >
      delta_x_max_start;          ///< point in space grid step stops increasing
  std::unique_ptr< double[] > z;  ///< atomic number for each nuclei
  std::unique_ptr< double[] > r0;          ///< SAE r_0 for each nuclei
  std::unique_ptr< double[] > c0;          ///< SAE C_0 for each nuclei
  std::unique_ptr< double[] > z_c;         ///< SAE z_0 for each nuclei
  std::unique_ptr< PetscInt[] > sae_size;  ///< number of elements in a and b
  std::unique_ptr< double[] >
      state_energy;  ///< theoretical eigenvalues for each state

  std::string
      experiment_type;  ///< type of experiment ///{"default", "streaking",
                        ///"transient"}
  double tau_delay;     ///< if streaking or ATAS, XUV needs delay
  std::unique_ptr< std::string[] >
      pulse_shape;  ///< pulse shape {"sin2","linear"}
  std::unique_ptr< PetscInt[] > pulse_shape_idx;  ///< index of pulse shape
  std::unique_ptr< double[] > cycles_on;          ///< ramp on cycles
  std::unique_ptr< double[] > cycles_plateau;     ///< plateau cycles
  std::unique_ptr< double[] > cycles_off;         ///< ramp off cycles
  std::unique_ptr< double[] > cycles_delay;       ///< delay in number of cycles
  std::unique_ptr< double[] >
      cep;  ///< carrier envelope phase fractions of a cycle
  std::unique_ptr< double[] > energy;     ///< photon energy
  std::unique_ptr< double[] > field_max;  ///< max amplitude
  std::unique_ptr< double[] >
      ellipticity;  ///< major_min/minor_max of the field
  std::unique_ptr< std::string[] > helicity;  ///< helicity of the field
  std::unique_ptr< PetscInt[] >
      helicity_idx;  ///< helicity index right:0 left:1

  /* Constructors */
  Parameters(std::string file_name);
  ~Parameters();

  void Validate();

  /* getters */
  PetscInt GetNumDims();
  PetscInt GetNumElectrons();
  PetscInt GetCoordinateSystemIdx();
  double GetDeltaT();

  PetscInt GetRestart();
  std::string GetTarget();
  PetscInt GetTargetIdx();
  PetscInt GetNumNuclei();
  double** GetLocation();
  double** GetA();
  double** GetB();
  double GetAlpha();
  PetscInt GetWriteFrequencyCheckpoint();
  PetscInt GetWriteFrequencyObservables();
  PetscInt GetWriteFrequencyEigenState();
  double GetGobbler();
  PetscInt GetOrder();
  double GetSigma();
  PetscInt GetNumStates();
  PetscInt GetStartState();
  double GetTol();
  PetscInt GetStateSolverIdx();
  std::string GetStateSolver();

  PetscInt GetPropagate();
  PetscInt GetFreePropagate();

  PetscInt GetNumPulses();
  double** GetPolarizationVector();
  double** GetPoyntingVector();
};