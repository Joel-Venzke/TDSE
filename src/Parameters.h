#pragma once
#include <cstdlib>
#include <memory>
#include <streambuf>
#include "Utils.h"

class Parameters : protected Utils
{
 private:
  /* numeric data */
  PetscInt num_dims;               ///< number of dimensions
  PetscInt num_electrons;          ///< number of dimensions
  double delta_t;                  ///< size of time step
  std::string coordinate_system;   ///< Cartesian or Cylindrical
  PetscInt coordinate_system_idx;  ///< Cartesian:0, Cylindrical:1, RBF:2,
                                   ///< Spherical:3

  PetscInt m_max;
  PetscInt l_max;

  PetscInt restart;     ///< simulation behavior restart mode (0 no restart, 1
                        /// restart) from file */
  std::string target;   ///< type of target {"He"}
  PetscInt target_idx;  ///< index of target
  PetscInt num_nuclei;  ///< number of nuclei in potential
  double** location;    ///< location of each nuclei
  double** exponential_r_0;         ///< exponential radial center
  double** exponential_amplitude;   ///< exponential  amplitude
  double** exponential_decay_rate;  ///< exponential  decay rate
  double** gaussian_r_0;            ///< Gaussian radial center
  double** gaussian_amplitude;      ///< Gaussian amplitude
  double** gaussian_decay_rate;     ///< Gaussian decay rate
  double** square_well_r_0;         ///< square well radial center
  double** square_well_amplitude;   ///< square well  amplitude
  double** square_well_width;       ///< square well width
  double** yukawa_r_0;              ///< yukawa radial center
  double** yukawa_amplitude;        ///< yukawa amplitude
  double** yukawa_decay_rate;       ///< yukawa decay rate
  double alpha;                     ///< atomic soft core
  double ee_soft_core;              ///< electron-electron repulsion soft core
  PetscInt write_frequency_checkpoint;   ///< how many steps between checkpoints
                                         /// during propagation
  PetscInt write_frequency_observables;  ///< how many steps between observable
                                         /// measurements during propagation
  PetscInt write_frequency_eigin_state;  ///< how many steps between checkpoints
                                         /// during eigen state calculations

  double gobbler;       ///< percent (1=100% and .9=90%) gobbler turns on at
  PetscInt order;       ///< What order finite differences (2,4,6,8,...)
  double sigma;         ///< std of wave function guess
  PetscInt num_states;  ///< number of ground states
  PetscInt num_start_state;       ///< number of states in super position
  PetscInt* start_state_idx;      ///< index of states in super position
  PetscInt* start_state_l_idx;    ///< index of states in super position
  PetscInt* start_state_m_idx;    ///< index of states in super position
  double* start_state_amplitude;  ///< amplitude of states in super position
  double* start_state_phase;      ///< phase of states in super position
  double tol;                     ///< tolerance in error
  std::string state_solver;       ///< name of the solver used to get states
  PetscInt state_solver_idx;      ///< index of state solver

  std::string gauge;   ///< name of the gauge used to get states
  PetscInt gauge_idx;  ///< index of gauge (0 velocity, 1 length)

  PetscInt propagate;       ///< 0: no propagation 1: propagation
  PetscInt free_propagate;  ///< How many free propagation steps (-1 means till
                            /// norm is converged)

  PetscInt field_max_states;  ///< 1 if states at max, 0 if field free

  /* pulse data */
  PetscInt num_pulses;           ///< number of pulses
  PetscInt frequency_shift;
  double** polarization_vector;  ///< polarization vector for each pulse
  double** poynting_vector;      ///< poynting vector of the field
  double* gaussian_length;

  void Setup(std::string file_name);
  void CheckParameter(int size, std::string doc_string);

 public:
  std::unique_ptr< double[] > dim_size;     ///< size of nth dimension in a.u.
  std::unique_ptr< double[] > delta_x_min;  ///< size of minimum grid step
  std::unique_ptr< double[] >
      delta_x_min_end;  ///< point in space grid step starts to increase
  std::unique_ptr< double[] > delta_x_max;  ///< largest grid step
  std::unique_ptr< double[] >
      delta_x_max_start;          ///< point in space grid step stops increasing
  std::unique_ptr< double[] > z;  ///< atomic number for each nuclei
  std::unique_ptr< PetscInt[] >
      square_well_size;  ///< number of square well elements
  std::unique_ptr< PetscInt[] >
      exponential_size;  ///< number of exponential elements
  std::unique_ptr< PetscInt[] > gaussian_size;  ///< number of Gaussian elements
  std::unique_ptr< PetscInt[] > yukawa_size;    ///< number of Gaussian elements
  std::unique_ptr< double[] >
      state_energy;  ///< theoretical eigenvalues for each state

  std::string
      experiment_type;  ///< type of experiment ///{"default", "streaking",
                        ///"transient"}
  double tau_delay;     ///< if streaking or ATAS, XUV needs delay
  std::unique_ptr< std::string[] >
      pulse_shape;  ///< pulse shape {"sin2","linear"}
  std::unique_ptr< PetscInt[] > pulse_shape_idx;  ///< index of pulse shape
  std::unique_ptr< PetscInt[] > power_on;         ///< power of sin^n shape on
  std::unique_ptr< PetscInt[] > power_off;        ///< power of sin^n shape off
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

  PetscInt GetMMax();
  PetscInt GetLMax();

  PetscInt GetRestart();
  std::string GetTarget();
  PetscInt GetTargetIdx();
  PetscInt GetNumNuclei();
  double** GetLocation();
  double** GetExponentialR0();
  double** GetExponentialAmplitude();
  double** GetExponentialDecayRate();
  double** GetGaussianR0();
  double** GetGaussianAmplitude();
  double** GetGaussianDecayRate();
  double** GetSquareWellR0();
  double** GetSquareWellAmplitude();
  double** GetSquareWellWidth();
  double** GetYukawaR0();
  double** GetYukawaAmplitude();
  double** GetYukawaDecayRate();
  double GetAlpha();
  double GetEESoftCore();
  PetscInt GetWriteFrequencyCheckpoint();
  PetscInt GetWriteFrequencyObservables();
  PetscInt GetWriteFrequencyEigenState();
  double GetGobbler();
  PetscInt GetOrder();
  double GetSigma();
  PetscInt GetNumStates();
  PetscInt GetNumStartState();
  PetscInt* GetStartStateIdx();
  PetscInt* GetStartStateLIdx();
  PetscInt* GetStartStateMIdx();
  double* GetStartStateAmplitude();
  double* GetStartStatePhase();
  double GetTol();
  PetscInt GetStateSolverIdx();
  std::string GetStateSolver();
  PetscInt GetGaugeIdx();

  PetscInt GetPropagate();
  PetscInt GetFreePropagate();

  PetscInt GetFieldMaxStates();

  PetscInt GetNumPulses();
  PetscInt GetFrequencyShift();
  double** GetPolarizationVector();
  double** GetPoyntingVector();
  double* GetGaussianLength();
};