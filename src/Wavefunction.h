#pragma once
#include "HDF5Wrapper.h"
#include "Parameters.h"
#include "Utils.h"
#include "ViewWrapper.h"

class Wavefunction : protected Utils
{
 private:
  PetscInt ierr;
  PetscInt num_dims;         /* number of dimensions */
  PetscInt num_electrons;    /* number of electrons in the system */
  double *dim_size;          /* sizes of each dimension in a.u. */
  double *delta_x_min;       /* step sizes of each dimension in a.u. */
  double *delta_x_min_end;   /* step sizes of each dimension in a.u. */
  double *delta_x_max;       /* step sizes of each dimension in a.u. */
  double *delta_x_max_start; /* step sizes of each dimension in a.u. */
  double delta_t;            /* time step in a.u. */
  PetscInt l_max;
  PetscInt m_max;
  PetscInt coordinate_system_idx;
  std::string target_file_name;
  PetscInt num_states;
  PetscInt *num_x;        /* number of grid points in each dimension */
  PetscInt num_psi_build; /* number of points in psi_1 and psi_2 */
  PetscInt num_psi;       /* number of points in psi */
  double **x_value;       /* location of grid point in each dimension */
  PetscInt *l_values;     /* l_values for the spherical code */
  PetscInt *m_values;     /* m_values for the spherical code */
  dcomp ***psi_build;     /* used for allocating new wave functions */
  Vec psi;                /* wavefunction for 2 electron system */
  Vec psi_small;          /* wavefunction for spherical eigen states */
  Vec psi_tmp;            /* wavefunction for 2 electron system */
  Vec psi_proj;           /* wavefunction for 2 electron system */
  Vec psi_tmp_cyl;        /* wavefunction for 2 electron system */
  Vec jacobian;
  Vec ECS;
  Vec *position_expectation;
  Vec *dipole_acceleration;
  Mat *position_mat;
  bool position_mat_alloc;
  bool psi_alloc_build;
  bool psi_alloc;
  /* false if its not the first time checkpointing the wavefunction */
  bool first_pass;
  double sigma;           /* std of gaussian guess */
  PetscInt **gobbler_idx; /* distance that starts gobbler */
  PetscInt order;

  PetscInt write_counter_checkpoint;
  PetscInt write_counter_observables;
  PetscInt write_counter_projections;

  PetscLogEvent time_norm;
  PetscLogEvent time_energy;
  PetscLogEvent time_position;
  PetscLogEvent time_dipole_acceration;
  PetscLogEvent time_gobbler;
  PetscLogEvent time_projections;
  PetscLogEvent time_block_pathways;
  PetscLogEvent time_insert_radial_psi;

  /* Block Pathways Stuff*/
  PetscInt num_block_state;
  PetscInt *block_state_idx;
  PetscInt *block_state_l_idx;
  PetscInt *block_state_m_idx;
  Vec *psi_block; /*Array of petsc vectors to project out at every timestep*/
                    /*Reading in from VecLoad causes major slowdown*/

  /* SAE stuff */
  double *z;                        ///< atomic number of each nuclei
  double **location;                ///< location of each nuclei
  double **exponential_r_0;         ///< exponential radial center
  double **exponential_amplitude;   ///< exponential  amplitude
  double **exponential_decay_rate;  ///< exponential  decay rate
  PetscInt *exponential_size;       ///< number of elements in exponential
  double **gaussian_r_0;            ///< Gaussian radial center
  double **gaussian_amplitude;      ///< Gaussian amplitude
  double **gaussian_decay_rate;     ///< Gaussian decay rate
  PetscInt *gaussian_size;          ///< number of elements in Gaussian
  double **square_well_r_0;         ///< square well radial center
  double **square_well_amplitude;   ///< square well  amplitude
  double **square_well_width;       ///< square well width
  PetscInt *square_well_size;       ///< number of elements in Gaussian
  double **yukawa_r_0;              ///< yukawa radial center
  double **yukawa_amplitude;        ///< yukawa amplitude
  double **yukawa_decay_rate;       ///< yukawa decay rate
  PetscInt *yukawa_size;            ///< number of elements in Gaussian
  double alpha;          /* soft core atomic */
  double alpha_2;        /* square of soft core atomic */
  double ee_soft_core;   /* soft core atomic */
  double ee_soft_core_2; /* square of soft core atomic */
  PetscInt num_nuclei; /* number of nuclei in potential */

  /* hidden from user for safety */
  void CreateGrid();
  void CreatePsi();
  void CreateObservables();
  void CleanUp();

  dcomp GetPsiVal(dcomp ***data, PetscInt idx);
  dcomp GetPositionVal(PetscInt idx, PetscInt elec_idx, PetscInt dim_idx,
                       bool integrate);
  dcomp GetDipoleAccerationVal(PetscInt idx, PetscInt elec_idx,
                               PetscInt dim_idx);
  dcomp GetGobblerVal(PetscInt idx);
  dcomp GetVolumeElement(PetscInt idx);

  double GetDistance(std::vector< PetscInt > idx_array, PetscInt elec_idx);

  std::vector< PetscInt > GetIntArray(PetscInt idx);

  void LoadRestart(HDF5Wrapper &h5_file, ViewWrapper &viewer_file,
                   PetscInt write_frequency_checkpoint,
                   PetscInt write_frequency_observables);

  void InsertRadialPsi(Vec &psi_radial, Vec &psi_total, PetscInt l_val,
                       PetscInt m_val);

 public:
  /* Constructor */
  Wavefunction(HDF5Wrapper &h5_file, ViewWrapper &view_file, Parameters &p);

  /* destructor */
  ~Wavefunction();

  /* IO */
  void Checkpoint(HDF5Wrapper &data_file, ViewWrapper &view_file, double time,
                  PetscInt checkpoint_psi = 0);
  void CheckpointPsi(ViewWrapper &view_file, PetscInt write_idx);
  void CheckpointPsiSmall(ViewWrapper &view_file, PetscInt write_idx,
                          PetscInt l_val);

  /* tools */
  void Normalize();
  void Normalize(Vec &data, double dx);
  double Norm();
  double Norm(Vec &data, double dx);
  double GetEnergy(Mat *h);
  double GetEnergy(Mat *h, Vec &p);
  double GetPosition(PetscInt elec_idx, PetscInt dim_idx);
  double GetDipoleAcceration(PetscInt elec_idx, PetscInt dim_idx);
  double GetGobbler();
  PetscInt GetProjectionSize();
  std::vector< dcomp > Projections(std::string file_name);
  void BlockPathways(std::string file_name);
  void ProjectOut(std::string file_name, HDF5Wrapper &h5_file,
                  ViewWrapper &viewer_file, double time);
  void LoadPsi(std::string file_name, PetscInt num_states,
               PetscInt num_start_state, PetscInt *start_state_idx,
               double *start_state_amplitude, double *start_state_phase);
  void LoadPsi(std::string file_name, PetscInt num_states,
               PetscInt num_start_state, PetscInt *start_state_idx,
               PetscInt *start_state_l_idx, PetscInt *start_state_m_idx,
               double *start_state_amplitude, double *start_state_phase);
  void ResetPsi();
  void ZeroPhasePsiSmall();
  void RadialHGroundPsiSmall();
  void SetPositionMat(Mat *input_mat);

  double SoftCoreDistance(double *location, PetscInt idx);
  double SoftCoreDistance(double *location, std::vector< PetscInt > &idx_array,
                          PetscInt elec_idx);
  double SoftCoreDistance(std::vector< PetscInt > &idx_array,
                          PetscInt elec_idx_1, PetscInt elec_idx_2);
  double EuclideanDistance(double *location, PetscInt idx);
  double EuclideanDistance(double *location, std::vector< PetscInt > &idx_array,
                           PetscInt elec_idx);
  double EuclideanDistance(std::vector< PetscInt > &idx_array,
                           PetscInt elec_idx_1, PetscInt elec_idx_2);
  std::vector< PetscInt > GetIndexArray(PetscInt idx_i);

  PetscInt *GetNumX();
  PetscInt GetNumPsi();
  PetscInt GetNumPsiBuild();
  Vec *GetPsi();
  Vec *GetPsiSmall();
  void SetPsiBlock();
  double **GetXValue();
  PetscInt *GetLValues();
  PetscInt *GetMValues();
  PetscInt **GetGobblerIdx();
  PetscInt GetWriteCounterCheckpoint();
};
