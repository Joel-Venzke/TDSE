#include "Hamiltonian.h"

#define dcomp std::complex<double>

Hamiltonian::Hamiltonian(Wavefunction& w, Pulse& pulse, HDF5Wrapper& data_file,
                         Parameters& p)
{
  if (world.rank() == 0) std::cout << "Creating Hamiltonian\n";
  num_dims   = p.GetNumDims();
  num_x      = w.GetNumX();
  num_psi    = w.GetNumPsi();
  num_psi_12 = w.GetNumPsi12();
  delta_x    = w.GetDeltaX();
  x_value    = w.GetXValue();
  z          = p.GetZ();
  alpha      = p.GetAlpha();
  a_field    = pulse.GetAField();

  /* set up time independent */
  CreateTimeIndependent();
  CreateTimeDependent();
  CreateTotalHamlitonian();

  if (world.rank() == 0) std::cout << "Hamiltonian Created\n";
}

void Hamiltonian::CreateTimeIndependent()
{
  double dx2 = delta_x[0] * delta_x[0]; /* dx squared */
  double diff;                          /* distance between x_1 and x_2 */
  dcomp off_diagonal(-1.0 / (2.0 * dx2), 0.0); /* off diagonal terms */
  dcomp diagonal(0.0, 0.0);                    /* diagonal terms */
  int idx_1;                                   /* index for psi_1 */
  int idx_2;                                   /* index for psi_2 */
  int i_val;                                   /* i index for matrix */
  int j_val;                                   /* j index for matrix */
  /* reserve right amount of memory to save storage */
  MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
               5, NULL, 5, NULL, &time_independent);
  if (world.rank() == 0)
  {
    for (int i = 0; i < num_psi; i++)
    {
      j_val = i;
      if (i - num_x[0] >= 0 and i - num_x[0] < num_psi)
      {
        i_val = i - num_x[0];
        MatSetValues(time_independent, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
      if (i - 1 >= 0 and i - 1 < num_psi)
      {
        i_val = i - 1;
        MatSetValues(time_independent, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
      if (i >= 0 and i < num_psi)
      {
        idx_1 = i / num_psi_12;
        idx_2 = i % num_psi_12;
        diff  = std::abs(x_value[0][idx_1] - x_value[0][idx_2]);

        /* kinetic term */
        diagonal = dcomp(2.0 / dx2, 0.0);
        /* nuclei electron 1 */
        diagonal -=
            dcomp(z / sqrt(x_value[0][idx_1] * x_value[0][idx_1] + alpha), 0.0);
        /* nuclei electron 2 */
        diagonal -=
            dcomp(z / sqrt(x_value[0][idx_2] * x_value[0][idx_2] + alpha), 0.0);
        /* e-e correlation */
        diagonal += dcomp(1 / sqrt(diff * diff + alpha), 0.0);

        i_val = i;
        MatSetValues(time_independent, 1, &i_val, 1, &j_val, &diagonal,
                     INSERT_VALUES);
      }
      if (i + 1 >= 0 and i + 1 < num_psi)
      {
        i_val = i + 1;
        MatSetValues(time_independent, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
      if (i + num_x[0] >= 0 and i + num_x[0] < num_psi)
      {
        i_val = i + num_x[0];
        MatSetValues(time_independent, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(time_independent, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(time_independent, MAT_FINAL_ASSEMBLY);
}

void Hamiltonian::CreateTimeDependent()
{
  double c = 1 / 7.2973525664e-3;
  dcomp off_diagonal(0.0, 1.0 / (2.0 * delta_x[0] * c));
  dcomp neg_off_diagonal = -1.0 * off_diagonal;
  int i_val; /* i index for matrix */
  int j_val; /* j index for matrix */

  MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
               4, NULL, 4, NULL, &time_dependent);

  if (world.rank() == 0)
  {
    for (int i = 0; i < num_psi; i++)
    {
      j_val = i;
      if (i - num_x[0] >= 0 and i - num_x[0] < num_psi)
      {
        i_val = i - num_x[0];
        MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &neg_off_diagonal,
                     INSERT_VALUES);
      }
      if (i - 1 >= 0 and i - 1 < num_psi)
      {
        i_val = i - 1;
        MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &neg_off_diagonal,
                     INSERT_VALUES);
      }
      if (i + 1 >= 0 and i + 1 < num_psi)
      {
        i_val = i + 1;
        MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
      if (i + num_x[0] >= 0 and i + num_x[0] < num_psi)
      {
        i_val = i + num_x[0];
        MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(time_dependent, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(time_dependent, MAT_FINAL_ASSEMBLY);
}

/* just fill the non zero with random values so the memory is allocated */
void Hamiltonian::CreateTotalHamlitonian()
{
  dcomp off_diagonal(-1.0, 0.0);
  dcomp diagonal(2.0, 0.0);
  int i_val; /* i index for matrix */
  int j_val; /* j index for matrix */

  MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
               5, NULL, 5, NULL, &total_hamlitonian);

  if (world.rank() == 0)
  {
    for (int i = 0; i < num_psi; i++)
    {
      j_val = i;
      if (i - num_x[0] >= 0 and i - num_x[0] < num_psi)
      {
        i_val = i - num_x[0];
        MatSetValues(total_hamlitonian, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
      if (i - 1 >= 0 and i - 1 < num_psi)
      {
        i_val = i - 1;
        MatSetValues(total_hamlitonian, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
      if (i >= 0 and i < num_psi)
      {
        i_val = i;
        MatSetValues(total_hamlitonian, 1, &i_val, 1, &j_val, &diagonal,
                     INSERT_VALUES);
      }
      if (i + 1 >= 0 and i + 1 < num_psi)
      {
        i_val = i + 1;
        MatSetValues(total_hamlitonian, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
      if (i + num_x[0] >= 0 and i + num_x[0] < num_psi)
      {
        i_val = i + num_x[0];
        MatSetValues(total_hamlitonian, 1, &i_val, 1, &j_val, &off_diagonal,
                     INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(total_hamlitonian, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(total_hamlitonian, MAT_FINAL_ASSEMBLY);
}

Mat* Hamiltonian::GetTotalHamiltonian(int time_idx)
{
  MatCopy(time_dependent, total_hamlitonian, DIFFERENT_NONZERO_PATTERN);
  MatAYPX(total_hamlitonian, a_field[time_idx], time_independent,
          DIFFERENT_NONZERO_PATTERN);
  // MatView(total_hamlitonian, PETSC_VIEWER_STDOUT_SELF);
  return &total_hamlitonian;
}

Mat* Hamiltonian::GetTimeIndependent() { return &time_independent; }

Hamiltonian::~Hamiltonian()
{
  if (world.rank() == 0) std::cout << "Deleting Hamiltonian\n";
  MatDestroy(&time_independent);
  MatDestroy(&time_dependent);
  MatDestroy(&total_hamlitonian);
}