#include "Hamiltonian.h"

Hamiltonian::Hamiltonian(Wavefunction& w, Pulse& pulse, HDF5Wrapper& data_file,
                         Parameters& p)
{
  if (world.rank() == 0) std::cout << "Creating Hamiltonian\n";
  num_dims      = p.GetNumDims();
  num_electrons = p.GetNumElectrons();
  num_nuclei    = p.GetNumNuclei();
  num_x         = w.GetNumX();
  num_psi       = w.GetNumPsi();
  num_psi_build = w.GetNumPsiBuild();
  delta_x       = p.delta_x.get();
  delta_x_2     = p.delta_x_2.get();
  x_value       = w.GetXValue();
  z             = p.z.get();
  location      = p.GetLocation();
  alpha         = p.GetAlpha();
  alpha_2       = alpha * alpha;
  a_field       = pulse.GetAField();

  /* set up time independent */
  CreateTimeIndependent();
  CreateTimeDependent();
  CreateTotalHamlitonian();

  if (world.rank() == 0) std::cout << "Hamiltonian Created\n";
}

void Hamiltonian::CreateTimeIndependent()
{
  double dx2 = delta_x[0] * delta_x[0];        /* dx squared */
  dcomp off_diagonal(-1.0 / (2.0 * dx2), 0.0); /* off diagonal terms */
  dcomp diagonal(0.0, 0.0);                    /* diagonal terms */
  dcomp val(0.0, 0.0);                         /* diagonal terms */
  int i_val;                                   /* i index for matrix */
  int j_val;                                   /* j index for matrix */
  int counter;
  /* reserve right amount of memory to save storage */
  MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
               5, NULL, 5, NULL, &time_independent);
  if (world.rank() == 0)
  {
    for (int i = 0; i < num_psi; i++)
    {
      i_val = i;
      // counter = 0;
      // for (int j = 0; j < num_psi; j++)
      // {
      //   j_val = j;
      //   val   = GetVal(i_val, j_val, false);
      //   if (val != dcomp(0.0, 0.0))
      //   {
      //     MatSetValues(time_independent, 1, &i_val, 1, &j_val, &val,
      //                  INSERT_VALUES);
      //     counter++;
      //   }
      // }
      // std::cout << i_val << " of " << num_psi << " with counter: " << counter
      //           << "\n";
      if (i - num_x[0] >= 0 and i - num_x[0] < num_psi)
      {
        j_val        = i - num_x[0];
        off_diagonal = GetVal(i_val, j_val, false);
        if (off_diagonal != dcomp(0.0, 0.0))
          MatSetValues(time_independent, 1, &i_val, 1, &j_val, &off_diagonal,
                       INSERT_VALUES);
      }
      if (i - 1 >= 0 and i - 1 < num_psi)
      {
        j_val        = i - 1;
        off_diagonal = GetVal(i_val, j_val, false);
        if (off_diagonal != dcomp(0.0, 0.0))
          MatSetValues(time_independent, 1, &i_val, 1, &j_val, &off_diagonal,
                       INSERT_VALUES);
      }
      if (i >= 0 and i < num_psi)
      {
        j_val = i;

        diagonal = GetVal(i_val, j_val, false);
        if (diagonal != dcomp(0.0, 0.0))
          MatSetValues(time_independent, 1, &i_val, 1, &j_val, &diagonal,
                       INSERT_VALUES);
      }
      if (i + 1 >= 0 and i + 1 < num_psi)
      {
        j_val        = i + 1;
        off_diagonal = GetVal(i_val, j_val, false);
        if (off_diagonal != dcomp(0.0, 0.0))
          MatSetValues(time_independent, 1, &i_val, 1, &j_val, &off_diagonal,
                       INSERT_VALUES);
      }
      if (i + num_x[0] >= 0 and i + num_x[0] < num_psi)
      {
        j_val        = i + num_x[0];
        off_diagonal = GetVal(i_val, j_val, false);
        if (off_diagonal != dcomp(0.0, 0.0))
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
  dcomp off_diagonal(0.0, 1.0 / (2.0 * delta_x[0] * c));
  dcomp neg_off_diagonal = -1.0 * off_diagonal;
  dcomp val(0.0, 0.0); /* diagonal terms */
  int i_val;           /* i index for matrix */
  int j_val;           /* j index for matrix */

  MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
               4, NULL, 4, NULL, &time_dependent);

  if (world.rank() == 0)
  {
    for (int i = 0; i < num_psi; i++)
    {
      i_val = i;
      if (i - num_x[0] >= 0 and i - num_x[0] < num_psi)
      {
        j_val = i - num_x[0];
        val   = GetVal(i_val, j_val, true);
        if (val != dcomp(0.0, 0.0))
          MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
      }
      if (i - 1 >= 0 and i - 1 < num_psi)
      {
        j_val = i - 1;
        val   = GetVal(i_val, j_val, true);
        if (val != dcomp(0.0, 0.0))
          MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
      }
      if (i + 1 >= 0 and i + 1 < num_psi)
      {
        j_val = i + 1;
        val   = GetVal(i_val, j_val, true);
        if (val != dcomp(0.0, 0.0))
          MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
      }
      if (i + num_x[0] >= 0 and i + num_x[0] < num_psi)
      {
        j_val = i + num_x[0];
        val   = GetVal(i_val, j_val, true);
        if (val != dcomp(0.0, 0.0))
          MatSetValues(time_dependent, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
      }
    }
  }
  MatAssemblyBegin(time_dependent, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(time_dependent, MAT_FINAL_ASSEMBLY);
}

/* just allocate the total Hamiltonian */
void Hamiltonian::CreateTotalHamlitonian()
{
  MatDuplicate(time_independent, MAT_DO_NOT_COPY_VALUES, &total_hamlitonian);
}

Mat* Hamiltonian::GetTotalHamiltonian(int time_idx)
{
  MatCopy(time_independent, total_hamlitonian, SAME_NONZERO_PATTERN);
  MatAXPY(total_hamlitonian, a_field[time_idx], time_dependent,
          SUBSET_NONZERO_PATTERN);
  // MatView(total_hamlitonian, PETSC_VIEWER_STDOUT_SELF);
  return &total_hamlitonian;
}

dcomp Hamiltonian::GetVal(int idx_i, int idx_j, bool time_dep)
{
  std::vector<int> idx_array  = GetIndexArray(idx_i, idx_j);
  std::vector<int> diff_array = GetDiffArray(idx_array);
  bool one_off_diagonal       = false; /* if this is an off diagonal value*/
  bool not_off_diagonal       = false; /* if this is an off diagonal value*/

  if (idx_i == idx_j)
  {
    return GetDiagonal(idx_array, time_dep);
  }

  for (int i = 0; i < num_dims * num_electrons; ++i)
  {
    if (diff_array[i] == -1 || diff_array[i] == 1)
    {
      if (one_off_diagonal)
      {
        not_off_diagonal = true;
      }
      one_off_diagonal = true;
    }
    else if (diff_array[i] != 0)
    {
      not_off_diagonal = true;
    }
  }

  if (one_off_diagonal and !not_off_diagonal)
  {
    return GetOffDiagonal(idx_array, diff_array, time_dep);
  }

  return dcomp(0.0, 0.0);
}

dcomp Hamiltonian::GetOffDiagonal(std::vector<int>& idx_array,
                                  std::vector<int>& diff_array, bool time_dep)
{
  dcomp off_diagonal(0.0, 0.0);
  for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (time_dep)
      {
        /* Upper diagonals */
        if (diff_array[elec_idx * num_dims + dim_idx] == 1)
        {
          off_diagonal += dcomp(0.0, 1.0 / (2.0 * delta_x[dim_idx] * c));
        }
        /* Lower diagonals */
        else if (diff_array[elec_idx * num_dims + dim_idx] == -1)
        {
          off_diagonal += dcomp(0.0, -1.0 / (2.0 * delta_x[dim_idx] * c));
        }
      }
      else
      {
        if (std::abs(diff_array[elec_idx * num_dims + dim_idx]) == 1)
        {
          off_diagonal += dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
      }
    }
  }
  return off_diagonal;
}

dcomp Hamiltonian::GetDiagonal(std::vector<int>& idx_array, bool time_dep)
{
  dcomp diagonal(0.0, 0.0);
  double diff; /* distance between x_1 and x_2 */
  int idx_1;   /* index for psi_1 */
  int idx_2;   /* index for psi_2 */
  idx_1 = idx_array[0];
  idx_2 = idx_array[2];
  diff  = std::abs(x_value[0][idx_1] - x_value[0][idx_2]);

  /* kinetic term */
  diagonal += GetKineticTerm();
  /* nuclei term */
  diagonal += GetNucleiTerm(idx_array);
  /* e-e correlation */
  diagonal += GetElectronElectronTerm(idx_array);
  return diagonal;
}

dcomp Hamiltonian::GetKineticTerm()
{
  dcomp kinetic(0.0, 0.0);
  /* Only num_dim terms per electron since it psi is a scalar function */
  for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    kinetic += dcomp(1.0 / delta_x_2[dim_idx], 0.0);
  }
  /* Same terms for each electron */
  kinetic *= num_electrons;
  return kinetic;
}

dcomp Hamiltonian::GetNucleiTerm(std::vector<int>& idx_array)
{
  dcomp nuclei(0.0, 0.0);
  /* loop over each electron */
  for (int elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    /* loop over each nuclei */
    for (int nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      nuclei -= dcomp(z[nuclei_idx] / SoftCoreDistance(location[nuclei_idx],
                                                       idx_array, elec_idx),
                      0.0);
    }
  }
  return nuclei;
}

dcomp Hamiltonian::GetElectronElectronTerm(std::vector<int>& idx_array)
{
  dcomp ee_val(0.0, 0.0);
  for (int elec_idx_1 = 0; elec_idx_1 < num_electrons; ++elec_idx_1)
  {
    for (int elec_idx_2 = elec_idx_1 + 1; elec_idx_2 < num_electrons;
         ++elec_idx_2)
    {
      ee_val +=
          dcomp(1 / SoftCoreDistance(idx_array, elec_idx_1, elec_idx_2), 0.0);
    }
  }
  return ee_val;
}

double Hamiltonian::SoftCoreDistance(double* location,
                                     std::vector<int>& idx_array, int elec_idx)
{
  double distance = alpha_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] - x_value[dim_idx][idx_array[2 * elec_idx]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Hamiltonian::SoftCoreDistance(std::vector<int>& idx_array,
                                     int elec_idx_1, int elec_idx_2)
{
  double distance = alpha_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = x_value[dim_idx][idx_array[2 * elec_idx_1]] -
           x_value[dim_idx][idx_array[2 * elec_idx_2]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

/* Returns the an array of alternating i,j components of the local matrix */
std::vector<int> Hamiltonian::GetIndexArray(int idx_i, int idx_j)
{
  int total_dims = num_electrons * num_dims;
  /* size of each dim */
  std::vector<int> num(total_dims);
  /* idx for return */
  std::vector<int> idx_array(total_dims * 2);
  for (int elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (int dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      num[elec_idx * num_dims + dim_idx] = num_x[dim_idx];
    }
  }
  for (int i = total_dims - 1; i >= 0; --i)
  {
    idx_array[2 * i] = idx_i % num[i];
    idx_i /= num[i];
    idx_array[2 * i + 1] = idx_j % num[i];
    idx_j /= num[i];
  }
  return idx_array;
}

std::vector<int> Hamiltonian::GetDiffArray(std::vector<int>& idx_array)
{
  std::vector<int> diff_array(num_dims * num_electrons);
  for (int i = 0; i < num_dims * num_electrons; ++i)
  {
    diff_array[i] = idx_array[2 * i + 1] - idx_array[2 * i];
  }
  return diff_array;
}

Mat* Hamiltonian::GetTimeIndependent() { return &time_independent; }

Hamiltonian::~Hamiltonian()
{
  if (world.rank() == 0) std::cout << "Deleting Hamiltonian\n";
  MatDestroy(&time_independent);
  MatDestroy(&time_dependent);
  MatDestroy(&total_hamlitonian);
}