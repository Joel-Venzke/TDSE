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
  a             = p.GetA();
  b             = p.GetB();
  r0            = p.r0.get();
  c0            = p.c0.get();
  z_c           = p.z_c.get();
  sae_size      = p.sae_size.get();
  alpha         = p.GetAlpha();
  alpha_2       = alpha * alpha;
  field         = pulse.GetField();
  eta           = pi / 4.0;

  gobbler_idx = new PetscInt*[num_dims];
  for (PetscInt i = 0; i < num_dims; ++i)
  {
    gobbler_idx[i] = new PetscInt[2];
    gobbler_idx[i][0] =
        (num_x[i] - PetscInt(num_x[i] * p.GetGobbler())) / 2 - 1;
    gobbler_idx[i][1] = num_x[i] - 1 - gobbler_idx[i][0];
  }

  /* set up time independent */
  CreateHamlitonian();

  if (world.rank() == 0) std::cout << "Hamiltonian Created\n";
}

void Hamiltonian::CreateHamlitonian()
{
  /* reserve right amount of memory to save storage */
  MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
               num_dims * num_electrons * 2 + 1, NULL,
               num_dims * num_electrons * 2 + 1, NULL, &hamiltonian);
  CalculateHamlitonian(-1);
}

void Hamiltonian::CalculateHamlitonian(PetscInt time_idx)
{
  dcomp val(0.0, 0.0); /* diagonal terms */
  PetscInt j_val;      /* j index for matrix */
  PetscInt offset;     /* offset of diagonal */
  PetscInt start, end; /* start end rows */
  bool time_dependent, insert_val;
  MatGetOwnershipRange(hamiltonian, &start, &end);
  if (time_idx < 0)
  {
    time_dependent = false;
  }
  else
  {
    time_dependent = true;
  }
  for (PetscInt i_val = start; i_val < end; i_val++)
  {
    /* Diagonal element */
    j_val = i_val;
    val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val);
    if (insert_val)
    {
      MatSetValues(hamiltonian, 1, &i_val, 1, &j_val, &val, INSERT_VALUES);
    }

    /* Loop over off diagonal elements */
    for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
    {
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        offset = GetOffset(elec_idx, dim_idx);

        /* Lower diagonal */
        if (i_val - offset >= 0 and i_val - offset < num_psi)
        {
          j_val = i_val - offset;
          val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val);
          if (insert_val)
          {
            MatSetValues(hamiltonian, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }

        /* Upper diagonal */
        if (i_val + offset >= 0 and i_val + offset < num_psi)
        {
          j_val = i_val + offset;
          val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val);
          if (insert_val)
          {
            MatSetValues(hamiltonian, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }
      }
    }
  }
  MatAssemblyBegin(hamiltonian, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(hamiltonian, MAT_FINAL_ASSEMBLY);
}

Mat* Hamiltonian::GetTotalHamiltonian(PetscInt time_idx)
{
  CalculateHamlitonian(time_idx);
  return &hamiltonian;
}

dcomp Hamiltonian::GetVal(PetscInt idx_i, PetscInt idx_j, bool time_dep,
                          PetscInt time_idx, bool& insert_val)
{
  /* Get arrays */
  std::vector<PetscInt> idx_array  = GetIndexArray(idx_i, idx_j);
  std::vector<PetscInt> diff_array = GetDiffArray(idx_array);
  PetscInt sum                     = 0;
  insert_val                       = true;

  /* Diagonal elements */
  if (idx_i == idx_j)
  {
    return GetDiagonal(idx_array, time_dep, time_idx);
  }

  /* Make sure there is exactly 1 non zero index so we can take care of the off
   * diagonal zeros */
  for (PetscInt i = 0; i < num_dims * num_electrons; ++i)
  {
    sum += std::abs(diff_array[i]);
  }

  /* if non zero off diagonal */
  if (sum == 1)
  {
    return GetOffDiagonal(idx_array, diff_array, time_dep, time_idx);
  }

  /* This is a true zero of the matrix */
  insert_val = false;

  /* Should be a zero in the matrix */
  return dcomp(0.0, 0.0);
}

dcomp Hamiltonian::GetOffDiagonal(std::vector<PetscInt>& idx_array,
                                  std::vector<PetscInt>& diff_array,
                                  bool time_dep, PetscInt time_idx)
{
  dcomp off_diagonal(0.0, 0.0);
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      /* DONT TOUCH FIELD WITH ECS */
      if (time_dep) /* Time dependent matrix */
      {
        /* Upper diagonals */
        /* DONT TOUCH FIELD WITH ECS */
        if (diff_array[elec_idx * num_dims + dim_idx] == 1)
        {
          /* Polarization vector for linear polarization */
          off_diagonal += field[dim_idx][time_idx] *
                          dcomp(0.0, 1.0 / (2.0 * delta_x[dim_idx] * c));
        }
        /* Lower diagonals */
        /* DONT TOUCH FIELD WITH ECS */
        else if (diff_array[elec_idx * num_dims + dim_idx] == -1)
        {
          /* Polarization vector for linear polarization */
          off_diagonal += field[dim_idx][time_idx] *
                          dcomp(0.0, -1.0 / (2.0 * delta_x[dim_idx] * c));
        }
      }

      /* Time independent portion*/
      /* upper diagonal */
      if (diff_array[elec_idx * num_dims + dim_idx] == 1)
      {
        if (idx_array[2 * (elec_idx * num_dims + dim_idx)] <
                gobbler_idx[dim_idx][0] ||
            idx_array[2 * (elec_idx * num_dims + dim_idx)] >
                gobbler_idx[dim_idx][1])
        {
          /* Imaginary part of ECS */
          off_diagonal += std::exp(-2.0 * imag * eta) *
                          dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
        else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] ==
                 gobbler_idx[dim_idx][0])
        {
          /* left discontinuity part of ECS */
          off_diagonal += (2.0 / (1.0 + std::exp(imag * eta))) *
                          dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
        else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] ==
                 gobbler_idx[dim_idx][1])
        {
          /* right discontinuity part of ECS */
          off_diagonal += (2.0 * std::exp(-1.0 * imag * eta) /
                           (1.0 + std::exp(imag * eta))) *
                          dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
        else
        {
          off_diagonal += dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
      }
      else if (diff_array[elec_idx * num_dims + dim_idx] == -1)
      {
        if (idx_array[2 * (elec_idx * num_dims + dim_idx)] <
                gobbler_idx[dim_idx][0] ||
            idx_array[2 * (elec_idx * num_dims + dim_idx)] >
                gobbler_idx[dim_idx][1])
        {
          /* Imaginary part of ECS */
          off_diagonal += std::exp(-2.0 * imag * eta) *
                          dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
        else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] ==
                 gobbler_idx[dim_idx][0])
        {
          /* left discontinuity part of ECS */
          off_diagonal += (2.0 * std::exp(-1.0 * imag * eta) /
                           (1.0 + std::exp(imag * eta))) *
                          dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
        else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] ==
                 gobbler_idx[dim_idx][1])
        {
          /* right discontinuity part of ECS */
          off_diagonal += (2.0 / (1.0 + std::exp(imag * eta))) *
                          dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
        else
        {
          off_diagonal += dcomp(-1.0 / (2.0 * delta_x_2[dim_idx]), 0.0);
        }
      }
    }
  }
  return off_diagonal;
}

dcomp Hamiltonian::GetDiagonal(std::vector<PetscInt>& idx_array, bool time_dep,
                               PetscInt time_idx)
{
  dcomp diagonal(0.0, 0.0);
  /* kinetic term */
  diagonal += GetKineticTerm(idx_array);
  /* nuclei term */
  diagonal += GetNucleiTerm(idx_array);
  /* e-e correlation */
  diagonal += GetElectronElectronTerm(idx_array);
  return diagonal;
}

dcomp Hamiltonian::GetKineticTerm(std::vector<PetscInt>& idx_array)
{
  dcomp kinetic(0.0, 0.0);
  /* Only num_dim terms per electron since it psi is a scalar function */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (idx_array[2 * (elec_idx * num_dims + dim_idx)] <
              gobbler_idx[dim_idx][0] ||
          idx_array[2 * (elec_idx * num_dims + dim_idx)] >
              gobbler_idx[dim_idx][1])
      {
        /* Imaginary part of ECS */
        kinetic +=
            std::exp(-2.0 * imag * eta) * dcomp(1.0 / delta_x_2[dim_idx], 0.0);
      }
      else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] ==
                   gobbler_idx[dim_idx][0] ||
               idx_array[2 * (elec_idx * num_dims + dim_idx)] ==
                   gobbler_idx[dim_idx][1])
      {
        /* discontinuities of ECS */
        kinetic +=
            std::exp(-1.0 * imag * eta) * dcomp(1.0 / delta_x_2[dim_idx], 0.0);
      }
      else
      {
        kinetic += dcomp(1.0 / delta_x_2[dim_idx], 0.0);
      }
    }
  }
  return kinetic;
}

dcomp Hamiltonian::GetNucleiTerm(std::vector<PetscInt>& idx_array)
{
  dcomp nuclei(0.0, 0.0);
  double r;
  /* loop over each electron */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    /* loop over each nuclei */
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (z[nuclei_idx] != 0.0) /* Column term */
      {
        nuclei -= dcomp(z[nuclei_idx] / SoftCoreDistance(location[nuclei_idx],
                                                         idx_array, elec_idx),
                        0.0);
      }
      else /* SAE */
      {
        r = SoftCoreDistance(location[nuclei_idx], idx_array, elec_idx);
        nuclei -= dcomp(c0[nuclei_idx] / r, 0.0);
        nuclei -= dcomp(z_c[nuclei_idx] * exp(-r0[nuclei_idx] * r) / r, 0.0);
        for (PetscInt i = 0; i < sae_size[nuclei_idx]; ++i)
        {
          nuclei -= dcomp(a[nuclei_idx][i] * exp(-b[nuclei_idx][i] * r), 0.0);
        }
      }
    }
  }
  return nuclei;
}

dcomp Hamiltonian::GetElectronElectronTerm(std::vector<PetscInt>& idx_array)
{
  dcomp ee_val(0.0, 0.0);
  /* loop over each correlation
   * (e_1 with e_2, e_1 with e_3, ... e_2 with e_3, ... ect.) */
  for (PetscInt elec_idx_1 = 0; elec_idx_1 < num_electrons; ++elec_idx_1)
  {
    /* make sure to not double count or calculate self terms */
    for (PetscInt elec_idx_2 = elec_idx_1 + 1; elec_idx_2 < num_electrons;
         ++elec_idx_2)
    {
      /* 1/r like pot */
      ee_val +=
          dcomp(1 / SoftCoreDistance(idx_array, elec_idx_1, elec_idx_2), 0.0);
    }
  }
  return ee_val;
}

double Hamiltonian::SoftCoreDistance(double* location,
                                     std::vector<PetscInt>& idx_array,
                                     PetscInt elec_idx)
{
  double distance = alpha_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] -
           x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx * num_dims)]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Hamiltonian::SoftCoreDistance(std::vector<PetscInt>& idx_array,
                                     PetscInt elec_idx_1, PetscInt elec_idx_2)
{
  double distance = alpha_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = x_value[dim_idx][idx_array[2 * elec_idx_1]] -
           x_value[dim_idx][idx_array[2 * elec_idx_2]];
    distance += diff * diff;
  }
  return sqrt(distance);
}

PetscInt Hamiltonian::GetOffset(PetscInt elec_idx, PetscInt dim_idx)
{
  PetscInt offset = 1;
  /* offset gets a factor of
   * num_psi_build = num_x[num_dims-1] * num_x[num_dims-2] * ... * num_x[0]
   * for each electron */
  if (elec_idx > 0)
  {
    for (PetscInt iter = 0; iter < elec_idx; ++iter)
    {
      offset *= num_psi_build;
    }
  }
  if (dim_idx > 0)
  {
    for (PetscInt iter = 0; iter < dim_idx; ++iter)
    {
      /* first offset is num_x[num_dims-1] and then next is
       * num_x[num_dims-1]*num_x[num_dims-2] and so on*/
      offset *= num_x[num_dims - 1 - iter];
    }
  }
  return offset;
}

/* Returns the an array of alternating i,j components of the local matrix */
std::vector<PetscInt> Hamiltonian::GetIndexArray(PetscInt idx_i, PetscInt idx_j)
{
  PetscInt total_dims = num_electrons * num_dims;
  /* size of each dim */
  std::vector<PetscInt> num(total_dims);
  /* idx for return */
  std::vector<PetscInt> idx_array(total_dims * 2);
  /* used for convenience. Could/should be optimized */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; elec_idx++)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
    {
      num[elec_idx * num_dims + dim_idx] = num_x[dim_idx];
    }
  }
  /* loop over dimensions backwards so psi becomes
   * psi[x_1,y_1,z_1,x_2,y_2,z_2,...]
   * where x_1 is the first dimension of the first electron and x_2 is the first
   * dimension of the second electron and so on */
  for (PetscInt i = total_dims - 1; i >= 0; --i)
  {
    idx_array[2 * i] = idx_i % num[i];
    idx_i /= num[i];
    idx_array[2 * i + 1] = idx_j % num[i];
    idx_j /= num[i];
  }
  return idx_array;
}

std::vector<PetscInt> Hamiltonian::GetDiffArray(
    std::vector<PetscInt>& idx_array)
{
  std::vector<PetscInt> diff_array(num_dims * num_electrons);
  /* Calculated difference between i and j indexes */
  for (PetscInt i = 0; i < num_dims * num_electrons; ++i)
  {
    diff_array[i] = idx_array[2 * i + 1] - idx_array[2 * i];
  }
  return diff_array;
}

Mat* Hamiltonian::GetTimeIndependent()
{
  CalculateHamlitonian(-1);
  return &hamiltonian;
}

Hamiltonian::~Hamiltonian()
{
  if (world.rank() == 0) std::cout << "Deleting Hamiltonian\n";
  MatDestroy(&hamiltonian);

  for (PetscInt i = 0; i < num_dims; ++i)
  {
    delete gobbler_idx[i];
  }
  delete[] gobbler_idx;
}
