#include "Hamiltonian.h"

Hamiltonian::Hamiltonian(Wavefunction& w, Pulse& pulse, HDF5Wrapper& data_file,
                         Parameters& p)
{
  if (world.rank() == 0) std::cout << "Creating Hamiltonian\n";
  num_dims              = p.GetNumDims();
  num_electrons         = p.GetNumElectrons();
  num_nuclei            = p.GetNumNuclei();
  coordinate_system_idx = p.GetCoordinateSystemIdx();
  num_x                 = w.GetNumX();
  num_psi               = w.GetNumPsi();
  num_psi_build         = w.GetNumPsiBuild();
  delta_x               = p.delta_x.get();
  delta_x_2             = p.delta_x_2.get();
  x_value               = w.GetXValue();
  z                     = p.z.get();
  location              = p.GetLocation();
  a                     = p.GetA();
  b                     = p.GetB();
  r0                    = p.r0.get();
  c0                    = p.c0.get();
  z_c                   = p.z_c.get();
  sae_size              = p.sae_size.get();
  alpha                 = p.GetAlpha();
  alpha_2               = alpha * alpha;
  field                 = pulse.GetField();
  eta                   = pi / 4.0;
  order                 = p.GetOrder();
  order_middle_idx      = order / 2;
  gobbler_idx           = w.GetGobblerIdx();

  /* call this after setting gobbler and before creating the Hamiltonian */
  SetUpCoefficients();

  /* set up time independent */
  CreateHamlitonian();

  if (world.rank() == 0) std::cout << "Hamiltonian Created\n";
}

void Hamiltonian::CreateHamlitonian()
{
  /* reserve right amount of memory to save storage */
  if (coordinate_system_idx == 1) /* Cylindrical needs 1 more for radial bc */
  {
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * (order + 3), NULL,
                 num_dims * num_electrons * (order + 3), NULL, &hamiltonian);
  }
  else
  {
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * order + 1, NULL,
                 num_dims * num_electrons * order + 1, NULL, &hamiltonian);
  }
  CalculateHamlitonian(-1, -1, true);
  // MatView(hamiltonian, PETSC_VIEWER_STDOUT_WORLD);
  // EndRun("lsdkjf");
}

void Hamiltonian::CalculateHamlitonian(PetscInt time_idx, PetscInt only_dim_idx,
                                       bool ecs)
{
  dcomp val(0.0, 0.0);  /* diagonal terms */
  PetscInt j_val;       /* j index for matrix */
  PetscInt base_offset; /* offset of diagonal */
  PetscInt offset;      /* offset of diagonal */
  PetscInt start, end;  /* start end rows */
  bool time_dependent, insert_val;
  std::vector< PetscInt > idx_array;

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
    val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val,
                 only_dim_idx, ecs);
    if (insert_val)
    {
      MatSetValues(hamiltonian, 1, &i_val, 1, &j_val, &val, INSERT_VALUES);
    }

    if (coordinate_system_idx == 1)
    {
      idx_array = GetIndexArray(i_val, j_val);
    }

    /* Loop over off diagonal elements */
    for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
    {
      for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
      {
        base_offset = GetOffset(elec_idx, dim_idx);
        if (coordinate_system_idx == 1 and dim_idx == 1 and
            idx_array[0] < order_middle_idx)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order + 1; ++diagonal_idx)
          {
            offset = (diagonal_idx + 1) * base_offset;
            /* Lower diagonal */
            if (i_val - offset >= 0 and i_val - offset < num_psi)
            {
              j_val = i_val - offset;
              val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val,
                           only_dim_idx, ecs);
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
              val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val,
                           only_dim_idx, ecs);
              if (insert_val)
              {
                MatSetValues(hamiltonian, 1, &i_val, 1, &j_val, &val,
                             INSERT_VALUES);
              }
            }
          }
        }
        else
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order_middle_idx;
               ++diagonal_idx)
          {
            offset = (diagonal_idx + 1) * base_offset;
            /* Lower diagonal */
            if (i_val - offset >= 0 and i_val - offset < num_psi)
            {
              j_val = i_val - offset;
              val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val,
                           only_dim_idx, ecs);
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
              val   = GetVal(i_val, j_val, time_dependent, time_idx, insert_val,
                           only_dim_idx, ecs);
              if (insert_val)
              {
                MatSetValues(hamiltonian, 1, &i_val, 1, &j_val, &val,
                             INSERT_VALUES);
              }
            }
          }
        }
      }
    }
  }
  MatAssemblyBegin(hamiltonian, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(hamiltonian, MAT_FINAL_ASSEMBLY);
}

void Hamiltonian::SetUpCoefficients()
{
  std::vector< dcomp > x_vals(order + 1, 0.0);
  double real_split;

  /* allocate space */
  left_ecs_coef.resize(
      num_dims, std::vector< std::vector< std::vector< dcomp > > >(
                    order, std::vector< std::vector< dcomp > >(
                               3, std::vector< dcomp >(order + 1, 0.0))));
  right_ecs_coef.resize(
      num_dims, std::vector< std::vector< std::vector< dcomp > > >(
                    order, std::vector< std::vector< dcomp > >(
                               3, std::vector< dcomp >(order + 1, 0.0))));

  real_coef.resize(num_dims, std::vector< std::vector< dcomp > >(
                                 3, std::vector< dcomp >(order + 1, 0.0)));

  for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    /* Set up real gird */
    for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
    {
      x_vals[coef_idx] = delta_x[dim_idx] * coef_idx;
    }
    /* Get real coefficients for each dimension */
    FDWeights(x_vals, 2, real_coef[dim_idx]);
    for (int discontinuity_idx = 0; discontinuity_idx < order;
         ++discontinuity_idx)
    {
      real_split = order - discontinuity_idx;
      for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
      {
        if (coef_idx < real_split)
        {
          x_vals[coef_idx] = delta_x[dim_idx] * coef_idx;
        }
        else
        {
          x_vals[coef_idx] = delta_x[dim_idx] * (real_split - 1 +
                                                 (coef_idx - real_split + 1) *
                                                     std::exp(imag * eta));
        }
      }
      FDWeights(x_vals, 2, right_ecs_coef[dim_idx][discontinuity_idx]);

      for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
      {
        if (coef_idx > discontinuity_idx)
        {
          x_vals[coef_idx] = delta_x[dim_idx] * coef_idx;
        }
        else
        {
          x_vals[coef_idx] =
              delta_x[dim_idx] *
              (discontinuity_idx + 1.0 -
               (discontinuity_idx - coef_idx + 1.0) * std::exp(imag * eta));
        }
      }
      FDWeights(x_vals, 2, left_ecs_coef[dim_idx][discontinuity_idx]);
    }
  }

  if (coordinate_system_idx == 1) /* Cylindrical boundary conditions */
  {
    radial_bc_coef.resize(order / 2,
                          std::vector< std::vector< dcomp > >(
                              3, std::vector< dcomp >(order + 1, 0.0)));
    std::vector< dcomp > x_vals_bc(order + 1, 0.0);
    /* Set up real gird for 1st and 2nd order derivatives */
    for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
    {
      if (coef_idx == 0)
        x_vals_bc[coef_idx] = 0.0;
      else
        x_vals_bc[coef_idx] = delta_x[0] * (coef_idx + 1);
    }

    for (int discontinuity_idx = 0; discontinuity_idx < order / 2;
         ++discontinuity_idx)
    {
      /* Get real coefficients for 2nd derivative (order+2 terms) */
      FDWeights(x_vals_bc, 2, radial_bc_coef[discontinuity_idx],
                discontinuity_idx + 1);
      radial_bc_coef[discontinuity_idx][1][0] +=
          radial_bc_coef[discontinuity_idx][1][1];
      radial_bc_coef[discontinuity_idx][2][0] +=
          radial_bc_coef[discontinuity_idx][2][1];
      for (int coef_idx = 1; coef_idx < order; ++coef_idx)
      {
        radial_bc_coef[discontinuity_idx][1][coef_idx] =
            radial_bc_coef[discontinuity_idx][1][coef_idx + 1];
        radial_bc_coef[discontinuity_idx][2][coef_idx] =
            radial_bc_coef[discontinuity_idx][2][coef_idx + 1];
      }
      radial_bc_coef[discontinuity_idx][1][order] = 0.0;
      radial_bc_coef[discontinuity_idx][2][order] = 0.0;
    }
    // for (int discontinuity_idx = 0; discontinuity_idx < order / 2;
    //      ++discontinuity_idx)
    // {
    //   for (int derivative_idx = 0; derivative_idx < 3; ++derivative_idx)
    //   {
    //     for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
    //     {
    //       radial_bc_coef[discontinuity_idx][derivative_idx][coef_idx] = 0.0;
    //       if ((order / 2) - discontinuity_idx + coef_idx < order + 1)
    //       {
    //         radial_bc_coef[discontinuity_idx][derivative_idx][coef_idx] +=
    //             real_coef[0][derivative_idx]
    //                      [(order / 2) - discontinuity_idx + coef_idx];
    //       }
    //       if ((order / 2) - discontinuity_idx - 1 - coef_idx >= 0)
    //       {
    //         radial_bc_coef[discontinuity_idx][derivative_idx][coef_idx] +=
    //             real_coef[0][derivative_idx]
    //                      [(order / 2) - discontinuity_idx - 1 - coef_idx];
    //       }
    //     }
    //   }
    // }
  }
}

Mat* Hamiltonian::GetTotalHamiltonian(PetscInt time_idx, PetscInt only_dim_idx,
                                      bool ecs)
{
  CalculateHamlitonian(time_idx, only_dim_idx, ecs);
  // if (only_dim_idx==-1 and time_idx==419) {
  //   MatView(hamiltonian, PETSC_VIEWER_STDOUT_WORLD);
  //   CalculateHamlitonian(time_idx, 0);
  //   MatView(hamiltonian, PETSC_VIEWER_STDOUT_WORLD);
  //   CalculateHamlitonian(time_idx, 1);
  //   MatView(hamiltonian, PETSC_VIEWER_STDOUT_WORLD);
  //   EndRun("lsdkjf");
  // }
  return &hamiltonian;
}

dcomp Hamiltonian::GetVal(PetscInt idx_i, PetscInt idx_j, bool time_dep,
                          PetscInt time_idx, bool& insert_val,
                          PetscInt only_dim_idx, bool ecs)
{
  /* Get arrays */
  std::vector< PetscInt > idx_array  = GetIndexArray(idx_i, idx_j);
  std::vector< PetscInt > diff_array = GetDiffArray(idx_array);
  PetscInt sum                       = 0;
  PetscInt non_zero_count            = 0;
  insert_val                         = true;

  /* Diagonal elements */
  if (idx_i == idx_j)
  {
    return GetDiagonal(idx_array, time_dep, time_idx, only_dim_idx, ecs);
  }

  /* Make sure there is exactly 1 non zero index so we can take care of the
   * off diagonal zeros */
  for (PetscInt i = 0; i < num_dims * num_electrons; ++i)
  {
    sum += std::abs(diff_array[i]);
    if (diff_array[i] != 0) non_zero_count++;
  }

  /* if non zero off diagonal */
  if (non_zero_count == 1)
  {
    if (sum <= order / 2)
    {
      return GetOffDiagonal(idx_array, diff_array, time_dep, time_idx,
                            only_dim_idx, ecs);
    }
    else if (coordinate_system_idx == 1 and diff_array[0] > 0 and
             sum < order - idx_array[0] and idx_array[0] < order_middle_idx)
    {
      return GetOffDiagonal(idx_array, diff_array, time_dep, time_idx,
                            only_dim_idx, ecs);
    }
  }

  /* This is a true zero of the matrix */
  insert_val = false;

  /* Should be a zero in the matrix */
  return dcomp(0.0, 0.0);
}

void Hamiltonian::FDWeights(std::vector< dcomp >& x_vals,
                            PetscInt max_derivative,
                            std::vector< std::vector< dcomp > >& coef,
                            PetscInt z_idx)
{
  /* get number of grid points given (order+1) */
  PetscInt x_size = x_vals.size();
  /* Find center */
  dcomp z;
  if (z_idx == -1)
  {
    z = x_vals[x_size / 2];
  }
  else
  {
    z = x_vals[z_idx];
  }
  /* Temporary variables */
  dcomp last_product, current_product, x_distance, z_distance,
      previous_z_distance;
  PetscInt mn;

  /* Start algorithm */
  last_product = 1.0;
  z_distance   = x_vals[0] - z; /* distance from z */
  coef[0][0]   = 1.0;
  for (PetscInt i = 0; i < x_size; ++i)
  {
    mn                  = fmin(i, max_derivative);
    current_product     = 1.0;
    previous_z_distance = z_distance;
    z_distance          = x_vals[i] - z; /* distance from z */
    for (PetscInt j = 0; j < i; ++j)
    {
      x_distance = x_vals[i] - x_vals[j];
      current_product *= x_distance;
      if (j == i - 1)
      {
        for (PetscInt k = mn; k > 0; --k)
        {
          coef[k][i] = last_product *
                       (dcomp(k, 0.0) * coef[k - 1][i - 1] -
                        previous_z_distance * coef[k][i - 1]) /
                       current_product;
        }
        coef[0][i] = -last_product * previous_z_distance * coef[0][i - 1] /
                     current_product;
      }
      for (PetscInt k = mn; k > 0; --k)
      {
        coef[k][j] =
            (z_distance * coef[k][j] - dcomp(k, 0.0) * coef[k - 1][j]) /
            x_distance;
      }
      coef[0][j] = z_distance * coef[0][j] / x_distance;
    }
    last_product = current_product;
  }
}

dcomp Hamiltonian::GetOffDiagonal(std::vector< PetscInt >& idx_array,
                                  std::vector< PetscInt >& diff_array,
                                  bool time_dep, PetscInt time_idx,
                                  PetscInt only_dim_idx, bool ecs)
{
  dcomp off_diagonal(0.0, 0.0);
  PetscInt discontinuity_idx = 0;
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (only_dim_idx == -1 or dim_idx == only_dim_idx)
      {
        if (diff_array[elec_idx * num_dims + dim_idx] != 0)
        {
          /* DONT TOUCH FIELD WITH ECS */
          if (time_dep) /* Time dependent matrix */
          {
            /* DONT TOUCH FIELD WITH ECS */
            /* Polarization vector for linear polarization */

            if (abs(diff_array[elec_idx * num_dims + dim_idx]) <
                order_middle_idx + 1)
            {
              off_diagonal -=
                  real_coef[dim_idx][1]
                           [order_middle_idx +
                            diff_array[elec_idx * num_dims + dim_idx]] *
                  dcomp(0.0, field[dim_idx][time_idx] / c);
            }
          }

          /* Time independent portion*/
          if (coordinate_system_idx == 1 and dim_idx == 0)
          {
            if (idx_array[2 * (elec_idx * num_dims + dim_idx)] <
                order_middle_idx)
            {
              discontinuity_idx =
                  idx_array[2 * (elec_idx * num_dims + dim_idx)];

              off_diagonal -=
                  radial_bc_coef[discontinuity_idx][2]
                                [discontinuity_idx +
                                 diff_array[elec_idx * num_dims + dim_idx]] /
                  2.0;
              off_diagonal -=
                  radial_bc_coef[discontinuity_idx][1]
                                [discontinuity_idx +
                                 diff_array[elec_idx * num_dims + dim_idx]] /
                  (2.0 * x_value[dim_idx][discontinuity_idx]);
            }
            /* right ECS */
            else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] >=
                         gobbler_idx[dim_idx][1] and
                     ecs)
            {
              discontinuity_idx =
                  fmin(idx_array[2 * (elec_idx * num_dims + dim_idx)] -
                           gobbler_idx[dim_idx][1],
                       order - 1);
              off_diagonal -=
                  right_ecs_coef[dim_idx][discontinuity_idx][2]
                                [order_middle_idx +
                                 diff_array[elec_idx * num_dims + dim_idx]] /
                  2.0;
              off_diagonal -=
                  right_ecs_coef[dim_idx][discontinuity_idx][1]
                                [order_middle_idx +
                                 diff_array[elec_idx * num_dims + dim_idx]] /
                  (2.0 *
                   x_value[dim_idx]
                          [idx_array[2 * (elec_idx * num_dims + dim_idx)]]);
            }
            else /* Real part */
            {
              off_diagonal -=
                  real_coef[dim_idx][2]
                           [order_middle_idx +
                            diff_array[elec_idx * num_dims + dim_idx]] /
                  2.0;
              off_diagonal -=
                  real_coef[dim_idx][1]
                           [order_middle_idx +
                            diff_array[elec_idx * num_dims + dim_idx]] /
                  (2.0 *
                   x_value[dim_idx]
                          [idx_array[2 * (elec_idx * num_dims + dim_idx)]]);
            }
          }
          else if (abs(diff_array[elec_idx * num_dims + dim_idx]) <
                   order_middle_idx + 1)
          {
            /* left ECS */
            if (idx_array[2 * (elec_idx * num_dims + dim_idx)] <=
                    gobbler_idx[dim_idx][0] and
                ecs)
            {
              discontinuity_idx =
                  fmin(gobbler_idx[dim_idx][0] -
                           idx_array[2 * (elec_idx * num_dims + dim_idx)],
                       order - 1);
              off_diagonal -=
                  left_ecs_coef[dim_idx][discontinuity_idx][2]
                               [order_middle_idx +
                                diff_array[elec_idx * num_dims + dim_idx]] /
                  2.0;
            }
            /* right ECS */
            else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] >=
                         gobbler_idx[dim_idx][1] and
                     ecs)
            {
              discontinuity_idx =
                  fmin(idx_array[2 * (elec_idx * num_dims + dim_idx)] -
                           gobbler_idx[dim_idx][1],
                       order - 1);
              off_diagonal -=
                  right_ecs_coef[dim_idx][discontinuity_idx][2]
                                [order_middle_idx +
                                 diff_array[elec_idx * num_dims + dim_idx]] /
                  2.0;
            }
            else /* real part */
            {
              off_diagonal -=
                  real_coef[dim_idx][2]
                           [order_middle_idx +
                            diff_array[elec_idx * num_dims + dim_idx]] /
                  2.0;
            }
          }
        }
      }
    }
  }
  return off_diagonal;
}

dcomp Hamiltonian::GetDiagonal(std::vector< PetscInt >& idx_array,
                               bool time_dep, PetscInt time_idx,
                               PetscInt only_dim_idx, bool ecs)
{
  dcomp diagonal(0.0, 0.0);
  /* kinetic term */
  diagonal += GetKineticTerm(idx_array, only_dim_idx, ecs);
  /* nuclei term */
  if (only_dim_idx == -1)
  {
    diagonal += GetNucleiTerm(idx_array);
    /* e-e correlation */
    diagonal += GetElectronElectronTerm(idx_array);
  }
  else
  {
    diagonal += GetNucleiTerm(idx_array) / (double)num_dims;
    /* e-e correlation */
    diagonal += GetElectronElectronTerm(idx_array) / (double)num_dims;
  }
  return diagonal;
}

dcomp Hamiltonian::GetKineticTerm(std::vector< PetscInt >& idx_array,
                                  PetscInt only_dim_idx, bool ecs)
{
  dcomp kinetic(0.0, 0.0);
  PetscInt discontinuity_idx = 0;
  /* Only num_dim terms per electron since it psi is a scalar function */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (only_dim_idx == -1 or only_dim_idx == dim_idx)
      {
        /* radial component is different */
        if (coordinate_system_idx == 1 and dim_idx == 0)
        {
          if (idx_array[2 * (elec_idx * num_dims + dim_idx)] < order_middle_idx)
          {
            discontinuity_idx = idx_array[2 * (elec_idx * num_dims + dim_idx)];

            kinetic -=
                radial_bc_coef[discontinuity_idx][2][discontinuity_idx] / 2.0;
            kinetic -= radial_bc_coef[discontinuity_idx][1][discontinuity_idx] /
                       (2.0 * x_value[dim_idx][discontinuity_idx]);
            // std::cout << discontinuity_idx << " "
            //           << " "
            //           <<
            // radial_bc_coef[discontinuity_idx][1][discontinuity_idx]
            //           << " "
            //           <<
            // radial_bc_coef[discontinuity_idx][2][discontinuity_idx]
            //           << " " << x_value[dim_idx][discontinuity_idx] << " "
            //           << x_value[dim_idx + 1][discontinuity_idx] << "\n";
          }
          /* right ECS */
          else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] >=
                       gobbler_idx[dim_idx][1] and
                   ecs)
          {
            discontinuity_idx =
                fmin(idx_array[2 * (elec_idx * num_dims + dim_idx)] -
                         gobbler_idx[dim_idx][1],
                     order - 1);
            kinetic -= right_ecs_coef[dim_idx][discontinuity_idx][2]
                                     [order_middle_idx] /
                       2.0;
            kinetic -=
                right_ecs_coef[dim_idx][discontinuity_idx][1]
                              [order_middle_idx] /
                (2.0 * x_value[dim_idx]
                              [idx_array[2 * (elec_idx * num_dims + dim_idx)]]);
          }
          else /* Real part */
          {
            kinetic -= real_coef[dim_idx][2][order_middle_idx] / 2.0;
            kinetic -=
                real_coef[dim_idx][1][order_middle_idx] /
                (2.0 * x_value[dim_idx]
                              [idx_array[2 * (elec_idx * num_dims + dim_idx)]]);
          }
        }
        else
        {
          /* left ECS */
          if (idx_array[2 * (elec_idx * num_dims + dim_idx)] <=
                  gobbler_idx[dim_idx][0] and
              ecs)
          {
            discontinuity_idx =
                fmin(gobbler_idx[dim_idx][0] -
                         idx_array[2 * (elec_idx * num_dims + dim_idx)],
                     order - 1);
            kinetic -=
                left_ecs_coef[dim_idx][discontinuity_idx][2][order_middle_idx] /
                2.0;
          }
          /* right ECS */
          else if (idx_array[2 * (elec_idx * num_dims + dim_idx)] >=
                       gobbler_idx[dim_idx][1] and
                   ecs)
          {
            discontinuity_idx =
                fmin(idx_array[2 * (elec_idx * num_dims + dim_idx)] -
                         gobbler_idx[dim_idx][1],
                     order - 1);
            kinetic -= right_ecs_coef[dim_idx][discontinuity_idx][2]
                                     [order_middle_idx] /
                       2.0;
          }
          else /* Real part */
          {
            kinetic -= real_coef[dim_idx][2][order_middle_idx] / 2.0;
          }
        }
      }
    }
  }
  return kinetic;
}

dcomp Hamiltonian::GetNucleiTerm(std::vector< PetscInt >& idx_array)
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

dcomp Hamiltonian::GetElectronElectronTerm(std::vector< PetscInt >& idx_array)
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
                                     std::vector< PetscInt >& idx_array,
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

double Hamiltonian::SoftCoreDistance(std::vector< PetscInt >& idx_array,
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
std::vector< PetscInt > Hamiltonian::GetIndexArray(PetscInt idx_i,
                                                   PetscInt idx_j)
{
  PetscInt total_dims = num_electrons * num_dims;
  /* size of each dim */
  std::vector< PetscInt > num(total_dims);
  /* idx for return */
  std::vector< PetscInt > idx_array(total_dims * 2);
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
   * where x_1 is the first dimension of the first electron and x_2 is the
   * first dimension of the second electron and so on */
  for (PetscInt i = total_dims - 1; i >= 0; --i)
  {
    idx_array[2 * i] = idx_i % num[i];
    idx_i /= num[i];
    idx_array[2 * i + 1] = idx_j % num[i];
    idx_j /= num[i];
  }
  return idx_array;
}

std::vector< PetscInt > Hamiltonian::GetDiffArray(
    std::vector< PetscInt >& idx_array)
{
  std::vector< PetscInt > diff_array(num_dims * num_electrons);
  /* Calculated difference between i and j indexes */
  for (PetscInt i = 0; i < num_dims * num_electrons; ++i)
  {
    diff_array[i] = idx_array[2 * i + 1] - idx_array[2 * i];
  }
  return diff_array;
}

Mat* Hamiltonian::GetTimeIndependent(PetscInt only_dim_idx, bool ecs)
{
  CalculateHamlitonian(-1, only_dim_idx, ecs);
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
