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
  gauge_idx             = p.GetGaugeIdx();
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
  ee_soft_core          = p.GetEESoftCore();
  ee_soft_core_2        = ee_soft_core * ee_soft_core;
  field                 = pulse.GetField();

  if (coordinate_system_idx != 2)
  {
    num_psi_build     = w.GetNumPsiBuild();
    delta_x_min       = p.delta_x_min.get();
    delta_x_min_end   = p.delta_x_min_end.get();
    delta_x_max       = p.delta_x_max.get();
    delta_x_max_start = p.delta_x_max_start.get();
    eta               = pi / 4.0;
    order             = p.GetOrder();
    order_middle_idx  = order / 2;
    gobbler_idx       = w.GetGobblerIdx();

    /* call this after setting gobbler and before creating the Hamiltonian */
    SetUpCoefficients();
  }
  /* set up time independent */
  CreateHamlitonian();

  if (world.rank() == 0) std::cout << "Hamiltonian Created\n";
}

void Hamiltonian::CreateHamlitonian()
{
  /* reserve right amount of memory to save storage */
  if (coordinate_system_idx == 2) /* RBF */
  {
    /* load node set hdf5 file*/
    HDF5Wrapper node_set("nodes.h5", "r");

    /* Find out how many nodes we need*/
    PetscInt stencil_size = node_set.GetLast("/parameters/stencil_size");
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 stencil_size, NULL, stencil_size, NULL, &hamiltonian);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 stencil_size, NULL, stencil_size, NULL, &hamiltonian_0);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 stencil_size, NULL, stencil_size, NULL, &hamiltonian_0_ecs);

    hamiltonian_laser = new Mat[num_dims];
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi,
                   num_psi, stencil_size, NULL, stencil_size, NULL,
                   &(hamiltonian_laser[dim_idx]));
    }
  }
  else if (coordinate_system_idx ==
           1) /* Cylindrical needs 1 more for radial bc */
  {
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * (order + 3), NULL,
                 num_dims * num_electrons * (order + 3), NULL, &hamiltonian);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * (order + 3), NULL,
                 num_dims * num_electrons * (order + 3), NULL, &hamiltonian_0);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * (order + 3), NULL,
                 num_dims * num_electrons * (order + 3), NULL,
                 &hamiltonian_0_ecs);

    hamiltonian_laser = new Mat[num_dims];
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi,
                   num_psi, num_dims * num_electrons * (order + 3), NULL,
                   num_dims * num_electrons * (order + 3), NULL,
                   &(hamiltonian_laser[dim_idx]));
    }
  }
  else
  {
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * order + 1, NULL,
                 num_dims * num_electrons * order + 1, NULL, &hamiltonian);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * order + 1, NULL,
                 num_dims * num_electrons * order + 1, NULL, &hamiltonian_0);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_dims * num_electrons * order + 1, NULL,
                 num_dims * num_electrons * order + 1, NULL,
                 &hamiltonian_0_ecs);

    hamiltonian_laser = new Mat[num_dims];
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi,
                   num_psi, num_dims * num_electrons * (order + 3), NULL,
                   num_dims * num_electrons * (order + 3), NULL,
                   &(hamiltonian_laser[dim_idx]));
    }
  }
  GenerateHamlitonian();
  // MatView(hamiltonian_0, PETSC_VIEWER_STDOUT_WORLD);
  // MatView(hamiltonian_0_ecs, PETSC_VIEWER_STDOUT_WORLD);
  // MatView(hamiltonian_laser[0], PETSC_VIEWER_STDOUT_WORLD);
  // MatView(hamiltonian_laser[1], PETSC_VIEWER_STDOUT_WORLD);
  // MatView(hamiltonian_laser[2], PETSC_VIEWER_STDOUT_WORLD);
  // EndRun("");
}

void Hamiltonian::GenerateHamlitonian()
{
  CalculateHamlitonian0();
  CalculateHamlitonian0ECS();
  CalculateHamlitonianLaser();
  MatCopy(hamiltonian_0, hamiltonian, DIFFERENT_NONZERO_PATTERN);
}

void Hamiltonian::CalculateHamlitonian0()
{
  dcomp val(0.0, 0.0); /* diagonal terms */
  PetscInt start, end; /* start end rows */

  MatGetOwnershipRange(hamiltonian_0, &start, &end);
  if (coordinate_system_idx == 2) /* RBF */
  {
    HDF5Wrapper node_set("nodes.h5", "r");
    PetscInt num_operators = node_set.GetLast("/parameters/num_operators");
    PetscInt* row_idx =
        node_set.GetFirstNInt("/operators/row_idx", num_operators);
    PetscInt* col_idx =
        node_set.GetFirstNInt("/operators/col_idx", num_operators);
    double* laplace = node_set.GetFirstN("/operators/laplace", num_operators);

    for (PetscInt idx = 0; idx < num_operators; idx++)
    {
      if (row_idx[idx] >= start and row_idx[idx] < end)
      {
        val = -1.0 * laplace[idx] / 2.0;
        if (row_idx[idx] == col_idx[idx])
        {
          val += GetNucleiTerm(row_idx[idx]);
        }
        MatSetValues(hamiltonian_0, 1, &row_idx[idx], 1, &col_idx[idx], &val,
                     INSERT_VALUES);
      }
    }
    delete row_idx;
    delete col_idx;
    delete laplace;
  }
  else
  {
    PetscInt j_val;       /* j index for matrix */
    PetscInt base_offset; /* offset of diagonal */
    PetscInt offset;      /* offset of diagonal */
    bool insert_val, ecs;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);

    ecs = false;
    for (PetscInt i_val = start; i_val < end; i_val++)
    {
      j_val     = i_val;
      idx_array = GetIndexArray(i_val, j_val);
      for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
      {
        /* avoid recalculating if the grid is uniform */
        if (delta_x_max[dim_idx] != delta_x_min[dim_idx])
        {
          /* Set up real gird */
          for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
          {
            if (idx_array[dim_idx * 2] < (order / 2 + 1) or
                num_x[dim_idx] - 1 - idx_array[dim_idx * 2] < (order / 2 + 1))
            {
              x_vals[coef_idx] = delta_x_max[dim_idx] * coef_idx;
            }
            else
            {
              x_vals[coef_idx] = x_value[dim_idx][coef_idx - order / 2 +
                                                  idx_array[dim_idx * 2]];
            }
          }
          /* Get real coefficients for each dimension */
          FDWeights(x_vals, 2, real_coef[dim_idx]);
        }
      }

      /* Diagonal element */
      val = GetVal(i_val, j_val, insert_val, ecs);
      if (insert_val)
      {
        MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val, INSERT_VALUES);
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
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                               INSERT_VALUES);
                }
              }

              /* Upper diagonal */
              if (i_val + offset >= 0 and i_val + offset < num_psi)
              {
                j_val = i_val + offset;
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
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
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                               INSERT_VALUES);
                }
              }

              /* Upper diagonal */
              if (i_val + offset >= 0 and i_val + offset < num_psi)
              {
                j_val = i_val + offset;
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                               INSERT_VALUES);
                }
              }
            }
          }
        }
      }
    }
  }
  MatAssemblyBegin(hamiltonian_0, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(hamiltonian_0, MAT_FINAL_ASSEMBLY);
}

void Hamiltonian::CalculateHamlitonian0ECS()
{
  dcomp val(0.0, 0.0); /* diagonal terms */
  PetscInt start, end; /* start end rows */

  MatGetOwnershipRange(hamiltonian_0, &start, &end);
  if (coordinate_system_idx == 2) /* RBF */
  {
    HDF5Wrapper node_set("nodes.h5", "r");
    PetscInt num_operators = node_set.GetLast("/parameters/num_operators");
    PetscInt* row_idx =
        node_set.GetFirstNInt("/operators/row_idx", num_operators);
    PetscInt* col_idx =
        node_set.GetFirstNInt("/operators/col_idx", num_operators);
    double* hyperviscosity =
        node_set.GetFirstN("/operators/hyperviscosity", num_operators);
    double* laplace_real =
        node_set.GetFirstN("/operators/laplace_ecs_real", num_operators);
    double* laplace_imag =
        node_set.GetFirstN("/operators/laplace_ecs_imag", num_operators);

    for (PetscInt idx = 0; idx < num_operators; idx++)
    {
      if (row_idx[idx] >= start and row_idx[idx] < end)
      {
        val = -1.0 * dcomp(laplace_real[idx], laplace_imag[idx]) / 2.0 -
              5e-3 * dcomp(0.0, hyperviscosity[idx]);
        if (row_idx[idx] == col_idx[idx])
        {
          val += GetNucleiTerm(row_idx[idx]);
        }
        MatSetValues(hamiltonian_0_ecs, 1, &row_idx[idx], 1, &col_idx[idx],
                     &val, INSERT_VALUES);
      }
    }
    delete row_idx;
    delete col_idx;
    delete hyperviscosity;
    delete laplace_real;
    delete laplace_imag;
  }
  else
  {
    PetscInt j_val;       /* j index for matrix */
    PetscInt base_offset; /* offset of diagonal */
    PetscInt offset;      /* offset of diagonal */
    bool insert_val, ecs;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);
    ecs = true;
    for (PetscInt i_val = start; i_val < end; i_val++)
    {
      j_val     = i_val;
      idx_array = GetIndexArray(i_val, j_val);
      for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
      {
        /* avoid recalculating if the grid is uniform */
        if (delta_x_max[dim_idx] != delta_x_min[dim_idx])
        {
          /* Set up real gird */
          for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
          {
            if (idx_array[dim_idx * 2] < (order / 2 + 1) or
                num_x[dim_idx] - 1 - idx_array[dim_idx * 2] < (order / 2 + 1))
            {
              x_vals[coef_idx] = delta_x_max[dim_idx] * coef_idx;
            }
            else
            {
              x_vals[coef_idx] = x_value[dim_idx][coef_idx - order / 2 +
                                                  idx_array[dim_idx * 2]];
            }
          }
          /* Get real coefficients for each dimension */
          FDWeights(x_vals, 2, real_coef[dim_idx]);
        }
      }

      /* Diagonal element */
      val = GetVal(i_val, j_val, insert_val, ecs);
      if (insert_val)
      {
        MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                     INSERT_VALUES);
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
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                               INSERT_VALUES);
                }
              }

              /* Upper diagonal */
              if (i_val + offset >= 0 and i_val + offset < num_psi)
              {
                j_val = i_val + offset;
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
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
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                               INSERT_VALUES);
                }
              }

              /* Upper diagonal */
              if (i_val + offset >= 0 and i_val + offset < num_psi)
              {
                j_val = i_val + offset;
                val   = GetVal(i_val, j_val, insert_val, ecs);
                if (insert_val)
                {
                  MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                               INSERT_VALUES);
                }
              }
            }
          }
        }
      }
    }
  }
  MatAssemblyBegin(hamiltonian_0_ecs, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(hamiltonian_0_ecs, MAT_FINAL_ASSEMBLY);
}

void Hamiltonian::CalculateHamlitonianLaser()
{
  dcomp val(0.0, 0.0);            /* diagonal terms */
  PetscInt start, end;            /* start end rows */
  if (coordinate_system_idx == 2) /* RBF */
  {
    HDF5Wrapper node_set("nodes.h5", "r");
    PetscInt num_operators = node_set.GetLast("/parameters/num_operators");
    PetscInt* row_idx =
        node_set.GetFirstNInt("/operators/row_idx", num_operators);
    PetscInt* col_idx =
        node_set.GetFirstNInt("/operators/col_idx", num_operators);

    for (PetscInt ham_dim_idx = 0; ham_dim_idx < num_dims; ham_dim_idx++)
    {
      MatGetOwnershipRange(hamiltonian_laser[ham_dim_idx], &start, &end);
      if (gauge_idx == 0) /* velocity gauge */
      {
        double* derivatives;

        if (ham_dim_idx == 0)
        {
          derivatives = node_set.GetFirstN("/operators/dx", num_operators);
        }
        else if (ham_dim_idx == 1)
        {
          derivatives = node_set.GetFirstN("/operators/dy", num_operators);
        }
        else if (ham_dim_idx == 2)
        {
          derivatives = node_set.GetFirstN("/operators/dz", num_operators);
        }
        else
        {
          EndRun("operator dimension does not exist for RBF.");
        }

        for (PetscInt idx = 0; idx < num_operators; idx++)
        {
          if (row_idx[idx] >= start and row_idx[idx] < end)
          {
            val = derivatives[idx];
            MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &row_idx[idx], 1,
                         &col_idx[idx], &val, INSERT_VALUES);
          }
        }
        delete derivatives;
      }
      else if (gauge_idx == 1) /* length gauge */
      {
        for (PetscInt idx = 0; idx < num_psi; idx++)
        {
          val = x_value[ham_dim_idx][idx];
          MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &idx, 1, &idx, &val,
                       INSERT_VALUES);
        }
      }
      else
      {
        EndRun("Bad gauge index");
      }
      MatAssemblyBegin(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
    }
    delete row_idx;
    delete col_idx;
  }
  else
  {
    PetscInt j_val;       /* j index for matrix */
    PetscInt base_offset; /* offset of diagonal */
    PetscInt offset;      /* offset of diagonal */
    bool insert_val;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);
    for (PetscInt ham_dim_idx = 0; ham_dim_idx < num_dims; ham_dim_idx++)
    {
      MatGetOwnershipRange(hamiltonian_laser[ham_dim_idx], &start, &end);
      for (PetscInt i_val = start; i_val < end; i_val++)
      {
        j_val     = i_val;
        idx_array = GetIndexArray(i_val, j_val);
        for (PetscInt dim_idx = 0; dim_idx < num_dims; dim_idx++)
        {
          /* avoid recalculating if the grid is uniform */
          if (delta_x_max[dim_idx] != delta_x_min[dim_idx])
          {
            /* Set up real gird */
            for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
            {
              if (idx_array[dim_idx * 2] < (order / 2 + 1) or
                  num_x[dim_idx] - 1 - idx_array[dim_idx * 2] < (order / 2 + 1))
              {
                x_vals[coef_idx] = delta_x_max[dim_idx] * coef_idx;
              }
              else
              {
                x_vals[coef_idx] = x_value[dim_idx][coef_idx - order / 2 +
                                                    idx_array[dim_idx * 2]];
              }
            }
            /* Get real coefficients for each dimension */
            FDWeights(x_vals, 2, real_coef[dim_idx]);
          }
        }

        /* Diagonal element */
        val = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
        if (insert_val)
        {
          MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1, &j_val,
                       &val, INSERT_VALUES);
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
              for (int diagonal_idx = 0; diagonal_idx < order + 1;
                   ++diagonal_idx)
              {
                offset = (diagonal_idx + 1) * base_offset;
                /* Lower diagonal */
                if (i_val - offset >= 0 and i_val - offset < num_psi)
                {
                  j_val = i_val - offset;
                  val   = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
                  if (insert_val)
                  {
                    MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                                 &j_val, &val, INSERT_VALUES);
                  }
                }

                /* Upper diagonal */
                if (i_val + offset >= 0 and i_val + offset < num_psi)
                {
                  j_val = i_val + offset;
                  val   = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
                  if (insert_val)
                  {
                    MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                                 &j_val, &val, INSERT_VALUES);
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
                  val   = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
                  if (insert_val)
                  {
                    MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                                 &j_val, &val, INSERT_VALUES);
                  }
                }

                /* Upper diagonal */
                if (i_val + offset >= 0 and i_val + offset < num_psi)
                {
                  j_val = i_val + offset;
                  val   = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
                  if (insert_val)
                  {
                    MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                                 &j_val, &val, INSERT_VALUES);
                  }
                }
              }
            }
          }
        }
      }

      MatAssemblyBegin(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
    }
  }
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
      x_vals[coef_idx] = delta_x_min[dim_idx] * coef_idx;
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
          x_vals[coef_idx] = delta_x_max[dim_idx] * coef_idx;
        }
        else
        {
          x_vals[coef_idx] =
              delta_x_max[dim_idx] *
              (real_split - 1 +
               (coef_idx - real_split + 1) * std::exp(imag * eta));
        }
      }
      FDWeights(x_vals, 2, right_ecs_coef[dim_idx][discontinuity_idx]);

      for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
      {
        if (coef_idx > discontinuity_idx)
        {
          x_vals[coef_idx] = delta_x_max[dim_idx] * coef_idx;
        }
        else
        {
          x_vals[coef_idx] =
              delta_x_max[dim_idx] *
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
      if (coef_idx == 0 and order > 2)
        x_vals_bc[coef_idx] = 0.0;
      else
        x_vals_bc[coef_idx] = delta_x_min[0] * (coef_idx + 1);
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
  }
}

Mat* Hamiltonian::GetTotalHamiltonian(PetscInt time_idx, bool ecs)
{
  if (ecs)
  {
    MatCopy(hamiltonian_0_ecs, hamiltonian, SAME_NONZERO_PATTERN);
  }
  else
  {
    MatCopy(hamiltonian_0, hamiltonian, SAME_NONZERO_PATTERN);
  }
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    if (field[dim_idx][time_idx] != 0.0)
    {
      MatAXPY(hamiltonian, field[dim_idx][time_idx], hamiltonian_laser[dim_idx],
              SUBSET_NONZERO_PATTERN);
    }
  }
  return &hamiltonian;
}

dcomp Hamiltonian::GetVal(PetscInt idx_i, PetscInt idx_j, bool& insert_val,
                          bool ecs)
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
    return GetDiagonal(idx_array, ecs);
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
    /* normal off diagonal */
    if (sum <= order / 2)
    {
      return GetOffDiagonal(idx_array, diff_array, ecs);
    }
    /* Cylindrical boundary condition */
    else if (coordinate_system_idx == 1 and diff_array[0] > 0 and
             sum < order - idx_array[0] and idx_array[0] < order_middle_idx)
    {
      return GetOffDiagonal(idx_array, diff_array, ecs);
    }
  }

  /* This is a true zero of the matrix */
  insert_val = false;

  /* Should be a zero in the matrix */
  return dcomp(0.0, 0.0);
}

dcomp Hamiltonian::GetValLaser(PetscInt idx_i, PetscInt idx_j, bool& insert_val,
                               PetscInt only_dim_idx)
{
  /* Get arrays */
  std::vector< PetscInt > idx_array  = GetIndexArray(idx_i, idx_j);
  std::vector< PetscInt > diff_array = GetDiffArray(idx_array);
  PetscInt sum                       = 0;
  PetscInt non_zero_count            = 0;
  insert_val                         = true;

  /* Diagonal elements */
  if (idx_i == idx_j and gauge_idx == 1) /* length gauge */
  {
    return GetDiagonalLaser(idx_array, only_dim_idx);
  }

  if (gauge_idx == 0) /* velocity gauge */
  {
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
      /* normal off diagonal */
      if (sum <= order / 2)
      {
        return GetOffDiagonalLaser(idx_array, diff_array, only_dim_idx);
      }
      /* Cylindrical boundary condition */
      else if (coordinate_system_idx == 1 and diff_array[0] > 0 and
               sum < order - idx_array[0] and idx_array[0] < order_middle_idx)
      {
        return GetOffDiagonalLaser(idx_array, diff_array, only_dim_idx);
      }
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

  for (int derivative_idx = 0; derivative_idx < max_derivative + 1;
       ++derivative_idx)
  {
    for (int i = 0; i < x_size; ++i)
    {
      coef[derivative_idx][i] = 0.0;
    }
  }

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
                                  std::vector< PetscInt >& diff_array, bool ecs)
{
  bool time_dep     = false;
  PetscInt time_idx = -1;
  dcomp off_diagonal(0.0, 0.0);
  PetscInt discontinuity_idx = 0;
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (diff_array[elec_idx * num_dims + dim_idx] != 0)
      {
        /* DONT TOUCH FIELD WITH ECS */
        if (time_dep and gauge_idx == 0) /* Time dependent matrix */
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
          if (idx_array[2 * (elec_idx * num_dims + dim_idx)] < order_middle_idx)
          {
            discontinuity_idx = idx_array[2 * (elec_idx * num_dims + dim_idx)];

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
                (2.0 * x_value[dim_idx]
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
                (2.0 * x_value[dim_idx]
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
  return off_diagonal;
}

dcomp Hamiltonian::GetOffDiagonalLaser(std::vector< PetscInt >& idx_array,
                                       std::vector< PetscInt >& diff_array,
                                       PetscInt only_dim_idx)
{
  dcomp off_diagonal(0.0, 0.0);
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (only_dim_idx == -1 or dim_idx == only_dim_idx)
      {
        if (diff_array[elec_idx * num_dims + dim_idx] != 0)
        {
          /* DONT TOUCH FIELD WITH ECS */
          if (gauge_idx == 0) /* Time dependent matrix */
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
                  dcomp(0.0, 1.0 / c);
            }
          }
        }
      }
    }
  }
  return off_diagonal;
}

dcomp Hamiltonian::GetDiagonal(std::vector< PetscInt >& idx_array, bool ecs)
{
  dcomp diagonal(0.0, 0.0);
  /* kinetic term */
  diagonal += GetKineticTerm(idx_array, ecs);
  /* nuclei term */
  diagonal += GetNucleiTerm(idx_array);
  /* e-e correlation */
  diagonal += GetElectronElectronTerm(idx_array);
  return diagonal;
}

dcomp Hamiltonian::GetDiagonalLaser(std::vector< PetscInt >& idx_array,
                                    PetscInt only_dim_idx)
{
  dcomp diagonal(0.0, 0.0);

  diagonal += GetLengthGauge(idx_array, only_dim_idx);

  return diagonal;
}

dcomp Hamiltonian::GetKineticTerm(std::vector< PetscInt >& idx_array, bool ecs)
{
  dcomp kinetic(0.0, 0.0);
  PetscInt discontinuity_idx = 0;
  /* Only num_dim terms per electron since it psi is a scalar function */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
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
          //           radial_bc_coef[discontinuity_idx][1][discontinuity_idx]
          //           << " "
          //           <<
          //           radial_bc_coef[discontinuity_idx][2][discontinuity_idx]
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
          kinetic -=
              right_ecs_coef[dim_idx][discontinuity_idx][2][order_middle_idx] /
              2.0;
          kinetic -=
              right_ecs_coef[dim_idx][discontinuity_idx][1][order_middle_idx] /
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
          kinetic -=
              right_ecs_coef[dim_idx][discontinuity_idx][2][order_middle_idx] /
              2.0;
        }
        else /* Real part */
        {
          kinetic -= real_coef[dim_idx][2][order_middle_idx] / 2.0;
        }
      }
    }
  }
  return kinetic;
}

dcomp Hamiltonian::GetLengthGauge(std::vector< PetscInt >& idx_array,
                                  PetscInt only_dim_idx)
{
  dcomp length_gauge(0.0, 0.0);
  /* Only num_dim terms per electron since it psi is a scalar function */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (only_dim_idx == -1 or only_dim_idx == dim_idx)
      {
        /* E dot x */
        length_gauge +=
            x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx * num_dims)]];
      }
    }
  }
  return length_gauge;
}

dcomp Hamiltonian::GetNucleiTerm(std::vector< PetscInt >& idx_array)
{
  dcomp nuclei(0.0, 0.0);
  double r_soft;
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
        r_soft = SoftCoreDistance(location[nuclei_idx], idx_array, elec_idx);
        r      = EuclideanDistance(location[nuclei_idx], idx_array, elec_idx);
        nuclei -= dcomp(c0[nuclei_idx] / r_soft, 0.0);
        nuclei -=
            dcomp(z_c[nuclei_idx] * exp(-r0[nuclei_idx] * r) / r_soft, 0.0);

        // Tong Lin He Only
        // clean up hack later
        // nuclei -= dcomp(-0.231 * exp(-0.480 * r) / r, 0.0);

        for (PetscInt i = 0; i < sae_size[nuclei_idx]; ++i)
        {
          nuclei -= dcomp(a[nuclei_idx][i] * exp(-b[nuclei_idx][i] * r), 0.0);
        }
      }
    }
  }
  return nuclei;
}

/* get nuclear term for rbf grid */
dcomp Hamiltonian::GetNucleiTerm(PetscInt idx)
{
  dcomp nuclei(0.0, 0.0);
  double r_soft;
  double r;
  /* loop over each electron */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    /* loop over each nuclei */
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (z[nuclei_idx] != 0.0) /* Column term */
      {
        nuclei -= dcomp(
            z[nuclei_idx] / SoftCoreDistance(location[nuclei_idx], idx), 0.0);
        // std::cout << "soft core" << SoftCoreDistance(location[nuclei_idx],
        // idx)
        //           << "\n";
      }
      else /* SAE */
      {
        r_soft = SoftCoreDistance(location[nuclei_idx], idx);
        r      = EuclideanDistance(location[nuclei_idx], idx);
        nuclei -= dcomp(c0[nuclei_idx] / r_soft, 0.0);
        nuclei -=
            dcomp(z_c[nuclei_idx] * exp(-r0[nuclei_idx] * r) / r_soft, 0.0);

        // Tong Lin He Only
        // clean up hack later
        // nuclei -= dcomp(-0.231 * exp(-0.480 * r) / r, 0.0);

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

double Hamiltonian::SoftCoreDistance(double* location, PetscInt idx)
{
  double distance = alpha_2; /* Make sure we include the soft core */
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] - x_value[dim_idx][idx];
    distance += diff * diff;
  }
  return sqrt(distance);
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
  double distance = ee_soft_core_2; /* Make sure we include the soft core */
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

double Hamiltonian::EuclideanDistance(double* location, PetscInt idx)
{
  double distance = 0.0;
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = location[dim_idx] - x_value[dim_idx][idx];
    distance += diff * diff;
  }
  return sqrt(distance);
}

double Hamiltonian::EuclideanDistance(double* location,
                                      std::vector< PetscInt >& idx_array,
                                      PetscInt elec_idx)
{
  double distance = 0.0;
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

double Hamiltonian::EuclideanDistance(std::vector< PetscInt >& idx_array,
                                      PetscInt elec_idx_1, PetscInt elec_idx_2)
{
  double distance = 0.0;
  double diff     = 0.0;
  /* loop over all dims */
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    diff = x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_1 * num_dims)]] -
           x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_2 * num_dims)]];
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

Mat* Hamiltonian::GetTimeIndependent(bool ecs)
{
  if (ecs)
  {
    MatCopy(hamiltonian_0_ecs, hamiltonian, SAME_NONZERO_PATTERN);
  }
  else
  {
    MatCopy(hamiltonian_0, hamiltonian, SAME_NONZERO_PATTERN);
  }
  return &hamiltonian;
}

Hamiltonian::~Hamiltonian()
{
  if (world.rank() == 0) std::cout << "Deleting Hamiltonian\n";
  MatDestroy(&hamiltonian);
  MatDestroy(&hamiltonian_0);
  MatDestroy(&hamiltonian_0_ecs);
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    MatDestroy(&(hamiltonian_laser[dim_idx]));
  }
  delete hamiltonian_laser;

  for (PetscInt i = 0; i < num_dims; ++i)
  {
    delete gobbler_idx[i];
  }
  delete[] gobbler_idx;
}
