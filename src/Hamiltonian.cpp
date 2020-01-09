#include "Hamiltonian.h"

Hamiltonian::Hamiltonian(Wavefunction& w, Pulse& pulse, HDF5Wrapper& data_file,
                         Parameters& p)
{
  if (world.rank() == 0) std::cout << "Creating Hamiltonian\n";
  wavefunction           = &w;
  num_dims               = p.GetNumDims();
  num_electrons          = p.GetNumElectrons();
  num_nuclei             = p.GetNumNuclei();
  coordinate_system_idx  = p.GetCoordinateSystemIdx();
  num_x                  = w.GetNumX();
  num_psi                = w.GetNumPsi();
  gauge_idx              = p.GetGaugeIdx();
  x_value                = w.GetXValue();
  z                      = p.z.get();
  location               = p.GetLocation();
  exponential_r_0        = p.GetExponentialR0();
  exponential_amplitude  = p.GetExponentialAmplitude();
  exponential_decay_rate = p.GetExponentialDecayRate();
  exponential_size       = p.exponential_size.get();
  gaussian_r_0           = p.GetGaussianR0();
  gaussian_amplitude     = p.GetGaussianAmplitude();
  gaussian_decay_rate    = p.GetGaussianDecayRate();
  gaussian_size          = p.gaussian_size.get();
  square_well_r_0        = p.GetSquareWellR0();
  square_well_amplitude  = p.GetSquareWellAmplitude();
  square_well_width      = p.GetSquareWellWidth();
  square_well_size       = p.square_well_size.get();
  yukawa_r_0             = p.GetYukawaR0();
  yukawa_amplitude       = p.GetYukawaAmplitude();
  yukawa_decay_rate      = p.GetYukawaDecayRate();
  yukawa_size            = p.yukawa_size.get();
  alpha                  = p.GetAlpha();
  alpha_2                = alpha * alpha;
  ee_soft_core           = p.GetEESoftCore();
  ee_soft_core_2         = ee_soft_core * ee_soft_core;
  field                  = pulse.GetField();

  PetscLogEventRegister("H_0", PETSC_VIEWER_CLASSID, &create_h_0);
  PetscLogEventRegister("H_0Diag", PETSC_VIEWER_CLASSID, &create_h_0_diag);
  PetscLogEventRegister("H_0FD", PETSC_VIEWER_CLASSID, &create_h_0_fd);
  PetscLogEventRegister("H_0Hyper", PETSC_VIEWER_CLASSID, &create_h_0_hyper);
  PetscLogEventRegister("H_0HyperLow", PETSC_VIEWER_CLASSID,
                        &create_h_0_hyper_low);
  PetscLogEventRegister("H_0HyperLowVal", PETSC_VIEWER_CLASSID,
                        &create_h_0_hyper_low_get_val);
  PetscLogEventRegister("H_0HyperHigh", PETSC_VIEWER_CLASSID,
                        &create_h_0_hyper_upper);
  PetscLogEventRegister("H_0_ecs", PETSC_VIEWER_CLASSID, &create_h_0_ecs);
  PetscLogEventRegister("H_laser", PETSC_VIEWER_CLASSID, &create_h_laser);
  PetscLogEventRegister("IdxArray", PETSC_VIEWER_CLASSID, &idx_array_time);
  PetscLogEventRegister("DiffArray", PETSC_VIEWER_CLASSID, &diff_array_time);
  PetscLogEventRegister("HyperPotential", PETSC_VIEWER_CLASSID,
                        &hyper_pot_time);
  PetscLogEventRegister("HyperCoulomb", PETSC_VIEWER_CLASSID,
                        &hyper_coulomb_time);
  PetscLogEventRegister("HyperLaser", PETSC_VIEWER_CLASSID, &hyper_laser_time);
  PetscLogEventRegister("BarrierH0", PETSC_VIEWER_CLASSID, &build_H_0);
  PetscLogEventRegister("BarrierH0ECS", PETSC_VIEWER_CLASSID, &build_H_0_ecs);

  if (coordinate_system_idx == 3)
  {
    l_max          = p.GetLMax();
    m_max          = p.GetMMax();
    l_values       = w.GetLValues();
    m_values       = w.GetMValues();
    k_max          = 0;
    eigen_values   = NULL;
    l_block_size   = NULL;
    max_block_size = 0;
  }
  else if (coordinate_system_idx == 4)
  {
    l_max          = p.GetLMax();
    m_max          = p.GetMMax();
    k_max          = p.GetKMax();
    eigen_values   = w.GetEigenValues();
    l_block_size   = w.GetLBlockSize();
    max_block_size = w.GetMaxBlockSize();
    l_values       = NULL;
    m_values       = NULL;
    num_ang        = 1e4;

    /* pre allocate arrays for integrals over hyper radius */
    angle.resize(num_ang);
    arg_vals.resize(num_ang);
    sphere_1.resize(num_ang);
    sphere_2.resize(num_ang);
  }
  else
  {
    l_max          = 0;
    m_max          = 0;
    k_max          = 0;
    l_values       = NULL;
    m_values       = NULL;
    eigen_values   = NULL;
    l_block_size   = NULL;
    max_block_size = 0;
  }

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
  else if (coordinate_system_idx == 3)
  {
    /* size is 3 off diagonals in l and m and one main diagonal for r each of
     * width order + 1 in velocity gauge and smaller in length gauge. This gives
     * an upper bound of num_electrons * 7 * (order + 1) non zero elements */
    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_electrons * 7 * (order + 1), NULL,
                 num_electrons * 7 * (order + 1), NULL, &hamiltonian);
    MatSetOption(hamiltonian, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_x[2],
                 num_x[2], num_dims * num_electrons * order + 1, NULL,
                 num_dims * num_electrons * order + 1, NULL, &hamiltonian_0);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 num_electrons * 7 * (order + 1), NULL,
                 num_electrons * 7 * (order + 1), NULL, &hamiltonian_0_ecs);

    hamiltonian_laser = new Mat[num_dims];
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi,
                   num_psi, num_electrons * 7 * (order + 1), NULL,
                   num_electrons * 7 * (order + 1), NULL,
                   &(hamiltonian_laser[dim_idx]));
    }
  }
  else if (coordinate_system_idx == 4)
  {
    /* nonzero for any sphere harm in that L block and radial part */
    // MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi,
    // num_psi,
    //              (order + 1) + max_block_size + 2, NULL,
    //              (order + 1) + max_block_size + 2, NULL, &hamiltonian);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 (order + 1) + max_block_size + 2, NULL,
                 (order + 1) + max_block_size + 2, NULL, &hamiltonian_0);

    MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi, num_psi,
                 (order + 1) + max_block_size + 2, NULL,
                 (order + 1) + max_block_size + 2, NULL, &hamiltonian_0_ecs);

    hamiltonian_laser = new Mat[num_dims];
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, num_psi,
                   num_psi, (order + 1) + max_block_size + 2, NULL,
                   (order + 1) + max_block_size + 2, NULL,
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
  PetscLogEventBegin(create_h_0, 0, 0, 0, 0);
  CalculateHamlitonian0();
  PetscLogEventEnd(create_h_0, 0, 0, 0, 0);
  PetscLogEventBegin(create_h_0_ecs, 0, 0, 0, 0);
  CalculateHamlitonian0ECS();
  PetscLogEventEnd(create_h_0_ecs, 0, 0, 0, 0);
  PetscLogEventBegin(create_h_laser, 0, 0, 0, 0);
  CalculateHamlitonianLaser();
  PetscLogEventEnd(create_h_laser, 0, 0, 0, 0);
  // MatAssemblyBegin(hamiltonian, MAT_FINAL_ASSEMBLY);
  // MatAssemblyEnd(hamiltonian, MAT_FINAL_ASSEMBLY);
  // MatCopy(hamiltonian_0_ecs, hamiltonian, DIFFERENT_NONZERO_PATTERN);
  // MatCopy(hamiltonian_0_ecs, hamiltonian, DIFFERENT_NONZERO_PATTERN);
  MatConvert(hamiltonian_0_ecs, MATSAME, MAT_INITIAL_MATRIX, &hamiltonian);
}

void Hamiltonian::CalculateHamlitonian0(PetscInt l_val)
{
  dcomp val(0.0, 0.0); /* diagonal terms */
  PetscInt start, end; /* start end rows */
  current_l_val = l_val;

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
  else if (coordinate_system_idx == 3) /* Spherical */
  {
    PetscInt j_val;       /* j index for matrix */
    PetscInt base_offset; /* offset of diagonal */
    PetscInt offset;      /* offset of diagonal */
    bool insert_val, ecs;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);
    PetscInt dim_idx;

    ecs = false;
    for (PetscInt i_val = start; i_val < end; i_val++)
    {
      j_val     = i_val;
      idx_array = GetIndexArray(i_val, j_val);
      dim_idx   = 0;
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
            x_vals[coef_idx] =
                x_value[dim_idx][coef_idx - order / 2 + idx_array[dim_idx * 2]];
          }
        }
        /* Get real coefficients for each dimension */
        FDWeights(x_vals, 2, real_coef[dim_idx]);
      }

      /* Diagonal element */
      val = GetVal(i_val, j_val, insert_val, ecs, l_val);
      if (insert_val)
      {
        MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val, INSERT_VALUES);
      }

      /* Loop over off diagonal elements */
      for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
      {
        dim_idx     = 0;
        base_offset = GetOffset(elec_idx, dim_idx);
        if (dim_idx == 0 and
            idx_array[2 * (2 + elec_idx * num_dims)] < order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order; ++diagonal_idx)
          {
            offset = (diagonal_idx + 1) * base_offset;
            /* Lower diagonal */
            if (i_val - offset >= 0 and i_val - offset < num_psi)
            {
              j_val = i_val - offset;
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
              if (insert_val)
              {
                MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                             INSERT_VALUES);
              }
            }
          }
        }
        else if (dim_idx == 0 and
                 num_x[2] - 1 - idx_array[2 * (2 + elec_idx * num_dims)] <
                     order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order - 1; ++diagonal_idx)
          {
            offset = (diagonal_idx + 1) * base_offset;
            /* Lower diagonal */
            if (i_val - offset >= 0 and i_val - offset < num_psi)
            {
              j_val = i_val - offset;
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
  else if (coordinate_system_idx == 4) /* Hyperpherical */
  {
    PetscInt j_val;       /* j index for matrix */
    PetscInt base_offset; /* offset of diagonal */
    PetscInt offset;      /* offset of diagonal */
    bool insert_val, ecs;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);
    PetscInt dim_idx;

    ecs = false;
    for (PetscInt i_val = start; i_val < end; i_val++)
    {
      j_val     = i_val;
      idx_array = GetIndexArray(i_val, j_val);
      dim_idx   = 0;
      if ((i_val == start or idx_array[4] == 0) and
          world.rank() == world.size() - 1)
      {
        std::cout << "Calculating " << idx_array[2] + 1 << " of " << num_x[1]
                  << "\n";
      }

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
            x_vals[coef_idx] =
                x_value[dim_idx][coef_idx - order / 2 + idx_array[dim_idx * 2]];
          }
        }
        /* Get real coefficients for each dimension */
        FDWeights(x_vals, 2, real_coef[dim_idx]);
      }

      PetscLogEventBegin(create_h_0_diag, 0, 0, 0, 0);
      /* Diagonal element */
      val = GetVal(i_val, j_val, insert_val, ecs, l_val);
      if (insert_val)
      {
        MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val, INSERT_VALUES);
      }
      PetscLogEventEnd(create_h_0_diag, 0, 0, 0, 0);

      PetscLogEventBegin(create_h_0_fd, 0, 0, 0, 0);
      /* Loop over off diagonal elements for KE opp */
      for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
      {
        dim_idx     = 0;
        base_offset = GetOffset(elec_idx, dim_idx);
        if (dim_idx == 0 and
            idx_array[2 * (2 + elec_idx * num_dims)] < order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order; ++diagonal_idx)
          {
            offset = (diagonal_idx + 1) * base_offset;
            /* Lower diagonal */
            if (i_val - offset >= 0 and i_val - offset < num_psi)
            {
              j_val = i_val - offset;
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
              if (insert_val)
              {
                MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                             INSERT_VALUES);
              }
            }
          }
        }
        else if (dim_idx == 0 and
                 num_x[2] - 1 - idx_array[2 * (2 + elec_idx * num_dims)] <
                     order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order - 1; ++diagonal_idx)
          {
            offset = (diagonal_idx + 1) * base_offset;
            /* Lower diagonal */
            if (i_val - offset >= 0 and i_val - offset < num_psi)
            {
              j_val = i_val - offset;
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
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
              val   = GetVal(i_val, j_val, insert_val, ecs, l_val);
              if (insert_val)
              {
                MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                             INSERT_VALUES);
              }
            }
          }
        }
      }
      PetscLogEventEnd(create_h_0_fd, 0, 0, 0, 0);

      PetscLogEventBegin(create_h_0_hyper, 0, 0, 0, 0);
      /* put in the <Y_k'|V|Y_k> terms */
      base_offset = num_x[2];
      for (int diagonal_idx = 0; diagonal_idx < max_block_size; ++diagonal_idx)
      {
        offset = (diagonal_idx + 1) * base_offset;

        j_val = i_val - offset;
        /* Lower diagonal */
        if (j_val >= 0 and j_val < num_psi)
        {
          PetscLogEventBegin(create_h_0_hyper_low, 0, 0, 0, 0);
          idx_array = GetIndexArray(i_val, j_val);
          PetscLogEventBegin(create_h_0_hyper_low_get_val, 0, 0, 0, 0);
          val = GetHyperspherePotential(idx_array);
          PetscLogEventEnd(create_h_0_hyper_low_get_val, 0, 0, 0, 0);
          if (abs(val) > 1e-16)
          {
            MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
          PetscLogEventEnd(create_h_0_hyper_low, 0, 0, 0, 0);
        }

        /* Upper diagonal */
        j_val = i_val + offset;
        if (j_val >= 0 and j_val < num_psi)
        {
          PetscLogEventBegin(create_h_0_hyper_upper, 0, 0, 0, 0);
          idx_array = GetIndexArray(i_val, j_val);

          val = GetHyperspherePotential(idx_array);
          if (abs(val) > 1e-16)
          {
            MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
          PetscLogEventEnd(create_h_0_hyper_upper, 0, 0, 0, 0);
        }
      }

      /* allocate for laser  <Y_k'|V|Y_k> terms */
      for (int diagonal_idx = 0; diagonal_idx < num_x[1]; ++diagonal_idx)
      {
        offset = (diagonal_idx + 1) * base_offset;

        j_val = i_val - offset;
        /* Lower diagonal */
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);
          val       = GetHypersphereLaser(idx_array);
          if (abs(val) > 1e-16)
          {
            val = 0.0;
            MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }

        /* Upper diagonal */
        j_val = i_val + offset;
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);

          val = GetHypersphereLaser(idx_array);
          if (abs(val) > 1e-16)
          {
            val = 0.0;
            MatSetValues(hamiltonian_0, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }
      }
      PetscLogEventEnd(create_h_0_hyper, 0, 0, 0, 0);
    }
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

  /* For profiling code
   * gives load balancing info vs assembly messages
   */
  PetscLogEventBegin(build_H_0, 0, 0, 0, 0);
  world.barrier();
  PetscLogEventEnd(build_H_0, 0, 0, 0, 0);
  MatAssemblyBegin(hamiltonian_0, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(hamiltonian_0, MAT_FINAL_ASSEMBLY);
}

void Hamiltonian::CalculateHamlitonian0ECS()
{
  dcomp val(0.0, 0.0); /* diagonal terms */
  PetscInt start, end; /* start end rows */

  MatGetOwnershipRange(hamiltonian_0_ecs, &start, &end);
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
  else if (coordinate_system_idx == 3) /* Spherical */
  {
    PetscInt j_val;             /* j index for matrix */
    PetscInt r_size = num_x[2]; /* r dimension size */
    PetscInt r_idx;             /* r_index */
    PetscInt diag_l_val;        /* l_value */
    PetscInt diag_m_val;        /* m_value */
    PetscInt base_offset;       /* offset of diagonal */
    PetscInt offset;            /* offset of diagonal */
    bool insert_val, ecs;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);
    PetscInt dim_idx;

    ecs = true;
    for (PetscInt i_val = start; i_val < end; i_val++)
    {
      j_val     = i_val;
      idx_array = GetIndexArray(i_val, j_val);
      dim_idx   = 0;
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
            x_vals[coef_idx] =
                x_value[dim_idx][coef_idx - order / 2 + idx_array[dim_idx * 2]];
          }
        }
        /* Get real coefficients for each dimension */
        FDWeights(x_vals, 2, real_coef[dim_idx]);
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
        dim_idx     = 0;
        base_offset = GetOffset(elec_idx, dim_idx);
        /* radial dimension handle boundary for first top rows (near r=0) */
        if (dim_idx == 0 and
            idx_array[2 * (2 + elec_idx * num_dims)] < order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order; ++diagonal_idx)
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
        /* radial dimension handle boundary for first bottom rows (near r=r_max)
         */
        else if (dim_idx == 0 and
                 num_x[2] - 1 - idx_array[2 * (2 + elec_idx * num_dims)] <
                     order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order - 1; ++diagonal_idx)
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
        /* standard finite difference stencils for rows where the entire FD
         * stencil fits on the grid*/
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
        /* Preallocate the non zero elements for the laser Hamiltonian */
        j_val      = i_val;
        idx_array  = GetIndexArray(i_val, j_val);
        r_idx      = idx_array[2 * (elec_idx * num_dims + 2)];
        diag_l_val = l_values[idx_array[2 * (elec_idx * num_dims + 1)]];
        diag_m_val = m_values[idx_array[2 * (elec_idx * num_dims + 1)]];
        /* make sure l+1 is part of our grid*/
        if (diag_l_val + 1 <= l_max)
        {
          /* get l -> l+1 here */
          j_val =
              GetIdxFromLM(diag_l_val + 1, diag_m_val, m_max) * r_size + r_idx;
          val = 0.0;
          MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
        }
        /* make sure l-1 is greater than zero and the m value exists*/
        if (diag_l_val - 1 >= 0 and diag_l_val - 1 >= std::abs(diag_m_val))
        {
          /* get the l -> l-1 */
          j_val =
              GetIdxFromLM(diag_l_val - 1, diag_m_val, m_max) * r_size + r_idx;
          val = 0.0;
          MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
        }
        /* make sure l+1 and m+1 is part of our grid*/
        if (diag_l_val + 1 <= l_max and std::abs(diag_m_val + 1) <= m_max)
        {
          /* get l -> l+1 and m -> m+1 */
          j_val = GetIdxFromLM(diag_l_val + 1, diag_m_val + 1, m_max) * r_size +
                  r_idx;
          val = 0.0;
          MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
        }
        /* make sure l+1 and m-1 is part of our grid*/
        if (diag_l_val + 1 <= l_max and std::abs(diag_m_val - 1) <= m_max)
        {
          /* get l -> l+1 and m -> m-1 */
          j_val = GetIdxFromLM(diag_l_val + 1, diag_m_val - 1, m_max) * r_size +
                  r_idx;
          val = 0.0;
          MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
        }
        /* make sure l-1 and m+1 is part of our grid */
        if (diag_l_val - 1 >= 0 and
            diag_l_val - 1 >= std::abs(diag_m_val + 1) and
            std::abs(diag_m_val + 1) <= m_max)
        {
          /* get l -> l+1 and m -> m-1 */
          j_val = GetIdxFromLM(diag_l_val - 1, diag_m_val + 1, m_max) * r_size +
                  r_idx;
          val = 0.0;
          MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
        }
        /* make sure l-1 and m-1 is part of our grid */
        if (diag_l_val - 1 >= 0 and
            diag_l_val - 1 >= std::abs(diag_m_val - 1) and
            std::abs(diag_m_val - 1) <= m_max)
        {
          /* get l -> l+1 and m -> m-1 */
          j_val = GetIdxFromLM(diag_l_val - 1, diag_m_val - 1, m_max) * r_size +
                  r_idx;
          val = 0.0;
          MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                       INSERT_VALUES);
        }
      }
    }
  }
  else if (coordinate_system_idx == 4) /* Hyperpherical */
  {
    PetscInt j_val;       /* j index for matrix */
    PetscInt base_offset; /* offset of diagonal */
    PetscInt offset;      /* offset of diagonal */
    bool insert_val, ecs;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);
    PetscInt dim_idx;

    ecs = true;
    for (PetscInt i_val = start; i_val < end; i_val++)
    {
      j_val     = i_val;
      idx_array = GetIndexArray(i_val, j_val);
      dim_idx   = 0;
      if ((i_val == start or idx_array[4] == 0) and
          world.rank() == world.size() - 1)
      {
        std::cout << "Calculating " << idx_array[2] + 1 << " of " << num_x[1]
                  << "\n";
      }

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
            x_vals[coef_idx] =
                x_value[dim_idx][coef_idx - order / 2 + idx_array[dim_idx * 2]];
          }
        }
        /* Get real coefficients for each dimension */
        FDWeights(x_vals, 2, real_coef[dim_idx]);
      }

      /* Diagonal element */
      val = GetVal(i_val, j_val, insert_val, ecs);
      if (insert_val)
      {
        MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                     INSERT_VALUES);
      }

      /* Loop over off diagonal elements for KE opp */
      for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
      {
        dim_idx     = 0;
        base_offset = GetOffset(elec_idx, dim_idx);
        if (dim_idx == 0 and
            idx_array[2 * (2 + elec_idx * num_dims)] < order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order; ++diagonal_idx)
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
        else if (dim_idx == 0 and
                 num_x[2] - 1 - idx_array[2 * (2 + elec_idx * num_dims)] <
                     order_middle_idx - 1)
        {
          /* loop over all off diagonals up to the order needed */
          for (int diagonal_idx = 0; diagonal_idx < order - 1; ++diagonal_idx)
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

      /* put in the <Y_k'|V|Y_k> terms */
      base_offset = num_x[2];
      for (int diagonal_idx = 0; diagonal_idx < max_block_size; ++diagonal_idx)
      {
        offset = (diagonal_idx + 1) * base_offset;

        j_val = i_val - offset;
        /* Lower diagonal */
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);
          val       = GetHyperspherePotential(idx_array);
          if (abs(val) > 1e-16)
          {
            MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }

        /* Upper diagonal */
        j_val = i_val + offset;
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);

          val = GetHyperspherePotential(idx_array);
          if (abs(val) > 1e-16)
          {
            MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }
      }

      /* allocate for laser  <Y_k'|V|Y_k> terms */
      for (int diagonal_idx = 0; diagonal_idx < num_x[1]; ++diagonal_idx)
      {
        offset = (diagonal_idx + 1) * base_offset;

        j_val = i_val - offset;
        /* Lower diagonal */
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);
          val       = GetHypersphereLaser(idx_array);
          if (abs(val) > 1e-16)
          {
            val = 0.0;
            MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }

        /* Upper diagonal */
        j_val = i_val + offset;
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);

          val = GetHypersphereLaser(idx_array);
          if (abs(val) > 1e-16)
          {
            val = 0.0;
            MatSetValues(hamiltonian_0_ecs, 1, &i_val, 1, &j_val, &val,
                         INSERT_VALUES);
          }
        }
      }
    }
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
  /* For profiling code
   * gives load balancing info vs assembly messages
   */
  PetscLogEventBegin(build_H_0_ecs, 0, 0, 0, 0);
  world.barrier();
  PetscLogEventEnd(build_H_0_ecs, 0, 0, 0, 0);
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
  else if (coordinate_system_idx == 3)
  {
    PetscInt j_val;             /* j index for matrix */
    PetscInt r_size = num_x[2]; /* r dimension size */
    PetscInt r_idx;             /* r_index */
    PetscInt diag_l_val;        /* l_value */
    PetscInt diag_m_val;        /* m_value */
    bool insert_val;
    std::vector< PetscInt > idx_array;
    std::vector< dcomp > x_vals(order + 1, 0.0);
    for (PetscInt ham_dim_idx = 0; ham_dim_idx < num_dims; ham_dim_idx++)
    {
      MatGetOwnershipRange(hamiltonian_laser[ham_dim_idx], &start, &end);
      for (PetscInt i_val = start; i_val < end; i_val++)
      {
        /* no true diagonal in spherical harmonics */

        /* Loop over off diagonal elements */
        for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
        {
          j_val      = i_val;
          idx_array  = GetIndexArray(i_val, j_val);
          r_idx      = idx_array[2 * (elec_idx * num_dims + 2)];
          diag_l_val = l_values[idx_array[2 * (elec_idx * num_dims + 1)]];
          diag_m_val = m_values[idx_array[2 * (elec_idx * num_dims + 1)]];

          /* for the z direction delta l=+-1 and delta m=0*/
          if (ham_dim_idx == 2)
          {
            /* make sure l+1 is part of our grid*/
            if (diag_l_val + 1 <= l_max)
            {
              /* get l -> l+1 here */
              j_val = GetIdxFromLM(diag_l_val + 1, diag_m_val, m_max) * r_size +
                      r_idx;
              val = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
              if (insert_val)
              {
                MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                             &j_val, &val, INSERT_VALUES);
              }
            }
            /* make sure l-1 is greater than zero and the m value exists*/
            if (diag_l_val - 1 >= 0 and diag_l_val - 1 >= std::abs(diag_m_val))
            {
              /* get the l -> l-1 */
              j_val = GetIdxFromLM(diag_l_val - 1, diag_m_val, m_max) * r_size +
                      r_idx;
              val = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
              if (insert_val)
              {
                MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                             &j_val, &val, INSERT_VALUES);
              }
            }
          }

          /* for the x and y direction delta l=+-1 and delta m=+-1 */
          if (ham_dim_idx == 0 or ham_dim_idx == 1)
          {
            /* make sure l+1 and m+1 is part of our grid*/
            if (diag_l_val + 1 <= l_max and std::abs(diag_m_val + 1) <= m_max)
            {
              /* get l -> l+1 and m -> m+1 */
              j_val =
                  GetIdxFromLM(diag_l_val + 1, diag_m_val + 1, m_max) * r_size +
                  r_idx;
              val = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
              // idx_array = GetIndexArray(i_val, j_val);
              // std::cout << "(" << diag_l_val << "," << diag_m_val << ")->"
              //           << "(" << l_values[idx_array[2 * (0 + 1) + 1]] << ","
              //           << m_values[idx_array[2 * (0 + 1) + 1]] << ") " <<
              //           val << " " << insert_val << "\n";
              if (insert_val)
              {
                MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                             &j_val, &val, INSERT_VALUES);
              }
            }
            /* make sure l+1 and m-1 is part of our grid*/
            if (diag_l_val + 1 <= l_max and std::abs(diag_m_val - 1) <= m_max)
            {
              /* get l -> l+1 and m -> m-1 */
              j_val =
                  GetIdxFromLM(diag_l_val + 1, diag_m_val - 1, m_max) * r_size +
                  r_idx;
              val = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
              // idx_array = GetIndexArray(i_val, j_val);
              // std::cout << "(" << diag_l_val << "," << diag_m_val << ")->"
              //           << "(" << l_values[idx_array[2 * (0 + 1) + 1]] << ","
              //           << m_values[idx_array[2 * (0 + 1) + 1]] << ") " <<
              //           val << " " << insert_val << "\n";
              if (insert_val)
              {
                MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                             &j_val, &val, INSERT_VALUES);
              }
            }
            /* make sure l-1 and m+1 is part of our grid */
            if (diag_l_val - 1 >= 0 and
                diag_l_val - 1 >= std::abs(diag_m_val + 1) and
                std::abs(diag_m_val + 1) <= m_max)
            {
              /* get l -> l+1 and m -> m-1 */
              j_val =
                  GetIdxFromLM(diag_l_val - 1, diag_m_val + 1, m_max) * r_size +
                  r_idx;
              val = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
              // idx_array = GetIndexArray(i_val, j_val);
              // std::cout << "(" << diag_l_val << "," << diag_m_val << ")->"
              //           << "(" << l_values[idx_array[2 * (0 + 1) + 1]] << ","
              //           << m_values[idx_array[2 * (0 + 1) + 1]] << ") " <<
              //           val << " " << insert_val << "\n";
              if (insert_val)
              {
                MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                             &j_val, &val, INSERT_VALUES);
              }
            }
            /* make sure l-1 and m-1 is part of our grid */
            if (diag_l_val - 1 >= 0 and
                diag_l_val - 1 >= std::abs(diag_m_val - 1) and
                std::abs(diag_m_val - 1) <= m_max)
            {
              /* get l -> l+1 and m -> m-1 */
              j_val =
                  GetIdxFromLM(diag_l_val - 1, diag_m_val - 1, m_max) * r_size +
                  r_idx;
              val = GetValLaser(i_val, j_val, insert_val, ham_dim_idx);
              // idx_array = GetIndexArray(i_val, j_val);
              // std::cout << "(" << diag_l_val << "," << diag_m_val << ")->"
              //           << "(" << l_values[idx_array[2 * (0 + 1) + 1]] << ","
              //           << m_values[idx_array[2 * (0 + 1) + 1]] << ") " <<
              //           val
              //           << " " << insert_val << "\n";
              if (insert_val)
              {
                MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1,
                             &j_val, &val, INSERT_VALUES);
              }
            }
          }
        }
      }
      MatAssemblyBegin(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
    }
    wavefunction->SetPositionMat(hamiltonian_laser);
  }
  else if (coordinate_system_idx == 4)
  {
    PetscInt j_val, offset;     /* j index for matrix */
    PetscInt r_size = num_x[2]; /* r dimension size */
    std::vector< PetscInt > idx_array;
    PetscInt ham_dim_idx = 2;
    MatGetOwnershipRange(hamiltonian_laser[ham_dim_idx], &start, &end);
    for (PetscInt i_val = start; i_val < end; i_val++)
    {
      /* put in the <Y_k'|V|Y_k> terms */
      for (int diagonal_idx = 0; diagonal_idx < num_x[1]; ++diagonal_idx)
      {
        offset = (diagonal_idx + 1) * r_size;

        j_val = i_val - offset;
        /* Lower diagonal */
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);
          val       = GetHypersphereLaser(idx_array);
          if (abs(val) > 1e-16)
          {
            MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1, &j_val,
                         &val, INSERT_VALUES);
          }
        }

        /* Upper diagonal */
        j_val = i_val + offset;
        if (j_val >= 0 and j_val < num_psi)
        {
          idx_array = GetIndexArray(i_val, j_val);

          val = GetHypersphereLaser(idx_array);
          if (abs(val) > 1e-16)
          {
            MatSetValues(hamiltonian_laser[ham_dim_idx], 1, &i_val, 1, &j_val,
                         &val, INSERT_VALUES);
          }
        }
      }
    }
    for (PetscInt ham_dim_idx = 0; ham_dim_idx < num_dims; ham_dim_idx++)
    {
      MatAssemblyBegin(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(hamiltonian_laser[ham_dim_idx], MAT_FINAL_ASSEMBLY);
    }

    wavefunction->SetPositionMat(hamiltonian_laser);
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
    if (coordinate_system_idx == 3)
    {
      wavefunction->SetPositionMat(hamiltonian_laser);
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
  if (coordinate_system_idx == 3 or
      coordinate_system_idx == 4) /* spherical boundary conditions */
  {
    /* We don't need forward and backward difference thus order-1 terms */
    radial_bc_coef.resize(order - 1,
                          std::vector< std::vector< dcomp > >(
                              3, std::vector< dcomp >(order + 1, 0.0)));
    std::vector< dcomp > x_vals_bc(order + 1, 0.0);
    /* Set up real gird for 1st and 2nd order derivatives */
    for (int coef_idx = 0; coef_idx < order + 1; ++coef_idx)
    {
      x_vals_bc[coef_idx] = delta_x_min[2] * (coef_idx);
    }

    /* We don't need forward and backward difference thus order-1 terms */
    for (int discontinuity_idx = 0; discontinuity_idx < order - 1;
         ++discontinuity_idx)
    {
      /* Get real coefficients for 2nd derivative (order+2 terms) */
      FDWeights(x_vals_bc, 2, radial_bc_coef[discontinuity_idx],
                discontinuity_idx + 1);
      if (discontinuity_idx < order / 2 - 1)
      {
        for (int coef_idx = 0; coef_idx < order; ++coef_idx)
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
}

Mat* Hamiltonian::GetTotalHamiltonian(PetscInt time_idx, bool ecs)
{
  if (ecs)
  {
    MatCopy(hamiltonian_0_ecs, hamiltonian, SAME_NONZERO_PATTERN);
  }
  else
  {
    if (coordinate_system_idx == 3)
    {
      EndRun(
          "Mat copy hamiltonian_0 to hamiltonian in GetTotalHamiltonian not "
          "supported due to different shapes in Spherical coordinates.");
    }
    else
    {
      MatCopy(hamiltonian_0, hamiltonian, SAME_NONZERO_PATTERN);
    }
  }
  for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
  {
    if (field[dim_idx][time_idx] != 0.0 and coordinate_system_idx == 4)
    {
      MatAXPY(hamiltonian, -2.0 * field[dim_idx][time_idx],
              hamiltonian_laser[dim_idx], SUBSET_NONZERO_PATTERN);
    }
    else if (field[dim_idx][time_idx] != 0.0)
    {
      MatAXPY(hamiltonian, field[dim_idx][time_idx], hamiltonian_laser[dim_idx],
              SUBSET_NONZERO_PATTERN);
    }
  }
  return &hamiltonian;
}

dcomp Hamiltonian::GetVal(PetscInt idx_i, PetscInt idx_j, bool& insert_val,
                          bool ecs, PetscInt l_val)
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
    /* allow for shifting l_values for H_0 in spherical coordinates (only
     * applies to diagonal term) */
    if (coordinate_system_idx == 3)
    {
      for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
      {
        idx_array[2 * (1 + elec_idx * num_dims)] = GetIdxFromLM(
            l_values[idx_array[2 * (1 + elec_idx * num_dims)]] + l_val, 0,
            m_max);
        idx_array[2 * (1 + elec_idx * num_dims) + 1] = GetIdxFromLM(
            l_values[idx_array[2 * (1 + elec_idx * num_dims) + 1]] + l_val, 0,
            m_max);
      }
    }
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
    else if (coordinate_system_idx == 3 or coordinate_system_idx == 4)
    {
      /* TODO update for more than one electron Probably a for loop */

      /* Handle the boundary condition at r=0 by doing forward difference */
      if (sum < order - idx_array[2 * (2 + 0 * num_dims)] and
          idx_array[2 * (2 + 0 * num_dims)] < order_middle_idx - 1)
      {
        return GetOffDiagonal(idx_array, diff_array, ecs);
      }
      /* Handle the boundary condition at r=r_max by doing backward difference
       */
      else if (sum < order -
                         (num_x[2] - 1 - idx_array[2 * (2 + 0 * num_dims)]) and
               (num_x[2] - 1 - idx_array[2 * (2 + 0 * num_dims)]) <
                   order_middle_idx - 1)
      {
        return GetOffDiagonal(idx_array, diff_array, ecs);
      }
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

  if (coordinate_system_idx == 3)
  {
    if (gauge_idx == 1) /* length gauge */
    {
      /* Calulating the z operator */
      if (only_dim_idx == 2)
      {
        /* ensure r_idx is not changed, l -> l+-1 and m -> m  */
        if (idx_array[2 * (0 + 2)] == idx_array[2 * (0 + 2) + 1] and
            std::abs(l_values[idx_array[2 * (0 + 1)]] -
                     l_values[idx_array[2 * (0 + 1) + 1]]) == 1 and
            m_values[idx_array[2 * (0 + 1)]] ==
                m_values[idx_array[2 * (0 + 1) + 1]])
        {
          return GetOffDiagonalLaser(idx_array, diff_array, only_dim_idx);
        }
      }
      /* Calulating the x or y operator */
      else if (only_dim_idx == 0 or only_dim_idx == 1)
      {
        /* ensure r_idx is not changed, l -> l+-1 and m -> m+-1  */
        if (idx_array[2 * (0 + 2)] == idx_array[2 * (0 + 2) + 1] and
            std::abs(l_values[idx_array[2 * (0 + 1)]] -
                     l_values[idx_array[2 * (0 + 1) + 1]]) == 1 and
            std::abs(m_values[idx_array[2 * (0 + 1)]] -
                     m_values[idx_array[2 * (0 + 1) + 1]]) == 1)
        {
          return GetOffDiagonalLaser(idx_array, diff_array, only_dim_idx);
        }
      }
    }
  }
  else
  {
    /* Diagonal elements */
    if (idx_i == idx_j and gauge_idx == 1) /* length gauge */
    {
      return GetDiagonalLaser(idx_array, only_dim_idx);
    }

    if (gauge_idx == 0) /* velocity gauge */
    {
      /* Make sure there is exactly 1 non zero index so we can take care of
       * the off diagonal zeros */
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
        else if ((coordinate_system_idx == 3 or coordinate_system_idx == 4) and
                 dim_idx == 2)
        {
          /* r=0  boundary condition */
          /* We use a forward difference like formula until the full stencil
           * can fit in the matrix (We include one ghost node that imposes
           * psi(0)=0) */
          if (idx_array[2 * (elec_idx * num_dims + dim_idx)] <
              order_middle_idx - 1)
          {
            discontinuity_idx = idx_array[2 * (elec_idx * num_dims + dim_idx)];

            off_diagonal -=
                radial_bc_coef[discontinuity_idx][2]
                              [discontinuity_idx +
                               diff_array[elec_idx * num_dims + dim_idx]] /
                2.0;
          }
          /* r=r_max  boundary condition */
          /* We use a forward difference like formula until the full stencil
           * can fit in the matrix (We include one ghost node that imposes
           * psi(r_max)=0) */
          else if ((num_x[dim_idx] - 1 -
                    idx_array[2 * (elec_idx * num_dims + dim_idx)]) <
                   order_middle_idx)
          {
            discontinuity_idx =
                order - 2 -
                (num_x[dim_idx] - 1 -
                 idx_array[2 * (elec_idx * num_dims + dim_idx)]);

            /* For ecs you need to divide by 1/exp(-i*eta*2) to account for
             * the 1/dx^2 in the Finite difference formula */
            if (ecs and idx_array[2 * (elec_idx * num_dims + dim_idx)] >=
                            gobbler_idx[dim_idx][1])
            {
              off_diagonal -=
                  radial_bc_coef[discontinuity_idx][2]
                                [discontinuity_idx +
                                 diff_array[elec_idx * num_dims + dim_idx] +
                                 1] /
                  (2.0 * std::exp(imag * 2.0 * eta));
            }
            else
            {
              off_diagonal -=
                  radial_bc_coef[discontinuity_idx][2]
                                [discontinuity_idx +
                                 diff_array[elec_idx * num_dims + dim_idx] +
                                 1] /
                  2.0;
            }
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

          else /* Real part */
          {
            off_diagonal -=
                real_coef[dim_idx][2]
                         [order_middle_idx +
                          diff_array[elec_idx * num_dims + dim_idx]] /
                2.0;
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
  PetscInt l0, l1, l_tot;
  PetscInt m0, m1, m_tot;
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    for (PetscInt dim_idx = 0; dim_idx < num_dims; ++dim_idx)
    {
      if (only_dim_idx == -1 or dim_idx == only_dim_idx)
      {
        /* spherical grid */
        if (coordinate_system_idx == 3 and gauge_idx == 1)
        {
          /* z direction */
          if (dim_idx == 2)
          {
            if (delta_x_min[2] != delta_x_max[2])
            {
              EndRun(
                  "Laser operator does not support nonunifrom grid in "
                  "Spherical coordinates.");
            }
            l0    = l_values[idx_array[2. * (elec_idx * num_dims + 1)]];
            l1    = 1;
            l_tot = l_values[idx_array[2. * (elec_idx * num_dims + 1) + 1]];
            m0    = m_values[idx_array[2. * (elec_idx * num_dims + 1)]];
            m1    = 0;
            m_tot = m_values[idx_array[2. * (elec_idx * num_dims + 1) + 1]];
            if (std::abs(l0 - l_tot) == 1 and m0 == m_tot)
            {
              off_diagonal +=
                  x_value[2][idx_array[2. * (elec_idx * num_dims + 2)]] *
                  std::sqrt(4.0 * pi / 3.0) *
                  std::sqrt((2 * l0 + 1) * (2 * l1 + 1) /
                            (4 * pi * (2 * l_tot + 1))) *
                  ClebschGordanCoef(l0, l1, l_tot, 0, 0, 0) *
                  ClebschGordanCoef(l0, l1, l_tot, m0, m1, m_tot);
            }
          }
          /* x direction */
          if (dim_idx == 0)
          {
            if (delta_x_min[2] != delta_x_max[2])
            {
              EndRun(
                  "Laser operator does not support nonunifrom grid in "
                  "Spherical coordinates.");
            }
            l0    = l_values[idx_array[2. * (elec_idx * num_dims + 1)]];
            l1    = 1;
            l_tot = l_values[idx_array[2. * (elec_idx * num_dims + 1) + 1]];
            m0    = m_values[idx_array[2. * (elec_idx * num_dims + 1)]];
            m_tot = m_values[idx_array[2. * (elec_idx * num_dims + 1) + 1]];
            m1    = m_tot - m0;
            if (std::abs(l0 - l_tot) == 1)
            {
              /* m -> m+1 */
              if (m1 == 1)
              {
                off_diagonal +=
                    -1.0 *
                    x_value[2][idx_array[2. * (elec_idx * num_dims + 2)]] *
                    std::sqrt(2.0 * pi / 3.0) *
                    std::sqrt((2 * l0 + 1) * (2 * l1 + 1) /
                              (4 * pi * (2 * l_tot + 1))) *
                    ClebschGordanCoef(l0, l1, l_tot, 0, 0, 0) *
                    ClebschGordanCoef(l0, l1, l_tot, m0, m1, m_tot);
              }
              /* m -> m-1 */
              else if (m1 == -1)
              {
                off_diagonal +=
                    x_value[2][idx_array[2. * (elec_idx * num_dims + 2)]] *
                    std::sqrt(2.0 * pi / 3.0) *
                    std::sqrt((2 * l0 + 1) * (2 * l1 + 1) /
                              (4 * pi * (2 * l_tot + 1))) *
                    ClebschGordanCoef(l0, l1, l_tot, 0, 0, 0) *
                    ClebschGordanCoef(l0, l1, l_tot, m0, m1, m_tot);
              }
            }
          }
          /* y direction */
          if (dim_idx == 1)
          {
            if (delta_x_min[2] != delta_x_max[2])
            {
              EndRun(
                  "Laser operator does not support nonunifrom grid in "
                  "Spherical coordinates.");
            }
            l0    = l_values[idx_array[2. * (elec_idx * num_dims + 1)]];
            l1    = 1;
            l_tot = l_values[idx_array[2. * (elec_idx * num_dims + 1) + 1]];
            m0    = m_values[idx_array[2. * (elec_idx * num_dims + 1)]];
            m_tot = m_values[idx_array[2. * (elec_idx * num_dims + 1) + 1]];
            m1    = m_tot - m0;
            if (std::abs(l0 - l_tot) == 1)
            {
              /* m -> m+1 */
              if (m1 == 1)
              {
                off_diagonal +=
                    x_value[2][idx_array[2. * (elec_idx * num_dims + 2)]] *
                    std::sqrt(2.0 * pi / 3.0) *
                    std::sqrt((2 * l0 + 1) * (2 * l1 + 1) /
                              (4 * pi * (2 * l_tot + 1))) *
                    ClebschGordanCoef(l0, l1, l_tot, 0, 0, 0) *
                    ClebschGordanCoef(l0, l1, l_tot, m0, m1, m_tot) / imag;
              }
              /* m -> m-1 */
              else if (m1 == -1)
              {
                off_diagonal +=
                    x_value[2][idx_array[2. * (elec_idx * num_dims + 2)]] *
                    std::sqrt(2.0 * pi / 3.0) *
                    std::sqrt((2 * l0 + 1) * (2 * l1 + 1) /
                              (4 * pi * (2 * l_tot + 1))) *
                    ClebschGordanCoef(l0, l1, l_tot, 0, 0, 0) *
                    ClebschGordanCoef(l0, l1, l_tot, m0, m1, m_tot) / imag;
              }
            }
          }
        }
        else if (diff_array[elec_idx * num_dims + dim_idx] != 0)
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
  if (coordinate_system_idx == 4)
  {
    /* kinetic term */
    diagonal += GetKineticTerm(idx_array, ecs);
    diagonal += GetCentrifugalTerm(idx_array);
    diagonal += GetHyperspherePotential(idx_array);
    // std::cout << GetKineticTerm(idx_array, ecs) +
    // GetCentrifugalTerm(idx_array)
    //           << " " << GetHyperspherePotential(idx_array) << " " << diagonal
    //           << "\n";
  }
  else
  {
    /* kinetic term */
    diagonal += GetKineticTerm(idx_array, ecs);
    /* nuclei term */
    diagonal += GetNucleiTerm(idx_array);
    /* e-e correlation */
    diagonal += GetElectronElectronTerm(idx_array);
    /* centrifugal term for radial equation */
    if (coordinate_system_idx == 3)
    {
      diagonal += GetCentrifugalTerm(idx_array);
    }
  }
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
    if (coordinate_system_idx == 3 or coordinate_system_idx == 4)
    {
      PetscInt dim_idx = 2;
      /* r=0  boundary condition */
      /* We use a forward difference like formula until the full stencil can
       * fit in the matrix (We include one ghost node that imposes
       * psi(0)=0) */
      if (idx_array[2 * (elec_idx * num_dims + dim_idx)] < order_middle_idx - 1)
      {
        discontinuity_idx = idx_array[2 * (elec_idx * num_dims + dim_idx)];

        kinetic -=
            radial_bc_coef[discontinuity_idx][2][discontinuity_idx] / 2.0;
      }
      /* r=r_max  boundary condition */
      /* We use a forward difference like formula until the full stencil can
       * fit in the matrix (We include one ghost node that imposes
       * psi(r_max)=0) */
      else if ((num_x[dim_idx] - 1 -
                idx_array[2 * (elec_idx * num_dims + dim_idx)]) <
               order_middle_idx)
      {
        discontinuity_idx = order - 2 -
                            (num_x[dim_idx] - 1 -
                             idx_array[2 * (elec_idx * num_dims + dim_idx)]);

        /* For ecs you need to divide by 1/exp(-i*eta*2) to account for the
         * 1/dx^2 in the Finite difference formula */
        if (ecs and idx_array[2 * (elec_idx * num_dims + dim_idx)] >=
                        gobbler_idx[dim_idx][1])
        {
          kinetic -=
              radial_bc_coef[discontinuity_idx][2][discontinuity_idx + 1] /
              (2.0 * std::exp(imag * 2.0 * eta));
        }
        else
        {
          kinetic -=
              radial_bc_coef[discontinuity_idx][2][discontinuity_idx + 1] / 2.0;
        }
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
    else
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
  double tmp;
  double tmp_soft;
  /* loop over each electron */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    /* loop over each nuclei */
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (coordinate_system_idx == 3)
      {
        r      = x_value[2][idx_array[2 * (2 + elec_idx * num_dims)]];
        r_soft = sqrt(r * r + alpha_2);
      }
      else
      {
        r_soft = SoftCoreDistance(location[nuclei_idx], idx_array, elec_idx);
        r      = EuclideanDistance(location[nuclei_idx], idx_array, elec_idx);
      }

      /* Coulomb term */
      nuclei -= dcomp(z[nuclei_idx] / r_soft, 0.0);

      /* Gaussian Donuts */
      for (PetscInt i = 0; i < gaussian_size[nuclei_idx]; ++i)
      {
        tmp = gaussian_decay_rate[nuclei_idx][i] *
              (r - gaussian_r_0[nuclei_idx][i]);
        nuclei -= dcomp(
            gaussian_amplitude[nuclei_idx][i] * exp(-0.5 * (tmp * tmp)), 0.0);
      }

      /* Exponential Donuts */
      for (PetscInt i = 0; i < exponential_size[nuclei_idx]; ++i)
      {
        tmp = exponential_decay_rate[nuclei_idx][i] *
              std::abs(r - exponential_r_0[nuclei_idx][i]);
        nuclei -= dcomp(exponential_amplitude[nuclei_idx][i] * exp(-tmp), 0.0);
      }

      /* Square Well Donuts */
      for (PetscInt i = 0; i < square_well_size[nuclei_idx]; ++i)
      {
        /* only apply square well between r_0 and r_0+width */
        if ((r >= square_well_r_0[nuclei_idx][i]) and
            (r <= (square_well_r_0[nuclei_idx][i] +
                   square_well_width[nuclei_idx][i])))
        {
          nuclei -= dcomp(square_well_amplitude[nuclei_idx][i], 0.0);
        }
      }

      /* Yukawa Donuts */
      for (PetscInt i = 0; i < yukawa_size[nuclei_idx]; ++i)
      {
        tmp = yukawa_decay_rate[nuclei_idx][i] *
              std::abs(r - yukawa_r_0[nuclei_idx][i]);
        tmp_soft = std::abs(r_soft - yukawa_r_0[nuclei_idx][i]);
        nuclei -=
            dcomp(yukawa_amplitude[nuclei_idx][i] * exp(-tmp) / tmp_soft, 0.0);
      }
    }
  }
  return nuclei;
}

dcomp Hamiltonian::GetHyperspherePotential(std::vector< PetscInt >& idx_array)
{
  dcomp nuclei(0.0, 0.0);
  double r;
  PetscLogEventBegin(hyper_pot_time, 0, 0, 0, 0);
  if (eigen_values[idx_array[2]][4] == eigen_values[idx_array[2 + 1]][4])
  {
    /* loop over each nuclei */
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      r = x_value[2][idx_array[2 * 2]];
      /* Coulomb term */
      nuclei += GetHypersphereCoulomb(eigen_values[idx_array[2]],
                                      eigen_values[idx_array[2 + 1]], r,
                                      z[nuclei_idx]);
      // std::cout << tmp << "\n";

      // /* Gaussian Donuts */
      // for (PetscInt i = 0; i < gaussian_size[nuclei_idx]; ++i)
      // {
      //   tmp = gaussian_decay_rate[nuclei_idx][i] *
      //         (r - gaussian_r_0[nuclei_idx][i]);
      //   nuclei -= dcomp(
      //       gaussian_amplitude[nuclei_idx][i] * exp(-0.5 * (tmp * tmp)),
      //       0.0);
      // }

      // /* Exponential Donuts */
      // for (PetscInt i = 0; i < exponential_size[nuclei_idx]; ++i)
      // {
      //   tmp = exponential_decay_rate[nuclei_idx][i] *
      //         std::abs(r - exponential_r_0[nuclei_idx][i]);
      //   nuclei -= dcomp(exponential_amplitude[nuclei_idx][i] * exp(-tmp),
      //   0.0);
      // }

      // /* Square Well Donuts */
      // for (PetscInt i = 0; i < square_well_size[nuclei_idx]; ++i)
      // {
      //   /* only apply square well between r_0 and r_0+width */
      //   if ((r >= square_well_r_0[nuclei_idx][i]) and
      //       (r <= (square_well_r_0[nuclei_idx][i] +
      //              square_well_width[nuclei_idx][i])))
      //   {
      //     nuclei -= dcomp(square_well_amplitude[nuclei_idx][i], 0.0);
      //   }
      // }

      // /* Yukawa Donuts */
      // for (PetscInt i = 0; i < yukawa_size[nuclei_idx]; ++i)
      // {
      //   tmp = yukawa_decay_rate[nuclei_idx][i] *
      //         std::abs(r - yukawa_r_0[nuclei_idx][i]);
      //   tmp_soft = std::abs(r_soft - yukawa_r_0[nuclei_idx][i]);
      //   nuclei -=
      //       dcomp(yukawa_amplitude[nuclei_idx][i] * exp(-tmp) / tmp_soft,
      //       0.0);
      // }
    }
  }
  PetscLogEventEnd(hyper_pot_time, 0, 0, 0, 0);
  return nuclei;
}

double Hamiltonian::GetHypersphereCoulomb(int* lambda_a, int* lambda_b,
                                          double r, double z)
{
  PetscLogEventBegin(hyper_coulomb_time, 0, 0, 0, 0);
  num_ang += num_ang % 2;
  double d_angle, matrix_element, pre_fac_a, pre_fac_b, result, tmp;
  int Ka, na, lxa, lya, La, Ma, Kb, nb, lxb, lyb, Lb, Mb;
  std::string key, internal_key;
  matrix_element = 0.0;
  Ka             = lambda_a[0];
  na             = lambda_a[1];
  lxa            = lambda_a[2];
  lya            = lambda_a[3];
  La             = lambda_a[4];
  Ma             = lambda_a[5];
  Kb             = lambda_b[0];
  nb             = lambda_b[1];
  lxb            = lambda_b[2];
  lyb            = lambda_b[3];
  Lb             = lambda_b[4];
  Mb             = lambda_b[5];
  key = to_string(Ka) + "_" + to_string(na) + "_" + to_string(lxa) + "_" +
        to_string(lya) + "_" + to_string(La) + "_" + to_string(Kb) + "_" +
        to_string(nb) + "_" + to_string(lxb) + "_" + to_string(lyb) + "_" +
        to_string(Lb) + "_" + to_string(z) + "_" + to_string(num_ang);

  /* check to see if this has been calculated already */
  if (hypersphere_coulomb_lookup.count(key) == 1)
  {
    PetscLogEventEnd(hyper_coulomb_time, 0, 0, 0, 0);
    return hypersphere_coulomb_lookup[key] / r;
  }

  if (La == Lb)
  {
    d_angle = pi / (2 * num_ang);
    for (int idx = 0; idx < num_ang; ++idx)
    {
      angle[idx]    = idx * d_angle + d_angle / 2.;
      arg_vals[idx] = cos(2. * angle[idx]);
    }
    for (int lx = 0; lx < min(Ka, Kb) + 1; ++lx)
    {
      for (int ly = 0; ly < min(Ka, Kb) + 1; ++ly)
      {
        if ((Ka - lx - ly) % 2 == 0 and (Kb - lx - ly) % 2 == 0)
        {
          pre_fac_a = RRC(Ka, La, lx, ly, lxa, lya, 0) *
                      RRC(Kb, Lb, lx, ly, lxb, lyb, 0);
          pre_fac_b = RRC(Ka, La, lx, ly, lxa, lya, 1) *
                      RRC(Kb, Lb, lx, ly, lxb, lyb, 1);
          if (abs(pre_fac_a) > 1e-16 or abs(pre_fac_b) > 1e-16)
          {
            internal_key = to_string(Ka) + "_" + to_string(lx) + "_" +
                           to_string(ly) + "_" + to_string(La) + "_" +
                           to_string(Ma) + "_" + to_string(Kb) + "_" +
                           to_string(Lb) + "_" + to_string(Mb) + "_" +
                           to_string(z) + "_" + to_string(num_ang);
            if (hypersphere_radial_int_lookup.count(internal_key) == 1)
            {
              result = hypersphere_radial_int_lookup[internal_key];
            }
            else
            {
              SpherHarm(Ka, (Ka - lx - ly) / 2, lx, ly, La, Ma, angle, sphere_1,
                        arg_vals);
              SpherHarm(Kb, (Kb - lx - ly) / 2, lx, ly, Lb, Mb, angle, sphere_2,
                        arg_vals);
              result = 0.0;
              for (int idx = 0; idx < num_ang; ++idx)
              {
                tmp = sin(angle[idx]);
                result +=
                    sphere_1[idx] * sphere_2[idx] * cos(angle[idx]) * tmp * tmp;
              }
              result *= z * d_angle;
              hypersphere_radial_int_lookup[internal_key] = result;
            }

            matrix_element -= result * pre_fac_a + result * pre_fac_b;
          }
        }
      }
    }
    if (lxa == lxb and lya == lyb)
    {
      SpherHarm(Ka, na, lxa, lya, La, Ma, angle, sphere_1, arg_vals);
      SpherHarm(Kb, nb, lxb, lyb, Lb, Mb, angle, sphere_2, arg_vals);
      result = 0.0;
      for (int idx = 0; idx < num_ang; ++idx)
      {
        tmp = sin(angle[idx]);
        result += sphere_1[idx] * sphere_2[idx] * cos(angle[idx]) * tmp * tmp;
      }
      result *= d_angle / sqrt(2.);
      matrix_element += result;
    }
  }
  hypersphere_coulomb_lookup[key] = matrix_element;
  PetscLogEventEnd(hyper_coulomb_time, 0, 0, 0, 0);
  return matrix_element / r;
}

dcomp Hamiltonian::GetHypersphereLaser(std::vector< PetscInt >& idx_array)
{
  dcomp ret_val(0.0, 0.0);
  double r = x_value[2][idx_array[2 * 2]];

  /* z axis */
  ret_val = GetHypersphereLaserVal(eigen_values[idx_array[2]],
                                   eigen_values[idx_array[2 + 1]], r);

  return ret_val;
}

double Hamiltonian::GetHypersphereLaserVal(int* lambda_a, int* lambda_b,
                                           double r)
{
  PetscLogEventBegin(hyper_laser_time, 0, 0, 0, 0);
  num_ang += num_ang % 2;
  double d_angle, matrix_element, result, m_sum, tmp_sin, tmp_cos;
  int Ka, na, lxa, lya, La, Ma, Kb, nb, lxb, lyb, Lb, Mb;
  std::string key;
  matrix_element = 0.0;
  Ka             = lambda_a[0];
  na             = lambda_a[1];
  lxa            = lambda_a[2];
  lya            = lambda_a[3];
  La             = lambda_a[4];
  Ma             = lambda_a[5];
  Kb             = lambda_b[0];
  nb             = lambda_b[1];
  lxb            = lambda_b[2];
  lyb            = lambda_b[3];
  Lb             = lambda_b[4];
  Mb             = lambda_b[5];
  key = to_string(Ka) + "_" + to_string(na) + "_" + to_string(lxa) + "_" +
        to_string(lya) + "_" + to_string(La) + "_" + to_string(Kb) + "_" +
        to_string(nb) + "_" + to_string(lxb) + "_" + to_string(lyb) + "_" +
        to_string(Lb) + "_" + to_string(num_ang);

  /* check to see if this has been calculated already */
  if (hypersphere_laser_lookup.count(key) == 1)
  {
    PetscLogEventEnd(hyper_laser_time, 0, 0, 0, 0);
    return hypersphere_laser_lookup[key] * r;
  }

  if (lxa == lxb and Ma == Mb and abs(lya - lyb) == 1 and abs(La - Lb) == 1)
  {
    d_angle = pi / (2 * num_ang);
    for (int idx = 0; idx < num_ang; ++idx)
    {
      angle[idx]    = idx * d_angle + d_angle / 2.;
      arg_vals[idx] = cos(2. * angle[idx]);
    }
    SpherHarm(Ka, na, lxa, lya, La, Ma, angle, sphere_1, arg_vals);
    SpherHarm(Kb, nb, lxb, lyb, Lb, Mb, angle, sphere_2, arg_vals);
    result = 0.0;
    for (int idx = 0; idx < num_ang; ++idx)
    {
      tmp_sin = sin(angle[idx]);
      tmp_sin *= tmp_sin * tmp_sin;
      tmp_cos = cos(angle[idx]);
      tmp_cos *= tmp_cos;
      result += sphere_1[idx] * sphere_2[idx] * tmp_cos * tmp_sin;
    }
    result *= d_angle;
    if (Ma != 0 or Mb != 0)
    {
      EndRun(
          "Hyperspherical laser opperator only supports M=0.\n"
          "A sum over m is needed for M!=0.");
    }
    result *= sqrt(((2. * lya + 1.)) / (2. * (2. * lyb + 1.)));
    m_sum = 0;
    for (int cur_m_val = -1 * min(La, Lb); cur_m_val <= min(La, Lb);
         ++cur_m_val)
    {
      m_sum += ClebschGordanCoef(lya, 1, lyb, 0, 0, 0) *
               ClebschGordanCoef(lya, 1, lyb, cur_m_val, 0, cur_m_val) *
               ClebschGordanCoef(lxa, lya, La, -cur_m_val, cur_m_val, Ma) *
               ClebschGordanCoef(lxb, lyb, Lb, -cur_m_val, cur_m_val, Mb);
    }
    result *= m_sum;

    matrix_element += result;
  }
  hypersphere_laser_lookup[key] = matrix_element;
  PetscLogEventEnd(hyper_laser_time, 0, 0, 0, 0);
  return matrix_element * r;
}

/* get nuclear term for rbf grid */
dcomp Hamiltonian::GetNucleiTerm(PetscInt idx)
{
  dcomp nuclei(0.0, 0.0);
  double r_soft;
  double r;
  double tmp;
  double tmp_soft;
  /* loop over each electron */
  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    /* loop over each nuclei */
    for (PetscInt nuclei_idx = 0; nuclei_idx < num_nuclei; ++nuclei_idx)
    {
      if (coordinate_system_idx == 3)
      {
        EndRun("This should only be used for RBF");
      }
      r_soft = SoftCoreDistance(location[nuclei_idx], idx);
      r      = EuclideanDistance(location[nuclei_idx], idx);

      /* Coulomb term */
      nuclei -= dcomp(z[nuclei_idx] / r_soft, 0.0);

      /* Gaussian Donuts */
      for (PetscInt i = 0; i < gaussian_size[nuclei_idx]; ++i)
      {
        tmp = gaussian_decay_rate[nuclei_idx][i] *
              (r - gaussian_r_0[nuclei_idx][i]);
        nuclei -= dcomp(
            gaussian_amplitude[nuclei_idx][i] * exp(-0.5 * (tmp * tmp)), 0.0);
      }

      /* Exponential Donuts */
      for (PetscInt i = 0; i < exponential_size[nuclei_idx]; ++i)
      {
        tmp = exponential_decay_rate[nuclei_idx][i] *
              std::abs(r - exponential_r_0[nuclei_idx][i]);
        nuclei -= dcomp(exponential_amplitude[nuclei_idx][i] * exp(-tmp), 0.0);
      }

      /* Square Well Donuts */
      for (PetscInt i = 0; i < square_well_size[nuclei_idx]; ++i)
      {
        /* only apply square well between r_0 and r_0+width */
        if ((r >= square_well_r_0[nuclei_idx][i]) and
            (r <= (square_well_r_0[nuclei_idx][i] +
                   square_well_width[nuclei_idx][i])))
        {
          nuclei -= dcomp(square_well_amplitude[nuclei_idx][i], 0.0);
        }
      }

      /* Yukawa Donuts */
      for (PetscInt i = 0; i < yukawa_size[nuclei_idx]; ++i)
      {
        tmp = yukawa_decay_rate[nuclei_idx][i] *
              std::abs(r - yukawa_r_0[nuclei_idx][i]);
        tmp_soft = std::abs(r_soft - yukawa_r_0[nuclei_idx][i]);
        nuclei -=
            dcomp(yukawa_amplitude[nuclei_idx][i] * exp(-tmp) / tmp_soft, 0.0);
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

dcomp Hamiltonian::GetCentrifugalTerm(std::vector< PetscInt >& idx_array)
{
  dcomp centrifugal_val(0.0, 0.0);
  double l_val;

  for (PetscInt elec_idx = 0; elec_idx < num_electrons; ++elec_idx)
  {
    if (coordinate_system_idx == 4)
    {
      l_val = eigen_values[idx_array[2 * (1 + elec_idx * num_dims)]][0];
      centrifugal_val += dcomp(
          (l_val * (l_val + 4) + (15. / 4.)) /
              (2.0 * x_value[2][idx_array[2 * (2 + elec_idx * num_dims)]] *
               x_value[2][idx_array[2 * (2 + elec_idx * num_dims)]]),
          0.0);
    }
    else if (coordinate_system_idx == 3)
    {
      l_val = l_values[idx_array[2 * (1 + elec_idx * num_dims)]];
      centrifugal_val += dcomp(
          l_val * (l_val + 1) /
              (2.0 * x_value[2][idx_array[2 * (2 + elec_idx * num_dims)]] *
               x_value[2][idx_array[2 * (2 + elec_idx * num_dims)]]),
          0.0);
    }
  }
  return centrifugal_val;
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
    diff = x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_1 * num_dims)]] -
           x_value[dim_idx][idx_array[2 * (dim_idx + elec_idx_2 * num_dims)]];
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
  for (PetscInt iter = 0; iter < elec_idx; ++iter)
  {
    offset *= num_psi_build;
  }
  for (PetscInt iter = 0; iter < dim_idx; ++iter)
  {
    /* first offset is num_x[num_dims-1] and then next is
     * num_x[num_dims-1]*num_x[num_dims-2] and so on*/
    offset *= num_x[num_dims - 1 - iter];
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
  PetscLogEventBegin(idx_array_time, 0, 0, 0, 0);
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
  PetscLogEventEnd(idx_array_time, 0, 0, 0, 0);
  return idx_array;
}

std::vector< PetscInt > Hamiltonian::GetDiffArray(
    std::vector< PetscInt >& idx_array)
{
  std::vector< PetscInt > diff_array;
  PetscLogEventBegin(diff_array_time, 0, 0, 0, 0);
  /* Calculated difference between i and j indexes */
  for (PetscInt i = 0; i < num_dims * num_electrons; ++i)
  {
    diff_array.push_back(idx_array[2 * i + 1] - idx_array[2 * i]);
  }
  PetscLogEventEnd(diff_array_time, 0, 0, 0, 0);
  return diff_array;
}

Mat* Hamiltonian::GetTimeIndependent(bool ecs, PetscInt l_val)
{
  if (ecs)
  {
    MatCopy(hamiltonian_0_ecs, hamiltonian, SAME_NONZERO_PATTERN);
    return &hamiltonian;
  }
  else
  {
    if (current_l_val != l_val)
    {
      CalculateHamlitonian0(l_val);
    }
    return &hamiltonian_0;
  }
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
