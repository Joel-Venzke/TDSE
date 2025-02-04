/**
 * @file Simulation.cpp
 * @brief Propagation and Eigen State calculations
 * @author Joel Venzke
 * @date 06/13/2017
 */

#include "Simulation.h"

/**
 * @brief Constructor for the simulation class
 *
 * @param hamiltonian_in hamiltonian class
 * @param w wavefunction
 * @param pulse_in pulse used in this simulation
 * @param h_file hdf5 wrapper
 * @param v_file petsc view wrapper
 * @param p parameter class
 */
Simulation::Simulation(Hamiltonian &hamiltonian_in, Wavefunction &w,
                       Pulse &pulse_in, HDF5Wrapper &h_file,
                       ViewWrapper &v_file, Parameters &p)
{
  if (world.rank() == 0) std::cout << "Creating Simulation\n";
  hamiltonian           = &hamiltonian_in;
  wavefunction          = &w;
  pulse                 = &pulse_in;
  h5_file               = &h_file;
  viewer_file           = &v_file;
  parameters            = &p;
  time                  = pulse_in.GetTime();
  time_length           = pulse_in.GetMaxPulseLength();
  coordinate_system_idx = p.GetCoordinateSystemIdx();
  num_x                 = w.GetNumX();

  /* allocated left and right */
  h = hamiltonian->GetTimeIndependent();
  MatDuplicate(*(hamiltonian->GetTimeIndependent()), MAT_DO_NOT_COPY_VALUES,
               &left);
  MatDuplicate(left, MAT_DO_NOT_COPY_VALUES, &right);
  psi = wavefunction->GetPsi();
  if (coordinate_system_idx == 3)
  {
    l_max     = p.GetLMax();
    m_max     = p.GetMMax();
    l_values  = wavefunction->GetLValues();
    m_values  = wavefunction->GetMValues();
    psi_small = wavefunction->GetPsiSmall();
  }
  else
  {
    l_max     = 0;
    m_max     = 0;
    l_values  = NULL;
    m_values  = NULL;
    psi_small = NULL;
  }
  VecDuplicate(*psi, &psi_right);

  /* Create the solver */
  KSPCreate(PETSC_COMM_WORLD, &ksp);

  PetscLogEventRegister("Time Step", PETSC_VIEWER_CLASSID, &time_step);
  PetscLogEventRegister("Hamiltonian", PETSC_VIEWER_CLASSID, &create_matrix);
  PetscLogEventRegister("Observables", PETSC_VIEWER_CLASSID,
                        &create_observables);
  PetscLogEventRegister("Checkpoint", PETSC_VIEWER_CLASSID, &create_checkpoint);

  if (world.rank() == 0) std::cout << "Simulation Created\n";
}

/**
 * @brief Main time propagation routine
 * @details Propagates in time using the whole Hamiltonian for each time step
 */
void Simulation::Propagate()
{
  if (world.rank() == 0) std::cout << "\nPropagating in time\n";
  clock_t t;
  clock_t step_time;
  std::ofstream timing_file;
  if (world.rank() == 0)
  {
    timing_file.open("timing.log");
    timing_file << "Iteration (time)\n";
  }
  /* if we are converged */
  bool converged = false;
  /* If we need end of pulse checkpoint */
  bool checkpoint_eof = true;
  /* error in norm */
  double norm = 1.0;
  /* how often do we write data */
  PetscInt write_frequency_checkpoint =
      parameters->GetWriteFrequencyCheckpoint();
  PetscInt write_frequency_observables =
      parameters->GetWriteFrequencyObservables();
  PetscInt free_propagate = parameters->GetFreePropagate();
  PetscInt i              = 1;
  /* pointer to actual psi in wavefunction object */
  psi = wavefunction->GetPsi();
  Vec psi_old;
  /* Allocate space for psi_old */
  VecDuplicate(*psi, &psi_old);
  /* time step */
  double delta_t = parameters->GetDeltaT();
  KSPDestroy(&ksp);
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOptionsPrefix(ksp, "prop_");
  /* time independent Hamiltonian */

  if (parameters->GetRestart() == 1)
  {
    /* set current iteration */
    i = std::round(h5_file->GetLast("/Wavefunction/time") / delta_t);
    i++;
    /* only checkpoint end of pulse if the simulation isn't in free propagation
     * already*/
    if (i > time_length)
    {
      checkpoint_eof = false;
    }
    if (world.rank() == 0)
    {
      std::cout << "Restarting at time step: " << i << "\n";
    }
  }

  if (world.rank() == 0)
    std::cout << "Total writes: "
              << (time_length + free_propagate) / write_frequency_checkpoint + 1
              << "\nTotal time steps: " << time_length + free_propagate
              << std::endl;

  /* Checkpoint file before propagation starts */
  if (parameters->GetRestart() != 1)
  {
    wavefunction->Checkpoint(*h5_file, *viewer_file, 0.0);
  }

  if (world.rank() == 0)
  {
    std::cout << "Starting propagation\n\n" << std::flush;
    t         = clock();
    step_time = clock();
  }
  for (; i < time_length; i++)
  {
    PetscLogEventBegin(time_step, 0, 0, 0, 0);
    CrankNicolson(delta_t, i, -1);

    /* only calculate observables so often */
    if (write_frequency_observables > 0 and
        i % write_frequency_observables == 0)
    {
      PetscLogEventBegin(create_observables, 0, 0, 0, 0);
      /* write a checkpoint */
      wavefunction->Checkpoint(*h5_file, *viewer_file, time[i], 1);
      PetscLogEventEnd(create_observables, 0, 0, 0, 0);
    }

    /* only checkpoint so often */
    if (i % write_frequency_checkpoint == 0)
    {
      PetscLogEventBegin(create_checkpoint, 0, 0, 0, 0);

      if (world.rank() == 0)
        std::cout << "\nIteration:  " << i << "\nPulse ends: " << time_length
                  << "\n"
                  << "Average time for time-step: "
                  << ((float)clock() - t) / CLOCKS_PER_SEC /
                         write_frequency_checkpoint
                  << "\n"
                  << std::flush;
      /* write a checkpoint */
      wavefunction->Checkpoint(*h5_file, *viewer_file, time[i]);
      if (world.rank() == 0) t = clock();
      PetscLogEventEnd(create_checkpoint, 0, 0, 0, 0);
    }

    if (world.rank() == 0)
    {
      timing_file << "Iteration: " << i << "\t("
                  << ((float)clock() - step_time) / CLOCKS_PER_SEC << "s)\n";
      step_time = clock();
    }
    PetscLogEventEnd(time_step, 0, 0, 0, 0);
  }

  if (checkpoint_eof)
  {
    /* Save frame after pulse ends*/
    wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * (i - 1));
  }

  if (free_propagate == -1) /* until norm stops changing */
  {
    if (world.rank() == 0)
      std::cout << "\nPropagating until norm stops changing\n";
    while (!converged) /* Should I add an upper bound to this? */
    {
      /* copy old state for convergence */
      if (i % write_frequency_checkpoint == 0)
      {
        VecCopy(*psi, psi_old);
      }

      CrankNicolson(delta_t, -1, -1);

      /* only calculate observables so often */
      if (i % write_frequency_observables == 0)
      {
        /* write a checkpoint */
        wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * i, 1);
      }

      /* only checkpoint so often */
      if (i % write_frequency_checkpoint == 0)
      {
        norm = wavefunction->Norm();
        if (world.rank() == 0)
          std::cout << "\nIteration: " << i << "\nPulse ended: " << time_length
                    << "\n"
                    << "Average time for time-step: "
                    << ((float)clock() - t) /
                           (CLOCKS_PER_SEC * write_frequency_checkpoint)
                    << "\nNorm: " << norm << "\n"
                    << std::flush;

        norm -= wavefunction->Norm(psi_old, 0.0);
        norm = std::abs(norm);
        if (world.rank() == 0) std::cout << "Norm error: " << norm << "\n";
        if (norm < 1e-14)
        {
          converged = true;
        }
        /* write a checkpoint */
        wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * i);
        if (world.rank() == 0) t = clock();
      }
      if (world.rank() == 0)
      {
        timing_file << "Iteration: " << i << "\t("
                    << ((float)clock() - step_time) / CLOCKS_PER_SEC << "s)\n"
                    << std::flush;
        step_time = clock();
      }
      i++;
    }
  }
  else /* Fixed number of steps */
  {
    if (world.rank() == 0)
      std::cout << "\nPropagating until step: " << time_length + free_propagate
                << "\n";
    while (i < time_length + free_propagate)
    {
      CrankNicolson(delta_t, -1, -1);

      /* only calculate observables so often */
      if (i % write_frequency_observables == 0)
      {
        /* write a checkpoint */
        wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * i, 1);
      }

      /* only checkpoint so often */
      if (i % write_frequency_checkpoint == 0)
      {
        if (world.rank() == 0)
          std::cout << "\nIteration: " << i
                    << "\nSimulation Ends: " << time_length + free_propagate
                    << "\n"
                    << "Average time for time-step: "
                    << ((float)clock() - t) /
                           (CLOCKS_PER_SEC * write_frequency_checkpoint)
                    << "\n"
                    << std::flush;
        /* write a checkpoint */
        wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * i);
        if (world.rank() == 0) t = clock();
      }
      if (world.rank() == 0)
      {
        timing_file << "Iteration: " << i << "\t("
                    << ((float)clock() - step_time) / CLOCKS_PER_SEC << "s)\n"
                    << std::flush;
        step_time = clock();
      }
      i++;
    }
    /* Save last Wavefunction since it might not end on a write frequency*/
    if (free_propagate > 0)
    {
      wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * (i - 1));
    }
  }

  if (world.rank() == 0)
  {
    timing_file.close();
  }

  VecDestroy(&psi_old);
}

/**
 * @brief Uses the "target".h5 file to read in the ground state
 * @details Uses the "target".h5 file to read in the ground state. It also makes
 * sure you have enough states for the projections
 *
 * @param num_states The number of states you wish to use for projections
 * @param return_state_idx the index of the state you want the Wavefunction
 * class to be set to upon return
 */
void Simulation::FromFile(PetscInt num_states)
{
  if (coordinate_system_idx == 3)
  {
    wavefunction->LoadPsi(
        parameters->GetTarget() + ".h5", num_states,
        parameters->GetNumStartState(), parameters->GetStartStateIdx(),
        parameters->GetStartStateLIdx(), parameters->GetStartStateMIdx(),
        parameters->GetStartStateAmplitude(), parameters->GetStartStatePhase());
  }
  else
  {
    wavefunction->LoadPsi(
        parameters->GetTarget() + ".h5", num_states,
        parameters->GetNumStartState(), parameters->GetStartStateIdx(),
        parameters->GetStartStateAmplitude(), parameters->GetStartStatePhase());
  }
}

/**
 * @brief Uses SLEPC to calculate Eigen States
 * @details Uses SLEPC to calculate Eigen States
 *
 * @param num_states The number of states you wish to calculate
 * @param return_state_idx the index of the state you want the Wavefunction
 * class to be set to upon return
 */
void Simulation::EigenSolve(PetscInt num_states)
{
  if (world.rank() == 0)
    std::cout << "\nCalculating the lowest " << num_states
              << " eigenvectors using SLEPC\n"
              << std::flush;
  double tol = parameters->GetTol();
  dcomp eigen_real;
  int nconv;

  /* Files for */
  ViewWrapper v_states_file(parameters->GetTarget() + ".h5"); /* PETSC viewer */
  /* create file so that the format works with PETSC */
  v_states_file.Open();
  v_states_file.Close();
  HDF5Wrapper h_states_file(parameters->GetTarget() + ".h5"); /* HDF5 viewer */

  if (coordinate_system_idx == 3)
  {
    /* only the radial part is needed for the states */
    psi = wavefunction->GetPsiSmall();
    for (int l_val = 0; l_val < l_max + 1; ++l_val)
    {
      /* make sure we still have states to calculate */
      /* for every increase in l, there is one less state in n */
      if (num_states > l_val)
      {
        if (parameters->GetFieldMaxStates())
        {
          EndRun("Field max states not supported in spherical coordinates");
        }
        else
        {
          h = hamiltonian->GetTimeIndependent(false, l_val);
        }
        EPS eps; /* eigen solver */
        EPSCreate(PETSC_COMM_WORLD, &eps);
        EPSSetOperators(eps, *(h), NULL);
        EPSSetProblemType(eps, EPS_NHEP);
        EPSSetTolerances(eps, tol, PETSC_DECIDE);
        EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
        /* EPS doesn't converge unless we increase the subspace */
        if (coordinate_system_idx == 3 and num_states > 10)
        {
          EPSSetDimensions(eps, num_states - l_val, PETSC_DECIDE,
                           num_x[2] * 0.1);
        }
        else /* make it faster for small calculations */
        {
          EPSSetDimensions(eps, num_states - l_val, 300, PETSC_DECIDE);
        }
        /* set up initial space */
        if (l_val == 0)
        {
          wavefunction->RadialHGroundPsiSmall();
          EPSSetInitialSpace(eps, 1, psi);
        }
        EPSSetFromOptions(eps);
        EPSSolve(eps);
        EPSGetConverged(eps, &nconv);
        if (world.rank() == 0)
        {
          std::cout << "\n";
        }
        for (int j = 0; j < num_states - l_val; j++)
        {
          EPSGetEigenpair(eps, j, &eigen_real, NULL, *psi, NULL);
          if (world.rank() == 0)
          {
            std::cout << "Eigen (n,l):  " << j + 1 + l_val << "," << l_val
                      << "\t" << eigen_real.real() << "\t" << eigen_real.imag()
                      << "\n";
          }
          CheckpointSmallState(h_states_file, v_states_file, j, eigen_real,
                               l_val);
        }
        EPSDestroy(&eps);
      }
    }
  }
  else if (coordinate_system_idx == 4 or coordinate_system_idx == 5)
  {
    psi = wavefunction->GetPsi();
    EPS eps; /* eigen solver */
    EPSCreate(PETSC_COMM_WORLD, &eps);
    if (parameters->GetFieldMaxStates())
    {
      h = hamiltonian->GetTotalHamiltonian(pulse->GetFieldMaxIdx(), false);
    }
    else
    {
      h = hamiltonian->GetTimeIndependent(false);
    }
    EPSSetOperators(eps, *(h), NULL);
    EPSSetProblemType(eps, EPS_NHEP);
    EPSSetTolerances(eps, tol, PETSC_DECIDE);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    if (num_x[2] > 3000)
    {
      EPSSetDimensions(eps, num_states, PETSC_DECIDE, num_x[2] * 0.1);
    }
    else /* make it faster for small calculations */
    {
      EPSSetDimensions(eps, num_states, 300, PETSC_DECIDE);
    }
    EPSSetFromOptions(eps);
    EPSSolve(eps);
    EPSGetConverged(eps, &nconv);

    for (int j = 0; j < num_states; j++)
    {
      EPSGetEigenpair(eps, j, &eigen_real, NULL, *psi, NULL);
      if (world.rank() == 0)
        std::cout << "Eigen: " << j << "\t" << eigen_real << "\n";
      wavefunction->Normalize();
      CheckpointState(h_states_file, v_states_file, j, h);
    }

    EPSDestroy(&eps);
  }
  else
  {
    psi = wavefunction->GetPsi();
    EPS eps; /* eigen solver */
    EPSCreate(PETSC_COMM_WORLD, &eps);
    if (parameters->GetFieldMaxStates())
    {
      h = hamiltonian->GetTotalHamiltonian(pulse->GetFieldMaxIdx(), false);
    }
    else
    {
      h = hamiltonian->GetTimeIndependent(false);
    }
    EPSSetOperators(eps, *(h), NULL);
    EPSSetProblemType(eps, EPS_NHEP);
    EPSSetTolerances(eps, tol, PETSC_DECIDE);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    EPSSetDimensions(eps, num_states, PETSC_DECIDE, PETSC_DECIDE);
    EPSSetFromOptions(eps);
    EPSSolve(eps);
    EPSGetConverged(eps, &nconv);

    for (int j = 0; j < nconv; j++)
    {
      EPSGetEigenpair(eps, j, &eigen_real, NULL, *psi, NULL);
      if (world.rank() == 0)
        std::cout << "Eigen: " << j << "\t" << eigen_real << "\n";
      wavefunction->Normalize();
      CheckpointState(h_states_file, v_states_file, j, h);
    }

    EPSDestroy(&eps);
  }
  FromFile(num_states);
}

/**
 * @brief Calculates one timestep of Crank Nicolson
 * @details Calculates one timestep of Crank Nicolson
 *
 * @param dt delta t (time step) used in the Crank Nicolson method
 * @param time_idx index into the laser pulse (-1 for time independent)
 * @param dim_idx dimension used in split operator (-1 for full Hamiltonian)
 */
void Simulation::CrankNicolson(double dt, PetscInt time_idx, PetscInt dim_idx)
{
  static PetscInt old_time_idx = -2;
  static PetscInt old_dim_idx  = -2;
  if (time_idx != old_time_idx or dim_idx != old_dim_idx)
  {
    PetscLogEventBegin(create_matrix, 0, 0, 0, 0);
    /* factor = (-i*dt/2) */
    dcomp factor = dcomp(0.0, 1.0) * dcomp(dt / 2.0, 0.0);
    if (time_idx < 0)
    {
      h = hamiltonian->GetTimeIndependent(dim_idx);
    }
    else
    {
      h = hamiltonian->GetTotalHamiltonian(time_idx, dim_idx);
    }

    MatCopy(*h, left, SAME_NONZERO_PATTERN);
    MatScale(left, factor);
    MatShift(left, 1.0);

    MatCopy(*h, right, SAME_NONZERO_PATTERN);
    MatScale(right, -1.0 * factor);
    MatShift(right, 1.0);

    KSPSetOperators(ksp, left, left);
    KSPSetTolerances(ksp, 1.e-15, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);

    old_dim_idx  = dim_idx;
    old_time_idx = time_idx;
    PetscLogEventEnd(create_matrix, 0, 0, 0, 0);
  }

  /* Get psi_right side */
  MatMult(right, *psi, psi_right);

  /* Solve Ax=b */
  KSPSolve(ksp, psi_right, *psi);

  /* Check for Divergence*/
  KSPGetConvergedReason(ksp, &reason);
  if (reason < 0)
  {
    std::cout << "Time step: " << time_idx << "\n";
    std::cout << "Divergence Reason: " << KSPConvergedReason(reason);
    EndRun("Divergence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  }
}

/**
 * @brief Inverse shifted power method for Eigen State calculations
 * @details Uses the Inverse shifted power method for Eigen State calculations.
 *
 * @param num_states Number of ground states to calculate
 * @param return_state_idx the index of the state you want the Wavefunction
 * class to be set to upon return
 */
void Simulation::PowerMethod(PetscInt num_states)
{
  if (world.rank() == 0)
    std::cout << "\nCalculating the lowest " << num_states
              << " eigenvectors using power method\n"
              << std::flush;
  clock_t t;
  bool converged   = false; /* if we are converged */
  bool gram_schmit = false; /* if we using gram_schmit */
  /* write index for checkpoints, Starts at 1 to avoid writing on first
   * iteration*/
  PetscInt i = 1;
  PetscInt write_frequency =
      parameters
          ->GetWriteFrequencyEigenState(); /* how often do we write data */
  double *state_energy = parameters->state_energy.get(); /* energy guesses */
  double energy; /* stores energy for IO */
  double norm;   /* stores norm for IO */

  /* Files for */
  ViewWrapper v_states_file(parameters->GetTarget() + ".h5"); /* PETSC viewer */
  /* create file so that the format works with PETSC */
  v_states_file.Open();
  v_states_file.Close();
  HDF5Wrapper h_states_file(parameters->GetTarget() + ".h5"); /* HDF5 viewer */

  std::vector< Vec > states; /* vector of currently converged states */
  Vec psi_old;               /* keeps track of last psi */
  Vec psi_tmp;               /* Used to place copies in states */

  /* Create the solver */
  KSPDestroy(&ksp);
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOptionsPrefix(ksp, "eigen_");

  /* pointer to actual psi in wavefunction object */
  psi = wavefunction->GetPsi();

  /* Allocate space for psi_old */
  VecDuplicate(*psi, &psi_old);

  /* loop over number of states wanted */
  for (PetscInt iter = 0; iter < num_states; iter++)
  {
    if (parameters->GetFieldMaxStates())
    {
      h = hamiltonian->GetTotalHamiltonian(pulse->GetFieldMaxIdx(), false);
    }
    else
    {
      h = hamiltonian->GetTimeIndependent(false);
    }
    MatCopy(*(h), left, SAME_NONZERO_PATTERN);
    /* Shift by eigen value */
    MatShift(left, -1.0 * state_energy[iter]);

    /* Tell user the guess */
    if (world.rank() == 0)
      std::cout << "Looking for state with Energy Guess: " << state_energy[iter]
                << "\n";

    /* Put matrix in solver
     * do this outside the loop since left never changes */
    KSPSetOperators(ksp, left, left);
    KSPSetTolerances(ksp, 1.e-15, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    /* Allow command line options */
    KSPSetFromOptions(ksp);

    /* Do we need to use gram schmidt */
    if (iter > 0 and
        std::abs(state_energy[iter] - state_energy[iter - 1]) < 1e-10)
    {
      if (world.rank() == 0) std::cout << "Starting Gram Schmit\n";

      gram_schmit = true;

      /* This is needed because the Power method converges
       * extremely quickly */
      ModifiedGramSchmidt(states);
    }
    if (world.rank() == 0) t = clock();
    /* loop until error is small enough */
    while (!converged)
    {
      /* copy old state for convergence */
      VecCopy(*psi, psi_old);

      /* Solve Ax=b */
      KSPSolve(ksp, psi_old, *psi);

      /* Check for Divergence*/
      KSPGetConvergedReason(ksp, &reason);
      if (reason < 0)
      {
        EndRun("Divergence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      }

      /* used to get higher states */
      if (gram_schmit) ModifiedGramSchmidt(states);

      /* Re normalize wave function */
      wavefunction->Normalize();

      /* only checkpoint so often */
      if (i % write_frequency == 0)
      {
        /* check convergence criteria */
        converged = CheckConvergence(psi[0], psi_old, parameters->GetTol());
        /* save this psi to ${target}.h5 */
        energy = wavefunction->GetEnergy(h);
        if (world.rank() == 0)
          std::cout << "Energy: " << energy << "\n" << std::flush;
        /* write a checkpoint */
        // wavefunction->Checkpoint(*h5_file, *viewer_file, -1.0, true);
        if (world.rank() == 0)
          std::cout << "Time: "
                    << ((float)clock() - t) / (CLOCKS_PER_SEC * write_frequency)
                    << "\n"
                    << std::flush;
        if (world.rank() == 0) t = clock();
      }
      /* increment counter */
      i++;
    }

    /* make sure all states are orthonormal for mgs */
    VecNormalize(*psi, &norm);
    VecDuplicate(*psi, &psi_tmp);
    VecCopy(*psi, psi_tmp);
    states.push_back(psi_tmp);

    /* save this psi to ${target}.h5 */
    CheckpointState(h_states_file, v_states_file, iter, h);

    /* new Gaussian guess */
    wavefunction->ResetPsi();

    /* reset for next state */
    converged = false;
    if (world.rank() == 0) std::cout << "\n";
  }

  FromFile(num_states);

  /* Clean up after ourselves */
  /* DO NOT delete psi_tmp since it is the same as states[states.size()-1] which
   * is being deleted already */
  VecDestroy(&psi_old);
  for (PetscInt i = 0; i < states.size(); ++i)
  {
    VecDestroy(&states[i]);
  }
}

/**
 * @brief Checks if psi is converge or not
 * @details calculates |psi|^2 and the energy and ensures it is less than the
 * tol provided from the Parameter class
 *
 * @param psi_1 [description]
 * @param psi_2 [description]
 * @param tol [description]
 * @return [description]
 */
bool Simulation::CheckConvergence(Vec &psi_1, Vec &psi_2, double tol)
{
  Mat *h              = hamiltonian->GetTimeIndependent();
  double wave_error   = 0.0;
  double wave_error_2 = 0.0;
  double energy_error = 0.0;

  energy_error =
      wavefunction->GetEnergy(h, psi_1) - wavefunction->GetEnergy(h, psi_2);

  /* psi_2 - psi_1 */
  VecAXPY(psi_2, -1.0, psi_1);
  VecNorm(psi_2, NORM_2, &wave_error);

  /* we want psi_2+psi_1 so we need to add 2*psi_1 */
  VecAXPY(psi_2, 2.0, psi_1);
  VecNorm(psi_2, NORM_2, &wave_error_2);

  if (wave_error_2 < wave_error) wave_error = wave_error_2;

  if (world.rank() == 0)
    std::cout << "Wavefunction Error: " << wave_error
              << "\nEnergy Error: " << energy_error << "\nTolerance: " << tol
              << "\n"
              << std::flush;

  return wave_error < tol and std::abs(energy_error) < tol;
}

/*
 *
 * https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-fall-1710/lecture-notes/MIT18_335JF10_lec10a_hand.pdf
 * Assumes all states are orthornormal and applies modified gram-schmit to
 * psi
 */

/**
 * @brief Preforms modified gram schmidt on psi in the Wavefunction
 * @details
 * https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-fall-1710/lecture-notes/MIT18_335JF10_lec10a_hand.pdf
 * Assumes all states are orthornormal and applies modified gram-schmit to
 * psi
 *
 * @param states vector of orthonormal vectors that are to be projected out of
 * psi
 */
void Simulation::ModifiedGramSchmidt(std::vector< Vec > &states)
{
  PetscInt size = states.size();
  dcomp coef;
  for (PetscInt i = 0; i < size; i++)
  {
    VecDot(*psi, states[i], &coef);
    VecAXPY(*psi, -1.0 * coef, states[i]);
  }
}

/**
 * @brief Writes psi to a eigen state file
 * @details Used to save eigen states to a file
 *
 * @param h_file HDF5Wapper file
 * @param v_file ViewWapper file
 * @param write_idx Index of the eigen state
 * @param energy the Energy of the state
 * @param l_val the l_value of the state
 */
void Simulation::CheckpointSmallState(HDF5Wrapper &h_file, ViewWrapper &v_file,
                                      PetscInt write_idx, dcomp energy,
                                      PetscInt l_val)
{
  // wavefunction->ZeroPhasePsiSmall();
  wavefunction->CheckpointPsiSmall(v_file, write_idx, l_val);
  h_file.WriteObject(&energy, 1, "/Energy_l_" + std::to_string(l_val),
                     "Energy of the corresponding state", write_idx);
}

/**
 * @brief Writes psi to a eigen state file
 * @details Used to save eigen states to a file
 *
 * @param h_file HDF5Wapper file
 * @param v_file ViewWapper file
 * @param write_idx Index of the eigen state
 * @param cur_hamiltonian The Hamiltonian used to get the energy
 */
void Simulation::CheckpointState(HDF5Wrapper &h_file, ViewWrapper &v_file,
                                 PetscInt write_idx, Mat *cur_hamiltonian)
{
  wavefunction->Normalize();
  wavefunction->CheckpointPsi(v_file, write_idx);
  if (parameters->GetFieldMaxStates())
  {
    h_file.WriteObject(wavefunction->GetEnergy(cur_hamiltonian), "/Energy",
                       "Energy of the corresponding state", write_idx);
  }
  else
  {
    h_file.WriteObject(wavefunction->GetEnergy(cur_hamiltonian), "/Energy",
                       "Energy of the corresponding state", write_idx);
  }
}

/**
 * @brief Destructor
 */
Simulation::~Simulation()
{
  MatDestroy(&left);
  MatDestroy(&right);
  VecDestroy(&psi_right);
  KSPDestroy(&ksp);
}
