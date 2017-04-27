#include "Simulation.h"

Simulation::Simulation(Hamiltonian &h, Wavefunction &w, Pulse &pulse_in,
                       HDF5Wrapper &h_file, ViewWrapper &v_file, Parameters &p)
{
  if (world.rank() == 0) std::cout << "Creating Simulation\n";
  hamiltonian  = &h;
  wavefunction = &w;
  pulse        = &pulse_in;
  h5_file      = &h_file;
  viewer_file  = &v_file;
  parameters   = &p;
  time         = pulse_in.GetTime();
  time_length  = pulse_in.GetMaxPulseLength();

  if (world.rank() == 0) std::cout << "Simulation Created\n";
}

void Simulation::Propagate()
{
  if (world.rank() == 0) std::cout << "\nPropagating in time\n";
  clock_t t;
  /* if we are converged */
  bool converged = false;
  /* error in norm */
  double norm = 1.0;
  /* iteration */
  int i = 1;
  /* steps in each direction */
  double *delta_x = parameters->delta_x.get();
  /* how often do we write data */
  int write_frequency = parameters->GetWriteFrequencyPropagation();
  /* pointer to actual psi in wavefunction object */
  psi = wavefunction->GetPsi();
  Vec psi_right;
  Vec psi_old;
  /* Allocate space for psi_old */
  VecDuplicate(*psi, &psi_right);
  VecDuplicate(*psi, &psi_old);
  /* time step */
  double delta_t = parameters->GetDeltaT();
  /* factor = i*(-i*dt/2) */
  dcomp factor = dcomp(0.0, 1.0) * dcomp(delta_t / 2.0, 0.0);
  KSP ksp; /* solver for Ax=b */
  /* Create the solver */
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOptionsPrefix(ksp, "prop_");
  KSPConvergedReason reason; /* reason for convergence check */
  /* time independent Hamiltonian */
  Mat *h = hamiltonian->GetTimeIndependent();
  /* left matrix in Ax=Cb it would be A */
  Mat left;  /* matrix on left side of Ax=b */
  Mat right; /* matrix on left side of Ax=Cb */
  MatDuplicate(*(hamiltonian->GetTimeIndependent()), MAT_DO_NOT_COPY_VALUES,
               &left);
  MatDuplicate(*(hamiltonian->GetTimeIndependent()), MAT_DO_NOT_COPY_VALUES,
               &right);

  if (world.rank() == 0)
    std::cout << "Total writes: " << time_length / write_frequency
              << "\nStarting propagation\n"
              << std::flush;

  /* Checkpoint file before propagation starts */
  wavefunction->Checkpoint(*h5_file, *viewer_file, 0.0);
  t = clock();
  for (i = 1; i < time_length; i++)
  {
    h = hamiltonian->GetTotalHamiltonian(i);

    MatCopy(*h, left, SAME_NONZERO_PATTERN);
    MatScale(left, factor);
    MatShift(left, 1.0);

    MatCopy(*h, right, SAME_NONZERO_PATTERN);
    MatScale(right, -1.0 * factor);
    MatShift(right, 1.0);

    /* Get psi_right side */
    MatMult(right, *psi, psi_right);

    KSPSetOperators(ksp, left, left);
    KSPSetTolerances(ksp, 1.e-15, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);
    /* Solve Ax=b */
    KSPSolve(ksp, psi_right, *psi);

    /* Check for Divergence*/
    KSPGetConvergedReason(ksp, &reason);
    if (reason < 0)
    {
      EndRun("Divergence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }

    /* only checkpoint so often */
    if (i % write_frequency == 0)
    {
      norm = wavefunction->Norm();
      if (world.rank() == 0)
        std::cout << "\nIteration: " << i << "\nPulse ends: " << time_length
                  << "\n"
                  << "Average time for time-step: "
                  << ((float)clock() - t) / (CLOCKS_PER_SEC * write_frequency)
                  << "\nNorm: " << norm << "\n"
                  << std::flush;
      /* write a checkpoint */
      wavefunction->Checkpoint(*h5_file, *viewer_file, time[i]);
      t = clock();
    }
  }

  /* Save frame after pulse ends*/
  wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * i);

  if (world.rank() == 0)
    std::cout << "\nPropagating until norm stops changing\n";

  h = hamiltonian->GetTimeIndependent();

  MatCopy(*h, left, SAME_NONZERO_PATTERN);
  MatScale(left, factor);
  MatShift(left, 1.0);

  MatCopy(*h, right, SAME_NONZERO_PATTERN);
  MatScale(right, -1.0 * factor);
  MatShift(right, 1.0);

  KSPSetOperators(ksp, left, left);
  KSPSetTolerances(ksp, 1.e-15, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  KSPSetFromOptions(ksp);

  while (!converged) /* Should I add an upper bound to this? */
  {
    /* copy old state for convergence */
    if (i % write_frequency == 0)
    {
      VecCopy(*psi, psi_old);
    }
    /* Get psi_right side */
    MatMult(right, *psi, psi_right);

    /* Solve Ax=b */
    KSPSolve(ksp, psi_right, *psi);

    /* Check for Divergence*/
    KSPGetConvergedReason(ksp, &reason);
    if (reason < 0)
    {
      EndRun("Divergence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }

    /* only checkpoint so often */
    if (i % write_frequency == 0)
    {
      norm = wavefunction->Norm();
      if (world.rank() == 0)
        std::cout << "\nIteration: " << i << "\nPulse ended: " << time_length
                  << "\n"
                  << "Average time for time-step: "
                  << ((float)clock() - t) / (CLOCKS_PER_SEC * write_frequency)
                  << "\nNorm: " << norm << "\n"
                  << std::flush;

      norm -= wavefunction->Norm(psi_old, delta_x[0]);
      norm = std::abs(norm);
      if (world.rank() == 0) std::cout << "Norm error: " << norm << "\n";
      if (norm < 1e-14)
      {
        converged = true;
        if (world.rank() == 0) std::cout << "Converged!!!\n";
      }
      /* write a checkpoint */
      wavefunction->Checkpoint(*h5_file, *viewer_file, delta_t * i);
      t = clock();
    }
    i++;
  }

  VecDestroy(&psi_right);
  VecDestroy(&psi_old);
  MatDestroy(&left);
  MatDestroy(&right);
  KSPDestroy(&ksp);
}

void Simulation::PowerMethod(int num_states, int return_state_idx)
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
  int i = 1;
  int write_frequency =
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

  std::vector<Vec> states; /* vector of currently converged states */
  Vec psi_old;             /* keeps track of last psi */
  Vec psi_tmp;             /* Used to place copies in states */
  Mat left;                /* matrix on left side of Ax=b */

  KSP ksp;                   /* solver for Ax=b */
  KSPConvergedReason reason; /* reason for convergence check */

  /* allocate mem for left */
  MatDuplicate(*(hamiltonian->GetTimeIndependent()), MAT_DO_NOT_COPY_VALUES,
               &left);

  /* Create the solver */
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOptionsPrefix(ksp, "eigen_");

  /* pointer to actual psi in wavefunction object */
  psi = wavefunction->GetPsi();

  /* Allocate space for psi_old */
  VecDuplicate(*psi, &psi_old);

  /* loop over number of states wanted */
  for (int iter = 0; iter < num_states; iter++)
  {
    /* Get Hamiltonian */
    MatCopy(*(hamiltonian->GetTimeIndependent()), left, SAME_NONZERO_PATTERN);
    /* Shift by eigen value */
    MatShift(left, -1.0 * state_energy[iter]);

    /* Tell user the guess */
    if (world.rank() == 0)
      std::cout << "Looking for state with Energy Guess: " << state_energy[iter]
                << "\n";

    /* Put matrix in solver
     * do this outside the loop since left never changes */
    KSPSetOperators(ksp, left, left);
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
    t = clock();
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
        converged = CheckConvergance(psi[0], psi_old, parameters->GetTol());
        /* save this psi to ${target}.h5 */
        energy = wavefunction->GetEnergy(hamiltonian->GetTimeIndependent());
        if (world.rank() == 0)
          std::cout << "Energy: " << energy << "\n" << std::flush;
        // /* write a checkpoint */
        // wavefunction->Checkpoint(*h5_file, *viewer_file, i /
        // write_frequency);
        if (world.rank() == 0)
          std::cout << "Time: "
                    << ((float)clock() - t) / (CLOCKS_PER_SEC * write_frequency)
                    << "\n"
                    << std::flush;
        t = clock();
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
    CheckpointState(h_states_file, v_states_file, iter);

    /* new Gaussian guess */
    wavefunction->ResetPsi();

    /* reset for next state */
    converged = false;
    if (world.rank() == 0) std::cout << "\n";
  }

  /* set psi to the return state */
  VecCopy(states[return_state_idx], *psi);
  wavefunction->Normalize();

  /* Clean up after ourselves */
  /* DO NOT delete psi_tmp since it is the same as states[states.size()-1] which
   * is being deleted already */
  KSPDestroy(&ksp);
  MatDestroy(&left);
  VecDestroy(&psi_old);
  for (int i = 0; i < states.size(); ++i)
  {
    VecDestroy(&states[i]);
  }
}

bool Simulation::CheckConvergance(Vec &psi_1, Vec &psi_2, double tol)
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

/* for details see:
 * https://ocw.mit.edu/courses/mathematics/18-335j-introduction-to-numerical-methods-fall-2010/lecture-notes/MIT18_335JF10_lec10a_hand.pdf
 * Assumes all states are orthornormal and applies modified gram-schmit to
 * psi
 */
void Simulation::ModifiedGramSchmidt(std::vector<Vec> &states)
{
  int size = states.size();
  dcomp coef;
  for (int i = 0; i < size; i++)
  {
    VecDot(*psi, states[i], &coef);
    VecAXPY(*psi, -1.0 * coef, states[i]);
  }
}

void Simulation::CheckpointState(HDF5Wrapper &h_file, ViewWrapper &v_file,
                                 int write_idx)
{
  wavefunction->Normalize();
  wavefunction->CheckpointPsi(v_file, write_idx);
  h_file.WriteObject(wavefunction->GetEnergy(hamiltonian->GetTimeIndependent()),
                     "/Energy", "Energy of the corresponding state", write_idx);
}
