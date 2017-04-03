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
  // clock_t t;
  // /* if we are converged */
  // bool converged = false;
  // /* error in norm */
  // double error = 1.0;
  // /* iteration */
  // int i = 1;
  // /* size of psi */
  // int num_psi = wavefunction->get_num_psi();
  // /* steps in each direction */
  // double *dx = wavefunction->get_delta_x();
  // /* how often do we write data */
  // int write_frequency = parameters->get_write_frequency();
  // /* pointer to actual psi in wavefunction object */
  // psi                      = wavefunction->get_psi();
  // Eigen::VectorXcd psi_old = *psi;
  // /* time step */
  // double dt = parameters->get_delta_t();
  // /* factor = i*(-i*dx/2) */
  // dcomp factor = dcomp(0.0, 1.0) * dcomp(dt / 2.0, 0.0);
  // /* solver for Ax=b */
  // Eigen::SparseLU<Eigen::SparseMatrix<dcomp>> solver;
  // /* time independent Hamiltonian */
  // Eigen::SparseMatrix<dcomp> *h = hamiltonian->get_time_independent();
  // /* left matrix in Ax=Cb it would be A */
  // Eigen::SparseMatrix<dcomp> left = *h;
  // /* right matrix in Ax=Cb it would be C */
  // Eigen::SparseMatrix<dcomp> right = left;

  // std::cout << "Total writes: " << time_length / write_frequency;
  // std::cout << "\nSetting up solver\n" << std::flush;
  // solver.analyzePattern(left);
  // std::cout << "Starting propagation\n" << std::flush;
  // for (i = 1; i < time_length; i++)
  // {
  //   std::cout << "Iteration: " << i << " of ";
  //   std::cout << time_length << "\n";
  //   t    = clock();
  //   h    = hamiltonian->get_total_hamiltonian(i);
  //   left = (idenity[0] + factor * h[0]);
  //   left.makeCompressed();
  //   right = (idenity[0] - factor * h[0]);
  //   right.makeCompressed();
  //   solver.factorize(left);
  //   psi[0] = solver.solve(right * psi[0]);

  //   wavefunction->gobble_psi();

  //   /* only checkpoint so often */
  //   if (i % write_frequency == 0)
  //   {
  //     std::cout << "\nNorm: " << wavefunction->norm() << "\n";
  //     /* write a checkpoint */
  //     wavefunction->checkpoint(*file, time[i]);
  //   }
  //   std::cout << "Time-step: ";
  //   std::cout << ((float)clock() - t) / CLOCKS_PER_SEC << "\n";
  //   std::cout << std::flush;
  // }

  // std::cout << "Propagating until norm stops changing\n";

  // h    = hamiltonian->get_time_independent();
  // left = (idenity[0] + factor * h[0]);
  // left.makeCompressed();
  // right = (idenity[0] - factor * h[0]);
  // right.makeCompressed();
  // solver.factorize(left);

  // while (!converged)
  // {
  //   /* copy old state for convergence */
  //   psi_old = psi[0];
  //   psi[0]  = solver.solve(right * psi_old);

  //   wavefunction->gobble_psi();
  //   if (i % write_frequency == 0)
  //   {
  //     wavefunction->checkpoint(*file, time[i]);
  //     error = wavefunction->norm();
  //     std::cout << "Norm: " << error << "\n";
  //     error -= wavefunction->norm(psi_old.data(), num_psi, dx[0]);
  //     error = std::abs(error);
  //     std::cout << "Norm error: " << error << "\n";
  //     if (error < 1e-14)
  //     {
  //       converged = true;
  //     }
  //     /* write a checkpoint */
  //     wavefunction->checkpoint(*file, i * dx[0]);
  //   }
  //   i++;
  // }
}

void Simulation::PowerMethod(int num_states)
{
  if (world.rank() == 0)
    std::cout << "\nCalculating the lowest " << num_states
              << " eigenvectors using power method\n"
              << std::flush;
  clock_t t;
  /* if we are converged */
  bool converged   = false;
  bool gram_schmit = false;
  /* write index for checkpoints */
  int i = 1;
  /* how often do we write data */
  int write_frequency = parameters->GetWriteFrequency();
  /* keeps track of last psi */
  Vec psi_old;
  /* matrix on left side of Ax=b */
  Mat left;
  MatDuplicate(*(hamiltonian->GetTimeIndependent()), MAT_DO_NOT_COPY_VALUES,
               &left);
  /* vector of currently converged states */
  std::vector<Vec> states;
  /* energy guesses */
  double *state_energy = parameters->state_energy.get();
  /* solver for Ax=b */
  KSPConvergedReason reason;
  int its;
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetFromOptions(ksp);
  /* file for converged states */
  if (world.rank() == 0) std::cout << "file one" << std::flush;
  ViewWrapper v_states_file(parameters->GetTarget() + ".h5");
  v_states_file.Open();
  v_states_file.Close();
  HDF5Wrapper h_states_file(parameters->GetTarget() + ".h5");

  /* pointer to actual psi in wavefunction object */
  psi = wavefunction->GetPsi();
  VecDuplicate(*psi, &psi_old);

  /* loop over number of states wanted */
  for (int iter = 0; iter < 1; iter++)
  {
    /* left side of power method */
    MatCopy(*(hamiltonian->GetTimeIndependent()), left, SAME_NONZERO_PATTERN);
    MatShift(left, state_energy[iter]);
    MatView(left, PETSC_VIEWER_STDOUT_WORLD);
    std::cout << "Looking for state with Energy: ";
    std::cout << state_energy[iter] << "\n";

    /* do this outside the loop since left never changes */
    std::cout << "Setting up solver"
              << "\n"
              << std::flush;
    KSPSetOperators(ksp, left, left);

    // flg_ilu     = PETSC_FALSE;
    // flg_superlu = PETSC_FALSE;
    // PetscOptionsGetBool(NULL, NULL, "-use_superlu_lu", &flg_superlu,
    // NULL); PetscOptionsGetBool(NULL, NULL, "-use_superlu_ilu", &flg_ilu,
    // NULL); if (flg_superlu || flg_ilu)
    // {
    //   KSPSetType(ksp, KSPPREONLY);
    //   KSPGetPC(ksp, &pc);
    //   if (flg_superlu)
    //   {
    //     PCSetType(pc, PCLU);
    //   }
    //   else if (flg_ilu)
    //   {
    //     PCSetType(pc, PCILU);
    //   }
    //   if (world.size() == 1)
    //   {
    //     PCFactorSetMatSolverPackage(pc, MATSOLVERSUPERLU);
    //   }
    //   else
    //   {
    //     PCFactorSetMatSolverPackage(pc, MATSOLVERSUPERLU_DIST);
    //   }
    //   PCFactorSetUpMatSolverPackage(pc); /* call MatGetFactor() to create
    //   F
    //   */ PCFactorGetMatrix(pc, &F); if (world.size() == 1)
    //   {
    //     MatSuperluSetILUDropTol(F, 1.e-8);
    //   }
    std::cout << "Solver set up"
              << "\n"
              << std::flush;
    // if (iter > 0 and
    //     std::abs(state_energy[iter] - state_energy[iter - 1]) < 1e-10)
    // {
    //   std::cout << "Starting Gram Schmit\n";
    //   gram_schmit = true;
    //   /* This is needed because the Power method converges */
    //   /* extremely quickly */
    //   modified_gram_schmidt(states);
    // }

    if (world.rank() == 0) t = clock();
    /* loop until error is small enough */
    // while (!converged)
    while (i < 5)
    {
      /* copy old state for convergence */
      VecCopy(*psi, psi_old);
      KSPSolve(ksp, psi_old, *psi);
      KSPGetConvergedReason(ksp, &reason);
      if (world.rank() == 0)
      {
        if (reason < 0)
        {
          printf("Divergence.\n");
        }
        else
        {
          KSPGetIterationNumber(ksp, &its);
          printf("Convergence in %d iterations.\n", (int)its);
        }
      }

      /* used to get higher states */
      // if (gram_schmit) modified_gram_schmidt(states);

      wavefunction->Normalize();

      /* only checkpoint so often */
      if (i % write_frequency == 0)
      {
        /* check convergence criteria */
        converged = CheckConvergance(psi[0], psi_old, parameters->GetTol());
        /* save this psi to ${target}.h5 */
        std::cout << "Energy: "
                  << wavefunction->GetEnergy(hamiltonian->GetTimeIndependent())
                  << "\n"
                  << std::flush;
        // /* write a checkpoint */
        wavefunction->Checkpoint(*h5_file, *viewer_file, i / write_frequency);
      }
      /* increment counter */
      i++;
      // converged = true;
    }
    if (world.rank() == 0)
      std::cout << "Time: " << ((float)clock() - t) / CLOCKS_PER_SEC << "\n";
    /* make sure all states are orthonormal for mgs */
    // states.push_back(psi[0] / psi->norm());
    // /* save this psi to ${target}.h5 */
    // checkpoint_state(states_file, iter);
    //  new Gaussian guess
    // wavefunction->reset_psi();
    // /* reset for next state */
    // converged = false;
    // std::cout << "\n";
  }
  // psi[0] = states[states.size() - 1];
  // wavefunction->normalize();
  KSPDestroy(&ksp);
  VecDestroy(&psi_old);
}

bool Simulation::CheckConvergance(Vec &psi_1, Vec &psi_2, double tol)
{
  Mat *h              = hamiltonian->GetTimeIndependent();
  double wave_error   = 0.0;
  double wave_error_2 = 0.0;
  double energy_error = 0.0;

  /* psi_2 - psi_1 */
  VecAXPY(psi_2, -1.0, psi_1);
  VecNorm(psi_2, NORM_2, &wave_error);

  /* we want psi_2+psi_1 so we need to add 2*psi_1 */
  VecAXPY(psi_2, 2.0, psi_1);
  VecNorm(psi_2, NORM_2, &wave_error);

  if (wave_error_2 < wave_error) wave_error = wave_error_2;

  energy_error = wavefunction->GetEnergy(h, psi_1);
  energy_error -= wavefunction->GetEnergy(h, psi_2);

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
// void Simulation::modified_gram_schmidt(std::vector<Eigen::VectorXcd>
// &states)
// {
//   int size = states.size();
//   for (int i = 0; i < size; i++)
//   {
//     psi[0] = psi[0] - psi->dot(states[i]) * states[i];
//   }
// }

void Simulation::CheckpointState(HDF5Wrapper &h_file, ViewWrapper &v_file,
                                 int write_idx)
{
  wavefunction->Normalize();
  wavefunction->CheckpointPsi(v_file, write_idx);
  h_file.WriteObject(wavefunction->GetEnergy(hamiltonian->GetTimeIndependent()),
                     "/Energy", "Energy of the corresponding state", write_idx);
}
