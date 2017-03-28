#include "Wavefunction.h"
#include <math.h>  // ceil()
#include <complex>
#include <iostream>

#define dcomp std::complex<double>

// prints error message, kills code and returns -1
void Wavefunction::end_run(std::string str)
{
  std::cout << "\n\nERROR: " << str << "\n" << std::flush;
  exit(-1);
}

// prints error message, kills code and returns exit_val
void Wavefunction::end_run(std::string str, int exit_val)
{
  std::cout << "\n\nERROR: " << str << "\n";
  exit(exit_val);
}

Wavefunction::Wavefunction(HDF5Wrapper& data_file, Parameters& p)
{
  std::cout << "Creating Wavefunction\n";

  // initialize values
  psi_12_alloc  = false;
  psi_alloc     = false;
  first_pass    = true;
  num_dims      = p.GetNumDims();
  dim_size      = p.GetDimSize();
  delta_x       = p.GetDeltaX();
  sigma         = p.GetSigma();
  num_psi_12    = 1;
  write_counter = 0;

  offset = dim_size[0] / 2.0 * p.GetGobbler();
  width  = pi / (2.0 * (dim_size[0] / 2.0 - offset));

  // validation
  if (num_dims > 1)
  {
    end_run("Only 1D is currently supported");
  }

  // allocate grid
  create_grid();
  std::cout << "grid built\n";

  // allocate psi_1, psi_2, and psi
  create_psi();

  // write out data
  checkpoint(data_file, 0.0);
  checkpoint(data_file, 0.0);
  checkpoint(data_file, 0.0);

  // delete psi_1 and psi_2
  cleanup();

  std::cout << "Wavefunction created\n";
}

void Wavefunction::checkpoint(HDF5Wrapper& data_file, double time)
{
  std::cout << "Checkpointing Wavefunction: " << write_counter;
  std::cout << "\n" << std::flush;
  std::string str;
  PetscViewer H5viewer;
  PetscInt ierr;
  const char* tmp;
  std::string name;
  std::string group_name = "/Wavefunction/";

  // open file
  // ierr =
  // PetscViewerHDF5Open(PETSC_COMM_WORLD,"TDSE.h5",FILE_MODE_APPEND,&H5viewer);
  // PetscViewerSetFromOptions(H5viewer);
  // PetscViewerHDF5PushGroup(H5viewer, "Wavefunction");
  // PetscViewerHDF5SetTimestep(H5viewer,write_counter);
  // VecView(psi,H5viewer);
  // PetscViewerDestroy(&H5viewer);

  // only write out at start
  if (first_pass)
  {
    std::cout << "wft mate 1\n";
    PetscErrorCode PetscViewerHDF5GetGroup(PetscViewer viewer,
                                           const char** name);

    // open file
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "TDSE.h5", FILE_MODE_APPEND,
                               &H5viewer);
    PetscViewerSetFromOptions(H5viewer);
    // move into group
    PetscViewerHDF5PushGroup(H5viewer, group_name.c_str());
    // set time step
    PetscViewerHDF5SetTimestep(H5viewer, write_counter);
    // write vector
    PetscObjectView((PetscObject)psi, H5viewer);
    // get object name
    PetscObjectGetName((PetscObject)psi, &tmp);
    name = tmp;
    PetscViewerHDF5WriteAttribute(H5viewer, (group_name + name).c_str(),
                                  "Attribute", PETSC_STRING,
                                  "Wavefunction for the two electron system");
    PetscViewerHDF5WriteAttribute(H5viewer, (group_name + name).c_str(), "Def",
                                  PETSC_STRING,
                                  "Wavefunction for the two electron system");
    // close file
    PetscViewerDestroy(&H5viewer);
    std::cout << "wft mate 2\n";
    // templating notes
    // http://en.cppreference.com/w/cpp/language/partial_specialization

    // size of each dim
    data_file.WriteObject(num_x, num_dims, "/Wavefunction/num_x",
                          "The number of physical dimension in the simulation");
    std::cout << "wft mate 3 \n";

    // write each dims x values
    for (int i = 0; i < num_dims; i++)
    {
      str = "x_value_";
      str += std::to_string(i);
      data_file.WriteObject(
          x_value[i], num_x[i], "/Wavefunction/" + str,
          "The coordinates of the " + std::to_string(i) + " dimension");
    }

    // write psi_1 and psi_2 if still allocated
    if (psi_12_alloc)
    {
      data_file.WriteObject(psi_1, num_psi_12, "/Wavefunction/psi_1",
                            "Wavefunction of first electron");
      data_file.WriteObject(psi_2, num_psi_12, "/Wavefunction/psi_2",
                            "Wavefunction of second electron");

      data_file.WriteObject(psi_1_gobbler, num_psi_12,
                            "/Wavefunction/psi_1_gobbler",
                            "Wavefunction of second electron");
      data_file.WriteObject(psi_2_gobbler, num_psi_12,
                            "/Wavefunction/psi_2_gobbler",
                            "Boundary potential of second electron");
    }

    // write data and attribute
    // data_file.WriteObject(psi->data(), num_psi,
    //     "/Wavefunction/psi",
    //     "Wavefunction for the two electron system",
    //     write_counter);

    // data_file.WriteObject(psi_gobbler->data(), num_psi,
    //     "/Wavefunction/psi_gobbler",
    //     "boundary potential for the two electron system");

    // write time and attribute
    data_file.WriteObject(time, "/Wavefunction/psi_time",
                          "Time step that psi was written to disk",
                          write_counter);

    // // write time and attribute
    // data_file.WriteObject(norm(), "/Wavefunction/norm",
    //     "Norm of wavefunction", write_counter);

    // allow for future passes to write psi only
    first_pass = false;
  }
  else
  {
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "TDSE.h5", FILE_MODE_APPEND,
                               &H5viewer);
    PetscViewerSetFromOptions(H5viewer);
    PetscViewerHDF5PushGroup(H5viewer, "Wavefunction");
    PetscViewerHDF5SetTimestep(H5viewer, write_counter);
    VecView(psi, H5viewer);
    PetscViewerDestroy(&H5viewer);
  }
  // } else {
  //     // write whenever this function is called
  //     data_file.WriteObject(psi->data(), num_psi,
  //         "/Wavefunction/psi", write_counter);

  //     // write time
  //     data_file.WriteObject(time, "/Wavefunction/psi_time",
  //         write_counter);

  //     data_file.WriteObject(norm(), "/Wavefunction/norm",
  //         write_counter);
  // }

  // keep track of what index we are on
  write_counter++;
}

void Wavefunction::checkpoint_psi(HDF5Wrapper& data_file, H5std_string var_path,
                                  int write_idx)
{
  std::cout << "Checkpointing Wavefunction: " << var_path;
  std::cout << " " << write_idx << "\n";

  // data_file.WriteObject(psi->data(), num_psi, var_path, write_idx);
}

void Wavefunction::create_grid()
{
  int center;        // idx of the 0.0 in the grid
  double current_x;  // used for setting grid

  // allocation
  num_x   = new int[num_dims];
  x_value = new double*[num_dims];

  // initialize for loop
  num_psi_12 = 1.0;

  // build grid
  for (int i = 0; i < num_dims; i++)
  {
    num_x[i] = ceil((dim_size[i]) / delta_x[i]) + 1;

    // odd number so it is even on both sides
    if (num_x[i] % 2 == 0) num_x[i]++;

    // size of 1d array for psi
    num_psi_12 *= num_x[i];

    // find center of grid
    center = num_x[i] / 2 + 1;

    // allocate grid
    x_value[i] = new double[num_x[i]];

    // store center
    x_value[i][center] = 0.0;

    // loop over all others
    for (int j = center - 1; j > 0; j--)
    {
      // get x value
      current_x = (j - center) * delta_x[i];

      // double checking index
      if (j - 1 < 0 || num_x[i] - j >= num_x[i])
      {
        end_run("Allocation error in grid");
      }

      // set negative side
      x_value[i][j - 1] = current_x;
      // set positive side
      x_value[i][num_x[i] - j] = -1 * current_x;
    }
  }
}

// builds psi from 2 Gaussian psi (one for each electron)
void Wavefunction::create_psi()
{
  double sigma2;  // variance squared for Gaussian in psi
  double x;       // x value squared
  double x2;      // x value squared
  int rank;
  PetscInt idx, ierr;
  PetscComplex val;
  PetscViewer H5viewer;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // allocate data
  if (!psi_12_alloc)
  {
    psi_1         = new dcomp[num_psi_12];
    psi_2         = new dcomp[num_psi_12];
    psi_1_gobbler = new dcomp[num_psi_12];
    psi_2_gobbler = new dcomp[num_psi_12];
    psi_12_alloc  = true;
  }

  sigma2 = sigma * sigma;
  // TODO: needs to be changed for more than one dim
  for (int i = 0; i < num_psi_12; i++)
  {
    // get x value squared
    x  = x_value[0][i];
    x2 = x * x;

    // Gaussian centered around 0.0 with variation sigma
    psi_1[i] = dcomp(exp(-1 * x2 / (2 * sigma2)), 0.0);
    psi_2[i] = dcomp(exp(-1 * x2 / (2 * sigma2)), 0.0);

    if (std::abs(x) - offset > 0)
    {
      psi_1_gobbler[i] =
          std::pow(cos((std::abs(x) - offset) * width), 1.0 / 8.0);
      psi_2_gobbler[i] =
          std::pow(cos((std::abs(x) - offset) * width), 1.0 / 8.0);
    }
    else
    {
      psi_1_gobbler[i] = dcomp(1.0, 0.0);
      psi_2_gobbler[i] = dcomp(1.0, 0.0);
    }
  }

  // get size of psi
  num_psi = num_psi_12 * num_psi_12;

  // allocate psi
  if (!psi_alloc)
  {
    VecCreate(PETSC_COMM_WORLD, &psi);
    VecSetSizes(psi, PETSC_DECIDE, num_psi);
    VecSetFromOptions(psi);
    ierr      = PetscObjectSetName((PetscObject)psi, "psi");
    psi_alloc = true;
  }

  // tensor product of psi_1 and psi_2
  if (rank == 0)
  {
    for (int i = 0; i < num_psi_12; i++)
    {  // e_2 dim
      for (int j = 0; j < num_psi_12; j++)
      {  // e_1 dim
        idx = i * num_psi_12 + j;
        val = psi_1[i] * psi_2[j];
        VecSetValues(psi, 1, &idx, &val, INSERT_VALUES);

        // psi_gobbler[0](i*num_psi_12+j) =
        //     psi_1_gobbler[i]*psi_2_gobbler[j];
      }
    }
  }
  VecAssemblyBegin(psi);
  VecAssemblyEnd(psi);

  // ierr =
  // PetscViewerHDF5Open(PETSC_COMM_WORLD,"gauss.h5",FILE_MODE_APPEND,&H5viewer);
  // PetscViewerSetFromOptions(H5viewer);
  // PetscViewerHDF5PushGroup(H5viewer, "Wavefunction");
  // VecView(psi,H5viewer);
  // PetscViewerDestroy(&H5viewer);
  // normalize all psi
  // normalize();
}

// delete psi_1 and psi_2 since they are not used later
void Wavefunction::cleanup()
{
  delete psi_1;
  delete psi_1_gobbler;
  delete psi_2;
  delete psi_2_gobbler;
  psi_12_alloc = false;
}

// normalize psi_1, psi_2, and psi
void Wavefunction::normalize()
{
  if (psi_12_alloc)
  {
    normalize(psi_1, num_psi_12, delta_x[0]);
    normalize(psi_2, num_psi_12, delta_x[0]);
  }
  // normalize(psi->data(), num_psi, delta_x[0]);
}

void Wavefunction::gobble_psi()
{
  // for (int i=0; i<num_psi; i++) {
  //     psi[0][i] *= psi_gobbler[0][i];
  // }
}

// normalizes the array provided
void Wavefunction::normalize(dcomp* data, int length, double dx)
{
  double total = norm(data, length, dx);

  // square root to get normalization factor
  total = sqrt(total);

  // normalize data
  for (int i = 0; i < length; i++)
  {
    data[i].real(data[i].real() / total);
    data[i].imag(data[i].imag() / total);
  }
}

// returns norm of psi
double Wavefunction::norm()
{
  return 2.0;
  // return norm(psi->data(), num_psi, delta_x[0]);
}

// returns norm of array using trapezoidal rule
double Wavefunction::norm(dcomp* data, int length, double dx)
{
  double total = 0;
  // lower end
  total += (std::conj(data[0]) * data[0]).real();
  // higher end
  total += (std::conj(data[length - 1]) * data[length - 1]).real();
  // middle points
  for (int i = 1; i < length - 1; i++)
  {
    total += 2.0 * (std::conj(data[i]) * data[i]).real();
  }
  total *= dx / 2.0;
  return total;
}

// double Wavefunction::get_energy(Eigen::SparseMatrix<dcomp> *h){
//     return (psi->dot(h[0]*psi[0])/psi->squaredNorm()).real();
// }

void Wavefunction::reset_psi()
{
  create_psi();
  cleanup();
}

int* Wavefunction::get_num_x() { return num_x; }

int Wavefunction::get_num_psi() { return num_psi; }

int Wavefunction::get_num_psi_12() { return num_psi_12; }

// Eigen::VectorXcd* Wavefunction::get_psi() {
//     return psi;
// }

double* Wavefunction::get_delta_x() { return delta_x; }

double** Wavefunction::get_x_value() { return x_value; }

// destructor
Wavefunction::~Wavefunction()
{
  std::cout << "Deleting Wavefunction\n";
  // do not delete dim_size or delta_x
  // since they belong to the Parameter class
  // and will be freed there
  delete num_x;
  for (int i = 0; i < num_dims; i++)
  {
    delete x_value[i];
  }
  delete[] x_value;
  if (psi_12_alloc)
  {
    cleanup();
  }
}
