#include "HDF5Wrapper.h"

/* constructor file_name needs ending ".h5" */
HDF5Wrapper::HDF5Wrapper(std::string f_name, Parameters &p)
{
  if (world.rank() == 0)
  {
    file_name = f_name;
    header    = false;
    file_open = false;
    if (p.GetRestart() == 1)
    {
      ReadRestart(p);
    }
    else
    {
      data_file = std::make_shared< H5::H5File >(file_name, H5F_ACC_RDWR);
      file_open = true;
      WriteHeader(p);
    }
    Close();
  }
}

/* constructor file_name needs ending ".h5" */
HDF5Wrapper::HDF5Wrapper(std::string f_name)
{
  if (world.rank() == 0)
  {
    file_name = f_name;
    header    = false;
    file_open = false;
    data_file = std::make_shared< H5::H5File >(file_name, H5F_ACC_RDWR);
    file_open = true;
    Close();
  }
}

/* constructor file_name needs ending ".h5" */
HDF5Wrapper::HDF5Wrapper(Parameters &p)
{
  if (world.rank() == 0)
  {
    header    = false;
    file_name = "TDSE.h5";
    file_open = false;
    if (p.GetRestart() == 1)
    {
      ReadRestart(p);
    }
    else
    {
      data_file = std::make_shared< H5::H5File >(file_name, H5F_ACC_TRUNC);
      file_open = true;
      WriteHeader(p);
    }
    Close();
  }
}

/* destructor */
HDF5Wrapper::~HDF5Wrapper()
{
  if (world.rank() == 0)
  {
    std::cout << "Deleting HDF5Wrapper: " << file_name << "\n";
    Close();
  }
}

/* end run after printing error string with exit value -1 */
void HDF5Wrapper::EndRun(std::string str)
{
  std::cout << "\n\nERROR: " << str << "\n" << std::flush;
  exit(-1);
}

/* end run after printing error string with exit_val */
void HDF5Wrapper::EndRun(std::string str, int exit_val)
{
  std::cout << "\n\nERROR: " << str << "\n";
  exit(exit_val);
}

std::unique_ptr< hsize_t[] > HDF5Wrapper::GetHsizeT(int &size, int *dims)
{
  /* size of array */
  auto h5_size = std::make_unique< hsize_t[] >(size);
  for (int i = 0; i < size; i++)
  {
    h5_size[i] = dims[i];
  }
  return h5_size;
}

std::unique_ptr< hsize_t[] > HDF5Wrapper::GetHsizeT(int &size)
{
  /* size of array */
  auto h5_size = std::make_unique< hsize_t[] >(size);
  for (int i = 0; i < size; i++)
  {
    h5_size[i] = 0;
  }
  return h5_size;
}

void HDF5Wrapper::WriteAttribute(H5std_string &var_path,
                                 H5std_string &attribute)
{
  Open();
  /* write attribute */
  H5::DataSet data_set(data_file->openDataSet(var_path));
  H5::StrType str_type(0, H5T_VARIABLE);
  H5::DataSpace att_space(H5S_SCALAR);
  H5::Attribute att =
      data_set.createAttribute("Attribute", str_type, att_space);
  att.write(str_type, attribute);
  Close();
}

template < typename T >
H5::PredType HDF5Wrapper::getter(T &data)
{
  if (std::is_same< T, int >::value)
  {
    return H5::PredType::NATIVE_INT;
  }
  else if (std::is_same< T, long long >::value)
  {
    return H5::PredType::NATIVE_LLONG;
  }
  else if (std::is_same< T, double >::value)
  {
    return H5::PredType::NATIVE_DOUBLE;
  }
  else
  {
    std::string type_name(typeid(data).name());
    EndRun("Unsupported datatype (" + type_name + ") in HDF5Wrapper");
    return H5::PredType::NATIVE_INT;
  }
}

/* Writes int to HDF5 takes int and var_path as inputs */
template < typename T >
void HDF5Wrapper::WriteObject(T data, H5std_string var_path)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of "array" */
    hsize_t h5_size[1];
    h5_size[0] = 1;

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, getter(data), h5_space);

    /* write data to file */
    data_set.write(&data, getter(data));

    Close();
  }
}

/* Writes 1D int array to HDF5 takes 1D int array, array size, and var_path as
 * inputs */
template < typename T >
void HDF5Wrapper::WriteObject(T data, int size, H5std_string var_path)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    hsize_t h5_size[1];
    h5_size[0] = size;

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, getter(*data), h5_space);

    /* write data to file */
    data_set.write(data, getter(*data));

    Close();
  }
}

/*Writes N-D int array to HDF5 takes N-D int array, array size, and var_path as
 * inputs*/
template < typename T >
void HDF5Wrapper::WriteObject(T data, int size, int *dims,
                              H5std_string var_path)
{
  if (world.rank() == 0)
  {
    Open();

    /* size of array */
    std::unique_ptr< hsize_t[] > h5_size = GetHsizeT(size, dims);
    /* make DataSpace for array */
    H5::DataSpace h5_space(size, h5_size.get());
    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, getter(*data), h5_space);
    /* write data to file */
    data_set.write(data, getter(*data));

    Close();
  }
}

/* Writes int to HDF5 takes int and var_path as inputs */
template < typename T >
void HDF5Wrapper::WriteObject(T data, H5std_string var_path,
                              H5std_string attribute)
{
  if (world.rank() == 0)
  {
    WriteObject(data, var_path);
    WriteAttribute(var_path, attribute);
  }
}

/* Writes 1D int array to HDF5 takes 1D int array, array size, and var_path as
 * inputs */
template < typename T >
void HDF5Wrapper::WriteObject(T data, int size, H5std_string var_path,
                              H5std_string attribute)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    hsize_t h5_size[1];
    h5_size[0] = size;

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, getter(*data), h5_space);

    /* write data to file */
    data_set.write(data, getter(*data));

    WriteAttribute(var_path, attribute);

    Close();
  }
}

/* Writes N-D int array to HDF5 takes N-D int array, array size, and var_path as
 * inputs */
template < typename T >
void HDF5Wrapper::WriteObject(T data, int size, int *dims,
                              H5std_string var_path, H5std_string attribute)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    std::unique_ptr< hsize_t[] > h5_size = GetHsizeT(size, dims);

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size.get());

    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, getter(*data), h5_space);

    /* write data to file */
    data_set.write(data, getter(*data));

    WriteAttribute(var_path, attribute);

    Close();
  }
}

template < typename T >
void HDF5Wrapper::WriteObject(T data, H5std_string var_path, int write_idx)
{
  if (world.rank() == 0)
  {
    Open();

    /* create object */
    if (write_idx == 0)
    {
      /* size of array */
      hsize_t h5_size[1];
      h5_size[0] = 1;

      /* max size of data set. */
      /* Built this way to make time the first dimension */
      hsize_t h5_max_size[1];
      h5_max_size[0] = H5S_UNLIMITED; /* time dim */

      hsize_t h5_chunk[1];
      h5_chunk[0] = 1;

      /* set up data space */
      H5::DataSpace h5_space(1, h5_size, h5_max_size);

      /* Modify dataset creation property to enable chunking */
      H5::DSetCreatPropList prop;
      prop.setChunk(1, h5_chunk);

      H5::DataSet data_set(
          data_file->createDataSet(var_path, getter(data), h5_space, prop));

      /* write data */
      data_set.write(&data, getter(data));
    }
    else
    {
      /* get data set pointer from array */
      H5::DataSet data_set(data_file->openDataSet(var_path));

      /* new dimension of dataset */
      hsize_t h5_size[1];
      h5_size[0] = write_idx + 1;

      /* sizes of extension */
      hsize_t h5_extend[1];
      h5_extend[0] = 1;

      /* sizes of extension */
      hsize_t h5_offset[1];
      h5_offset[0] = write_idx;

      /* extend data set */
      data_set.extend(h5_size);

      /* get hyperslab */
      H5::DataSpace filespace(data_set.getSpace());
      filespace.selectHyperslab(H5S_SELECT_SET, h5_extend, h5_offset);

      /* get memory space */
      H5::DataSpace memspace(1, h5_extend, NULL);

      /* write data */
      data_set.write(&data, getter(data), memspace, filespace);
    }

    Close();
  }
}

template < typename T >
void HDF5Wrapper::WriteObject(T data, H5std_string var_path,
                              H5std_string attribute, int write_idx)
{
  if (world.rank() == 0)
  {
    Open();

    /* create object */
    if (write_idx == 0)
    {
      /* size of array */
      hsize_t h5_size[1];
      h5_size[0] = 1;

      /* max size of data set. */
      /* Built this way to make time the first dimension */
      hsize_t h5_max_size[1];
      h5_max_size[0] = H5S_UNLIMITED;

      hsize_t h5_chunk[1];
      h5_chunk[0] = 1;

      /* set up data space */
      H5::DataSpace h5_space(1, h5_size, h5_max_size);

      /* Modify dataset creation property to enable chunking */
      H5::DSetCreatPropList prop;
      prop.setChunk(1, h5_chunk);

      H5::DataSet data_set(
          data_file->createDataSet(var_path, getter(data), h5_space, prop));

      /* write data */
      data_set.write(&data, getter(data));

      WriteAttribute(var_path, attribute);
    }
    else
    {
      /* get data set pointer from array */
      H5::DataSet data_set(data_file->openDataSet(var_path));

      /* new dimension of dataset */
      hsize_t h5_size[1];
      h5_size[0] = write_idx + 1;

      /* sizes of extension */
      hsize_t h5_extend[1];
      h5_extend[0] = 1;

      /* sizes of extension */
      hsize_t h5_offset[1];
      h5_offset[0] = write_idx;

      /* extend data set */
      data_set.extend(h5_size);

      /* get hyperslab */
      H5::DataSpace filespace(data_set.getSpace());
      filespace.selectHyperslab(H5S_SELECT_SET, h5_extend, h5_offset);

      /* get memory space */
      H5::DataSpace memspace(1, h5_extend, NULL);

      /* write data */
      data_set.write(&data, getter(data), memspace, filespace);
    }

    Close();
  }
}

PetscInt HDF5Wrapper::GetTime(H5std_string var_path)
{
  PetscInt ret_val;
  if (world.rank() == 0)
  {
    Open();
    H5::DataSet data_set(data_file->openDataSet(var_path));
    H5::DataSpace memspace               = data_set.getSpace();
    PetscInt ndims                       = memspace.getSimpleExtentNdims();
    std::unique_ptr< hsize_t[] > h5_size = GetHsizeT(ndims);
    memspace.getSimpleExtentDims(h5_size.get());
    Close();
    ret_val = h5_size[0] - 1;
  }
  broadcast(world, ret_val, 0);
  return ret_val;
}

void HDF5Wrapper::CreateGroup(H5std_string group_path)
{
  if (world.rank() == 0)
  {
    Open();

    H5::Group new_group(data_file->createGroup(group_path));

    Close();
  }
}

/* writes out header for Parameters and builds the various groups that will be
 * used by other classes */
void HDF5Wrapper::WriteHeader(Parameters &p)
{
  if (world.rank() == 0)
  {
    PetscInt num_dims   = p.GetNumDims();
    PetscInt num_pulses = p.GetNumPulses();
    header              = true;

    CreateGroup("/Parameters/");

    /* write out header values */
    WriteObject(num_dims, "/Parameters/num_dims",
                "Number of dimension in simulation");
    WriteObject(p.GetNumElectrons(), "/Parameters/num_electrons",
                "Number of electrons in the simulation");
    WriteObject(p.dim_size.get(), num_dims, "/Parameters/dim_size",
                "The length of that dimension in atomic units.");
    WriteObject(p.delta_x.get(), num_dims, "/Parameters/delta_x",
                "The step sizes in that dimension in atomic units.");
    WriteObject(p.GetCoordinateSystemIdx(), "/Parameters/coordinate_system_idx",
                "Index of coordinate system: Cartesian:0, Cylindrical:1");
    WriteObject(p.GetDeltaT(), "/Parameters/delta_t",
                "The size of the time step in atomic units.");
    WriteObject(p.GetTargetIdx(), "/Parameters/target_idx",
                "The index of the target. He:0 ");
    WriteObject(p.z.get(), p.GetNumNuclei(), "/Parameters/z",
                "Atomic number used in Hamiltonian ");
    WriteObject(p.z_c.get(), p.GetNumNuclei(), "/Parameters/z_c",
                "Z_c in the SAE potential for each nuclei");
    WriteObject(p.c0.get(), p.GetNumNuclei(), "/Parameters/c0",
                "C_0 in the SAE potential for each nuclei");
    WriteObject(p.r0.get(), p.GetNumNuclei(), "/Parameters/r0",
                "R_0 in the SAE potential for each nuclei");
    for (int i = 0; i < p.GetNumNuclei(); ++i)
    {
      WriteObject(p.GetLocation()[i], p.GetNumDims(),
                  "/Parameters/location_" + std::to_string(i),
                  "location of nuclei " + std::to_string(i));
      WriteObject(p.GetA()[i], p.sae_size[i],
                  "/Parameters/a_" + std::to_string(i),
                  "SAE a terms for nuclei " + std::to_string(i));
      WriteObject(p.GetB()[i], p.sae_size[i],
                  "/Parameters/b_" + std::to_string(i),
                  "SAE a terms for nuclei " + std::to_string(i));
    }
    WriteObject(p.GetAlpha(), "/Parameters/alpha",
                "Soft core used in atomic term of Hamiltonian");
    WriteObject(
        p.GetWriteFrequencyObservables(),
        "/Parameters/write_frequency_observables",
        "How often are observables are printed done during propagation");
    WriteObject(p.GetWriteFrequencyCheckpoint(),
                "/Parameters/write_frequency_checkpoint",
                "How often are checkpoints done during propagation");
    WriteObject(
        p.GetWriteFrequencyEigenState(),
        "/Parameters/write_frequency_eigin_state",
        "How often are checkpoints done during eigen state calculations");
    WriteObject(p.GetGobbler(), "/Parameters/gobbler",
                "The percent of the grid that is real and not part of the ECS "
                "boundary potential, (1=100 and 0.9=90)");
    WriteObject(p.GetSigma(), "/Parameters/sigma", "STD of wavefunction guess");
    WriteObject(p.GetTol(), "/Parameters/tol", "Error tolerance in psi");
    WriteObject(p.GetStateSolverIdx(), "/Parameters/state_solver_idx",
                "Index of solver: File:0, ITP:1, Power:2");
    WriteObject(num_pulses, "/Parameters/num_pulses",
                "The number of pulses from the input file");
    for (int pulse_idx = 0; pulse_idx < p.GetNumPulses(); ++pulse_idx)
    {
      WriteObject(
          p.GetPolarizationVector()[pulse_idx], num_dims,
          "/Parameters/polarization_vector_" + std::to_string(pulse_idx),
          "The vector used to define the polarization direction for the " +
              std::to_string(pulse_idx) + " pulse");
      if (num_dims == 3)
      {
        WriteObject(
            p.GetPoyntingVector()[pulse_idx], num_dims,
            "/Parameters/poynting_vector_" + std::to_string(pulse_idx),
            "The vector used to define the poynting direction for the " +
                std::to_string(pulse_idx) + " pulse");
      }
      WriteObject(p.ellipticity.get()[pulse_idx],
                  "/Parameters/ellipticity_" + std::to_string(pulse_idx),
                  "The ellipticity for the " + std::to_string(pulse_idx) +
                      " pulse (major/minor)");
      WriteObject(p.helicity_idx.get()[pulse_idx],
                  "/Parameters/helicity_idx_" + std::to_string(pulse_idx),
                  "The helicity idx for the " + std::to_string(pulse_idx) +
                      " pulse right:0, left:1");
    }
    WriteObject(p.pulse_shape_idx.get(), num_pulses,
                "/Parameters/pulse_shape_idx",
                "The index of the pulse shape. Sin2:0");
    WriteObject(p.cycles_on.get(), num_pulses, "/Parameters/cycles_on",
                "Number of cycles the pulse ramps on for");
    WriteObject(p.cycles_plateau.get(), num_pulses,
                "/Parameters/cycles_plateau",
                "Number of cycles the pulse stays at max amplitude for");
    WriteObject(p.cycles_off.get(), num_pulses, "/Parameters/cycles_off",
                "Number of cycles the pulse ramps off for");
    WriteObject(p.cycles_delay.get(), num_pulses, "/Parameters/cycles_delay",
                "Number of cycles before the pulse starts");
    WriteObject(p.cep.get(), num_pulses, "/Parameters/cep",
                "The carrying phase envelope of the pulse. It is defined at "
                "the end of cycles on as a fraction of a cycle (i.e. 0.5 -> "
                "90^0 or pi phase)");
    WriteObject(p.energy.get(), num_pulses, "/Parameters/energy",
                "The fundamental angular frequency of the pulse. Corresponds "
                "to the energy of the photons in atomic units.");
    WriteObject(p.field_max.get(), num_pulses, "/Parameters/field_max",
                "The maximum amplitude of the pulse in atomic units.");

    header = false;
    Close();
  }
}

/* TDOD: set up restart */
void HDF5Wrapper::ReadRestart(Parameters &p)
{
  if (world.rank() == 0)
  {
    /* TODO(jove7731): put parameter check here */
  }
}

void HDF5Wrapper::Open()
{
  if (world.rank() == 0)
  {
    if (!header && !file_open)
    {
      data_file = std::make_unique< H5::H5File >(file_name, H5F_ACC_RDWR);
      file_open = true;
    }
  }
}

void HDF5Wrapper::Close()
{
  if (world.rank() == 0)
  {
    if (!header && file_open)
    {
      data_file->close();
      file_open = false;
    }
  }
}

void HDF5Wrapper::SetHeader(bool h)
{
  if (world.rank() == 0)
  {
    header = h;
  }
}

template void HDF5Wrapper::WriteObject< double >(double data,
                                                 H5std_string var_path,
                                                 int write_idx);
template void HDF5Wrapper::WriteObject< double >(double data,
                                                 H5std_string var_path,
                                                 H5std_string attribute,
                                                 int write_idx);
template void HDF5Wrapper::WriteObject< int * >(int *, int, std::string,
                                                std::string);
