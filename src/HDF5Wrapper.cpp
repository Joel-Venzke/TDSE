#include "HDF5Wrapper.h"
#include <complex>

#define dcomp std::complex<double>

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

hsize_t *HDF5Wrapper::GetHsizeT(int size, int *dims)
{
  /* size of array */
  hsize_t *h5_size = new hsize_t[size];
  for (int i = 0; i < size; i++)
  {
    h5_size[i] = dims[i];
  }
  return h5_size;
}

/* Writes int to HDF5 takes int and var_path as inputs */
void HDF5Wrapper::WriteObject(int data, H5std_string var_path)
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
        data_file->createDataSet(var_path, H5::PredType::NATIVE_INT, h5_space);

    /* write data to file */
    data_set.write(&data, H5::PredType::NATIVE_INT);

    Close();
  }
}

/* Writes double to HDF5 takes double and var_path as inputs */
void HDF5Wrapper::WriteObject(double data, H5std_string var_path)
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
    H5::DataSet data_set = data_file->createDataSet(
        var_path, H5::PredType::NATIVE_DOUBLE, h5_space);

    /* write data to file */
    data_set.write(&data, H5::PredType::NATIVE_DOUBLE);

    Close();
  }
}

/* Writes 1D int array to HDF5 takes 1D int array, array size, and var_path as
 * inputs */
void HDF5Wrapper::WriteObject(int *data, int size, H5std_string var_path)
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
        data_file->createDataSet(var_path, H5::PredType::NATIVE_INT, h5_space);

    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_INT);

    Close();
  }
}

/*Writes N-D int array to HDF5 takes N-D int array, array size, and var_path as
 * inputs*/
void HDF5Wrapper::WriteObject(int *data, int size, int *dims,
                              H5std_string var_path)
{
  if (world.rank() == 0)
  {
    Open();

    /* size of array */
    hsize_t *h5_size = GetHsizeT(size, dims);
    /* make DataSpace for array */
    H5::DataSpace h5_space(size, h5_size);
    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, H5::PredType::NATIVE_INT, h5_space);
    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_INT);

    delete[] h5_size;

    Close();
  }
}

/* Writes 1D double array to HDF5 takes 1D double array, array size, and
 * var_path as inputs */
void HDF5Wrapper::WriteObject(double *data, int size, H5std_string var_path)
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
    H5::DataSet data_set = data_file->createDataSet(
        var_path, H5::PredType::NATIVE_DOUBLE, h5_space);

    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_DOUBLE);

    Close();
  }
}

/* Writes N-Dim double array to HDF5 takes N-Dim double array, array size, and
 * var_path as inputs */
void HDF5Wrapper::WriteObject(double *data, int size, int *dims,
                              H5std_string var_path)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    hsize_t *h5_size = GetHsizeT(size, dims);

    /* make DataSpace for array */
    H5::DataSpace h5_space(size, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set = data_file->createDataSet(
        var_path, H5::PredType::NATIVE_DOUBLE, h5_space);

    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_DOUBLE);

    delete[] h5_size;

    Close();
  }
}

/* Writes 1D complex double array to HDF5 takes 1D complex double array, array
 * size, and var_path as inputs */
void HDF5Wrapper::WriteObject(dcomp *data, int size, H5std_string var_path)
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
        data_file->createDataSet(var_path, complex_data_type[0], h5_space);

    /* write data to file */
    data_set.write(data, complex_data_type[0]);

    Close();
  }
}

void HDF5Wrapper::WriteObject(dcomp *data, int size, int *dims,
                              H5std_string var_path)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    hsize_t *h5_size = GetHsizeT(size, dims);

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, complex_data_type[0], h5_space);

    /* write data to file */
    data_set.write(data, complex_data_type[0]);

    delete[] h5_size;

    Close();
  }
}

/* Writes int to HDF5 takes int and var_path as inputs */
void HDF5Wrapper::WriteObject(int data, H5std_string var_path,
                              H5std_string attribute)
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
        data_file->createDataSet(var_path, H5::PredType::NATIVE_INT, h5_space);

    /* write data to file */
    data_set.write(&data, H5::PredType::NATIVE_INT);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    Close();
  }
}

/* Writes double to HDF5 takes double and var_path as inputs */
void HDF5Wrapper::WriteObject(double data, H5std_string var_path,
                              H5std_string attribute)
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
    H5::DataSet data_set = data_file->createDataSet(
        var_path, H5::PredType::NATIVE_DOUBLE, h5_space);

    /* write data to file */
    data_set.write(&data, H5::PredType::NATIVE_DOUBLE);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    Close();
  }
}

/* Writes 1D int array to HDF5 takes 1D int array, array size, and var_path as
 * inputs */
void HDF5Wrapper::WriteObject(int *data, int size, H5std_string var_path,
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
        data_file->createDataSet(var_path, H5::PredType::NATIVE_INT, h5_space);

    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_INT);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    Close();
  }
}

/* Writes N-D int array to HDF5 takes N-D int array, array size, and var_path as
 * inputs */
void HDF5Wrapper::WriteObject(int *data, int size, int *dims,
                              H5std_string var_path, H5std_string attribute)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    hsize_t *h5_size = GetHsizeT(size, dims);

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, H5::PredType::NATIVE_INT, h5_space);

    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_INT);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    delete h5_size;

    Close();
  }
}

/* Writes 1D double array to HDF5 takes 1D double array, array size, and
 * var_path as inputs */
void HDF5Wrapper::WriteObject(double *data, int size, H5std_string var_path,
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
    H5::DataSet data_set = data_file->createDataSet(
        var_path, H5::PredType::NATIVE_DOUBLE, h5_space);

    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_DOUBLE);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    Close();
  }
}

/* Writes N-D double array to HDF5 takes N-D double array, array size, and
 * var_path as inputs */
void HDF5Wrapper::WriteObject(double *data, int size, int *dims,
                              H5std_string var_path, H5std_string attribute)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    hsize_t *h5_size = GetHsizeT(size, dims);

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set = data_file->createDataSet(
        var_path, H5::PredType::NATIVE_DOUBLE, h5_space);

    /* write data to file */
    data_set.write(data, H5::PredType::NATIVE_DOUBLE);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    delete h5_size;

    Close();
  }
}

/* Writes 1D complex double array to HDF5 takes 1D complex double array, array
 * size, and var_path as inputs */
void HDF5Wrapper::WriteObject(dcomp *data, int size, H5std_string var_path,
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
        data_file->createDataSet(var_path, complex_data_type[0], h5_space);

    /* write data to file */
    data_set.write(data, complex_data_type[0]);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    Close();
  }
}

/* Writes N-D complex double array to HDF5 takes N-D complex double array, array
 * size, and var_path as inputs */
void HDF5Wrapper::WriteObject(dcomp *data, int size, int *dims,
                              H5std_string var_path, H5std_string attribute)
{
  if (world.rank() == 0)
  {
    Open();
    /* size of array */
    hsize_t *h5_size = GetHsizeT(size, dims);

    /* make DataSpace for array */
    H5::DataSpace h5_space(1, h5_size);

    /* build the header for the data entry */
    H5::DataSet data_set =
        data_file->createDataSet(var_path, complex_data_type[0], h5_space);

    /* write data to file */
    data_set.write(data, complex_data_type[0]);

    /* write attribute */
    H5::StrType str_type(0, H5T_VARIABLE);
    H5::DataSpace att_space(H5S_SCALAR);
    H5::Attribute att =
        data_set.createAttribute("Attribute", str_type, att_space);
    att.write(str_type, attribute);

    delete h5_size;

    Close();
  }
}

void HDF5Wrapper::WriteObject(dcomp *data, int size, H5std_string var_path,
                              int write_idx)
{
  if (world.rank() == 0)
  {
    Open();

    /* create object */
    if (write_idx == 0)
    {
      /* size of array */
      hsize_t h5_size[2];
      h5_size[0] = 1;
      h5_size[1] = size;

      /* max size of data set. */
      /* Built this way to make time the first dimension */
      hsize_t h5_max_size[2];
      h5_max_size[0] = H5S_UNLIMITED; /* time dim */
      h5_max_size[1] = H5S_UNLIMITED; /* spacial dim */

      hsize_t h5_chunk[2];
      h5_chunk[0] = 1;
      h5_chunk[1] = size;

      /* set up data space */
      H5::DataSpace h5_space(2, h5_size, h5_max_size);

      /* Modify dataset creation property to enable chunking */
      H5::DSetCreatPropList prop;
      prop.setChunk(2, h5_chunk);

      H5::DataSet *data_set = new H5::DataSet(data_file->createDataSet(
          var_path, complex_data_type[0], h5_space, prop));

      /* write data */
      data_set->write(data, complex_data_type[0]);
      delete data_set;
    }
    else
    {
      /* get data set pointer from array */
      H5::DataSet *data_set = new H5::DataSet(data_file->openDataSet(var_path));

      /* new dimension of dataset */
      hsize_t h5_size[2];
      h5_size[0] = write_idx + 1;
      h5_size[1] = size;

      /* sizes of extension */
      hsize_t h5_extend[2];
      h5_extend[0] = 1;
      h5_extend[1] = size;

      /* sizes of extension */
      hsize_t h5_offset[2];
      h5_offset[0] = write_idx;
      h5_offset[1] = 0;

      /* extend data set */
      data_set->extend(h5_size);

      /* get hyperslab */
      H5::DataSpace *filespace = new H5::DataSpace(data_set->getSpace());
      filespace->selectHyperslab(H5S_SELECT_SET, h5_extend, h5_offset);

      /* get memory space */
      H5::DataSpace *memspace = new H5::DataSpace(2, h5_extend, NULL);

      /* write data */
      data_set->write(data, complex_data_type[0], *memspace, *filespace);

      /* clean up */
      delete filespace;
      delete memspace;
      delete data_set;
    }

    Close();
  }
}

void HDF5Wrapper::WriteObject(dcomp *data, int size, H5std_string var_path,
                              H5std_string attribute, int write_idx)
{
  if (world.rank() == 0)
  {
    Open();

    /* create object */
    if (write_idx == 0)
    {
      /* size of array */
      hsize_t h5_size[2];
      h5_size[0] = 1;
      h5_size[1] = size;

      /* max size of data set. */
      /* Built this way to make time the first dimension */
      hsize_t h5_max_size[2];
      h5_max_size[0] = H5S_UNLIMITED; /* time dim */
      h5_max_size[1] = H5S_UNLIMITED; /* spacial dim */

      hsize_t h5_chunk[2];
      h5_chunk[0] = 1;
      h5_chunk[1] = size;

      /* set up data space */
      H5::DataSpace h5_space(2, h5_size, h5_max_size);

      /* Modify dataset creation property to enable chunking */
      H5::DSetCreatPropList prop;
      prop.setChunk(2, h5_chunk);

      H5::DataSet *data_set = new H5::DataSet(data_file->createDataSet(
          var_path, complex_data_type[0], h5_space, prop));

      /* write data */
      data_set->write(data, complex_data_type[0]);

      /* write attribute */
      H5::StrType str_type(0, H5T_VARIABLE);
      H5::DataSpace att_space(H5S_SCALAR);
      H5::Attribute att =
          data_set->createAttribute("Attribute", str_type, att_space);
      att.write(str_type, attribute);
    }
    else
    {
      /* get data set pointer from array */
      H5::DataSet *data_set = new H5::DataSet(data_file->openDataSet(var_path));

      /* new dimension of dataset */
      hsize_t h5_size[2];
      h5_size[0] = write_idx + 1;
      h5_size[1] = size;

      /* sizes of extension */
      hsize_t h5_extend[2];
      h5_extend[0] = 1;
      h5_extend[1] = size;

      /* sizes of extension */
      hsize_t h5_offset[2];
      h5_offset[0] = write_idx;
      h5_offset[1] = 0;

      /* extend data set */
      data_set->extend(h5_size);

      /* get hyperslab */
      H5::DataSpace *filespace = new H5::DataSpace(data_set->getSpace());
      filespace->selectHyperslab(H5S_SELECT_SET, h5_extend, h5_offset);

      /* get memory space */
      H5::DataSpace *memspace = new H5::DataSpace(2, h5_extend, NULL);

      /* write data */
      data_set->write(data, complex_data_type[0], *memspace, *filespace);

      /* clean up */
      delete filespace;
      delete memspace;
      delete data_set;
    }

    Close();
  }
}

void HDF5Wrapper::WriteObject(double data, H5std_string var_path, int write_idx)
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

      H5::DataSet *data_set = new H5::DataSet(data_file->createDataSet(
          var_path, H5::PredType::NATIVE_DOUBLE, h5_space, prop));

      /* write data */
      data_set->write(&data, H5::PredType::NATIVE_DOUBLE);
      delete data_set;
    }
    else
    {
      /* get data set pointer from array */
      H5::DataSet *data_set = new H5::DataSet(data_file->openDataSet(var_path));

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
      data_set->extend(h5_size);

      /* get hyperslab */
      H5::DataSpace *filespace = new H5::DataSpace(data_set->getSpace());
      filespace->selectHyperslab(H5S_SELECT_SET, h5_extend, h5_offset);

      /* get memory space */
      H5::DataSpace *memspace = new H5::DataSpace(1, h5_extend, NULL);

      /* write data */
      data_set->write(&data, H5::PredType::NATIVE_DOUBLE, *memspace,
                      *filespace);

      /* clean up */
      delete filespace;
      delete memspace;
      delete data_set;
    }

    Close();
  }
}

void HDF5Wrapper::WriteObject(double data, H5std_string var_path,
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

      H5::DataSet *data_set = new H5::DataSet(data_file->createDataSet(
          var_path, H5::PredType::NATIVE_DOUBLE, h5_space, prop));

      /* write data */
      data_set->write(&data, H5::PredType::NATIVE_DOUBLE);

      /* write attribute */
      H5::StrType str_type(0, H5T_VARIABLE);
      H5::DataSpace att_space(H5S_SCALAR);
      H5::Attribute att =
          data_set->createAttribute("Attribute", str_type, att_space);
      att.write(str_type, attribute);
      delete data_set;
    }
    else
    {
      /* get data set pointer from array */
      H5::DataSet *data_set = new H5::DataSet(data_file->openDataSet(var_path));

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
      data_set->extend(h5_size);

      /* get hyperslab */
      H5::DataSpace *filespace = new H5::DataSpace(data_set->getSpace());
      filespace->selectHyperslab(H5S_SELECT_SET, h5_extend, h5_offset);

      /* get memory space */
      H5::DataSpace *memspace = new H5::DataSpace(1, h5_extend, NULL);

      /* write data */
      data_set->write(&data, H5::PredType::NATIVE_DOUBLE, *memspace,
                      *filespace);

      /* clean up */
      delete filespace;
      delete memspace;
      delete data_set;
    }

    Close();
  }
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
    int num_dims   = p.GetNumDims();
    int num_pulses = p.GetNumPulses();
    header         = true;

    CreateGroup("/Parameters");

    /* write out header values */
    WriteObject(num_dims, "/Parameters/num_dims",
                "Number of dimension in simulation");
    WriteObject(p.dim_size.get(), num_dims, "/Parameters/dim_size",
                "The length of that dimension in atomic units.");
    WriteObject(p.delta_x.get(), num_dims, "/Parameters/delta_x",
                "The step sizes in that dimension in atomic units.");
    WriteObject(p.GetDeltaT(), "/Parameters/delta_t",
                "The size of the time step in atomic units.");
    WriteObject(p.GetTargetIdx(), "/Parameters/target_idx",
                "The index of the target. He:0 ");
    WriteObject(p.GetZ(), "/Parameters/z",
                "Atomic number used in Hamiltonian ");
    WriteObject(p.GetAlpha(), "/Parameters/alpha",
                "Soft core used in atomic term of Hamiltonian");
    WriteObject(p.GetWriteFrequencyPropagation(),
                "/Parameters/write_frequency_propagation",
                "How often are checkpoints done during propagation");
    WriteObject(
        p.GetWriteFrequencyEigenState(),
        "/Parameters/write_frequency_eigin_state",
        "How often are checkpoints done during eigen state calculations");
    WriteObject(
        p.GetGobbler(), "/Parameters/gobbler",
        "The point at which the gobbler turns on at, (1=100 and 0.9=90)");
    WriteObject(p.GetSigma(), "/Parameters/sigma", "STD of wavefunction guess");
    WriteObject(p.GetTol(), "/Parameters/tol", "Error tolerance in psi");
    WriteObject(p.GetStateSolverIdx(), "/Parameters/state_solver_idx",
                "Index of solver: File:0, ITP:1, Power:2");
    WriteObject(num_pulses, "/Parameters/num_pulses",
                "The number of pulses from the input file");
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
                "the time the pulse starts to turn on.");
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
    std::string err_str;
    /* Check if restart can use current file */
    if (p.GetRestart() == 1)
    {
      err_str += "Restart not implemented yet \n";
    }
    EndRun(err_str);
  }
}

/* TODO(jove7731): more of this restart stuff */
void HDF5Wrapper::ReadRestart(Parameters &p, std::string f_name)
{
  if (world.rank() == 0)
  {
    std::string err_str;
    /* Check if restart can use current file */
    if (p.GetRestart() == 1)
    {
      err_str += "Restart not implemented yet \n";
    }
    EndRun(err_str);
  }
}

/* constructor file_name needs ending ".h5" */
HDF5Wrapper::HDF5Wrapper(std::string f_name, Parameters &p)
{
  if (world.rank() == 0)
  {
    file_name = f_name;
    header    = false;
    if (p.GetRestart() == 1)
    {
      ReadRestart(p, file_name);
    }
    else
    {
      data_file = new H5::H5File(file_name, H5F_ACC_RDWR);
      file_open = true;
      WriteHeader(p);
    }
    DefineComplex();
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
    data_file = new H5::H5File(file_name, H5F_ACC_RDWR);
    file_open = true;
    DefineComplex();
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
    if (p.GetRestart() == 1)
    {
      ReadRestart(p);
    }
    else
    {
      data_file = new H5::H5File(file_name, H5F_ACC_TRUNC);
      file_open = true;
      WriteHeader(p);
    }
    DefineComplex();
    Close();
  }
}

void HDF5Wrapper::DefineComplex()
{
  if (world.rank() == 0)
  {
    complex_data_type = new H5::CompType(sizeof(dcomp(1.0, 1.0)));
    complex_data_type->insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
    complex_data_type->insertMember("i", sizeof(double),
                                    H5::PredType::NATIVE_DOUBLE);
  }
}

void HDF5Wrapper::Open()
{
  if (world.rank() == 0)
  {
    if (!header && !file_open)
    {
      data_file = new H5::H5File(file_name, H5F_ACC_RDWR);
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
      delete data_file;
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

/* destructor */
HDF5Wrapper::~HDF5Wrapper()
{
  if (world.rank() == 0)
  {
    std::cout << "Deleting HDF5Wrapper: " << file_name << "\n";
    Close();
  }
}