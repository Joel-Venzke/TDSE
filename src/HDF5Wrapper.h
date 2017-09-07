#pragma once
#include <petsc.h>
#include <complex>
#include <memory>
#include <vector>
#include "H5Cpp.h"
#include "Parameters.h"
#include "Utils.h"
#include "hdf5.h"

class HDF5Wrapper : protected Utils
{
 private:
  mpi::communicator world;
  std::shared_ptr< H5::H5File > data_file;
  std::string file_name;
  bool file_open;
  bool header;

  std::unique_ptr< hsize_t[] > GetHsizeT(int size, int *dims, bool complex);
  std::unique_ptr< hsize_t[] > GetHsizeT(int size, bool complex);
  void WriteAttribute(H5std_string &var_path, H5std_string &attribute);

 public:
  /* ructor */
  HDF5Wrapper(Parameters &p);
  HDF5Wrapper(std::string f_name, Parameters &p);
  HDF5Wrapper(std::string f_name, std::string mode = "wr");

  /* destructor */
  ~HDF5Wrapper();

  void Open();
  void Close();

  template < typename T >
  H5::PredType getter(T &data);

  void SetHeader(bool h);

  void CreateGroup(H5std_string group_path);

  /* begin template*/
  template < typename T >
  void WriteObject(T data, H5std_string var_path);
  template < typename T >
  void WriteObject(T data, H5std_string var_path, H5std_string attribute);

  template < typename T >
  void WriteObject(T data, int size, H5std_string var_path);
  template < typename T >
  void WriteObject(T data, int size, H5std_string var_path,
                   H5std_string attribute);

  template < typename T >
  void WriteObject(T data, int size, int *dims, H5std_string var_path);
  template < typename T >
  void WriteObject(T data, int size, int *dims, H5std_string var_path,
                   H5std_string attribute);

  template < typename T >
  void WriteObject(T data, H5std_string var_path, int write_idx);
  template < typename T >
  void WriteObject(T data, H5std_string var_path, H5std_string attribute,
                   int write_idx);

  template < typename T >
  void WriteObject(T data, int size, H5std_string var_path, int write_idx);
  template < typename T >
  void WriteObject(T data, int size, H5std_string var_path,
                   H5std_string attribute, int write_idx);

  PetscInt GetTimeIdx(H5std_string var_path, bool complex = false);
  double GetLast(H5std_string var_path);

  /* write for parameters */
  void WriteHeader(Parameters &p);

  /* kill run */
  void EndRun(std::string str);
  void EndRun(std::string str, int exit_val);
};