#pragma once
#include "H5Cpp.h"
#include "Parameters.h"
// #include "hdf5.h"
#include <petsc.h>
#include <complex>
#include <vector>

#define dcomp std::complex<double>

class HDF5Wrapper
{
 private:
  PetscInt rank;
  H5::H5File *data_file;
  std::string file_name;
  bool file_open;
  bool header;
  H5::CompType *complex_data_type;

  hsize_t *GetHsizeT(int size, int *dims);
  void DefineComplex();

 public:
  /* Constructor */
  HDF5Wrapper(Parameters &p);
  HDF5Wrapper(std::string f_name, Parameters &p);
  HDF5Wrapper(std::string f_name);

  /* destructor */
  ~HDF5Wrapper();

  void Open();
  void Close();

  void SetHeader(bool h);

  void CreateGroup(H5std_string group_path);

  // write single entry
  void WriteObject(int data, H5std_string var_path);
  void WriteObject(int data, H5std_string var_path, H5std_string attribute);

  void WriteObject(double data, H5std_string var_path);
  void WriteObject(double data, H5std_string var_path, H5std_string attribute);

  void WriteObject(int *data, int size, H5std_string var_path);
  void WriteObject(int *data, int size, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(int *data, int size, int *dims, H5std_string var_path);
  void WriteObject(int *data, int size, int *dims, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(double *data, int size, H5std_string var_path);
  void WriteObject(double *data, int size, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(double *data, int size, int *dims, H5std_string var_path);
  void WriteObject(double *data, int size, int *dims, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(dcomp *data, int size, H5std_string var_path);
  void WriteObject(dcomp *data, int size, H5std_string var_path,
                   H5std_string attribute);

  void WriteObject(dcomp *data, int size, int *dims, H5std_string var_path);
  void WriteObject(dcomp *data, int size, int *dims, H5std_string var_path,
                   H5std_string attribute);

  /* time series writes */
  void WriteObject(dcomp *data, int size, H5std_string var_path, int write_idx);
  void WriteObject(dcomp *data, int size, H5std_string var_path,
                   H5std_string attribute, int write_idx);

  void WriteObject(double data, H5std_string var_path, int write_idx);
  void WriteObject(double data, H5std_string var_path, H5std_string attribute,
                   int write_idx);

  /* write for parameters */
  void WriteHeader(Parameters &p);

  /* reads restart and validates file */
  void ReadRestart(Parameters &p);
  void ReadRestart(Parameters &p, std::string f_name);

  /* kill run */
  void EndRun(std::string str);
  void EndRun(std::string str, int exit_val);
};