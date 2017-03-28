#pragma once
#include "H5Cpp.h"
#include "Parameters.h"
// #include "hdf5.h"
#include <petsc.h>
#include <complex>
#include <vector>

#define dcomp std::complex<double>

using namespace H5;

class HDF5Wrapper
{
 private:
  PetscInt rank;
  H5File *data_file;
  std::string file_name;
  bool file_open;
  bool header;
  CompType *complex_data_type;

  hsize_t *get_hsize_t(int size, int *dims);
  void define_complex();

 public:
  // Constructor
  HDF5Wrapper(Parameters &p);
  HDF5Wrapper(std::string f_name, Parameters &p);
  HDF5Wrapper(std::string f_name);

  // destructor
  ~HDF5Wrapper();

  void reopen();
  void close();

  void set_header(bool h);

  void create_group(H5std_string group_path);

  // write single entry
  void write_object(int data, H5std_string var_path);
  void write_object(int data, H5std_string var_path, H5std_string attribute);

  void write_object(double data, H5std_string var_path);
  void write_object(double data, H5std_string var_path, H5std_string attribute);

  void write_object(int *data, int size, H5std_string var_path);
  void write_object(int *data, int size, H5std_string var_path,
                    H5std_string attribute);

  void write_object(int *data, int size, int *dims, H5std_string var_path);
  void write_object(int *data, int size, int *dims, H5std_string var_path,
                    H5std_string attribute);

  void write_object(double *data, int size, H5std_string var_path);
  void write_object(double *data, int size, H5std_string var_path,
                    H5std_string attribute);

  void write_object(double *data, int size, int *dims, H5std_string var_path);
  void write_object(double *data, int size, int *dims, H5std_string var_path,
                    H5std_string attribute);

  void write_object(dcomp *data, int size, H5std_string var_path);
  void write_object(dcomp *data, int size, H5std_string var_path,
                    H5std_string attribute);

  void write_object(dcomp *data, int size, int *dims, H5std_string var_path);
  void write_object(dcomp *data, int size, int *dims, H5std_string var_path,
                    H5std_string attribute);

  // time series writes
  void write_object(dcomp *data, int size, H5std_string var_path,
                    int write_idx);
  void write_object(dcomp *data, int size, H5std_string var_path,
                    H5std_string attribute, int write_idx);

  void write_object(double data, H5std_string var_path, int write_idx);
  void write_object(double data, H5std_string var_path, H5std_string attribute,
                    int write_idx);

  // write for parameters
  void write_header(Parameters &p);

  // reads restart and validates file
  void read_restart(Parameters &p);
  void read_restart(Parameters &p, std::string f_name);

  // kill run
  void end_run(std::string str);
  void end_run(std::string str, int exit_val);
};