#pragma once
#include "Parameters.h"
#include "H5Cpp.h"
#include <complex>
#include <vector>

#define dcomp std::complex<double>

using namespace H5;

class HDF5Wrapper {
private:
    H5File                   *data_file;
    CompType                 *complex_data_type;
    // saves datasets for expendable variables
    std::vector<DataSet*>    extendable_dataset_complex;
    std::vector<std::string> extendable_string_complex;
    std::vector<DataSet*>    extendable_dataset_double;
    std::vector<std::string> extendable_string_double;
public:
    // Constructor
    HDF5Wrapper(Parameters& p);
    HDF5Wrapper(std::string file_name, Parameters& p);

    // destructor
    ~HDF5Wrapper();

    // write single entry and 1D variables
    void write_object(int data, H5std_string var_path);
    void write_object(double data, H5std_string var_path);
    void write_object(int *data, int size, H5std_string var_path);
    void write_object(double *data, int size, H5std_string var_path);
    void write_object(dcomp *data, int size, H5std_string var_path);

    void write_object(int data, H5std_string var_path,
                      H5std_string attribute);
    void write_object(double data, H5std_string var_path,
                      H5std_string attribute);
    void write_object(int *data, int size, H5std_string var_path,
                      H5std_string attribute);
    void write_object(double *data, int size, H5std_string var_path,
                      H5std_string attribute);
    void write_object(dcomp *data, int size, H5std_string var_path,
                      H5std_string attribute);

    // time series writes
    void write_object(dcomp *data, int size, H5std_string var_path,
                      int write_idx);
    void write_object(dcomp *data, int size, H5std_string var_path,
                      H5std_string attribute, int write_idx);
    void write_object(double data, H5std_string var_path,
                      int write_idx);
    void write_object(double data, H5std_string var_path,
                      H5std_string attribute, int write_idx);



    // write for parameters
    void write_header(Parameters& p);

    // reads restart and validates file
    void read_restart(Parameters& p);
    void read_restart(Parameters& p, std::string file_name);

    // kill run
    void end_run(std::string str);
    void end_run(std::string str, int exit_val);
};
