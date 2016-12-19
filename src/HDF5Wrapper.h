#pragma once
#include "Parameters.h"

class HDF5Wrapper {
private:
	H5File * data_file;
public:
	// Constructor
	HDF5Wrapper(Parameters& p);
	HDF5Wrapper(std::string file_name, Parameters& p);

	// destructor
	~HDF5Wrapper();

	// write 1D variables
	void write_object(int data, H5std_string var_path);
	void write_object(double data, H5std_string var_path);
	void write_object(int *data, int size, H5std_string var_path);
	void write_object(double *data, int size, H5std_string var_path);


	// write for parameters
	void write_header(Parameters& p);
	void read_restart(Parameters& p);
	void read_restart(Parameters& p, std::string file_name);

	// kill run
	void end_run(std::string str);
	void end_run(std::string str, int exit_val);
};