#pragma once
#include <iostream>
#include "Parameters.h"
#include "HDF5Wrapper.h"

class Pulse {
private:
	int test;
public:
	// Constructor
	Pulse(HDF5Wrapper& data_file, Parameters& p);
};