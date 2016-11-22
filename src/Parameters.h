#pragma once
#include <iostream>

class Parameters {
private:
	double delta_t;
public:
	// Constructor
	Parameters(std::string file_name);

	void set_delta_t(double dt);

	double get_delta_t();
};