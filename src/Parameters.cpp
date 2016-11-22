#include <iostream>
#include <fstream>
#include <streambuf>
#include <math.h>    // ceil
#include "json.cpp"
#include "Parameters.h"
#include <netcdfcpp.h>

// for convenience
using json = nlohmann::json;

/*
* Reads file and returns a string
*/
std::string file_to_string(std::string file_name) {
    // TODO: check if file exists
    std::ifstream t(file_name);
    std::string str((std::istreambuf_iterator<char>(t)),
                     std::istreambuf_iterator<char>());
    return str;
}

/*
* Reads file and returns a json object
*/
json file_to_json(std::string file_name) {
    auto j = json::parse(file_to_string(file_name));
    return j;
}

// Constructor
Parameters::Parameters(std::string file_name) {
    std::cout << "Reading input file: "<< file_name <<"\n";
    std::cout << std::flush;
    // read data from file
	json data  = file_to_json(file_name);

    // get numeric information
    delta_t    = data["delta_t"]; 
    num_dims   = data["dimensions"].size();
    grid_dims  = new NcDim*[num_dims];
    dim_size   = new double[num_dims];
    delta_x    = new double[num_dims];
    size_of_x  = new int[num_dims];
    for (int i = 0; i < num_dims; ++i)
    {
        dim_size[i]  = data["dimensions"][i]["dim_size"];
        delta_x[i]   = data["dimensions"][i]["delta_x"];
        size_of_x[i] = ceil(dim_size[i]/delta_x[i])+1;

        // std::cout << dim_size[i] << " ";
        // std::cout << delta_x[i] << " ";
        // std::cout << size_of_x[i] << "\n";
    }

    // get simulation behavior;
    restart    = data["restart"];
    target     = data["target"];

    // get pulse information
    num_pulses = data["pulses"].size();

    // allocate memory
    pulse_shape    = new std::string[num_pulses];
    cycles_on      = new double[num_pulses];
    cycles_plateau = new double[num_pulses];
    cycles_off     = new double[num_pulses];
    cep            = new double[num_pulses];
    energy         = new double[num_pulses];
    e_max          = new double[num_pulses];

    // read data
    for (int i = 0; i < num_pulses; ++i)
    {
        pulse_shape[i]    = data["pulses"][i]["pulse_shape"];
        cycles_on[i]      = data["pulses"][i]["cycles_on"];
        cycles_plateau[i] = data["pulses"][i]["cycles_plateau"];
        cycles_off[i]     = data["pulses"][i]["cycles_off"];
        cep[i]            = data["pulses"][i]["cep"];
        energy[i]         = data["pulses"][i]["energy"];
        e_max[i]          = data["pulses"][i]["e_max"];

        // std::cout << pulse_shape[i] << " ";
        // std::cout << cycles_on[i] << " ";
        // std::cout << cycles_plateau[i] << " ";
        // std::cout << cycles_off[i] << " ";
        // std::cout << cep[i] << " ";
        // std::cout << energy[i] << " ";
        // std::cout << e_max[i] << "\n";
    }

    // write_header(nc_data_file);

    std::cout << "Input file read\n" << std::flush;
}


// write netcdf header
void Parameters::write_header(NcFile * nc_data_file) {
    std::string str;
    data_file = nc_data_file;
    data_file->add_dim("num_dims", num_dims);
    for (int i = 0; i < num_dims; ++i) {
        str = "size_of_"+std::to_string(i);
        grid_dims[i] = data_file->add_dim(str.c_str(), size_of_x[i]);
    }
}


// getters
double Parameters::get_delta_t() {
	return delta_t;
}

int Parameters::get_num_dims(){
    return num_dims;
}

double* Parameters::get_dim_size(){
    return dim_size;
}

double* Parameters::get_delta_x(){
    return delta_x;
}

int* Parameters::get_size_of_x(){
    return size_of_x;
}


int Parameters::get_restart() {
    return restart;
}

std::string Parameters::get_target() {
    return target;
}

int Parameters::get_num_pulses() {
    return num_pulses;
}

std::string* Parameters::get_pulse_shape() {
    return pulse_shape;
}


double* Parameters::get_cycles_on() {
    return cycles_on;
}

double* Parameters::get_cycles_plateau() {
    return cycles_plateau;
}

double* Parameters::get_cycles_off() {
    return cycles_off;
}

double* Parameters::get_cep() {
    return cep;
}

double* Parameters::get_energy() {
    return energy;
}

double* Parameters::get_e_max() {
    return e_max;
}
