#include <fstream>
#include <streambuf>
#include "json.cpp"
#include "Parameters.h"
#include <cstdlib>     // exit
#include "H5Cpp.h"     // hdf5

// for convenience
using json = nlohmann::json;
using namespace H5;


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


// prints error message, kills code and returns -1
void Parameters::end_run(std::string str) {
    std::cout << "\n\nERROR: " << str << "\n" << std::flush;
    exit(-1);
}

// prints error message, kills code and returns exit_val
void Parameters::end_run(std::string str, int exit_val) {
    std::cout << "\n\nERROR: " << str << "\n";
    exit(exit_val);
}

// Constructor
Parameters::Parameters(std::string file_name) {
    std::cout << "Reading input file: "<< file_name <<"\n";
    std::cout << std::flush;
    // read data from file
    json data  = file_to_json(file_name);

    std::cout << "input file: "<< file_name <<"\n";
    std::cout << std::flush;

    // get numeric information
    delta_t    = data["delta_t"];
    num_dims   = data["dimensions"].size();
    dim_size   = new double[num_dims];
    delta_x    = new double[num_dims];
    for (int i = 0; i < num_dims; ++i) {
        dim_size[i]  = data["dimensions"][i]["dim_size"];
        delta_x[i]   = data["dimensions"][i]["delta_x"];

        // this should be small
        std::cout << "Dim-" << i << " dx/dt^2 = ";
        std::cout << delta_x[i]/delta_t*delta_t << "\n";
    }

    // get simulation behavior;
    restart         = data["restart"];
    target          = data["target"];
    alpha           = data["alpha"];
    beta            = data["beta"];
    write_frequency = data["write_frequency"];
    sigma           = data["sigma"];
    tol             = data["tol"];
    state_solver    = data["state_solver"];
    num_states      = data["states"].size();

    state_energy    = new double[num_states];

    for (int i=0; i<num_states; i++) {
        state_energy[i] = data["states"][i]["energy"];
    }

    // index is used throughout code for efficiency
    // and ease of writing to hdf5
    if (target=="He") {
        target_idx = 0;
        z          = 2.0; // He atomic number
    }

    if (state_solver == "File") {
        state_solver_idx = 0;
    } else if ( state_solver == "ITP" ) {
        state_solver_idx = 1;
    } else if ( state_solver == "Power" ) {
        state_solver_idx = 2;
    }

    // get pulse information
    num_pulses = data["pulses"].size();

    // allocate memory
    pulse_shape     = new std::string[num_pulses];
    pulse_shape_idx = new int[num_pulses];
    cycles_on       = new double[num_pulses];
    cycles_plateau  = new double[num_pulses];
    cycles_off      = new double[num_pulses];
    cycles_delay    = new double[num_pulses];
    cep             = new double[num_pulses];
    energy          = new double[num_pulses];
    e_max           = new double[num_pulses];

    // read data
    for (int i = 0; i < num_pulses; ++i) {
        pulse_shape[i]    = data["pulses"][i]["pulse_shape"];
        // index used similar target_idx
        if (pulse_shape[i]=="sin2") {
            pulse_shape_idx[i] = 0;
        } else if (pulse_shape[i]=="linear") {
            pulse_shape_idx[i] = 1;
        }

        cycles_on[i]      = data["pulses"][i]["cycles_on"];
        cycles_plateau[i] = data["pulses"][i]["cycles_plateau"];
        cycles_off[i]     = data["pulses"][i]["cycles_off"];
        cycles_delay[i]   = data["pulses"][i]["cycles_delay"];
        cep[i]            = data["pulses"][i]["cep"];
        energy[i]         = data["pulses"][i]["energy"];
        e_max[i]          = data["pulses"][i]["e_max"];

    }

    // ensure input is good
    validate();

    std::cout << "Reading input complete\n" << std::flush;
}

Parameters::~Parameters(){
    std::cout << "Deleting Parameters\n" << std::flush;
    delete dim_size;
    delete delta_x;
    delete[] pulse_shape;
    delete pulse_shape_idx;
    delete cycles_on;
    delete cycles_plateau;
    delete cycles_off;
    delete cycles_delay;
    delete cep;
    delete energy;
    delete e_max;
}

// checks important input parameters for errors
void Parameters::validate(){

    std::cout << "Validating input\n";
    std::string err_str;
    bool error_found = false; // set to true if error is found

    // Check pulses
    for (int i=0; i<num_pulses; i++ ){
        // Check pulse shapes
        if (pulse_shape[i]!="sin2"){
            error_found = true;
            err_str += "\nPulse ";
            err_str += std::to_string(i);
            err_str += " has unsupported pulse shape: \"";
            err_str += pulse_shape[i]+"\"\n";
            err_str += "Current support includes: ";
            err_str += "\"sin2\"\n";
        }

        // check e_max
        if (e_max[i] <= 0) {
            error_found = true;
            err_str += "\nPulse ";
            err_str += std::to_string(i);
            err_str += " has unsupported e_max: \"";
            err_str += std::to_string(e_max[i])+"\"\n";
            err_str += "e_max should be greater than 0\n";
        }

        if (cycles_on[i] < 0) {
            error_found = true;
            err_str += "\nPulse ";
            err_str += std::to_string(i);
            err_str += " has cycles_on: \"";
            err_str += std::to_string(cycles_on[i])+"\"\n";
            err_str += "cycles_on should be >= 0\n";
        }

        if (cycles_plateau[i] < 0) {
            error_found = true;
            err_str += "\nPulse ";
            err_str += std::to_string(i);
            err_str += " has cycles_plateau: \"";
            err_str += std::to_string(cycles_plateau[i])+"\"\n";
            err_str += "cycles_plateau should be >= 0\n";
        }

        if (cycles_off[i] < 0) {
            error_found = true;
            err_str += "\nPulse ";
            err_str += std::to_string(i);
            err_str += " has cycles_off: \"";
            err_str += std::to_string(cycles_off[i])+"\"\n";
            err_str += "cycles_off should be >= 0\n";
        }

        // exclude delay because it is zero anyways
        // pulses must exist so we don't run supper long time scales
        double p_length = cycles_on[i] + cycles_off[i] +
                          cycles_plateau[i];
        if (p_length<=0) {
            error_found = true;
            err_str += "\nPulse ";
            err_str += std::to_string(i);
            err_str += " has length: \"";
            err_str += std::to_string(p_length)+"\"\n";
            err_str += "the length should be > 0\n";
        }
    }

    // target
    if (target!="He") {
        error_found = true;
        err_str += "\nInvalid target: \"";
        err_str += target;
        err_str += "\" \nvalid targets are \"He\"";
    }

    if (state_solver=="file") {
        error_found = true;
        err_str += "\nInvalid state solver: \"";
        err_str += "States from file is not supported yet\"";
    } else if (state_solver!="ITP" && state_solver!="Power") {
        error_found = true;
        err_str += "\nInvalid state solver: \"";
        err_str += state_solver;
        err_str += "\" \nvalid solvers are \"ITP\" and \"Power\"\n";
    }

    // exit here to get all errors in one run
    if (error_found) {
        end_run(err_str);
    } else {
        std::cout << "Input valid\n";
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

int Parameters::get_restart() {
    return restart;
}

std::string Parameters::get_target() {
    return target;
}

int Parameters::get_target_idx(){
    return target_idx;
}

double Parameters::get_z() {
    return z;
}

double Parameters::get_alpha() {
    return alpha;
}

double Parameters::get_beta() {
    return beta;
}

int Parameters::get_write_frequency() {
    return write_frequency;
}

double Parameters::get_sigma() {
    return sigma;
}

int Parameters::get_num_states() {
    return num_states;
}

double* Parameters::get_state_energy() {
    return state_energy;
}

double Parameters::get_tol() {
    return tol;
}

int Parameters::get_state_solver_idx() {
    return state_solver_idx;
}

std::string Parameters::get_state_solver() {
    return state_solver;
}

int Parameters::get_num_pulses() {
    return num_pulses;
}

std::string* Parameters::get_pulse_shape() {
    return pulse_shape;
}

int* Parameters::get_pulse_shape_idx(){
    return pulse_shape_idx;
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

double* Parameters::get_cycles_delay() {
    return cycles_delay;
}
