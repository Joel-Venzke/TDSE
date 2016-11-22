#include <iostream>
#include <fstream>
#include <streambuf>
#include "json.cpp"
#include "Parameters.h"


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
	json data = file_to_json(file_name);
	std::cout << data.dump();
}

void Parameters::set_delta_t(double dt) {
	delta_t = dt;
}

double Parameters::get_delta_t() {
	return delta_t;
}
