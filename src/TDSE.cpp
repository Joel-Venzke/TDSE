#include <iostream>
#include "Hamiltonian.h"
#include "Observables.h"
#include "Parameters.h"
#include "Pulse.h"
#include "Simulation.h"
#include "Wavefuction.h"


int main() {

    Hamiltonian h;
    Observables o;
    Parameters p("input.json");
    p.set_delta_t(3.14);
    std::cout << p.get_delta_t() << "\n";
    Pulse p1;
    Simulation s;
    Wavefuction w;

    // std::cout << "Hello\n";
    // json j;
    // j["array_test"] = {2,3,4};
    // j["arrays"]["pulse_1"] = {2,3,4};
    // j["arrays"]["pulse_2"] = {2,3,4};
    // j["arrays"]["pulse_2"].push_back({3,2,3,4,5});
    
    // int a[3] = {2,3,4};
    // for ( int i=0; i<3; i++ ){
    //     std::cout << a[i] << "\n";
    //     j["arrays"]["pulse_2"].push_back(a[i]);
    // }

    // std::cout << j.dump() << std::endl;
    // std::cout << j["arrays"]["pulse_2"][2] << std::endl;

    // std::cout << file_to_string("file.txt") << "\n";

    // json j2 = file_to_string("file.txt");
    // std::string s = j2.dump(4);
    // std::cout << s << "\n";
    return 0;
}
