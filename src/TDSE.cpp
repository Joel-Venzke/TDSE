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

    return 0;
}
