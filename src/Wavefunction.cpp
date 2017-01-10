#include <iostream>
#include "Wavefunction.h"

Wavefunction::Wavefunction( Parameters & p) {
	test = 1;
	std::cout << "Wavefunction\n";
}

// destructor
Wavefunction::~Wavefunction(){
	std::cout << "Deleting Wavefunction\n";
}
