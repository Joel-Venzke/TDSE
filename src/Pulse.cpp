#include "Pulse.h"
#include <math.h>    // ceil()

Pulse::Pulse(HDF5Wrapper& data_file, Parameters& p) {
	std::cout << "Creating pulses\n" << std::flush;
	num_pulses       = p.get_num_pulses();
	delta_t          = p.get_delta_t();
	max_pulse_length = 0;

	pulse_shape_idx = new int[num_pulses];
    cycles_on       = new double[num_pulses];
    cycles_plateau  = new double[num_pulses];
    cycles_off      = new double[num_pulses];
    cycles_delay    = new double[num_pulses];
    cycles_total    = new double[num_pulses];
    cep             = new double[num_pulses];
    energy          = new double[num_pulses];
    e_max           = new double[num_pulses];
    pulse_length    = new int[num_pulses];
    pulse_value     = new double*[num_pulses];
    pulse_envelope  = new double*[num_pulses];

    for (int i = 0; i < num_pulses; ++i) {
        pulse_shape_idx[i] = p.get_pulse_shape_idx()[i];
        cycles_on[i]       = p.get_cycles_on()[i];
        cycles_plateau[i]  = p.get_cycles_plateau()[i];
        cycles_off[i]      = p.get_cycles_off()[i];
        cycles_delay[i]    = p.get_cycles_delay()[i];
        cycles_total[i]    = cycles_delay[i]+cycles_on[i]+
                             cycles_plateau[i]+cycles_off[i];
        cep[i]             = p.get_cep()[i];
        energy[i]          = p.get_energy()[i];
        e_max[i]           = p.get_e_max()[i];
        pulse_length[i]    = ceil(2.0*pi*cycles_total[i]/
        	                 (energy[i]*delta_t));
        if (pulse_length[i]>max_pulse_length){
        	max_pulse_length = pulse_length[i];
        }
    }

    initialize_time();
    initialize_pulse();

    checkpoint(data_file);
	std::cout << "Pulses created" << std::flush;
}

Pulse::~Pulse() {
	std::cout << "Deleting Pulse\n" << std::flush;
	delete pulse_shape_idx;
    delete cycles_on;
    delete cycles_plateau;
    delete cycles_off;
    delete cycles_delay;
    delete cycles_total;
    delete cep;
    delete energy;
    delete e_max;
    delete pulse_length;
    delete time;
    for (int i = 0; i < num_pulses; ++i)
    {
    	delete pulse_value[i];
    	delete pulse_envelope[i];
    }
    delete[] pulse_value;
    delete[] pulse_envelope;
}

void Pulse::initialize_time() {
	time = new double[max_pulse_length];
	for (int i = 0; i < max_pulse_length; ++i)
	{
		time[i] = i*delta_t;
	}
}

void Pulse::initialize_pulse(int idx){
	int on_start, plateau_start, off_start, off_end;
	double period = 2*pi/energy[idx];
	double s1;

	// index that turns pulse on
	on_start      = ceil(period*cycles_delay[idx]/
		                 (delta_t));

	// index that holds pulse at max
	plateau_start = ceil(period*(cycles_on[idx]+cycles_delay[idx])/
		                 (delta_t));

	// index that turns pulse off
	off_start     = ceil(period*
		                (cycles_plateau[idx]+cycles_on[idx]+
		                cycles_delay[idx])/
		                (delta_t));	

	// index that holds pulse at 0
	off_end       = ceil(period*
		                (cycles_off[idx]+cycles_plateau[idx]+
		      	        cycles_on[idx]+cycles_delay[idx])/
		      	        (delta_t));

	pulse_envelope[idx] = new double[max_pulse_length];
	pulse_value[idx] = new double[max_pulse_length];
	for (int i = 0; i < max_pulse_length; ++i)
	{

		if (i<on_start){ // pulse still off
			pulse_envelope[idx][i] = 0.0;
		} else if (i < plateau_start) { // pulse ramping on
			s1 = sin(energy[idx]*delta_t*(i-on_start)/
				    (4.0*cycles_on[idx]));
			pulse_envelope[idx][i] = s1*s1;
		} else if (i < off_start) { // pulse at max
			pulse_envelope[idx][i] = 1.0;
		} else if (i < off_end) { // pulse ramping off
			s1 = sin(energy[idx]*delta_t*(i-off_start)/
				    (4.0*cycles_off[idx]));
			pulse_envelope[idx][i] = 1-(s1*s1);
		} else { // pulse is off
			pulse_envelope[idx][i] = 0.0;
		}

		pulse_value[idx][i] = e_max[idx]*pulse_envelope[idx][i]*
		                       sin(energy[idx]*
		                       	delta_t*(i-on_start)+
		                      	cep[idx]*2*pi);
	}
}

void Pulse::initialize_pulse(){
	for (int i = 0; i < num_pulses; ++i) {
		initialize_pulse(i);
	}

	a_field = new double[max_pulse_length];
	for (int i = 0; i < max_pulse_length; ++i) {
		a_field[i] = 0;
		for (int j = 0; j < num_pulses; ++j) {
			a_field[i] += pulse_value[j][i];
		}
	}
}

void Pulse::checkpoint(HDF5Wrapper& data_file) {
	data_file.write_object(time,max_pulse_length,"/Pulse/time");
	data_file.write_object(a_field,max_pulse_length,"/Pulse/a_field");
	for (int i = 0; i < num_pulses; ++i) {
		data_file.write_object(pulse_envelope[i],max_pulse_length,
			"/Pulse/Pulse_envelope_"+std::to_string(i));
		data_file.write_object(pulse_value[i],max_pulse_length,
			"/Pulse/Pulse_value_"+std::to_string(i));
	}
}
