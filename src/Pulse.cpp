#include "Pulse.h"
#include <math.h>    // ceil()

Pulse::Pulse(HDF5Wrapper& data_file, Parameters& p) {
    int pulse_length = 0;

    std::cout << "Creating pulses\n" << std::flush;

    // get number of pulses and dt from Parameters
    pulse_alloc      = false;
    num_pulses       = p.get_num_pulses();
    delta_t          = p.get_delta_t();
    max_pulse_length = 0; // stores longest pulse

    // allocate arrays
    pulse_shape_idx = new int[num_pulses];
    cycles_on       = new double[num_pulses];
    cycles_plateau  = new double[num_pulses];
    cycles_off      = new double[num_pulses];
    cycles_delay    = new double[num_pulses];
    cycles_total    = new double[num_pulses];
    cep             = new double[num_pulses];
    energy          = new double[num_pulses];
    e_max           = new double[num_pulses];

    // get data from Parameters
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

        // calculate length (number of array cells) of each pulse
        pulse_length       = ceil(2.0*pi*cycles_total[i]/
                             (energy[i]*delta_t))+1;

        // find the largest
        if (pulse_length>max_pulse_length){
            max_pulse_length = pulse_length;
        }
    }

    initialize_time();
    initialize_pulse();
    initialize_a_field();

    checkpoint(data_file);

    deallocate_pulses();

    std::cout << "Pulses created\n" << std::flush;
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
    delete time;
    if (pulse_alloc) {
        for (int i = 0; i < num_pulses; ++i) {
            delete pulse_value[i];
            delete pulse_envelope[i];
        }
        delete[] pulse_value;
        delete[] pulse_envelope;
    }
    delete   a_field;
}

// Build array with time in au
void Pulse::initialize_time() {
    time = new double[max_pulse_length];
    for (int i = 0; i < max_pulse_length; ++i) {
        time[i] = i*delta_t;
    }
}

// build the nth pulse
void Pulse::initialize_pulse(int n){
    int on_start, plateau_start, off_start, off_end;
    double period = 2*pi/energy[n];
    double s1;

    // index that turns pulse on
    on_start      = ceil(period*cycles_delay[n]/
                         (delta_t))+1;

    // index that holds pulse at max
    plateau_start = ceil(period*(cycles_on[n]+cycles_delay[n])/
                         (delta_t))+1;

    // index that turns pulse off
    off_start     = ceil(period*
                        (cycles_plateau[n]+cycles_on[n]+
                        cycles_delay[n])/
                        (delta_t))+1;

    // index that holds pulse at 0
    off_end       = ceil(period*
                        (cycles_off[n]+cycles_plateau[n]+
                          cycles_on[n]+cycles_delay[n])/
                          (delta_t))+1;
    if (! pulse_alloc) {
        pulse_envelope[n] = new double[max_pulse_length];
        pulse_value[n] = new double[max_pulse_length];
    }
    for (int i = 0; i < max_pulse_length; ++i) {
        if (i<on_start){ // pulse still off
            pulse_envelope[n][i] = 0.0;
        } else if (i < plateau_start) { // pulse ramping on
            s1 = sin(energy[n]*delta_t*(i-on_start)/
                    (4.0*cycles_on[n]));
            pulse_envelope[n][i] = e_max[n]*s1*s1;
        } else if (i < off_start) { // pulse at max
            pulse_envelope[n][i] = e_max[n];
        } else if (i < off_end) { // pulse ramping off
            s1 = sin(energy[n]*delta_t*(i-off_start)/
                    (4.0*cycles_off[n]));
            pulse_envelope[n][i] = e_max[n]*(1-(s1*s1));
        } else { // pulse is off
            pulse_envelope[n][i] = 0.0;
        }

        // calculate the actual pulse
        pulse_value[n][i] = pulse_envelope[n][i]*
                               sin(energy[n]*
                                   delta_t*(i-on_start)+
                                   cep[n]*2*pi);
    }
}

// sets up all pulses and calculates the a_field
void Pulse::initialize_pulse(){
    // set up the input pulses
    if (! pulse_alloc) {
        pulse_value     = new double*[num_pulses];
        pulse_envelope  = new double*[num_pulses];
    }
    for (int i = 0; i < num_pulses; ++i) {
        initialize_pulse(i);
    }
    pulse_alloc = true;
}

void Pulse::initialize_a_field() {
    // calculate the a_field by summing each pulse
    // TODO: add support for setting e_field
    if (! pulse_alloc) {
        initialize_pulse();
    }
    a_field = new double[max_pulse_length];
    for (int i = 0; i < max_pulse_length; ++i) {
        a_field[i] = 0;
        for (int j = 0; j < num_pulses; ++j) {
            a_field[i] += pulse_value[j][i];
        }
    }
}

// write out the state of the pulse
void Pulse::checkpoint(HDF5Wrapper& data_file) {
    // write time, a_field, and a_field_envelope to hdf5
    data_file.write_object(time,max_pulse_length,"/Pulse/time",
        "The time for each index of the pulse in a.u.");
    data_file.write_object(a_field,max_pulse_length,"/Pulse/a_field",
        "The value of the A field at each point in time in a.u.");

    if (pulse_alloc) {
        // write each pulse both value and envelope
        for (int i = 0; i < num_pulses; ++i) {
            data_file.write_object(pulse_envelope[i],max_pulse_length,
                "/Pulse/Pulse_envelope_"+std::to_string(i),
                "The envelope function for the "+std::to_string(i)+
                " pulse in the input file");
            data_file.write_object(pulse_value[i],max_pulse_length,
                "/Pulse/Pulse_value_"+std::to_string(i),
                "The pulse value for the "+std::to_string(i)+
                " pulse in the input file");
        }
    }
}

void Pulse::deallocate_pulses() {
    if (pulse_alloc) {
        for (int i = 0; i < num_pulses; ++i) {
            delete pulse_value[i];
            delete pulse_envelope[i];
        }
        delete[] pulse_value;
        delete[] pulse_envelope;
        pulse_alloc = false;
    }
}

double* Pulse::get_a_field() {
    return a_field;
}

double* Pulse::get_time(){
    return time;
}

int Pulse::get_max_pulse_length(){
    return max_pulse_length;
}
