import json
import numpy as np
import os

# copy base input file here
data = {
 "start_state": {
  "phase": [
   0.0
  ],
  "index": [
   0
  ],
  "amplitude": [
   1.0
  ]
 },
 "states": 6,
 "write_frequency_checkpoint": 100,
 "gauge": "Velocity",
 "write_frequency_observables": 1,
 "dimensions": [
  {
   "delta_x_max_start": 4.0,
   "delta_x_min": 0.1,
   "delta_x_min_end": 4.0,
   "dim_size": 5000.0,
   "delta_x_max": 0.1
  }
 ],
 "delta_t": 0.1,
 "tol": 1e-10,
 "num_electrons": 1,
 "gobbler": 0.95,
 "propagate": 1,
 "alpha": 1.0,
 "restart": 0,
 "state_solver": "File",
 "write_frequency_eigin_state": 1,
 "laser": {
 "experiment_type": "streaking",
 "pulses": [
   {
    "power_on": 2.0,
    "cycles_on": 1.5,
    "cycles_off": 1.5,
    "pulse_shape": "sin",
    "cycles_delay": 0.0,
    "polarization_vector": [
     0.0,
     1.0,
     0.0
    ],
    "energy": 0.057,
    "power_off": 2.0,
    "intensity": 100000000000.0,
    "cycles_plateau": 0.0,
    "helicity": "left",
    "cep": 0.0,
    "poynting_vector": [
     0.0,
     0.0,
     1.0
    ],
    "ellipticity": 0.0
   },
   {
   "power_on": 1.0,
   "cycles_on": 2.0,
   "cycles_off": 2.0,
   "pulse_shape": "gaussian",
   "tau_delay": 0.0,
   "polarization_vector": [
                           0.0,
                           1.0,
                           0.0
                           ],
   "energy": 3.01,
   "power_off": 2.0,
   "intensity": 1e13,
   "cycles_plateau": 0.0,
   "helicity": "left",
   "cep": 0.0,
   "poynting_vector": [
                       0.0,
                       0.0,
                       1.0
                       ],
   "ellipticity": 0.0
   }
  ]
 },
 "target": {
  "nuclei": [
   {
    "z": 3.0,
    "location": [
     0.0,
     0.0,
     0.0
    ],
    "SAE": {
     "a": [
      0.66294407
     ],
     "z_c": 1.0,
     "b": [
      4.073942
     ],
     "c0": 1.0,
     "r0": 2.353059
    }
   }
  ],
  "name": "soft"
 },
 "coordinate_system": "Cartesian",
 "field_max_states": 0,
 "free_propagate": 0,
 "sigma": 3.0,
 "order": 2
}
 
# loop over thing you want to change
for i in np.arange(-150.0, 160.0, 10.0):
  # make a folder
    fold =  "tau%.2f" % i
    print fold
    os.mkdir(fold)
  
  # update the parameter of interest
    data["laser"]["pulses"][0]["tau_delay"] \
    = i * data["laser"]["pulses"][0]["energy"] \
      / (2 * np.pi)
  # write input file
    with open(fold + "/input.json", 'w') as f:
        f.write(json.dumps(data, sort_keys='True', indent=2))
