import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5", "r")
p = h5py.File("Pulse.h5", "r")
pulses = p["Pulse"]
p_time = pulses["time"][:]
num_dims = f["Parameters"]["num_dims"][0]
num_electrons = f["Parameters"]["num_electrons"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]

print "Plotting Pulses E"
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fieldX = -1.0 * np.gradient(pulses["field_" + str(0)][:], 
                           f["Parameters"]["delta_t"][0]) * 7.2973525664e-3
fieldY = -1.0 * np.gradient(pulses["field_" + str(1)][:],
                           f["Parameters"]["delta_t"][0]) * 7.2973525664e-3
plt.plot(fieldX, fieldY, p_time)
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("Pulses")
plt.legend()
plt.show()