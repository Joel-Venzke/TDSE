import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import h5py

f = h5py.File("TDSE.h5", "r")
p = h5py.File("Pulse.h5", "r")
pulses = p["Pulse"]
p_time = pulses["time"][:]
num_dims = f["Parameters"]["num_dims"][0]
num_electrons = f["Parameters"]["num_electrons"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
print "Plotting E Field"
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax  = fig.add_subplot(111, projection='polar')

# for dim_idx in range(num_dims):
fieldX = -1.0 * np.gradient(pulses["field_" + str(0)][:], 
                           f["Parameters"]["delta_t"][0]) * 7.2973525664e-3
fieldY = -1.0 * np.gradient(pulses["field_" + str(1)][:],
                           f["Parameters"]["delta_t"][0]) * 7.2973525664e-3

radius = np.sqrt(fieldX**2 + fieldY**2)
theta  = np.arctan2(fieldY, fieldX)
print radius[np.argmax(radius)]
print theta[np.argmax(theta)]
# ar1 = ax.arrow(theta[0], 0, theta[0], 
#   radius[0], length_includes_head=True, width=0.001)
# 
# im  = ax.plot(theta, radius)
import pylab as plb
plt.plot(p_time, fieldX, p_time, fieldY)
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("E Field")
plb.legend(['x', 'y'])
plt.show()