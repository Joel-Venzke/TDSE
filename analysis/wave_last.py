import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# read data
f = h5py.File("TDSE.h5","r")
psi_value = f["Wavefunction"]["psi"][-1]
x         = f["Wavefunction"]["x_value_0"][:]

# # calculate location for time to be printed
time_x    = np.min(x)*0.95
time_y    = np.max(x)*0.9

# # calculate color bounds
max_val   = np.max(psi_value[:,0])
min_val   = np.min(psi_value[:,0])
print "min plot: ", min_val
print "max plot: ", max_val

# # shape into a 3d array with time as the first axis
p_sqrt   = np.sqrt(psi_value.shape[0])
print "dim size:", p_sqrt, "Should be integer"
p_sqrt          = int(p_sqrt)
psi_value.shape = (p_sqrt,p_sqrt,2)

# set up initial figure with color bar
fig = plt.figure()
plt.imshow(psi_value[:,:,0], cmap='viridis', vmin=min_val, vmax=max_val,
               origin='lower', extent=[x[0],x[-1],x[0],x[-1]])
# color bar doesn't change during the video so only set it here
plt.colorbar()
plt.xlabel("Electron 2 a.u.")
plt.ylabel("Electron 1 a.u.")
plt.title("Wave Function")
fig.savefig("figs/Wave_last.jpg")
