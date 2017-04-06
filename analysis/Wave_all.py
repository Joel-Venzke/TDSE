import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

# read data
f = h5py.File("TDSE.h5","r")
psi_value = f["Wavefunction"]["psi"]
psi_time  = f["Wavefunction"]["psi_time"][:]
x         = f["Wavefunction"]["x_value_0"][:]

# calculate location for time to be printed
time_x    = np.min(x)*0.95
time_y    = np.max(x)*0.9

max_val = 0
# calculate color bounds
for psi in psi_value:
    psi = psi[:,0] + 1j*psi[:,1]
    max_val_tmp   = np.max(np.absolute(psi))
    if (max_val_tmp > max_val):
        max_val = max_val_tmp

# shape into a 3d array with time as the first axis
p_sqrt   = np.sqrt(psi_value[0].shape[0])
print "dim size:", p_sqrt, "Should be integer"
p_sqrt          = int(p_sqrt)

fig = plt.figure()

for i, psi in enumerate(psi_value):
    print "plotting", i
    # set up initial figure with color bar
    psi = psi[:,0] + 1j*psi[:,1]
    psi.shape = (p_sqrt,p_sqrt)
    plt.imshow(np.absolute(psi), cmap='viridis', origin='lower',
               extent=[x[0],x[-1],x[0],x[-1]],
               norm=LogNorm(vmin=1e-10, vmax=max_val))
    plt.text(time_x, time_y, "Time: "+str(psi_time[i])+" a.u.",
                color='white')
    # color bar doesn't change during the video so only set it here
    plt.colorbar()
    plt.xlabel("Electron 2 a.u.")
    plt.ylabel("Electron 1 a.u.")
    plt.title("Wave Function")
    fig.savefig("figs/Wave_"+str(i).zfill(8)+".png")
    plt.clf()
    plt.clf()
