import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# read data
f = h5py.File("TDSE.h5","r")
psi_value = f["Wavefunction"]["psi"][:,:]
psi_time  = f["Wavefunction"]["psi_time"][:]
x         = f["Wavefunction"]["x_value_0"][:]

# calculate location for time to be printed
time_x    = np.min(x)*0.95
time_y    = np.max(x)*0.9

# calculate color bounds
max_val   = np.max(psi_value[:].real)
min_val   = np.min(psi_value[:].real)

# shape into a 3d array with time as the first axis
p_sqrt   = np.sqrt(psi_value.shape[1])
print "dim size:", p_sqrt, "Should be integer"
p_sqrt          = int(p_sqrt)
psi_value.shape = (psi_value.shape[0],p_sqrt,p_sqrt)
print "total plots:", psi_value.shape[0]
fig = plt.figure()

for i, psi in enumerate(psi_value):
    print "plotting", i
    # set up initial figure with color bar
    max_val   = np.max(abs(psi.real))
    min_val   = -1*max_val
    plt.pcolor(x, x, psi.real, cmap='bwr',vmin=min_val, vmax=max_val)
    # color bar doesn't change during the video so only set it here
    plt.colorbar()
    plt.xlabel("Electron 2 a.u.")
    plt.ylabel("Electron 1 a.u.")
    plt.title("Wave Function")
    fig.savefig("figs/Wave_"+str(i)+".jpg")
    plt.clf()
    plt.clf()