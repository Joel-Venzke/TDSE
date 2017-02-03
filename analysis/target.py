import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import h5py

# read data
target = h5py.File("He.h5","r")
f = h5py.File("TDSE.h5","r")
psi_value = target["States"]
energy    = target["Energy"]
psi_time  = f["Wavefunction"]["psi_time"][:]
x         = f["Wavefunction"]["x_value_0"][:]


# shape into a 3d array with time as the first axis
p_sqrt   = np.sqrt(psi_value.shape[1])
print("dim size:", p_sqrt, "Should be integer")
p_sqrt          = int(p_sqrt)

fig = plt.figure()

for i, psi in enumerate(psi_value):
    print("plotting", i)
    psi.shape = (p_sqrt,p_sqrt)
    # set up initial figure with color bar
    max_val   = np.max(abs(psi.real))
    min_val   = -1*max_val
    plt.clf()
    plt.imshow(psi.real, cmap='bwr', vmin=-1*max_val, vmax=max_val,
               origin='lower', extent=[x[0],x[-1],x[0],x[-1]])
    # color bar doesn't change during the video so only set it here
    plt.colorbar()
    plt.xlabel("Electron 2 a.u.")
    plt.ylabel("Electron 1 a.u.")
    plt.title("Wave Function - Energy "+str(energy[i]))
    fig.savefig("figs/He_bwr_state_"+str(i).zfill(3)+".jpg")

    plt.clf()
    plt.imshow(np.abs(psi.real), cmap='viridis', origin='lower',
        extent=[x[0],x[-1],x[0],x[-1]],
        norm=LogNorm(vmin=1e-10, vmax=np.abs(psi.real).max()))
    # plt.pcolor(x, x, abs(psi.real),
    #     norm=LogNorm(vmin=1e-16,
    #         vmax=abs(psi.real).max()), cmap='viridis')
    plt.colorbar()
    plt.xlabel("Electron 2 a.u.")
    plt.ylabel("Electron 1 a.u.")
    plt.title("Wave Function - Energy "+str(energy[i]))
    fig.savefig("figs/He_log_state_"+str(i).zfill(3)+".jpg")
