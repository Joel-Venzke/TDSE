import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# read data
f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
psi_time = f["Wavefunction"]["time"][:]
x = f["Wavefunction"]["x_value_0"][:]
y = f["Wavefunction"]["x_value_1"][:]
shape = f["Wavefunction"]["num_x"][:]
gobbler = f["Parameters"]["gobbler"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx
# calculate location for time to be printed
time_x = np.min(y[lower_idx[1]:upper_idx[1]]) * 0.95
time_y = np.max(x[lower_idx[0]:upper_idx[0]]) * 0.9

# shape into a 3d array with time as the first axis
fig = plt.figure()
font = {'size': 18}
matplotlib.rc('font', **font)
for i, psi in enumerate(psi_value):
    if i > 0:  # the zeroth wave function is the guess and not relevant
        print "plotting", i
        plt.text(
            time_x,
            time_y,
            "Time: " + str(psi_time[i]) + " a.u.",
            color='white')
        # set up initial figure with color bar
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = tuple(shape)
        if f["Parameters"]["coordinate_system_idx"][0] == 1:
            x_min_idx = 0
        else:
            x_min_idx = lower_idx[0]
        x_max_idx = upper_idx[0]
        y_min_idx = lower_idx[1]
        y_max_idx = upper_idx[1]
        x_max_idx = -1
        y_min_idx = 0
        y_max_idx = -1
        psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
        data = None
        if f["Parameters"]["coordinate_system_idx"][0] == 1:
            psi = np.multiply(x[x_min_idx:x_max_idx], psi.transpose()).transpose()
        data = plt.imshow(
            np.abs(np.fft.fftshift(np.fft.fft2(psi))),
            cmap='viridis',
            origin='lower',
            norm=LogNorm(vmin=1e-15))
        plt.text(
            time_x,
            time_y,
            "Time: " + str(psi_time[i]) + " a.u.",
            color='white')
        # color bar doesn't change during the video so only set it here
        plt.xlabel("X-axis (a.u.)")
        plt.ylabel("Y-axis  (a.u.)")
        # plt.axis('off')
        plt.colorbar()
        fig.savefig("figs/2d_fft_" + str(i).zfill(8) + ".png")
        plt.clf()
