import numpy as np
import h5py
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax

# read data
f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]
x = f["Wavefunction"]["x_value_0"][:]
kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))

if len(shape) > 1:
    y = f["Wavefunction"]["x_value_1"][:]

gobbler = f["Parameters"]["gobbler"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx

#how much (in a.u.) do you wish to cut off?
cut_left = 5
cut_right = 5
# cut_left = 2250
# cut_right = 2500

if len(shape) > 1:
    time_x = np.min(y[lower_idx[1]:upper_idx[1]]) * 0.95
else:
    time_x = np.min(x[lower_idx[0]:upper_idx[0]]) * 0.95
time_y = np.max(x[lower_idx[0]:upper_idx[0]]) * 0.9

max_val = 0
# calculate color bounds
for i, psi in enumerate(psi_value):
    if i > 0 and i < 2:  # the zeroth wave function is the guess and not relevant
        psi = psi[:, 0] + 1j * psi[:, 1]
        max_val_tmp = np.max(np.absolute(psi))
        if (max_val_tmp > max_val):
            max_val = max_val_tmp

if len(shape) == 1:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pylab as plb
    from matplotlib.colors import LogNorm

    font = {'size': 18}
    matplotlib.rc('font', **font)
    for i, psi in enumerate(psi_value):
        #Which frame?
        if i == 10:  #THIS IS ARBITRARY at the moment
            print "plotting cut version", i
            # set up initial figure with color bar
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)

            for j, val in enumerate(x):
                if val >= -cut_left and val <= cut_right:
                    psi[j] = 0.0 + 1j * 0.0

            data = None
            data = plt.semilogy(x, np.abs(psi))
            plt.text(
                time_x,
                time_y,
                "Time: " + str(psi_time[i]) + " a.u.",
                color='black')
            plb.xlabel('x (a.u.)')
            plb.ylabel('psi (arb. u)')
            plt.ylim(ymin=1e-15)
            plt.savefig("figs/Wave_cut" + str(i).zfill(8) + ".png")
            plt.clf()

            dataft = np.abs(np.fft.fftshift(np.fft.fft(psi)))
            # dataF = plt.semilogy(kx, dataft)
            dataF = plt.plot(kx, dataft)
            plt.text(
                time_x,
                time_y,
                "Time: " + str(psi_time[i]) + " a.u.",
                color='k')
            # color bar doesn't change during the video so only set it here
            plt.xlabel("$k_x$ (a.u.)")
            plt.ylabel("DFT($\psi$) (arb.)")
            # plb.xlim([-2.1, 2.1])

            #print peaks for last one

            kx_peaks = []
            peak_values = []
            thresh = 0.00001

            for element in argrelmax(dataft)[0]:
                if (dataft[element] > thresh):
                    kx_peaks.append(kx[element])
                    peak_values.append(dataft[element])
            kx_peaks = np.array(kx_peaks)
            peak_values = np.array(peak_values)

            for elem in argrelmax(peak_values)[0]:
                print "k_x:", kx_peaks[elem], peak_values[elem]
            plb.xlim([0, 2])
            plt.savefig("figs/2d_fft_cut" + str(i).zfill(8) + ".png")
            plt.clf()
