import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import pylab as plb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax

# read data
f = h5py.File("TDSE.h5", "r")
psi_value = f["Projections"]["psi"]
psi_time = f["Projections"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]
gobbler = f["Parameters"]["gobbler"][0]
dt = f["Parameters"]["delta_t"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx
# calculate location for time to be printed

x = f["Wavefunction"]["x_value_0"][:]
kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))
if f["Parameters"]["coordinate_system_idx"][0] == 1:
    kx = np.pad(kx, (kx.shape[0], 0), 'reflect', reflect_type='odd') / 2.0

if len(shape) == 1:
    time_x = 0
    time_y = 2.5
elif len(shape) == 2:
    y = f["Wavefunction"]["x_value_1"][:]
    ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
    time_x = np.min(ky[lower_idx[1]:upper_idx[1]]) * 0.95
    time_y = np.max(kx[lower_idx[0]:upper_idx[0]]) * 0.9
else:
    exit("Only 1D and 2D versions are supported currently")

font = {'size': 18}
matplotlib.rc('font', **font)
fig = plt.figure()
for i, psi in enumerate(psi_value):
    if i > 0:  # the zeroth wave function is the guess and not relevant
        print "plotting", i
        # set up initial figure with color bar
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = tuple(shape)
        data = None

        if len(shape) == 2:
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                psi = np.pad(psi, ((psi.shape[0], 0), (0, 0)), 'symmetric')
                psi = np.abs(np.fft.fftshift(np.fft.fft2(psi)))
                # cross_sec = (ky**2 / 2.0 + 0.50188)**(7.0 / 2.0)
                # cross_sec[cross_sec > 1e4] = 1.0
                # psi *= cross_sec
                max_idx = np.unravel_index(
                    np.argmax(psi), (psi.shape[0], psi.shape[1]))
                print np.sqrt(ky[max_idx[0]] * ky[max_idx[0]] + kx[max_idx[1]]
                              * kx[max_idx[1]]), kx[max_idx[1]], ky[max_idx[
                                  0]], max_idx, kx.shape, ky.shape, kx.min(
                                  ), kx.max(), ky.min(), ky.max()
                data = plt.imshow(
                    psi,
                    # np.abs(psi),
                    cmap='viridis',
                    origin='lower',
                    # norm=LogNorm(vmin=1e-5),
                    extent=[ky.min(), ky.max(),
                            kx.min(), kx.max()])
            else:
                psi = np.abs(np.fft.fftshift(np.fft.fft2(psi)))
                max_idx = np.unravel_index(
                    np.argmax(psi), (psi.shape[0], psi.shape[1]))
                print np.sqrt(ky[max_idx[0]] * ky[max_idx[0]] + ky[max_idx[1]]
                              * ky[max_idx[1]]), ky[max_idx[1]], ky[max_idx[0]]
                data = plt.imshow(
                    # psi * (ky**2 / 2.0 + 0.50188)**(7.0 / 2.0),
                    np.abs(psi),
                    cmap='viridis',
                    origin='lower',
                    #norm=LogNorm(vmin=1e-5),
                    vmin=np.abs(psi).max()-0.2,
                    extent=[ky.min(), ky.max(),
                            kx.min(), kx.max()])
            plt.text(
                time_x,
                time_y,
                "Time: " + str(psi_time[i]) + " a.u.",
                color='white')
            # color bar doesn't change during the video so only set it here
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                plt.xlabel("$k_\\rho$ (a.u.)")
                plt.ylabel("$k_z$ (a.u.)")
            else:
                plt.xlabel("$k_x$ (a.u.)")
                plt.ylabel("$k_y$  (a.u.)")
            plt.colorbar()
            fig.savefig("figs/2d_fft_full_" + str(i).zfill(8) + ".png")
            plb.xlim([-2.0, 2.0])
            plb.ylim([-2.0, 2.0])
            fig.savefig("figs/2d_fft_" + str(i).zfill(8) + ".png")
            plt.clf()

        elif len(shape) == 1:
            dataft = np.abs(np.fft.fftshift(np.fft.fft(psi)))
            # cross_sec = (kx**2 / 2.0 + 0.50188)**(7.0 / 2.0)
            # cross_sec[cross_sec > 1e4] = 1.0
            # dataft *= cross_sec
            data = plt.semilogy(kx, dataft)
            plt.text(
                time_x,
                time_y,
                "Time: " + str(psi_time[i]) + " a.u.",
                color='k')
            # color bar doesn't change during the video so only set it here
            plt.xlabel("$k_x$ (a.u.)")
            plt.ylabel("DFT($\psi$) (arb.)")
            plb.xlim([-5, 5])

            #print peaks for last one

            kx_peaks = []
            peak_values = []
            thresh = 0.00005

            for element in argrelmax(dataft)[0]:
                if (dataft[element] > thresh):
                    kx_peaks.append(kx[element])
                    peak_values.append(dataft[element])
            kx_peaks = np.array(kx_peaks)
            peak_values = np.array(peak_values)

            for elem in argrelmax(peak_values)[0]:
                print "k_x:", kx_peaks[elem], peak_values[elem]

            plt.savefig("figs/2d_fft_" + str(i).zfill(8) + ".png")
            plt.clf()
