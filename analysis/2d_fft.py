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
time_y = np.max(x[lower_idx[0]:upper_idx[0]]) * 0.9
kx = np.zeros(x.size)

if len(shape) > 1:
    y = f["Wavefunction"]["x_value_1"][:]
    time_x = np.min(y[lower_idx[1]:upper_idx[1]]) * 0.95
    ky = np.zeros(y.size)
    i = 0
    for j in y:
        ky[i] = (-(y.shape[0] / 2) + i) * 2 * np.pi / (x.shape[0] *
                                                       (y[1] - y[0]))
        i += 1
    fig = plt.figure()

else:
    time_x = np.min(x[lower_idx[0]:upper_idx[0]]) * 0.95

i = 0
for j in x:
    kx[i] = (-(x.shape[0] / 2) + i) * 2 * np.pi / (x.shape[0] * (x[1] - x[0]))
    i += 1

# shape into a 3d array with time as the first axis

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

        if len(shape) > 1:
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
                    psi = np.multiply(x[x_min_idx:x_max_idx],
                                      psi.transpose()).transpose()
                    data = plt.imshow(
                        np.abs(np.fft.fftshift(np.fft.fft2(psi), axes=1)),
                        cmap='viridis',
                        origin='lower',
                        norm=LogNorm(vmin=1e-15),
                        extent=[ky.min(),
                                ky.max(),
                                kx.min(),
                                kx.max()])
                    plt.text(
                        time_x,
                        time_y,
                        "Time: " + str(psi_time[i]) + " a.u.",
                        color='white')
                    # color bar doesn't change during the video so only set it here
                    plt.xlabel("X-axis (a.u.)")
                    plt.ylabel("Y-axis  (a.u.)")
                    plb.xlim([-5, 5])
                    plb.ylim([0, 5])
                    # plt.axis('off')
                    plt.colorbar()
                    fig.savefig("figs/2d_fft_" + str(i).zfill(8) + ".png")
                    plt.clf()

        elif len(shape) == 1:
            data = None
            dataft = np.abs(np.fft.fftshift(np.fft.fft(psi)))
            data = plt.plot(kx, np.log10(dataft))
            plt.text(
                time_x,
                time_y,
                "Time: " + str(psi_time[i]) + " a.u.",
                color='white')
            # color bar doesn't change during the video so only set it here
            plt.xlabel("X-axis (a.u.)")
            plt.ylabel("Y-axis  (a.u.)")
            plb.xlim([-5, 5])
            plb.ylim([-5, 5])
            # plt.axis('off')

            #print peaks for last one

            pp = []
            ppa = []
            pAp = []
            ftt = []
            ftta = []
            fAp = []
            thresh = 0.0005

            # for element in argrelmax(dataft)[0]:
            #     if (dataft[element] > thresh):
            #         pp.append(kx[element])
            #         ftt.append(dataft[element])
            #         ppa = np.array(pp)
            #         ftta = np.array(ftt)

            # for elem in argrelmax(ftta)[0]:
            #     pAp.append(ppa[elem])
            #     fAp.append(ftta[elem])
            #     print str(ppa[elem]) + '\t'
            #     print str(ftta[elem]) + '\n'

            plb.xlabel('$k_{x}$ (a.u.)')
            plb.ylabel('log10(FT(psi)) (arb. u)')
            plt.savefig("figs/2d_fft_" + str(i).zfill(8) + ".png")
            plt.clf()
