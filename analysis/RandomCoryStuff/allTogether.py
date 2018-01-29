import numpy as np
import h5py
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax
import os

print "start"

# read data
f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]

gobbler = f["Parameters"]["gobbler"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx
dx = f["Parameters"]["delta_x_max"][0]

x = f["Wavefunction"]["x_value_0"][:]
kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))
kxc = kx[lower_idx[0]:upper_idx[0]]

if len(shape) > 1:
    y = f["Wavefunction"]["x_value_1"][:]
    ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
    kyc = ky[lower_idx[1]:upper_idx[1]]

dkx = kxc[1] - kxc[0]
dky = kyc[1] - kyc[0]
#how much (in a.u.) do you wish to cut off?
cut_left = 5
cut_right = 5
r_critical = 50 

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
            plt.savefig("figs/1d_fft_cut" + str(i).zfill(8) + ".png")
            plt.clf()

elif len(shape) == 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pylab as plb
    from matplotlib.colors import LogNorm
    # shape into a 3d array with time as the first axis
    y = f["Wavefunction"]["x_value_1"][:]
    ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
    time_x = np.min(ky[lower_idx[1]:upper_idx[1]]) * 0.95
    time_y = np.max(kx[lower_idx[0]:upper_idx[0]]) * 0.9
    fig = plt.figure()
    font = {'size': 18}
    matplotlib.rc('font', **font)


    for i, psi in enumerate(psi_value):
        if i == 11:  # arbitrary at moment
            print "cut version", i
            # set up initial figure with color bar
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                x_min_idx = 0
            else:
                x_min_idx = lower_idx[0]
            x_max_idx = upper_idx[0]
            x_min_idx = lower_idx[0]
            y_min_idx = lower_idx[1]
            y_max_idx = upper_idx[1]
            

            psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
            xc = x[x_min_idx:x_max_idx]
            yc = y[y_min_idx:y_max_idx]
            # cut based on r_critical
            alpha = 0.075
            for j, val in enumerate(xc):
                for k, valy in enumerate(yc):
                    r = np.sqrt(val**2 + valy**2)
                    if r <= r_critical:
                        psi[j][k] = psi[j][k] \
                        * (np.exp(-alpha * (r - r_critical)**2) + 0j)

            data = None
            dataft = None
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                psi = np.absolute(
                    np.multiply(
                        np.conjugate(psi),
                        np.multiply(x[x_min_idx:x_max_idx], psi.transpose())
                        .transpose()))
               
            else:
                print psi.shape

                i_vector = np.unravel_index(np.argmax(psi), (psi.shape[0], psi.shape[1]))
                angle = np.arctan2(yc[i_vector[0]], xc[i_vector[1]])

                print "angle of position max is " + str(angle * 180 / np.pi) + \
                        " degrees at y = " + str(yc[i_vector[0]]) + ", x = " + \
                        str(xc[i_vector[1]])

            data = plt.imshow(
                np.absolute(psi),
                cmap='viridis',
                origin='lower',
                extent=[
                    y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                ],
                norm=LogNorm(vmin=1e-10, vmax=max_val))
            np.savetxt("cutWave.txt", np.absolute(psi))
            # np.savetxt("cutWave.txt", psi.view(float))

            plt.text(
                time_x,
                time_y,
                "Time: " + str(psi_time[i]) + " a.u.",
                color='white')
            # color bar doesn't change during the video so only set it here
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                plt.xlabel("z-axis (a.u.)")
                plt.ylabel("$\\rho$-axis  (a.u.)")
            else:
                plt.xlabel("X-axis (a.u.)")
                plt.ylabel("Y-axis  (a.u.)")
            # plt.axis('off')
            plt.colorbar()
            fig.savefig("figs/Wave_cut" + str(i).zfill(8) + ".png")
            plt.clf()

            saver = np.loadtxt('centralCycle.txt')
            t_central = saver[0][:]
            x_central = saver[1][:]
            y_central = saver[2][:]
            fieldTheta = np.arctan2(y_central, x_central)

            # plt.plot(fieldTheta, fieldIntensity)
            # plt.show()

            tol = 0.02
            r_current = 0
            theta_current = 0
            z = 0
            data_adjusted = np.zeros(psi.shape, 'complex')

            #for each point in data, divide 
            #by field Intensity corresponding to correct angle
            print "dividing out intensity...."
            for j, val in enumerate(xc):
                for k, valy in enumerate(yc):
                    theta_current = np.arctan2(valy, val)
                    z = np.argmin(np.abs(fieldTheta - theta_current))
                    data_adjusted[j][k] = \
                    psi[j][k] / (saver[1][z]**2 + saver[2][z]**2 + 0j)


            psi = np.abs(data_adjusted)

            # i_vector = np.unravel_index(np.argmax(data_adjusted),
            #                     (data_adjusted.shape[0], data_adjusted.shape[1]))
            i_vector = np.unravel_index(np.argmax(psi),
                                (psi.shape[0], psi.shape[1]))
            print "max location is y = " + str(yc[i_vector[0]]) \
                + "a.u., " + "x = " + str(xc[i_vector[1]]) + "\n"
            angle = np.arctan2(yc[i_vector[0]], xc[i_vector[1]])
            print "location max value = " + str(data_adjusted[i_vector[0]][i_vector[1]]) 
            print "angle of position max is " + str(angle * 180 / np.pi) +\
                  " degrees."
            # plt.plot(fieldTheta, fieldIntensity)
            datapos = plt.imshow(
                                psi,
                                cmap='viridis',
                                origin='lower',
                                # vmin=0.2,vmax=0.61,
                                norm=LogNorm(vmin=1e-10),
                                extent=[yc.min(), yc.max(),
                                        xc.min(), xc.max()])

            plt.xlabel("$y$ (a.u.)")
            plt.ylabel("$x$  (a.u.)")
            # plb.xlim([-10, 10])
            # plb.ylim([-10, 10])
            plt.colorbar()
            # fig.savefig("figs/adjusted_cut" + "_full.png")
            # plb.xlim([-2, 2])
            # plb.ylim([-2, 2])
            fig.savefig("figs/adjusted_cut" + ".png")
            plt.clf()
            ft_full = None

            print "FT adjusted cut Wavefunction"
            ft_full = np.abs(np.fft.fftshift(np.fft.fft2(data_adjusted)))**2
            print ft_full
            # print "cutting again"
            # r_critical = 0.2
            # alpha = 2500
            # for j, val in enumerate(kxc):
            #     for k, valy in enumerate(kyc):
            #         r = np.sqrt(val**2 + valy**2)
            #         if r <= r_critical:
            #             ft_full[j][k] = ft_full[j][k] \
            #             * np.exp(-alpha * (r - r_critical)**2)

            # np.savetxt('fftAdjusted.txt', ft_full, delimiter=',')
            print "Calculating rotation angle..."
            i_vector = np.unravel_index(np.argmax(ft_full),
                (ft_full.shape[0], ft_full.shape[1]))
            print "max momentum is at k_y = " + str(kyc[i_vector[0]]) \
                + "a.u., " + "k_x = " + str(kxc[i_vector[1]]) + "\n"
            angle = np.arctan2(kyc[i_vector[0]], kxc[i_vector[1]])

            print " angle of momentum max is " + str(angle * 180 / np.pi) +\
                  " degrees. Now plotting full spectrum"

            dataft = plt.imshow(
            # np.abs(np.fft.fftshift(np.fft.fft2(data_adjusted)))
            np.sqrt(ft_full),
            cmap='viridis',
            origin='lower',
            # norm=LogNorm(vmin=1e-3),
            # vmin=0.125,vmax=0.175,
            vmin=10000,
            extent=[kyc.min(), kyc.max(),
                    kxc.min(), kxc.max()])

            plt.xlabel("$k_y$ (a.u.)")
            plt.ylabel("$k_x$  (a.u.)")
            plb.xlim([-10, 10])
            plb.ylim([-10, 10])
            plt.colorbar()
            fig.savefig("figs/adjusted_fftcutPos" + "_full.png")
            plb.xlim([-2, 2])
            plb.ylim([-2, 2])
            fig.savefig("figs/adjusted_fftcutPos" + ".png")
            plt.clf()




