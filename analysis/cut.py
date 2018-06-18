import numpy as np
import h5py
from matplotlib.colors import LogNorm
from scipy.signal import argrelmax
from scipy.interpolate import interp2d
import os
from scipy.special import sph_harm


def project_onto_ylm(theta, data):
    delta_theta = theta[1] - theta[0]
    data_conj = np.conjugate(data)
    for l in range(0, 4):
        for m in range(-l, l + 1):
            projection = (data_conj * sph_harm(m, l, theta, np.pi / 2.0)
                          ).sum() * delta_theta
            if np.abs(projection) > 1e-10:
                print "l=" + str(l) + " m=" + str(m) + " amp=" + str(
                    np.abs(projection)) + " phi=" + str(np.angle(projection))


print "start"

ground_state_energy = -0.499639

# read data
f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]
ellipticity = f["Parameters"]["ellipticity_0"][0]
pulse_energy = f["Parameters"]["energy"][0]

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

if len(shape) == 2:
    import matplotlib
    # matplotlib.use('Agg')
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

    if f["Parameters"]["coordinate_system_idx"][0] == 1:
        x_min_idx = 0
    else:
        x_min_idx = lower_idx[0]
    x_max_idx = upper_idx[0]
    x_min_idx = lower_idx[0]
    y_min_idx = lower_idx[1]
    y_max_idx = upper_idx[1]

    xc = x[x_min_idx:x_max_idx]
    yc = y[y_min_idx:y_max_idx]
    x_mesh, y_mesh = np.meshgrid(xc, yc)
    r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
    cut_filter = np.ones((xc.shape[0], yc.shape[0]))
    cut_array = np.where(r_mesh <= r_critical)
    alpha = 0.075
    cut_filter[cut_array] *= (np.exp(-alpha *
                                     (r_mesh[cut_array] - r_critical)**2))
    # alpha = 0.075
    # for j, val in enumerate(xc):
    #     for k, valy in enumerate(yc):
    #         r = np.sqrt(val**2 + valy**2)
    #         if r <= r_critical:
    #             cut_filter[j][k] = (np.exp(-alpha * (r - r_critical)**2))

    for i, psi in enumerate(psi_value):
        if i == 8:
            k_max = None
            print "Plotting", i
            # set up initial figure with color bar
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)
            psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
            # cut based on r_critical

            # for j, val in enumerate(xc):
            #     for k, valy in enumerate(yc):
            #         r = np.sqrt(val**2 + valy**2)
            #         if r <= r_critical:
            #             psi[j][k] = psi[j][k] \
            #             * (np.exp(-alpha * (r - r_critical)**2) + 0j)
            psi *= cut_filter
            data = None
            dataft = None
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                psi = np.absolute(
                    np.multiply(
                        np.conjugate(psi),
                        np.multiply(x[x_min_idx:x_max_idx], psi.transpose())
                        .transpose()))

            else:
                i_vector = np.unravel_index(
                    np.argmax(psi), (psi.shape[0], psi.shape[1]))
                angle = np.arctan2(yc[i_vector[1]], xc[i_vector[0]])

                # print "angle of position max is " + str(angle * 180 / np.pi) + \
                #         " degrees at y = " + str(yc[i_vector[1]]) + ", x = " + \
                #         str(xc[i_vector[0]])

            data = plt.imshow(
                np.absolute(psi),
                cmap='viridis',
                origin='lower',
                extent=[
                    y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                ],
                norm=LogNorm(vmin=1e-10))
            # np.savetxt("cutWave.txt", np.absolute(psi))
            # np.savetxt('cutWaveReal.txt', psi.real)
            # np.savetxt('cutWaveImag.txt', psi.imag)
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
            fig.savefig("figs/Wave_cut_" + str(i).zfill(8) + ".png")
            plt.clf()

            # Now Fourier transform the cut data
            ft_full = None
            ft_left = None
            ft_right = None
            asm = 0
            asymmetry = 0
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                psi = np.pad(psi, ((psi.shape[0], 0), (0, 0)), 'symmetric')
                ft_full = np.abs(np.fft.fftshift(np.fft.fft2(psi)))**2
                half = ceil(ft_full.shape[0] / 2.0)
                ft_left = ft_full[:, :int(half)]
                ft_right = ft_full[:, int(half):]
                kycl, kycr = kyc[:int(half)], kyc[int(half):]
                p_l = np.sum(ft_left) * dkx * dky
                p_r = np.sum(ft_right) * dkx * dky
                asm = (p_l - p_r) / (p_l + p_r)
                print "asymmetry: " + str(asm)
                dataft = plt.imshow(
                    np.sqrt(ft_full),
                    cmap='viridis',
                    origin='lower',
                    vmin=3.0,
                    vmax=3.5,  #norm=LogNorm(vmin=1e-10),
                    extent=[
                        ky.min(),
                        ky.max(), -1.0 * kx.max() / 2.0,
                        kx.max() / 2.0
                    ])
            else:
                pad_x = 2**np.ceil(np.log2(psi.shape[0] * 1))
                pad_y = 2**np.ceil(np.log2(psi.shape[1] * 1))
                psi = np.pad(
                    psi, ((int(np.floor((pad_x - psi.shape[0]) / 2.0)),
                           int(np.ceil((pad_x - psi.shape[0]) / 2.0))),
                          (int(np.floor((pad_y - psi.shape[1]) / 2.0)),
                           int(np.ceil((pad_y - psi.shape[1]) / 2.0)))),
                    'constant',
                    constant_values=((0, 0), (0, 0)))
                padded_shape = psi.shape[0]
                x = np.arange(
                    int(-0.5 * padded_shape) * dx,
                    int(0.5 * padded_shape) * dx, dx)

                kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) *
                                        (x[1] - x[0]))
                kxc = kx

                y = np.arange(
                    int(-0.5 * padded_shape) * dx,
                    int(0.5 * padded_shape) * dx, dx)
                ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) *
                                        (y[1] - y[0]))
                kyc = ky

                dkx = kxc[1] - kxc[0]
                dky = kyc[1] - kyc[0]

                psi_fft = np.fft.fftshift(np.fft.fft2(psi))
                ft_full = np.abs(psi_fft)**2
                full = ft_full.shape[0]
                half = np.ceil(full / 2.0)
                ft_left = ft_full[:, :int(half)]
                ft_right = ft_full[:, int(half):]
                kycl, kycr = kyc[:int(half)], kyc[int(half):]

                p_l = np.sum(ft_left) * dkx * dky
                p_r = np.sum(ft_right) * dkx * dky
                asymmetry = (p_l - p_r) / (p_l + p_r)
                i_vector = np.unravel_index(
                    np.argmax(ft_full), (ft_full.shape[0], ft_full.shape[1]))
                #NOTE: for i_vector 0 is y, and 1 is x
                # print "max location is y = " + str(kyc[i_vector[1]]) \
                #     + "a.u., " + "x = " + str(kxc[i_vector[0]]) + "\n"
                angle = np.arctan2(kyc[i_vector[1]], kxc[i_vector[0]])
                k_max = np.sqrt(kyc[i_vector[1]]**2 + kxc[i_vector[0]]**2)
                print "asymmetry: " + str(asymmetry)
                print "Angle: " + str(angle * 180 / np.pi)
                print "Angle mod 180: " + str((angle * 180 / np.pi) % 180)

                #testing whether rotation or not
                # print "o.g. value is " + str(psi_fft[half][half])

                #uncomment to save fft
                # print "outputting FFT"
                # np.savetxt('fft.txt', ft_full, delimiter=',')
                # np.savetxt('fftcomplex.txt', psi_fft.view(float))

                dataft = plt.imshow(
                    #np.sqrt(ft_full),
                    ft_full.transpose(),
                    cmap='viridis',
                    origin='lower',
                    # norm=LogNorm(vmin=1e-8),
                    #vmin=5,vmax=9,
                    extent=[kx.min(), kx.max(),
                            ky.min(), ky.max()])

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
            plb.xlim([-10, 10])
            plb.ylim([-10, 10])
            plt.title("$\epsilon =$ %.2f" % ellipticity)
            plt.colorbar()
            plt.tight_layout()
            fig.savefig("figs/2d_fft_cut_lin_full_" + str(i).zfill(8) + ".png")
            plb.xlim([-1.5, 1.5])
            plb.ylim([-1.5, 1.5])
            plt.tight_layout()
            fig.savefig("figs/momentum_elip_" + str(ellipticity) + "_" +
                        str(i).zfill(8) + ".png")
            plt.clf()

            interp = interp2d(kx, ky, ft_full, kind='cubic')

            interp_real = interp2d(kx, ky, psi_fft.real, kind='cubic')
            interp_imag = interp2d(kx, ky, psi_fft.imag, kind='cubic')

            theta_max = 10
            delta_theta = 0.001
            theta = np.arange(-theta_max * np.pi / 180.,
                              theta_max * np.pi / 180 + delta_theta,
                              delta_theta)
            x_interp = k_max * np.cos(theta)
            y_interp = k_max * np.sin(theta)
            data = []
            data_complex = []
            for theta_idx in range(x_interp.shape[0]):
                data.append(interp(x_interp[theta_idx], y_interp[theta_idx]))
                data_complex.append(
                    interp_real(x_interp[theta_idx],
                                y_interp[theta_idx]) + 1.0j * interp_imag(
                                    x_interp[theta_idx], y_interp[theta_idx]))
            data = np.array(data)
            data_complex = np.array(data_complex)
            project_onto_ylm(theta, data_complex)
            print "Predicted vs actual max:", np.sqrt(
                2 * (pulse_energy - np.abs(ground_state_energy))), k_max
            print -1.0 * theta[np.argmax(data)] * 180 / np.pi
            plt.plot(-1.0 * theta * 180 / np.pi, data)
            plt.axvline(0.0, color='k')
            plt.axvline(-1.0 * theta[np.argmax(data)] * 180 / np.pi, color='r')
            plt.xlabel("angle (degrees)")
            plt.ylabel("yield (arb. units.)")
            plt.xlim([-theta_max, theta_max])
            # plt.ylim(ymin=0.0)
            plt.tight_layout()
            fig.savefig("figs/rotation_" + str(ellipticity) + "_" +
                        str(i).zfill(8) + ".png")
            plt.clf()
