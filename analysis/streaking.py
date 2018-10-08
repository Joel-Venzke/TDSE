import numpy as np
import matplotlib
import find_peaks_2d
from scipy.signal import argrelmax
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as plb
import h5py

taus, moms, folders = [], [], []
kx_peaks = []
peak_values = []

# here set based on cycles_delay in 
# sweep submission script
for tau in np.arange(-150.0, 160.0, 10.0):
    key = "tau%.2f" % tau
    folders.append(key)
    taus.append(tau)

ti = np.arange(-150.0, 151.0, 0.1)
# Get vector potential of streaking field
# Obtain parameters for fit
states = h5py.File("states" + "/soft.h5", "r")
f_ir = h5py.File(folders[0] + "/TDSE.h5", "r")
cep = f_ir["Parameters"]["cep"][0]
cycles = f_ir["Parameters"]["cycles_on"][0] +\
            f_ir["Parameters"]["cycles_off"][0] 
freq = f_ir["Parameters"]["energy"][0]
xuv_freq = f_ir["Parameters"]["energy"][1]

print "XUV photon energy is " + str(xuv_freq)

fmax = f_ir["Parameters"]["field_max"][0]
shape = f_ir["Parameters"]["pulse_shape_idx"][0]

c = 137.0
ftau = 2 * np.pi * cycles / freq
Ip = -states["Energy"][0]

# For the moment, this will only work 
# for a sine squared streaking pulse
def trace_fit_sinsq(t, alpha, k0, delay):
    return k0 - alpha * (c * fmax / freq) * \
    np.sin(np.pi * (t + delay) / ftau)**2 * np.sin(freq *\
     (t + delay) + 2 * np.pi * cep) 

# For Gaussian streaking pulse
def trace_fit_gauss(t, alpha, k0, delay):
    return k0 - alpha * (c * fmax / freq) * \
    np.exp(-(t + delay)**2 / (2 * ftau**2)) * np.sin(freq *\
     (t + delay) + 2 * np.pi * cep) 

# for each folder containing a
# sim with a different XUV-IR delay
for fold in folders:
    # read data
    f = h5py.File(fold + "/TDSE.h5", "r")
    psi = f["Wavefunction"]["psi"][-1] #taking only last time step
    psi_time = f["Wavefunction"]["time"][-1]
    shape = f["Wavefunction"]["num_x"][:]
    tau_delay = f["Parameters"]["cycles_delay"][1]
    gobbler = f["Parameters"]["gobbler"][0]
    upper_idx = (shape * gobbler - 1).astype(int)
    lower_idx = shape - upper_idx
    print "tau_delay " + str(tau_delay)

    # set up 1D grid
    x = f["Wavefunction"]["x_value_0"][:]
    dx = f["Parameters"]["delta_x_max"][0]
    kx = x * 2.0 * np.pi / (x.shape[0] * (x[1] - x[0]) * (x[1] - x[0]))
    kxc = kx[lower_idx[0]:upper_idx[0]]
    dkx = kxc[1] - kxc[0]

    # if 2D, add second dimension
    if len(shape) > 1:
        y = f["Wavefunction"]["x_value_1"][:]
        ky = y * 2.0 * np.pi / (y.shape[0] * (y[1] - y[0]) * (y[1] - y[0]))
        kyc = ky[lower_idx[1]:upper_idx[1]]
        dky = kyc[1] - kyc[0]

    # how much (in a.u.) do you wish to cut off?
    # i.e., where does the bound part end?
    r_critical = 100.0
    max_val = 0

    # set 
    if len(shape) > 1:
        time_x = np.min(y[lower_idx[1]:upper_idx[1]]) * 0.95
    else:
        time_x = np.min(x[lower_idx[0]:upper_idx[0]]) * 0.95
    time_y = np.max(x[lower_idx[0]:upper_idx[0]]) * 0.9

    max_val = 0
    if len(shape) == 1:
        cut_filter = np.where(np.abs(x) <= r_critical, 0, 1)
        k_max = None
        # define the complex wavefunction and 
        # sort properly and set color bounds
        psi = psi[:, 0] + 1j * psi[:, 1]
        max_val_tmp = np.max(np.absolute(psi))
        if (max_val_tmp > max_val):
            max_val = max_val_tmp
        # apply cut filter
        psi *= cut_filter
        plt.plot(x, np.log10(np.abs(psi)))
        # plt.xlim([-2, 2])
        plt.ylim([-10, 0])
        plt.savefig(fold+"/figs/cut_" + str(psi_time).zfill(8) + ".png")
        plt.clf()
        data = None
        dataft = None
        dataft = np.abs(np.fft.fftshift(np.fft.fft(psi)))

        # plot the photoelectron momentum spectrum
        data = plt.semilogy(kx, dataft)
        plt.text(
            time_x,
            time_y,
            "Time: " + str(psi_time) + " a.u.",
            color='k')
        plt.xlabel("$k_x$ (a.u.)")
        plt.ylabel("DFT($\psi$) (arb.)")
        plb.xlim([-2, 2])
        plt.savefig(fold + "/figs/1d_fft_" + str(psi_time).zfill(8) + ".png")
        plt.clf()

        # get maxima, but only for positive momenta
        # add to list for streaking trace
        # to be taken at end of script
        dataft_right = np.where(kx >= 0, dataft, 0)
        nmax = kx[np.argmax(dataft_right)]
        norm = np.sum(dataft_right**2)
        kx_avg = np.sum(np.multiply(kx, dataft_right**2)) / norm
        kx_peaks.append(kx_avg)
        print "max momentum = " + str(nmax)
        print "<kx> = " + str(kx_avg)

    elif len(shape) == 2:
        import matplotlib
        import matplotlib.pyplot as plt
        import pylab as plb
        from matplotlib.colors import LogNorm

        print "WARNING: 2d code not finished. No\
        streaking delay will be found"

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
        # define cut filter based on r_critical
        # alpha may be changed arbitrarily
        # depending on the case
        cut_filter = np.ones((xc.shape[0], yc.shape[0]))
        cut_array = np.where(r_mesh <= r_critical)
        alpha = 0.075
        cut_filter[cut_array] *= (np.exp(-alpha *
                                         (r_mesh[cut_array] - r_critical)**2))

        k_max = None
        print "Plotting", i
        # define the complex wavefunction and 
        # sort properly
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = tuple(shape)
        psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
        # apply cut filter
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

        data = plt.imshow(
            np.absolute(psi),
            cmap='viridis',
            origin='lower',
            extent=[
                y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
            ],
            norm=LogNorm(vmin=1e-10))

        plt.text(
            time_x,
            time_y,
            "Time: " + str(psi_time[i]) + " a.u.",
            color='white')
        
        if f["Parameters"]["coordinate_system_idx"][0] == 1:
            plt.xlabel("z-axis (a.u.)")
            plt.ylabel("$\\rho$-axis  (a.u.)")
        else:
            plt.xlabel("X-axis (a.u.)")
            plt.ylabel("Y-axis  (a.u.)")
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
            zed  = ft_full[:, int(half)]
            plt.plot(kxc, zed)
            plt.xlim([np.min(kxc), np.max(kxc)])
            plt.savefig("figs/z_mom.png")
            plt.clf()
            i_vector = np.argmax(zed)

            print "k_f: " + str(kxc[i_vector])

t_adj = taus + 0.5 * ftau
ti_adj = ti + 0.5 * ftau

alpha_guess, delay_guess = 1.0, -1.0
if Ip < xuv_freq:
    k0_guess = np.sqrt(2 * (xuv_freq - Ip))
else:
    k0_guess = np.sqrt(2 * (2 * xuv_freq - Ip))

guess = np.array([alpha_guess, k0_guess, delay_guess])

# if shape == 0:
popt, pcov = curve_fit(trace_fit_sinsq, t_adj, kx_peaks, p0=guess)
# elif shape == 1:
    # popt, pcov = curve_fit(trace_fit_gauss, t_adj, kx_peaks, p0=guess)
print popt
print "Streaking time delay is " + str(popt[2])
print "Now plotting trace..."
plt.plot(taus, kx_peaks, '*', ti,\
         trace_fit_sinsq(ti_adj, *popt))
plt.xlabel('XUV-IR delay (a.u.)')
plt.ylabel('final momentum (a.u.)')
plt.savefig("trace" + ".png")
plt.clf()
print "Done, please see trace.png in the current directory"
