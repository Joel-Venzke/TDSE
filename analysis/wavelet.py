from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
from matplotlib.colors import LogNorm
from obspy.signal.tf_misfit import cwt

font = {'size': 18}

matplotlib.rc('font', **font)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

w_0 = 0.057
ground_state = np.abs(-0.503829)
min_harm = 5
max_harm = 12

w_list = range(3, 100, 5)
# w_list = range(38, 100, 5)
w_list += range(100, 1000, 100)
w_list += list(10**np.arange(3, 5))

# w_list = range(53, 54)

v_max_list = 5 * 10.0**np.arange(-3, -4, -1)
v_min_list = 1 * 10.0**np.arange(-4, -5, -1)
f_0 = w_0 / 2.0 / np.pi
f_min = f_0 * min_harm
f_max = f_0 * max_harm
plot_pad_time = 0

f = h5py.File("Observables.h5", "r")
observables = f["Observables"]
time = observables["time"][1:]
time_len = time.shape[0]
dt = time[1] - time[0]
pulse_envelope = f["Pulse"]["Pulse_envelope_0"][1:]
pulse_time = f["Pulse"]["time"][1:]
u_p = (pulse_envelope**2) * 7.2973525664e-3 * 7.2973525664e-3 / 4.0
state_shift = ground_state + u_p

plot_pad = int(plot_pad_time / dt)

data = observables["dipole_acceleration_0_1"][1:]
padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
data = np.lib.pad(
    data, (int(np.floor((padd2 - data.shape[0]) / 2)),
           int(np.ceil((padd2 - data.shape[0]) / 2))),
    'constant',
    constant_values=(0.0, 0.0))
print w_list
for scale_type in ["lin"]:
    # for scale_type in ["lin", "log"]:
    for num in w_list[rank::size]:
        print "rank", rank, "plotting", num
        scalogram = cwt(data, dt, num, f_min, f_max, nf=400)
        lower_idx = scalogram.shape[1] / 2 - (time_len / 2 + plot_pad)
        upper_idx = scalogram.shape[1] / 2 + (time_len / 2 + plot_pad)
        scalogram = scalogram[:, lower_idx:upper_idx]
        time = np.linspace(0, dt * scalogram.shape[1],
                           scalogram.shape[1]) - plot_pad_time
        fig = plt.figure()
        x, y = np.meshgrid(time,
                           np.logspace(
                               np.log10(f_min),
                               np.log10(f_max), scalogram.shape[0]))
        y /= f_0
        if scale_type == "lin":
            plt.contourf(x, y, np.abs(scalogram), 15, cmap='viridis')
        elif scale_type == "log":
            min_val = 1e-6
            max_val = np.abs(scalogram).max()
            contour_vals = list(10**np.arange(
                -6.,
                np.ceil(np.log10(max_val)) + (np.ceil(np.log10(max_val)) + 6.)
                / 15., (np.ceil(np.log10(max_val)) + 6.) / 15.))
            plot_data = np.abs(scalogram)
            plot_data[plot_data < min_val] = min_val
            plt.contourf(
                x,
                y,
                plot_data,
                contour_vals,
                cmap='viridis',
                norm=LogNorm(),
                rasterized=True)
        else:
            exit("scale fail")
        plt.plot(pulse_time, state_shift / w_0, 'r-')
        plt.plot(pulse_time, (state_shift - 0.125) / w_0, 'w-')
        plt.plot(pulse_time, (state_shift - 0.0556058) / w_0, 'w-')
        plt.plot(pulse_time, (state_shift - 0.0312736) / w_0, 'w-')
        plt.text(
            100, (state_shift[0] - 0.12552) / w_0 - 0.5,
            "2p",
            color='w',
            fontsize=12)
        plt.text(
            100, (state_shift[0] - 0.0556058) / w_0 - 0.5,
            "3p",
            color='w',
            fontsize=12)
        plt.text(
            100, (state_shift[0] - 0.0312736) / w_0 - 0.5,
            "4p",
            color='w',
            fontsize=12)
        plt.text(
            100, (state_shift[0]) / w_0 + 0.2, "$I_p$", color='r', fontsize=12)
        ax = plt.gca()
        plt.xlabel("time (a.u)")
        plt.ylabel("harmonic order")
        plt.yticks(range(1, max_harm + 1, 2))
        plt.ylim(y.min(), y.max())
        plt.colorbar()
        plt.grid(c='gray', ls='--')
        plt.tight_layout()
        if scale_type == "lin":
            plt.savefig("wavelet_" + str(num).zfill(6) + "_lin_gird.png")
        elif scale_type == "log":
            plt.savefig("wavelet_" + str(num).zfill(6) + "_log_gird.png")
        plt.close()
