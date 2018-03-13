import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm

font = {'size': 18}

matplotlib.rc('font', **font)

f = h5py.File("Observables.h5", "r")
observables = f["Observables"]
pulses = f["Pulse"]
p_time = pulses["time"][:]
time = observables["time"][1:]
num_dims = f["Parameters"]["num_dims"][0]
num_electrons = f["Parameters"]["num_electrons"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]

# HHG Spectrum
print "Plotting HHG Spectrum"
fig = plt.figure(figsize=(24, 18), dpi=80)
# fig = plt.figure()
energy = f["Parameters"]["energy"][0]
energy = 0.057
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            data = observables["dipole_acceleration_"
                               + str(elec_idx) + "_" + str(dim_idx)][:]
            data = data * np.blackman(data.shape[0])
            padd2 = 2**np.ceil(np.log2(data.shape[0] * 16))
            paddT = np.max(time) * padd2 / data.shape[0]
            dH = 2 * np.pi / paddT / energy
            if np.max(data) > 1e-19:
                data = np.absolute(
                    np.fft.fft(
                        np.lib.pad(
                            data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                   int(np.ceil((padd2 - data.shape[0]) / 2))),
                            'constant',
                            constant_values=(0.0, 0.0))))
                data /= data.max()
                plt.semilogy(
                    np.arange(data.shape[0]) * dH,
                    data,
                    label="Total E:" + str(elec_idx) + " Dim:" + str(dim_idx))

            if (np.max(time) > np.max(p_time)):
                data = observables[
                    "dipole_acceleration_"
                    + str(elec_idx) + "_" + str(dim_idx)][1:len(
                        pulses["field_" + str(dim_idx)]
                        [checkpoint_frequency::checkpoint_frequency]) + 1]
                data = data * np.blackman(data.shape[0])
                padd2 = 2**np.ceil(np.log2(data.shape[0] * 16))
                paddT = np.max(p_time) * padd2 / data.shape[0]
                dH = 2 * np.pi / paddT / energy
                if np.max(data) > 1e-19:
                    data = np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data,
                                (int(np.floor((padd2 - data.shape[0]) / 2)),
                                 int(np.ceil((padd2 - data.shape[0]) / 2))),
                                'constant',
                                constant_values=(0.0, 0.0))))
                    data /= data.max()
                    plt.semilogy(
                        np.arange(data.shape[0]) * dH,
                        data,
                        label="Pulse E:" + str(elec_idx) + " Dim:" +
                        str(dim_idx))

plt.ylabel("HHG Spectrum (a.u.)")
plt.title("HHG Spectrum")
# plt.legend()
x_min = 0
x_max = 20
plt.xticks(np.arange(int(x_min), x_max + 1, 1.0))
plt.xlim([x_min, x_max])
plt.ylim([1e-4, 1e0])
plt.grid(True, which='both')
plt.tight_layout()
fig.savefig("figs/HHG_Spectrum.png")
plt.clf()
plt.close(fig)

# HHG Spectrum
print "Plotting HHG Spectrum No Blackman"
fig = plt.figure(figsize=(24, 18), dpi=80)
# fig = plt.figure()
energy = f["Parameters"]["energy"][0]
energy = 0.057
state_energies = [
    -0.518536, -0.127738, -0.125544, -0.0564105, -0.0557645, -0.0556327,
    -0.03162, -0.031348, -0.0312945, -0.0312691, -0.0201923, -0.0200532,
    -0.0200265, -0.0200132, -0.0200069, -0.0140013, -0.0139208, -0.0139056,
    -0.0138979, -0.0138942, -0.013892, -0.0102754, -0.0102247, -0.0102153,
    -0.0102103, -0.010208, -0.0102066, -0.0102057, -0.00786051, -0.00782659,
    -0.0078203, -0.00781699, -0.00781544, -0.00781449, -0.00781385,
    -0.00781339, -0.00620669, -0.00618288, -0.00617849, -0.00617616,
    -0.00617507, -0.0061744, -0.00617395, -0.00617363, -0.00617338,
    -0.00502476, -0.0050074, -0.00500422, -0.00500252, -0.00500172,
    -0.00500123, -0.00500091, -0.00500067, -0.00500049, -0.00500035,
    -0.00415088, -0.00413784, -0.00413547, -0.00413418, -0.00413358,
    -0.00413322, -0.00413297, -0.00413279, -0.00413266, -0.00413255,
    -0.00413246, -0.00348662, -0.00347658, -0.00347475, -0.00347376,
    -0.0034733, -0.00347302, -0.00347283, -0.00347269, -0.00347259, -0.0034725,
    -0.00347244, -0.00347238, -0.00296993, -0.00296203, -0.0029606,
    -0.00295982, -0.00295945, -0.00295923, -0.00295908, -0.00295897,
    -0.00295889, -0.00295883, -0.00295877, -0.00295873, -0.0029587,
    -0.00256012, -0.00255379, -0.00255265, -0.00255203, -0.00255174,
    -0.00255156, -0.00255144, -0.00255135, -0.00255129, -0.00255123,
    -0.00255119, -0.00255116, -0.00255113, -0.00255111, -0.00222963,
    -0.00222449, -0.00222356
]
# energy = 0.0625

p_states = [2]
for n in range(2, 100):
    p_states.append(p_states[-1] + n)

# for state_idx, e in enumerate(state_energies):
#     if state_idx in p_states:
#         plt.axvline(np.abs(e - state_energies[0]) / energy, c='gray', lw=1)
# else:
#     plt.axvline(
#         np.abs(e - state_energies[0]) / energy, c='silver', ls=":", lw=1)

for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            data = observables["dipole_acceleration_"
                               + str(elec_idx) + "_" + str(dim_idx)][:]
            padd2 = 2**np.ceil(np.log2(data.shape[0] * 16))
            paddT = np.max(time) * padd2 / data.shape[0]
            dH = 2 * np.pi / paddT / energy
            if np.max(data) > 1e-19:
                data = np.absolute(
                    np.fft.fft(
                        np.lib.pad(
                            data, (int(np.floor((padd2 - data.shape[0]) / 2)),
                                   int(np.ceil((padd2 - data.shape[0]) / 2))),
                            'constant',
                            constant_values=(0.0, 0.0))))
                data /= data.max()
                plt.semilogy(
                    np.arange(data.shape[0]) * dH,
                    data,
                    label="Total E:" + str(elec_idx) + " Dim:" + str(dim_idx))

            if (np.max(time) > np.max(p_time)):
                data = observables[
                    "dipole_acceleration_"
                    + str(elec_idx) + "_" + str(dim_idx)][1:len(
                        pulses["field_" + str(dim_idx)]
                        [checkpoint_frequency::checkpoint_frequency]) + 1]
                padd2 = 2**np.ceil(np.log2(data.shape[0] * 16))
                paddT = np.max(p_time) * padd2 / data.shape[0]
                dH = 2 * np.pi / paddT / energy
                if np.max(data) > 1e-19:
                    data = np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data,
                                (int(np.floor((padd2 - data.shape[0]) / 2)),
                                 int(np.ceil((padd2 - data.shape[0]) / 2))),
                                'constant',
                                constant_values=(0.0, 0.0))))
                    data /= data.max()
                    plt.semilogy(
                        np.arange(data.shape[0]) * dH,
                        data,
                        '--',
                        label="Pulse E:" + str(elec_idx) + " Dim:" +
                        str(dim_idx))

plt.xlabel("harmonic order (0.057 a.u.)")
plt.ylabel("HHG spectrum (arb.)")
# plt.title("HHG Spectrum No Blackman")
# plt.legend()
x_min = 0
x_max = 20
plt.xticks(np.arange(int(x_min), x_max + 1, 1.0))
plt.xlim([x_min, x_max])
plt.ylim([1e-4, 1e0])
plt.grid()
# plt.grid(True, which='both')
plt.tight_layout()
fig.savefig("figs/HHG_Spectrum_no_blackman.png")
plt.clf()
plt.close(fig)

# HHG Spectrum
print "Plotting HHG Diff"
fig = plt.figure(figsize=(24, 18), dpi=80)
energy = f["Parameters"]["energy"][0]
energy = 0.057
print energy
state_energies = [
    -0.518536, -0.127738, -0.125544, -0.0564105, -0.0557645, -0.0556327,
    -0.03162, -0.031348, -0.0312945, -0.0312691, -0.0201923, -0.0200532,
    -0.0200265, -0.0200132, -0.0200069, -0.0140013, -0.0139208, -0.0139056,
    -0.0138979, -0.0138942, -0.013892, -0.0102754, -0.0102247, -0.0102153,
    -0.0102103, -0.010208, -0.0102066, -0.0102057, -0.00786051, -0.00782659,
    -0.0078203, -0.00781699, -0.00781544, -0.00781449, -0.00781385,
    -0.00781339, -0.00620669, -0.00618288, -0.00617849, -0.00617616,
    -0.00617507, -0.0061744, -0.00617395, -0.00617363, -0.00617338,
    -0.00502476, -0.0050074, -0.00500422, -0.00500252, -0.00500172,
    -0.00500123, -0.00500091, -0.00500067, -0.00500049, -0.00500035,
    -0.00415088, -0.00413784, -0.00413547, -0.00413418, -0.00413358,
    -0.00413322, -0.00413297, -0.00413279, -0.00413266, -0.00413255,
    -0.00413246, -0.00348662, -0.00347658, -0.00347475, -0.00347376,
    -0.0034733, -0.00347302, -0.00347283, -0.00347269, -0.00347259, -0.0034725,
    -0.00347244, -0.00347238, -0.00296993, -0.00296203, -0.0029606,
    -0.00295982, -0.00295945, -0.00295923, -0.00295908, -0.00295897,
    -0.00295889, -0.00295883, -0.00295877, -0.00295873, -0.0029587,
    -0.00256012, -0.00255379, -0.00255265, -0.00255203, -0.00255174,
    -0.00255156, -0.00255144, -0.00255135, -0.00255129, -0.00255123,
    -0.00255119, -0.00255116, -0.00255113, -0.00255111, -0.00222963,
    -0.00222449, -0.00222356
]

p_states = [2]
for n in range(2, 100):
    p_states.append(p_states[-1] + n)

for state_idx, e in enumerate(state_energies):
    if state_idx in p_states:
        plt.axvline(np.abs(e - state_energies[0]) / energy, c='gray', lw=1)
    # else:
    #     plt.axvline(
    #         np.abs(e - state_energies[0]) / energy, c='silver', ls=":", lw=1)
    # energy = 0.057003883037

for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0
                 and f["Parameters"]["coordinate_system_idx"][0] == 1)):
            if (np.max(time) > np.max(p_time)):
                total = observables["dipole_acceleration_"
                                    + str(elec_idx) + "_" + str(dim_idx)][1:]
                pulse = observables[
                    "dipole_acceleration_"
                    + str(elec_idx) + "_" + str(dim_idx)][1:len(
                        pulses["field_" + str(dim_idx)]
                        [checkpoint_frequency::checkpoint_frequency]) + 1]
                free_prop = total.shape[0] - pulse.shape[0]
                padd2 = 2**np.ceil(np.log2(total.shape[0] * 16))
                paddT = np.max(time) * padd2 / total.shape[0]
                dH = 2 * np.pi / paddT / energy
                total = np.abs(
                    np.fft.fft(
                        np.lib.pad(
                            total, (int(
                                np.floor((padd2 - total.shape[0]) / 2.)), int(
                                    np.ceil((padd2 - total.shape[0]) / 2.))),
                            'constant',
                            constant_values=(0.0, 0.0))))
                total /= total.max()
                plot_grid = np.arange(total.shape[0]) * dH
                pulse = np.abs(
                    np.fft.fft(
                        np.lib.pad(
                            pulse, (int(
                                np.floor((padd2 - (
                                    pulse.shape[0] + free_prop)) / 2)), int(
                                        np.ceil((padd2 -
                                                 (pulse.shape[0] + free_prop))
                                                / 2 + free_prop))),
                            'constant',
                            constant_values=(0.0, 0.0))))
                pulse /= pulse.max()
                plot_data = np.abs(total / pulse)
                plt.plot(
                    plot_grid,
                    plot_data / plot_data.max(),
                    label="Pulse E:" + str(elec_idx) + " Dim:" + str(dim_idx))

plt.ylabel("HHG Spectrum (a.u.)")
plt.title("HHG Spectrum No Blackman")
plt.legend()
x_min = 6.5
x_max = 9.5
plt.xticks(np.arange(int(x_min), x_max + 1, 1.0))
plt.xlim([x_min, x_max])
plt.ylim([0, 1])
# plt.ylim([1e-5, 1e-2])
# plt.grid(True, which='both')
plt.tight_layout()
fig.savefig("figs/HHG_Spectrum_diff_w_a.png")
plt.clf()
plt.close(fig)