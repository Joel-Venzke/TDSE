import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
f = None
p = None
try:
    f = h5py.File("TDSE.h5", "r")
    p = h5py.File("Pulse.h5", "r")
except:
    f = h5py.File("Observables.h5", "r")
    p = f
pulses = p["Pulse"]
p_time = pulses["time"][:]
num_dims = f["Parameters"]["num_dims"][0]
num_electrons = f["Parameters"]["num_electrons"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
energy = f["Parameters"]["energy"][0]
energy = 0.057

# Field Plot
print "Plotting A Field"
fig = plt.figure()
for dim_idx in range(num_dims):
    print np.max(np.abs(pulses["field_" + str(dim_idx)][:]))
    plt.plot(
        p_time,
        pulses["field_" + str(dim_idx)][:],
        label="field " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("A Field")
plt.legend()
fig.savefig("figs/Pulse_total_A_field.png")

print "Plotting E Field"
fig = plt.figure()
for dim_idx in range(num_dims):
    plt.plot(
        p_time,
        -1.0 * np.gradient(pulses["field_" + str(dim_idx)][:],
                           f["Parameters"]["delta_t"][0]) * 7.2973525664e-3,
        label="field " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("E Field")
plt.legend()
fig.savefig("figs/Pulse_total_E_field.png")

# Field Plot
print "Plotting Pulses A"
fig = plt.figure()
for pulse_idx in range(num_pulses):
    plt.plot(p_time, pulses["Pulse_envelope_" + str(pulse_idx)][:], 'r--')
    plt.plot(p_time, -1.0 * pulses["Pulse_envelope_" + str(pulse_idx)][:],
             'r--')
    for dim_idx in range(num_dims):
        plt.plot(
            p_time,
            pulses["Pulse_value_" + str(pulse_idx) + "_" + str(dim_idx)][:],
            label="Pulse " + str(pulse_idx) + " Dim " + str(dim_idx))
        print(p_time[2] - p_time[1]) * (
            pulses["Pulse_value_"
                   + str(pulse_idx) + "_" + str(dim_idx)][:]).sum()

plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("Pulses")
plt.legend()
fig.savefig("figs/Pulses_A_field.png")

print "Plotting Envelope derivatives"
fig = plt.figure()
for pulse_idx in range(num_pulses):
    plt.plot(p_time,
             np.gradient(pulses["Pulse_envelope_" + str(pulse_idx)][:],
                         f["Parameters"]["delta_t"][0]), 'r--')
    plt.plot(p_time, pulses["Pulse_envelope_" + str(pulse_idx)][:] /
             np.max(pulses["Pulse_envelope_" + str(pulse_idx)][:]), 'r--')
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("Pulses")
fig.savefig("figs/Pulses_Envelope_Derivative.png")

# print "Plotting Pulses E"
# fig = plt.figure()
# for pulse_idx in range(num_pulses):
#     plt.plot(p_time, pulses["Pulse_envelope_" + str(pulse_idx)][:] *
#              7.2973525664e-3 * f["Parameters"]["energy"][pulse_idx], 'k')
#     plt.plot(
#         p_time,
#         -1.0 * pulses["Pulse_envelope_" + str(pulse_idx)][:] * 7.2973525664e-3
#         * f["Parameters"]["energy"][pulse_idx],
#         'k',
#         label="$\\frac{\omega}{c}\\times A(t)$ Envelope")

#     plt.plot(
#         p_time,
#         -1.0 * np.gradient(pulses["Pulse_value_" + str(pulse_idx) + "_1"][:],
#                            f["Parameters"]["delta_t"][0]) * 7.2973525664e-3,
#         '--',
#         label="E")

#     plt.plot(
#         p_time,
#         pulses["Pulse_value_" + str(pulse_idx) + "_1"][:] * 7.2973525664e-3 *
#         f["Parameters"]["energy"][pulse_idx],
#         label="$\\frac{\omega}{c}\\times A(t)$")
#     # for dim_idx in range(num_dims):
# # plt.plot(
# #     p_time,
# #     -1.0 *
# #     np.gradient(pulses["Pulse_value_"
# #                        + str(pulse_idx) + "_" + str(dim_idx)][:],
# #                 f["Parameters"]["delta_t"][0]) * 7.2973525664e-3,
# #     label="Pulse " + str(pulse_idx) + " Dim " + str(dim_idx))
# # print(p_time[2] - p_time[1]) * ((-1.0 * np.gradient(
# #     pulses["Pulse_value_" + str(pulse_idx) + "_" + str(dim_idx)][:], f[
# #         "Parameters"]["delta_t"][0]) * 7.2973525664e-3)**2).sum()
# # print(p_time[2] - p_time[1]) * ((-1.0 * np.gradient(
# #     pulses["Pulse_value_" + str(pulse_idx) + "_" + str(dim_idx)][:], f[
# #         "Parameters"]["delta_t"][0]) * 7.2973525664e-3)).sum()
# plt.xlabel("Time (a.u.)")
# plt.ylabel("Field (a.u.)")
# plt.title("Pulses")
# plt.legend()
# fig.savefig("figs/Pulses_E_field.png")

# Spectrum
print "Plotting Spectrum"
grid_max = 0.0
fig = plt.figure()
energy = f["Parameters"]["energy"][0]
for dim_idx in range(num_dims):
    # data = -1.0 * np.gradient(pulses["field_" + str(dim_idx)][:],
    #                           f["Parameters"]["delta_t"][0]) * 7.2973525664e-3
    # if np.max(data) > 1e-10:
    #     data_fft = np.absolute(
    #         np.fft.fft(
    #             np.lib.pad(
    #                 data, (300 * data.shape[0], 300 * data.shape[0]),
    #                 'constant',
    #                 constant_values=(0.0, 0.0))))
    #     spec_time = np.arange(data_fft.shape[0]) * 2.0 * np.pi / (
    #         data_fft.shape[0] * (p_time[1] - p_time[0]))
    #     plt.plot(
    #         spec_time, data_fft / data_fft.max(), '--', label="E($\omega$)")
    #     grid_max = max(spec_time[np.argmax(data_fft[:data_fft.shape[0] / 2])],
    #                    grid_max)
    #     print spec_time[2] - spec_time[1]
    data = pulses["field_" + str(dim_idx)][:]
    if np.max(data) > 1e-10:
        data_fft = np.absolute(
            np.fft.fft(
                np.lib.pad(
                    data, (10 * data.shape[0], 10 * data.shape[0]),
                    'constant',
                    constant_values=(0.0, 0.0))))
        spec_time = np.arange(data_fft.shape[0]) * 2.0 * np.pi / (
            data_fft.shape[0] * (p_time[1] - p_time[0]))
        plt.plot(spec_time, data_fft / data_fft.max(), label="E($\omega$)")
        grid_max = max(spec_time[np.argmax(data_fft[:data_fft.shape[0] / 2])],
                       grid_max)
    # plt.axvline(x=energy, color='k')
    # plt.axvline(x=grid_max, color='r')
plt.axvline(x=energy, color='k')
plt.ylabel("Field Spectrum (arb)")
plt.xlabel("$\omega$ (a.u.)")
plt.title("Field Spectrum")
plt.xlim([0, 2 * energy])
# plt.ylim([0.5, 1])
# plt.ylim(ymin=0)
plt.grid()
plt.legend()
fig.savefig("figs/Spectrum.png")
print "Omega Error (A vs E):", grid_max - energy, grid_max