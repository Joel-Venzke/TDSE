import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

# f = h5py.File("Observables.h5", "r")
# p = f
f = h5py.File("TDSE.h5", "r")
p = h5py.File("Pulse.h5", "r")
pulses = p["Pulse"]
p_time = pulses["time"][:]
num_dims = f["Parameters"]["num_dims"][0]
num_electrons = f["Parameters"]["num_electrons"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
energy = f["Parameters"]["energy"][0]

# Field Plot
print "Plotting A Field"
fig = plt.figure()
for dim_idx in range(num_dims):
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
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("Pulses")
plt.legend()
fig.savefig("figs/Pulses_A_field.png")

print "Plotting Pulses E"
fig = plt.figure()
for pulse_idx in range(num_pulses):
    plt.plot(p_time, pulses["Pulse_envelope_" + str(pulse_idx)][:] *
             7.2973525664e-3 * f["Parameters"]["energy"][pulse_idx], 'r--')
    plt.plot(p_time, -1.0 * pulses["Pulse_envelope_" + str(pulse_idx)][:] *
             7.2973525664e-3 * f["Parameters"]["energy"][pulse_idx], 'r--')
    for dim_idx in range(num_dims):
        plt.plot(
            p_time,
            -1.0 *
            np.gradient(pulses["Pulse_value_"
                               + str(pulse_idx) + "_" + str(dim_idx)][:],
                        f["Parameters"]["delta_t"][0]) * 7.2973525664e-3,
            label="Pulse " + str(pulse_idx) + " Dim " + str(dim_idx))
        print(p_time[2] - p_time[1]) * ((-1.0 * np.gradient(
            pulses["Pulse_value_" + str(pulse_idx) + "_" + str(dim_idx)][:], f[
                "Parameters"]["delta_t"][0]) * 7.2973525664e-3)**2).sum()
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("Pulses")
plt.legend()
fig.savefig("figs/Pulses_E_field.png")

# Spectrum
print "Plotting Spectrum"
grid_max = 0.0
fig = plt.figure()
for dim_idx in range(num_dims):
    data = -1.0 * np.gradient(pulses["field_" + str(dim_idx)][:],
                              f["Parameters"]["delta_t"][0]) * 7.2973525664e-3
    if np.max(data) > 1e-10:
        data_fft = np.absolute(
            np.fft.fft(
                np.lib.pad(
                    data, (10 * data.shape[0], 10 * data.shape[0]),
                    'constant',
                    constant_values=(0.0, 0.0))))
        spec_time = np.arange(data_fft.shape[0]) * 2.0 * np.pi / (
            data_fft.shape[0] * (p_time[1] - p_time[0]))
        plt.semilogy(spec_time, data_fft, label="field " + str(dim_idx))
        grid_max = max(spec_time[np.argmax(data_fft[:data_fft.shape[0] / 2])],
                       grid_max)
plt.axvline(x=energy, color='k')
plt.axvline(x=grid_max, color='r')
plt.ylabel("Field Spectrum (arb)")
plt.xlabel("$\omega$ (a.u.)")
plt.title("Field Spectrum")
plt.xlim([0, grid_max * 8.0])
plt.legend()
fig.savefig("figs/Spectrum.png")
print "Omega Error (A vs E):", grid_max - energy, grid_max