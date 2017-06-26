import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5", "r")
observables = f["Observables"]
pulses = f["Pulse"]
p_time = pulses["time"][:]
time = observables["time"][1:]
num_dims = f["Parameters"]["num_dims"][0]
num_electrons = f["Parameters"]["num_electrons"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]

# Field Plot
print "Plotting Field"
fig = plt.figure()
for dim_idx in range(num_dims):
    plt.plot(
        p_time,
        pulses["field_" + str(dim_idx)][:],
        label="field " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("Field")
plt.legend()
fig.savefig("figs/Pulse_field.png")

# Field Plot
print "Plotting Pulses"
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
fig.savefig("figs/Pulses.png")

# Spectrum
print "Plotting Spectrum"
fig = plt.figure()
for dim_idx in range(num_dims):
    data = pulses["field_" + str(dim_idx)][:]
    if np.max(data) > 1e-10:
        plt.semilogy(
            np.absolute(
                np.fft.fft(
                    np.lib.pad(
                        data, (10 * data.shape[0], 10 * data.shape[0]),
                        'constant',
                        constant_values=(0.0, 0.0)))),
            label="field " + str(dim_idx))
plt.ylabel("Field Spectrum")
plt.title("Field Spectrum")
plt.xlim([0, 1000])
plt.legend()
fig.savefig("figs/Spectrum.png")
