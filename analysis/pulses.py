import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5", "r")
p = h5py.File("Pulse.h5", "r")
observables = f["Observables"]
pulses = p["Pulse"]
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
grid_max = 0.0
fig = plt.figure()
for dim_idx in range(num_dims):
    data = pulses["field_" + str(dim_idx)][:]
    if np.max(data) > 1e-10:
        data_fft = np.absolute(
                np.fft.fft(
                    np.lib.pad(
                        data, (10 * data.shape[0], 10 * data.shape[0]),
                        'constant',
                        constant_values=(0.0, 0.0))))
        # 2*pi/(dt*N)
        spec_time = np.arange(data_fft.shape[0])*2.0*np.pi/(data_fft.shape[0]*(p_time[1]-p_time[0]))
        plt.semilogy(spec_time,
            data_fft,
            label="field " + str(dim_idx))
        grid_max  = max(spec_time[np.argmax(data_fft[:data_fft.shape[0]/2])], grid_max)        

plt.ylabel("Field Spectrum (arb)")
plt.xlabel("$\omega$ (a.u.)")
plt.title("Field Spectrum")
plt.xlim([0, grid_max * 2.0])
plt.legend()
fig.savefig("figs/Spectrum.png")
