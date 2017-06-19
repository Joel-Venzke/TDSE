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

# Norm
print "Plotting Norm"
fig = plt.figure()
plt.plot(time, observables["norm"][1:])
plt.xlabel("Time (a.u.)")
plt.ylabel("Norm")
plt.title("Norm")
plt.tight_layout()
fig.savefig("figs/Norm.png")
plt.clf()

# Ionization
print "Plotting Ionization"
fig = plt.figure()
plt.plot(time, 1.0 - observables["norm"][1:])
plt.xlabel("Time (a.u.)")
plt.ylabel("Ionization")
plt.title("Ionization")
plt.tight_layout()
fig.savefig("figs/Ionization.png")
plt.clf()

# Ionization
print "Plotting Ionization Rate"
fig = plt.figure()
plt.plot(time, np.gradient(1.0 - observables["norm"][1:]))
plt.xlabel("Time (a.u.)")
plt.ylabel("Ionization Rate")
plt.title("Ionization Rate")
plt.tight_layout()
fig.savefig("figs/Ionization_rate.png")
plt.clf()

# Gobbler
print "Plotting ECS Population"
fig = plt.figure()
if np.max(observables["gobbler"][1:]) > 1e-10:
    plt.semilogy(time, observables["gobbler"][1:])
    plt.xlabel("Time (a.u.)")
    plt.ylabel("ECS Population")
    plt.title("ECS Population")
    plt.tight_layout()
    fig.savefig("figs/ECS_population.png")
plt.clf()

# Dipole
print "Plotting Dipole"
fig = plt.figure()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0 and
                 f["Parameters"]["coordinate_system_idx"][0] == 1)):
            plt.plot(
                time,
                observables["position_expectation_" +
                            str(elec_idx) + "_" + str(dim_idx)][1:],
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Dipole (a.u.)")
plt.title("Dipole Moment")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Dipole.png")
plt.clf()

# dipole with ionization
print "Plotting Dipole with ionization"
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0 and
                 f["Parameters"]["coordinate_system_idx"][0] == 1)):
            ax1.plot(
                time,
                observables["position_expectation_" +
                            str(elec_idx) + "_" + str(dim_idx)][1:],
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
ax2.plot(time, 1.0 - observables["norm"][1:], 'r--', label="Ionization")
ax1.set_xlabel("Time a.u.")
ax1.set_ylabel("Dipole (a.u.)")
ax2.set_ylim(ymin=0)
ax1.legend(loc=2)
ax2.legend(loc=1)
fig2.savefig("figs/Dipole_with_ionization.png")

# Dipole with envelope
print "Plotting Dipole with field Envelope"
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0 and
                 f["Parameters"]["coordinate_system_idx"][0] == 1)):
            ax1.plot(
                time,
                np.abs(observables["position_expectation_" + str(elec_idx) +
                                   "_" + str(dim_idx)][1:]),
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
for pulse_idx in range(num_pulses):
    ax2.plot(
        p_time,
        pulses["Pulse_envelope_" + str(pulse_idx)],
        'r--',
        label="Pulse " + str(pulse_idx))
ax1.set_xlabel("Time a.u.")
ax1.set_ylabel("Dipole (a.u.)")
ax2.set_ylabel("Field (a.u.)")
ax1.legend(loc=2)
ax2.legend(loc=1)
fig2.savefig("figs/Dipole_with_field_envelope.png")

# Dipole acceleration
print "Plotting Dipole Acceleration"
fig = plt.figure()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0 and
                 f["Parameters"]["coordinate_system_idx"][0] == 1)):
            plt.plot(
                time,
                observables["dipole_acceleration_" +
                            str(elec_idx) + "_" + str(dim_idx)][1:],
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.xlabel("Time (a.u.)")
plt.ylabel("Dipole Acceleration (a.u.)")
plt.title("Dipole Acceleration")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Dipole_acceleration.png")
plt.clf()

# HHG Spectrum
print "Plotting HHG Spectrum"
fig = plt.figure()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0 and
                 f["Parameters"]["coordinate_system_idx"][0] == 1)):
            data = observables["dipole_acceleration_" +
                               str(elec_idx) + "_" + str(dim_idx)][1:]
            data = data * np.blackman(data.shape[0])
            if np.max(data) > 1e-10:
                plt.semilogy(
                    np.absolute(
                        np.fft.fft(
                            np.lib.pad(
                                data, (2 * data.shape[0], 2 * data.shape[0]),
                                'constant',
                                constant_values=(0.0, 0.0)))),
                    label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.ylabel("HHG Spectrum (a.u.)")
plt.title("HHG Spectrum")
plt.legend()
plt.xlim([0, 500])
plt.tight_layout()
fig.savefig("figs/HHG_Spectrum.png")
plt.clf()

# Dipole acceleration with envelope
print "Plotting Dipole Acceleration with field Envelope"
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0 and
                 f["Parameters"]["coordinate_system_idx"][0] == 1)):
            ax1.plot(
                time,
                np.abs(observables["dipole_acceleration_" + str(elec_idx) + "_"
                                   + str(dim_idx)][1:]),
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
for pulse_idx in range(num_pulses):
    ax2.plot(
        p_time,
        pulses["Pulse_envelope_" + str(pulse_idx)],
        'r--',
        label="Pulse " + str(pulse_idx))
ax1.set_xlabel("Time (a.u.)")
ax1.set_ylabel("Dipole Acceleration (a.u.)")
ax2.set_ylabel("Field (a.u.)")
ax1.legend(loc=2)
ax2.legend(loc=1)
fig2.savefig("figs/Dipole_acceleration_with_field_envelope.png")

# Linearity
print "Plotting Linearity"
fig = plt.figure()
for elec_idx in range(num_electrons):
    for dim_idx in range(num_dims):
        if (not (dim_idx == 0 and
                 f["Parameters"]["coordinate_system_idx"][0] == 1)):
            plt.plot(
                observables["dipole_acceleration_" +
                            str(elec_idx) + "_" + str(dim_idx)][1:],
                pulses["field_" + str(dim_idx)][checkpoint_frequency::
                                                checkpoint_frequency],
                label="Electron " + str(elec_idx) + " Dim " + str(dim_idx))
plt.xlabel("Field in (a.u.)")
plt.ylabel("Dipole Acceleration (a.u.)")
plt.title("Linearity")
plt.legend()
plt.tight_layout()
fig.savefig("figs/Linearity.png")
plt.clf()
