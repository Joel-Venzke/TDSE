import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import json

with open('input.json', 'r') as data_file:
    data = json.load(data_file)
target_name = data["target"]["name"]
target = h5py.File(target_name + ".h5", "r")
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
energies = target["Energy"][:]


print "Plotting Spectrum"
fig = plt.figure()
data = pulses["field_1"][:]
if np.max(data) > 1e-10:
    data_fft = np.absolute(
            np.fft.fft(
                np.lib.pad(
                    data, (10 * data.shape[0], 10 * data.shape[0]),
                    'constant',
                    constant_values=(0.0, 0.0))))
    data_fft /= data_fft.max()
    # 2*pi/(dt*N)
    spec_time = np.arange(data_fft.shape[0])*2.0*np.pi/(data_fft.shape[0]*(p_time[1]-p_time[0]))
    for n in range(1,int(abs(energies[0]/spec_time[np.argmax(data_fft[:data_fft.shape[0]/2])]))+1):
      print "Plotting:", n
      plt.semilogy(spec_time*n+energies[0],
          data_fft,
          label="field*"+str(n))

for energy in energies:
    print "Plotting:",energy
    plt.axvline(x=energy, color='k')
plt.ylabel("Field Spectrum (arb)")
plt.xlabel("$\omega$ (a.u.)")
plt.title("Field Spectrum")
plt.xlim([energies[0], 0.0])
plt.ylim([1e-4, 1.0])
plt.legend()
fig.savefig("figs/Photons.png")
