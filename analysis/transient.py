import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
import h5py

f = h5py.File("TDSE.h5", "r")
p = h5py.File("Pulse.h5", "r")
pulses = p["Pulse"]
observables = f["Observables"]
p_time = pulses["time"][:]
tau = pulses
num_dims = f["Parameters"]["num_dims"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]

dipole = np.zeros(p_time.shape)
dipole_fft = np.zeros(p_time.shape)
print "Time to plot the transient absorption spectrum"
grid_max = 0.0
fig = plt.figure()

# Dipole
print "Calculating dipole spectrum"
if(num_dims > 1): 
	exit('only supports 1D currently')
dim_idx = 0

dipole = -1.0 * observables["position_expectation_" + str(dim_idx) + '_' + str(dim_idx)][1:]
dipole_fft = np.fft.fft(
    np.lib.pad(
        dipole, (10 * dipole.shape[0], 10 * dipole.shape[0]),
        'constant',
        constant_values=(0.0, 0.0)))
spec_time_dip = np.arange(dipole_fft.shape[0]) * 2.0 * np.pi / (
    dipole_fft.shape[0] * (p_time[1] - p_time[0]))
spec_time_dip = spec_time_dip[:-8]
dipole_fft = dipole_fft[:-8]
f_dip = interpolate.interp1d(spec_time_dip, dipole_fft)
xnew = np.arange(0, spec_time_dip.max(), 0.00001)
ynew_dip = f_dip(xnew) 
grid_max = max(spec_time_dip[np.argmax(dipole_fft[:dipole_fft.shape[0] / 2])],
               grid_max)


# Spectrum
print "Calculating field spectrum"
data = -1.0 * np.gradient(pulses["field_" + str(dim_idx)][:],
                          f["Parameters"]["delta_t"][0]) * 7.2973525664e-3
data_fft = \
    np.fft.fft(
        np.lib.pad(
            data, (10 * data.shape[0], 10 * data.shape[0]),
            'constant',
            constant_values=(0.0, 0.0)))
spec_time = np.arange(data_fft.shape[0]) * 2.0 * np.pi / (
    data_fft.shape[0] * (p_time[1] - p_time[0]))
f = interpolate.interp1d(spec_time, data_fft)
ynew = f(xnew)
print "Plotting Spectrum"
plt.plot(xnew, -2 * (np.conj(ynew) * ynew_dip).imag, label="ATAS")


plt.ylabel("Field Spectrum (arb)")
plt.xlabel("$\omega$ (a.u.)")
plt.title("Field Spectrum")
plt.xlim([0.3, 2])
plt.ylim([-50, 50])
plt.legend()
print "saving figure....may take a while. please hold"
fig.savefig("figs/transient.png")

