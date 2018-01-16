import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
from matplotlib.colors import LogNorm
import h5py

z = []
time = []
intensity = [
    "2.71e13", "2.75e13", "3.00e13", "3.25e13", "3.50e13", "3.75e13", "4.00e13",
    "4.25e13", "4.50e13", "4.75e13", "5.00e13", "5.25e13", "5.31e13", "5.5e13", "5.75e13", "6.0e13",
    "6.25e13", "6.5e13", "6.75e13", "7.0e13", "7.25e13", "7.5e13", "7.75e13",
    "7.91e13"
]

folders = []

for i in intensity:
    key = i + "/"
    folders.append(key)

# HHG Spectrum
print "Plotting HHG Spectrum"
count = 0
harm_value = 0
for fold in folders:
    print fold
    f = h5py.File(fold + "TDSE.h5", "r")
    p = h5py.File(fold + "Pulse.h5", "r")
    observables = f["Observables"]
    pulses = p["Pulse"]
    p_time = pulses["time"][:]
    time = observables["time"][1:]
    num_dims = f["Parameters"]["num_dims"][0]
    num_electrons = f["Parameters"]["num_electrons"][0]
    num_pulses = f["Parameters"]["num_pulses"][0]
    checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
    energy = f["Parameters"]["energy"][0]
    data = f["Wavefunction"]["projections"][:]
    data = np.absolute(data[:, :, 0] + 1j * data[:, :, 1])
    data *= data
    #z.append(data[-1,36:45])
    print data.shape
    z.append(data[-1,91:])
    time = np.arange(z[-1].shape[0]+1)

intensity = np.array(intensity, dtype=float)
z = np.array(z)
#interp = scipy.interpolate.RectBivariateSpline(intensity,time,  z)
#points_y = 100.
#intensity = np.arange(intensity.min(),intensity.max(), (intensity.max()-intensity.min())/points_y)
#print "getting new z"
#z = interp(intensity, time)
#z[z<=0] = 1e-20
time, intensity = np.meshgrid(time, intensity)
fig = plt.figure()
plt.pcolormesh(time, intensity, z, cmap='viridis', norm=LogNorm(vmin=1e-10))
ax = plt.gca()
ax.set_xticks(np.arange(0.5, z.shape[1]+1, 1))
ax.set_xticklabels(np.arange(0, z.shape[1]+1, 1))
plt.xlim([time.min(),time.max()])
plt.ylim([intensity.min(),intensity.max()])
plt.xlabel("l value")
plt.ylabel("Intensity $(W/cm^2)$")
cbar = plt.colorbar()
cbar.set_label('Population')
#plt.savefig("figs/Proj_heat_8_.png")
plt.savefig("figs/Proj_heat_14_.png")
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print time.shape, intensity.shape, z.shape
#z = np.log10(z)
#for a in np.arange(0, 360, 5):
#    print a
#    surf = ax.plot_surface(
#        time,
#        intensity,
#        z,
#        rstride=1,
#        cstride=1,
#        linewidth=0,
#        antialiased=False,
#        cmap="viridis")
#    ax.autoscale_view()
#    ax.view_init(30, a)
#    plt.savefig("figs/Proj_3D_14_" + str(a).zfill(5) + ".png")
#    ax.cla()
