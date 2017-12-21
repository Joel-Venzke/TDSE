import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import h5py

z = []
time = []
intensity = [
    "2.71e13", "2.75e13", "3.00e13", "3.25e13", "3.50e13", "3.75e13", "4.00e13", "4.25e13", "4.50e13", "4.75e13", "5.00e13", "5.25e13",
    "5.31e13", "5.5e13", "5.75e13", "6.0e13", "6.25e13", "6.5e13", "6.75e13", "7.0e13",
    "7.25e13", "7.5e13", "7.75e13", "7.91e13"
]
#intensity = [
#    "5.31e13", "6.5e13", "7.91e13"
#]

folders = []

for i in intensity:
    key = i + "/"
    folders.append(key)

# HHG Spectrum
print "Plotting HHG Spectrum"
count = 0
harm_value = 0
harm_match = 1
energy = 0.057
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
    #energy = f["Parameters"]["energy"][0]
    for elec_idx in range(num_electrons):
        for dim_idx in range(num_dims):
            if (not (dim_idx == 0
                     and f["Parameters"]["coordinate_system_idx"][0] == 1)):
                data = observables[
                    "dipole_acceleration_"
                    + str(elec_idx) + "_" + str(dim_idx)][1:len(
                        pulses["field_" + str(dim_idx)]
                        [checkpoint_frequency::checkpoint_frequency]) + 1]
                data = data * np.blackman(data.shape[0])
                padd2 = 2**np.ceil(np.log2(data.shape[0] * 4))
                paddT = np.max(time) * padd2 / data.shape[0]
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
                    #data /= data.max()
                    #if count == 0:
                    #    count = 1
                    #    harm_value = data[np.argmin(
                    #        np.abs(np.arange(data.shape[0]) * dH - harm_match))]
                    #else:
                    #    data *= harm_value / data[np.argmin(
                    #        np.abs(np.arange(data.shape[0]) * dH - harm_match))]
                    z.append(data[:data.shape[0]/2])
                    time = np.arange(data[:data.shape[0]/2].shape[0]) * dH

min_harm = 6.
max_harm = 30.
start_idx = np.argmin(np.abs(time-min_harm))
end_idx = np.argmin(np.abs(time-max_harm))
print start_idx, end_idx
time = time[start_idx:end_idx]
intensity = np.array(intensity, dtype=float)
z = np.log10(np.array(z)[:,start_idx:end_idx])
print "setting up interp", z.shape, time.shape, intensity.shape
interp = scipy.interpolate.RectBivariateSpline(intensity,time,  z)
points_y = 100.
intensity = np.arange(intensity.min(),intensity.max(), (intensity.max()-intensity.min())/points_y)
print "getting new z"
z = interp(intensity, time)
time, intensity = np.meshgrid(time, intensity)
min_val = -2
z[z<min_val] = min_val
z_log = np.array(z)
z_log_min = z_log.min()
z_log_max = z_log.max()
z = 10**z
#print "printing"
#fig = plt.figure()
#plt.pcolormesh(time, intensity, z, cmap='viridis')
#plt.colorbar()
#plt.savefig("figs/HHG_heat_all.png")
fig = plt.figure()
for c in range(5,60):
    v = 10**np.arange(z_log_min,z_log_max+np.abs(z_log_max-z_log_min)/(2*c), np.abs(z_log_max-z_log_min)/c)
    v_lab = 10**np.arange(z_log_min,z_log_max)
    print v
    plt.contourf(time, intensity, z, v, cmap='viridis', norm=LogNorm())
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)))
    cbar.set_label('HHG Intensity (arb.)')
    plt.ylim([4e13,6e13])
    small = 6
    large = 16
    plt.xlim([small,large])
    plt.xticks(range(small, large+1))
    #plt.grid()
    #plt.axhline(2.3472102925e+13, c='r', ls='-')
    plt.axhline(4.94691079523e+13, c='r', ls='-')
    plt.axhline(7.54661129797e+13, c='r', ls='-')
    plt.axvline(0.518536/energy, c='r', ls='-')
    plt.axvline((0.518536-0.125544)/energy, c='w', ls='-')
    plt.axvline((0.518536-0.0557645)/energy, c='w', ls='-')
    plt.axvline((0.518536-0.031348)/energy, c='w', ls='-')
    plt.text(0.518536/energy, 6.1e13, "$I_p$")
    plt.text((0.518536-0.125544)/energy, 6.1e13, "2p")
    plt.text((0.518536-0.0557645)/energy, 6.1e13, "3p")
    plt.text((0.518536-0.031348)/energy, 6.1e13, "4p")
    plt.xlabel("Harmonic Order (800nm)")
    plt.ylabel("Laser Intensity ($W/cm^2$)")
    plt.savefig("figs/HHG_cont_all_"+str(c).zfill(3)+".png")
    plt.clf()
    plt.contourf(time, intensity, z, v, cmap='viridis', norm=LogNorm())
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)))
    cbar.set_label('HHG Intensity (arb.)')
    small = 6
    large = 10
    plt.xlim([small,large])
    plt.xticks(range(small, large+1))
    #plt.grid()
    plt.xlabel("Harmonic Order (800nm)")
    plt.ylabel("Laser Intensity ($W/cm^2$)")
    #plt.axhline(2.3472102925e+13, c='r', ls='-')
    plt.axhline(4.94691079523e+13, c='r', ls='-')
    plt.axhline(7.54661129797e+13, c='r', ls='-')
    plt.axvline(0.518536/energy, c='r', ls='-')
    plt.axvline((0.518536-0.125544)/energy, c='w', ls='-')
    plt.axvline((0.518536-0.0557645)/energy, c='w', ls='-')
    plt.axvline((0.518536-0.031348)/energy, c='w', ls='-')
    plt.axvline((0.518536-0.0200532)/energy, c='w', ls='-')
    plt.text(0.518536/energy, 8e13, "$I_p$")
    plt.text((0.518536-0.125544)/energy, 8e13, "2p")
    plt.text((0.518536-0.0557645)/energy, 8e13, "3p")
    plt.text((0.518536-0.031348)/energy, 8e13, "4p")
    plt.text((0.518536-0.0200532)/energy, 8e13, "5p")
    plt.savefig("figs/HHG_cont_8_"+str(c).zfill(3)+".png")
    plt.clf()
    #plt.contourf(time, intensity, z, v, cmap='viridis', norm=LogNorm())
    #cbar = plt.colorbar(ticks = LogLocator(subs=range(10)))
    #cbar.set_label('HHG Intensity (arb.)')
    #small = 12
    #large = 18
    #plt.xlim([small,large])
    #plt.xticks(range(small, large+1))
    #plt.grid()
    #plt.xlabel("Harmonic Order (800nm)")
    #plt.ylabel("Laser Intensity ($W/cm^2$)")
    #plt.savefig("figs/HHG_cont_high_"+str(c).zfill(3)+".png")
    #plt.clf()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#print time.shape, intensity.shape, z.shape
#for a in np.arange(0,360,5):
#    print a
#    surf = ax.plot_surface(
#        time, intensity, z, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap="viridis")
#    ax.autoscale_view()
#    ax.set_xlim([max_harm, min_harm])
#    ax.view_init(30, a)
#    plt.savefig("figs/HHG_3D_all_"+str(a).zfill(5)+".png")
#    ax.cla()
