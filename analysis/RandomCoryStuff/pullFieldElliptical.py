import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5", "r")
p = h5py.File("Pulse.h5", "r")
pulses = p["Pulse"]
p_time = pulses["time"][:]
freq = f["Parameters"]["energy"][0]
num_dims = f["Parameters"]["num_dims"][0]
num_pulses = f["Parameters"]["num_pulses"][0]
checkpoint_frequency = f["Parameters"]["write_frequency_observables"][0]
field = np.zeros([num_dims, p_time.shape[0]])
cycles_on = f["Parameters"]["cycles_on"]
ellipticity = f["Parameters"]["ellipticity_0"][0]

dt = f["Parameters"]["delta_t"][0]
period = 2 * np.pi / freq
t1 = (p_time.max() - period) / 2
t2 = (p_time.max() + period) / 2
saverSize = int((t2 - t1) / dt + 1)
central = np.zeros([num_dims + 1, saverSize - 1])

for dim_idx in range(num_dims):
        field[dim_idx][:] = -1.0 * np.gradient(pulses["field_" + str(dim_idx)][:],
                           f["Parameters"]["delta_t"][0]) * 7.2973525664e-3
i = 0
for k, t in enumerate(p_time):
	if t < t2 and t > t1:
		central[0][i] = p_time[k]
		central[1][i] = field[0][k]
		central[2][i] = field[1][k]
		i += 1

t_central = central[0][:]
x_central = central[1][:]
y_central = central[2][:]
strength  = np.sqrt(x_central**2 + y_central**2)
angle     = np.arctan2(y_central, x_central)

plt.plot(t_central, x_central, t_central, y_central)
# plt.plot(angle, strength)
plt.xlabel("Time (a.u.)")
plt.ylabel("Field (a.u.)")
plt.title("E Field")
plt.legend(['x', 'y'])
np.savetxt('centralCycle.txt', central)

# fig.savefig("figs/Pulse_total_E_field.png")
plt.show()

# radius = np.sqrt(field[0][:]**2 + field[1][:]**2)
# plt.plot(radius)
# plt.show()