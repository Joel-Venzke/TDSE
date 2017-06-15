import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5", "r")
pulses = f["Pulse"]
p_time = pulses["time"][:]
dims = f["Parameters"]["num_dims"][:]

fig = plt.figure()
for i in range(dims[0]):
    plt.plot(p_time, pulses["field_" + str(i)][:], label="field " + str(i))
plt.xlabel("Time a.u.")
plt.ylabel("A a.u.")
plt.title("Pulse - Field")

plt.legend()
fig.savefig("figs/Pulse_field.png")
