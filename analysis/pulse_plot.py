import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5","r")
pulses = f["Pulse"]
p_time = pulses["time"][:]
dims = f["Parameters"]["num_dims"][:]

fig = plt.figure()
# for i in range(f["Parameters"]["num_pulses"][0]):
#     plt.plot(p_time,pulses["Pulse_envelope_"+str(i)][:])
#     plt.plot(p_time,-1*pulses["Pulse_envelope_"+str(i)][:])

#     plt.plot(p_time,pulses["Pulse_value_"+str(i)][:], label=i)
for i in range(dims[0]):
  plt.plot(p_time,pulses["field_"+str(i)][:], label="field "+str(i))
plt.xlabel("Time a.u.")
plt.ylabel("A a.u.")
plt.title("Pulse - Field")

plt.legend()
fig.savefig("figs/Pulse_field.pdf")