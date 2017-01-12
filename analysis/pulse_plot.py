import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File("TDSE.h5","r")
pulses = f["Pulse"]
p_time = pulses["time"][:]

fig = plt.figure()
for i in range(f["Parameters"]["num_pulses"][0]):
    plt.plot(p_time,pulses["Pulse_envelope_"+str(i)][:])
    plt.plot(p_time,-1*pulses["Pulse_envelope_"+str(i)][:])

    plt.plot(p_time,pulses["Pulse_value_"+str(i)][:], label=i)
plt.plot(p_time,pulses["a_field"][:], label="a_field")
plt.xlabel("Time a.u.")
plt.ylabel("A a.u.")
plt.title("Pulse - A Field")

plt.legend()
fig.savefig("figs/Pulse_all.pdf")

# A field
plt.cla()
plt.xlabel("Time a.u.")
plt.ylabel("A")
plt.title("Pulse - A Field")
plt.plot(p_time,pulses["a_field"][:])
fig.savefig("figs/Pulse_A_field.pdf")