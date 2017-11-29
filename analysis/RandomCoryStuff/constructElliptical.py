import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
import h5py

c = 1 / 7.2973525664e-3
f = h5py.File("TDSE.h5", "r")
freq = f["Parameters"]["energy"][0]
afieldMax = f["Parameters"]["field_max"][0]
fieldMax = freq * afieldMax / c
cyclesOn = f["Parameters"]["cycles_on"][0]
ellipticity = f["Parameters"]["ellipticity_0"][0]


dt = 0.005
N = 120.0
t = np.arange(0.0, N, dt)
fx = np.zeros(t.shape[0])
fy = np.zeros(t.shape[0])
env = np.zeros(t.shape[0])
fac = np.sqrt(1 + ellipticity**2)
tot = np.zeros([t.shape[0], t.shape[0]])

x1 = ((N - 1) / 2) - (np.pi / freq)
x2 = ((N - 1) / 2) + (np.pi / freq)
saverSize = int((x2 - x1) / dt + 1)
saver = np.zeros([saverSize, 3])


i = 0
for j, x in enumerate(t):
	s1 = freq * (x - (N / 2)) \
	/ (2 * np.pi * 2 * np.sqrt(np.log(2)) * cyclesOn)
	env[j] = fieldMax * np.exp(-1.0 * s1 * s1)
	fy[j]  = -env[j] * np.cos(freq * x) / fac 
	fx[j]  = env[j] * ellipticity * np.sin(freq * x) / fac

	if x >= x1 and x <= x2:
		saver[i][0] = x
		saver[i][1] = fx[j]
		saver[i][2] = fy[j]
		i += 1

# for j, x in enumerate(fx):
# 	for k, y in enumerate(fy):
# 		tot[j][k] = np.sqrt(x**2 + y**2)
	
radiusC = np.sqrt(fy**2 + fx**2)
thetaC  = np.arctan2(fy, fx)
radius = np.sqrt(saver[:,1]**2 + saver[:,2]**2)
theta  = np.arctan2(saver[:,2], saver[:, 1])
np.savetxt('centralCycle.txt', saver)
# plt.plot(t, env * fx, t, env * fy)
# plt.plot(saver[:, 0], saver[:,1], saver[:, 0], saver[:,2])
plt.plot(theta, radius)
# plt.plot(radiusC)
# plt.imshow(
# 	       tot,
# 	       origin='lower',
# 	       extent=[t.min(), t.max(),
# 	               t.min(), t.max()])
plt.show()