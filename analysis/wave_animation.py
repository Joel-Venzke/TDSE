import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# read data
f = h5py.File("../examples/basic/TDSE.h5","r")
psi_time = f["Wavefunction"]["psi"][:,:]
x        = f["Wavefunction"]["x_value_0"][:]

# shape into a 3d array with time as the first axis
p_sqrt   = np.sqrt(psi_time.shape[1])
psi_time.shape = (psi_time.shape[0],p_sqrt,p_sqrt)

fig = plt.figure()

ims = []
i=0
for psi in psi_time:
    print i
    i+=1
    ims.append((plt.pcolor(x, x, psi.real),))

im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
    blit=False)
im_ani.save('im.mp4', metadata={'artist':'Joel Venzke'})
