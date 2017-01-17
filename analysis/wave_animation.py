import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

# read data
f = h5py.File("TDSE.h5","r")
psi_value = f["Wavefunction"]["psi"][:,:]
psi_time  = f["Wavefunction"]["psi_time"][:]
x         = f["Wavefunction"]["x_value_0"][:]

# calculate location for time to be printed
time_x    = np.min(x)*0.95
time_y    = np.max(x)*0.9

# calculate color bounds
max_val   = np.max(psi_value[:].real)
min_val   = np.min(psi_value[:].real)
print "min: ", min_val
print "max: ", max_val

# shape into a 3d array with time as the first axis
p_sqrt   = np.sqrt(psi_value.shape[1])
psi_value.shape = (psi_value.shape[0],p_sqrt,p_sqrt)

# set up initial figure with color bar
fig = plt.figure()
plt.pcolor(x, x, psi_value[0].real, cmap='viridis',vmin=min_val, vmax=max_val)
# color bar doesn't change during the video so only set it here
plt.colorbar()
plt.xlabel("Electron 2 a.u.")
plt.ylabel("Electron 1 a.u.")
plt.title("Wave Function")

ims = []
i=0
# for psi, time in zip(psi_value[:4],psi_time[:4]):
for psi, time in zip(psi_value,psi_time):
    print i, time
    i+=1
    # add frames
    ims.append((plt.pcolor(x, x, psi.real, cmap='viridis',
        vmin=min_val, vmax=max_val),
        plt.text(time_x, time_y, "Time: "+str(time)+" a.u.", color='white'),))

# animate
im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000,
    blit=False)

#save
im_ani.save('figs/Wavefunction.mp4', metadata={'artist':'Joel Venzke'})