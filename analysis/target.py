import numpy as np
import h5py
target_name = "He-SAE"
# read data
target = h5py.File(target_name+".h5","r")
f = h5py.File("TDSE.h5","r")
psi_value = target["psi"]
# psi_value = f["Wavefunction"]["psi"]
energy    = target["Energy"]
psi_time  = f["Wavefunction"]["time"][:]
shape         = f["Wavefunction"]["num_x"][:]
print shape
x         = f["Wavefunction"]["x_value_0"][:]
y         = f["Wavefunction"]["x_value_1"][:]

if len(shape) == 3: 
    from mayavi import mlab
    z         = f["Wavefunction"]["x_value_2"][:]
    mlab.figure(bgcolor=(1.0,1.0,1.0),fgcolor=(0.0,0.0,0.0),size=(1000,1000))
    for i, psi in enumerate(psi_value):
        psi = psi[:,0]+1j*psi[:,1]
        psi.shape = tuple(shape)
        # print np.log10(np.abs(psi))
        mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(np.log10(np.abs(psi))),
            vmin=-05,
            opacity=0.3,
            colormap="viridis",
            contours=[-3,-2,-1])
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(np.log10(np.abs(psi))),vmin=-15,
        #    plane_orientation='z_axes',
        #    transparent=True,colormap="viridis",
        #    slice_index=shape[2]/2)
        mlab.colorbar(nb_labels=4,orientation="vertical")
        # mlab.show()
        mlab.savefig("figs/target_"+str(i).zfill(8)+".png")
        mlab.clf()

elif len(shape) == 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    fig = plt.figure()
    for i, psi in enumerate(psi_value):
        print("plotting", i)
        psi = psi[:,0]+1j*psi[:,1]
        psi.shape = tuple(shape)
        if len(shape) == 3:
            psi = psi[:,:,shape[-1]/2]
        # set up initial figure with color bar
        max_val   = np.max(abs(psi.real))
        min_val   = -1*max_val
        plt.clf()
        plt.imshow(psi.real, cmap='bwr', vmin=-1*max_val, vmax=max_val,
                   origin='lower', extent=[y[0],y[-1],x[0],x[-1]])
        # color bar doesn't change during the video so only set it here
        plt.colorbar()
        plt.xlabel("Electron 2 a.u.")
        plt.ylabel("Electron 1 a.u.")
        plt.title("Wave Function - Energy "+str(energy[i]))
        fig.savefig("figs/"+target_name+"_bwr_state_"+str(i).zfill(3)+".jpg")
        plt.clf()
        plt.imshow(psi.imag, cmap='bwr', vmin=-1*max_val, vmax=max_val,
                   origin='lower', extent=[y[0],y[-1],x[0],x[-1]])
        # color bar doesn't change during the video so only set it here
        plt.colorbar()
        plt.xlabel("Electron 2 a.u.")
        plt.ylabel("Electron 1 a.u.")
        plt.title("Wave Function - Energy "+str(energy[i]))
        fig.savefig("figs/"+target_name+"_bwr_state_imag_"+str(i).zfill(3)+".jpg")

        plt.clf()
        if f["Parameters"]["coordinate_system_idx"]==1:
            plt.imshow(np.vdot(psi,np.multiply(x,psi.transpose()).transpose()), cmap='viridis', origin='lower',
                extent=[y[0],y[-1],x[0],x[-1]],
                norm=LogNorm(vmin=1e-15, vmax=np.abs(psi.real).max()))
        else:
            plt.imshow(np.abs(psi), cmap='viridis', origin='lower',
                extent=[y[0],y[-1],x[0],x[-1]],
                norm=LogNorm(vmin=1e-15, vmax=np.abs(psi.real).max()))
        # plt.pcolor(x, x, abs(psi.real),
        #     norm=LogNorm(vmin=1e-16,
        #         vmax=abs(psi.real).max()), cmap='viridis')
        plt.colorbar()
        plt.xlabel("Electron 2 a.u.")
        plt.ylabel("Electron 1 a.u.")
        plt.title("Wave Function - Energy "+str(energy[i]))
        fig.savefig("figs/"+target_name+"_log_state_"+str(i).zfill(3)+".jpg")
