import numpy as np
import h5py
target_name = "He-SAE"
# read data
target = h5py.File(target_name + ".h5", "r")
f = h5py.File("TDSE.h5", "r")
psi_value = target["psi"]
# psi_value = f["Wavefunction"]["psi"]
energy = target["Energy"]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]
print shape
x = f["Wavefunction"]["x_value_0"][:]
y = f["Wavefunction"]["x_value_1"][:]

if len(shape) == 3:
    from mayavi import mlab
    z = f["Wavefunction"]["x_value_2"][:]
    mlab.figure(
        bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.0, 0.0, 0.0), size=(1000, 1000))
    for i, psi in enumerate(psi_value):
        psi = psi[:, 0] + 1j * psi[:, 1]
        psi.shape = tuple(shape)
        # print np.log10(np.abs(psi))
        mlab.pipeline.iso_surface(
            mlab.pipeline.scalar_field(np.log10(np.abs(psi))),
            vmin=-05,
            opacity=0.3,
            colormap="viridis",
            contours=[-3, -2, -1])
        #mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(np.log10(np.abs(psi))),vmin=-15,
        #    plane_orientation='z_axes',
        #    transparent=True,colormap="viridis",
        #    slice_index=shape[2]/2)
        mlab.colorbar(nb_labels=4, orientation="vertical")
        # mlab.show()
        mlab.savefig("figs/target_" + str(i).zfill(8) + ".png")
        mlab.clf()

elif len(shape) == 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    fig = plt.figure()
    for i, psi in enumerate(psi_value):
        if i > 0:  # the zeroth wave function is the guess and not relevant
            print "plotting", i
            # set up initial figure with color bar
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi.shape = tuple(shape)
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                x_min_idx = 0
            else:
                x_min_idx = lower_idx[0]
            x_max_idx = upper_idx[0]
            y_min_idx = lower_idx[1]
            y_max_idx = upper_idx[1]
            psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
            if f["Parameters"]["coordinate_system_idx"][0] == 1:
                psi = np.absolute(
                    np.multiply(
                        np.conjugate(psi),
                        np.multiply(x[x_min_idx:x_max_idx], psi.transpose())
                        .transpose()))
                plt.imshow(
                    psi,
                    cmap='viridis',
                    origin='lower',
                    extent=[
                        y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                    ],
                    norm=LogNorm(vmin=1e-12, vmax=max_val))
            else:
                plt.imshow(
                    np.absolute(psi),
                    cmap='viridis',
                    origin='lower',
                    extent=[
                        y[y_min_idx], y[y_max_idx], x[x_min_idx], x[x_max_idx]
                    ],
                    norm=LogNorm(vmin=1e-12, vmax=max_val))
            # color bar doesn't change during the video so only set it here
            plt.colorbar()
            plt.xlabel("X-axis (a.u.)")
            plt.ylabel("Y-axis  (a.u.)")
            plt.title("Wave Function - Energy " + str(energy[i]))
            fig.savefig("figs/" + target_name + "_log_state_" + str(i).zfill(3)
                        + ".jpg")
