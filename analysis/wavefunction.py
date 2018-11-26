import numpy as np
import h5py

# read data
f = h5py.File("TDSE.h5", "r")
psi_value = f["Wavefunction"]["psi"]
psi_time = f["Wavefunction"]["time"][:]
shape = f["Wavefunction"]["num_x"][:]
num_electrons = f["Parameters"]["num_electrons"][0]
num_dims = len(shape)
x = f["Wavefunction"]["x_value_0"][:]

if len(shape) > 1:
    y = f["Wavefunction"]["x_value_1"][:]

gobbler = f["Parameters"]["gobbler"][0]
upper_idx = (shape * gobbler - 1).astype(int)
lower_idx = shape - upper_idx
# calculate location for time to be printed
if len(shape) > 1:
    time_x = np.min(y[lower_idx[1]:upper_idx[1]]) * 0.95
else:
    time_x = np.min(x[lower_idx[0]:upper_idx[0]]) * 0.95
time_y = np.max(x[lower_idx[0]:upper_idx[0]]) * 0.9

max_val = 0
# calculate color bounds
for i, psi in enumerate(psi_value):
    if i > 0 and i < 2:  # the zeroth wave function is the guess and not relevant
        psi = psi[:, 0] + 1j * psi[:, 1]
        max_val_tmp = np.max(np.absolute(psi))
        if (max_val_tmp > max_val):
            max_val = max_val_tmp

if num_electrons == 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    import pylab as plb
    font = {'size': 22}
    matplotlib.rc('font', **font)

    shape_2_elecron = np.concatenate([shape, shape])
    num_axis = num_dims * num_electrons
    for i, psi in enumerate(psi_value):
        if i > 0:
            psi_save = np.array(psi)
            print "plotting", i
            # set up initial figure with color bar
            psi = psi[:, 0] + 1j * psi[:, 1]
            psi = (psi * np.conjugate(psi)).real
            psi.shape = tuple(shape_2_elecron)

            # plot projections onto each axis
            for electron_idx in np.arange(num_electrons):
                for dim_idx in np.arange(num_dims):
                    plot_name = "figs/Wave_projection_on_dim_" + str(
                        dim_idx) + "_electron_" + str(
                            electron_idx) + "_state_" + str(i).zfill(8) + ".png"
                    current_dim = electron_idx * num_dims + dim_idx
                    axis_sum_list = []
                    for this_dim in range(num_axis):
                        if this_dim != current_dim:
                            axis_sum_list.append(this_dim)
                    axis_sum_list = tuple(axis_sum_list)
                    fig = plt.figure()
                    data = np.sum(psi, axis=axis_sum_list)
                    data = (data * np.conjugate(data)).real
                    axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
                    plt.plot(axis, data / data.max())
                    if dim_idx == 0:
                        plt.axvline(-0.7, color='k')
                        plt.axvline(0.7, color='k')
                    else:
                        plt.axvline(0.0, color='k')
                    plt.xlabel("Dim:" + str(dim_idx) + "   Electron:" +
                               str(electron_idx))
                    plt.tight_layout()
                    plt.savefig(plot_name)
                    plt.close(fig)

                    # plot on log scale without recalculating
                    plot_name = "figs/Wave_log_projection_on_dim_" + str(
                        dim_idx) + "_electron_" + str(
                            electron_idx) + "_state_" + str(i).zfill(8) + ".png"
                    fig = plt.figure()
                    plt.semilogy(axis, data / data.max())
                    if dim_idx == 0:
                        plt.axvline(-0.7, color='k')
                        plt.axvline(0.7, color='k')
                    else:
                        plt.axvline(0.0, color='k')
                    plt.xlabel("Dim:" + str(dim_idx) + "   Electron:" +
                               str(electron_idx))
                    plt.tight_layout()
                    plt.savefig(plot_name)
                    plt.close(fig)

            if num_dims == 2:
                dim_0 = f["Wavefunction"]["x_value_0"][:]
                dim_1 = f["Wavefunction"]["x_value_1"][:]
                # plot electron 0
                plot_name = "figs/Wave_2d_electron_0_state_" + str(i).zfill(
                    8) + ".png"
                axis_sum_list = tuple([2, 3])
                fig = plt.figure()
                data = np.sum(psi, axis=axis_sum_list)
                axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]])
                plt.ylabel("Dim: 0")
                plt.xlabel("Dim: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                # log scale electron 0 without recalculating
                plot_name = "figs/Wave_log_2d_electron_0_state_" + str(
                    i).zfill(8) + ".png"
                fig = plt.figure()
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]],
                    norm=LogNorm(vmin=1e-10))
                plt.ylabel("Dim: 0")
                plt.xlabel("Dim: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                # plot electron 1
                plot_name = "figs/Wave_2d_electron_1_state_" + str(i).zfill(
                    8) + ".png"
                axis_sum_list = tuple([0, 1])
                fig = plt.figure()
                data = np.sum(psi, axis=axis_sum_list)
                axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]])
                plt.ylabel("Dim: 0")
                plt.xlabel("Dim: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                # log scale electron 1 without recalculating
                plot_name = "figs/Wave_log_2d_electron_1_state_" + str(
                    i).zfill(8) + ".png"
                fig = plt.figure()
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_1[-1], dim_1[1], dim_0[-1], dim_0[1]],
                    norm=LogNorm(vmin=1e-10))
                plt.ylabel("Dim: 0")
                plt.xlabel("Dim: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                # plot dim 0 for both electrons
                plot_name = "figs/Wave_2d_dim_0_state_" + str(i).zfill(
                    8) + ".png"
                axis_sum_list = tuple([1, 3])
                fig = plt.figure()
                data = np.sum(psi, axis=axis_sum_list)
                axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]])
                plt.ylabel("Dim: 0    Electron: 0")
                plt.xlabel("Dim: 0    Electron: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                # log scale dim 0 for both electrons without recalculating
                plot_name = "figs/Wave_log_2d_dim_0_state_" + str(i).zfill(
                    8) + ".png"
                fig = plt.figure()
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]],
                    norm=LogNorm(vmin=1e-10))
                plt.ylabel("Dim: 0    Electron: 0")
                plt.xlabel("Dim: 0    Electron: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                # plot dim 1 for both electrons
                plot_name = "figs/Wave_2d_dim_1_state_" + str(i).zfill(
                    8) + ".png"
                axis_sum_list = tuple([0, 2])
                fig = plt.figure()
                data = np.sum(psi, axis=axis_sum_list)
                axis = f["Wavefunction"]["x_value_" + str(dim_idx)][:]
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]])
                plt.ylabel("Dim: 1    Electron: 0")
                plt.xlabel("Dim: 1    Electron: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

                # log scale dim 1 for both electrons without recalculating
                plot_name = "figs/Wave_log_2d_dim_1_state_" + str(i).zfill(
                    8) + ".png"
                fig = plt.figure()
                plt.imshow(
                    data / data.max(),
                    cmap='viridis',
                    extent=[dim_0[-1], dim_0[1], dim_0[-1], dim_0[1]],
                    norm=LogNorm(vmin=1e-10))
                plt.ylabel("Dim: 1    Electron: 0")
                plt.xlabel("Dim: 1    Electron: 1")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plot_name)
                plt.close(fig)

else:
    if len(shape) == 3:
        from mayavi import mlab
        # mlab.options.offscreen = True
        z = f["Wavefunction"]["x_value_2"][:]
        mlab.figure(
            bgcolor=(1.0, 1.0, 1.0),
            fgcolor=(0.0, 0.0, 0.0),
            size=(1000, 1000))
        for i, psi in enumerate(psi_value):
            if i > 0:  # the zeroth wave function is the guess and not relevant
                print "plotting", i
                psi = psi[:, 0] + 1j * psi[:, 1]
                psi.shape = tuple(shape)
                psi = psi[lower_idx[0]:upper_idx[0], lower_idx[1]:upper_idx[1],
                          lower_idx[2]:upper_idx[2]]
                psi = np.log10(np.abs(psi))
                # mlab.pipeline.iso_surface(mlab.pipeline.scalar_field(psi),
                #     vmin=-10,
                #     vmax=0.0,
                #     opacity=0.3,
                #     colormap="viridis",
                #     contours=11)
                # mlab.colorbar(nb_labels=11,orientation="vertical")
                # mlab.orientation_axes()
                # mlab.view(azimuth=0.0,distance='auto',elevation=90.0)
                # mlab.savefig("figs/Wave_iso_x_"+str(i).zfill(8)+".png")
                # mlab.view(azimuth=90.0,distance='auto',elevation=90.0)
                # mlab.savefig("figs/Wave_iso_y_"+str(i).zfill(8)+".png")
                # mlab.view(azimuth=0.0,distance='auto',elevation=0.0)
                # mlab.savefig("figs/Wave_iso_z_"+str(i).zfill(8)+".png")
                # mlab.view(azimuth=45.0,distance='auto',elevation=45.0)
                # mlab.savefig("figs/Wave_iso_"+str(i).zfill(8)+".png")

                mlab.clf()
                mlab.pipeline.iso_surface(
                    mlab.pipeline.scalar_field(psi),
                    vmin=-10.0,
                    vmax=0.0,
                    opacity=0.3,
                    colormap="viridis",
                    contours=[1.0])
                mlab.pipeline.volume(
                    mlab.pipeline.scalar_field(psi), vmin=-10.0, vmax=0.0)
                mlab.colorbar(nb_labels=10, orientation="vertical")
                mlab.orientation_axes()
                mlab.view(azimuth=0.0, distance='auto', elevation=90.0)
                mlab.savefig("figs/Wave_density_10_cross_x_" + str(i).zfill(8)
                             + ".png")
                mlab.view(azimuth=90.0, distance='auto', elevation=90.0)
                mlab.savefig("figs/Wave_density_10_cross_y_" + str(i).zfill(8)
                             + ".png")
                mlab.view(azimuth=0.0, distance='auto', elevation=0.0)
                mlab.savefig("figs/Wave_density_10_cross_z_" + str(i).zfill(8)
                             + ".png")
                mlab.view(azimuth=45.0, distance='auto', elevation=45.0)
                mlab.savefig("figs/Wave_density_10_cross_" + str(i).zfill(8) +
                             ".png")
                mlab.clf()

    elif len(shape) == 2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import pylab as plb
        # shape into a 3d array with time as the first axis
        fig = plt.figure()
        font = {'size': 18}
        matplotlib.rc('font', **font)
        for i, psi in enumerate(psi_value):
            if i > 0:  # the zeroth wave function is the guess and not relevant
                print "plotting", i
                psi_save = np.array(psi)
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
                x_max_idx = -1
                y_min_idx = 0
                y_max_idx = -1
                psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
                data = None
                if f["Parameters"]["coordinate_system_idx"][0] == 1:
                    psi = np.absolute(
                        np.multiply(
                            np.conjugate(psi),
                            np.multiply(x[x_min_idx:x_max_idx],
                                        psi.transpose()).transpose()))
                    data = plt.imshow(
                        psi,
                        cmap='viridis',
                        origin='lower',
                        extent=[
                            y[y_min_idx], y[y_max_idx], x[x_min_idx],
                            x[x_max_idx]
                        ],
                        norm=LogNorm(vmin=1e-15, vmax=max_val))
                else:
                    data = plt.imshow(
                        np.absolute(psi),
                        cmap='viridis',
                        origin='lower',
                        extent=[
                            y[y_min_idx], y[y_max_idx], x[x_min_idx],
                            x[x_max_idx]
                        ],
                        norm=LogNorm(vmin=1e-15, vmax=max_val))
                plt.text(
                    time_x,
                    time_y,
                    "Time: " + str(psi_time[i]) + " a.u.",
                    color='white')
                # color bar doesn't change during the video so only set it here
                if f["Parameters"]["coordinate_system_idx"][0] == 1:
                    plt.xlabel("z-axis (a.u.)")
                    plt.ylabel("$\\rho$-axis  (a.u.)")
                else:
                    plt.xlabel("X-axis (a.u.)")
                    plt.ylabel("Y-axis  (a.u.)")
                # plt.axis('off')
                plt.colorbar()
                fig.savefig("figs/Wave_" + str(i).zfill(8) + ".png")
                plt.clf()
                fig.savefig("figs/Wave_" + str(i).zfill(8) + ".png")
                plb.xlim([-100, 100])
                plb.ylim([-100, 100])
                plt.tight_layout()
                fig.savefig("figs/Wave_small_" + str(i).zfill(8) + ".png")
                plt.clf()

                plt.semilogy(x[0:-1], np.abs(psi[:, psi.shape[1] / 2]))
                plt.xlabel("X-axis  (a.u.)")
                plt.xlim([-100, 100])
                plt.ylim([1e-10, 1])
                fig.savefig("figs/Wave_x_cut_" + str(i).zfill(8) + ".png")
                plt.clf()

                plt.semilogy(y[0:-1], np.abs(psi[psi.shape[0] / 2, :]))
                plt.xlabel("Y-axis  (a.u.)")
                plt.xlim([-100, 100])
                plt.ylim([1e-10, 1])
                fig.savefig("figs/Wave_y_cut_" + str(i).zfill(8) + ".png")
                plt.clf()

                psi = psi_save
                psi = psi[:, 0] + 1j * psi[:, 1]
                psi.shape = tuple(shape)
                if f["Parameters"]["coordinate_system_idx"][0] == 1:
                    x_min_idx = 0
                else:
                    x_min_idx = lower_idx[0]
                x_max_idx = upper_idx[0]
                y_min_idx = lower_idx[1]
                y_max_idx = upper_idx[1]
                x_max_idx = -1
                y_min_idx = 0
                y_max_idx = -1
                psi = psi[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
                data = None
                if f["Parameters"]["coordinate_system_idx"][0] == 1:
                    psi = np.angle(psi)
                    data = plt.imshow(
                        psi,
                        cmap='hsv',
                        origin='lower',
                        extent=[
                            y[y_min_idx], y[y_max_idx], x[x_min_idx],
                            x[x_max_idx]
                        ])
                else:
                    data = plt.imshow(
                        np.angle(psi),
                        cmap='hsv',
                        origin='lower',
                        extent=[
                            y[y_min_idx], y[y_max_idx], x[x_min_idx],
                            x[x_max_idx]
                        ])
                plt.text(
                    time_x,
                    time_y,
                    "Time: " + str(psi_time[i]) + " a.u.",
                    color='white')
                # color bar doesn't change during the video so only set it here
                if f["Parameters"]["coordinate_system_idx"][0] == 1:
                    plt.xlabel("z-axis (a.u.)")
                    plt.ylabel("$\\rho$-axis  (a.u.)")
                else:
                    plt.xlabel("X-axis (a.u.)")
                    plt.ylabel("Y-axis  (a.u.)")
                # plt.axis('off')
                plt.colorbar()
                fig.savefig("figs/Wave_phase_" + str(i).zfill(8) + ".png")
                plt.clf()

    elif len(shape) == 1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pylab as plb
        from matplotlib.colors import LogNorm

        font = {'size': 18}
        matplotlib.rc('font', **font)
        for i, psi in enumerate(psi_value):
            if i > 0:  # the zeroth wave function is the guess and not relevant
                print "plotting", i
                # set up initial figure with color bar
                psi = psi[:, 0] + 1j * psi[:, 1]
                psi.shape = tuple(shape)

                data = None

                data = plt.semilogy(x, np.abs(psi))
                plt.text(
                    time_x,
                    time_y,
                    "Time: " + str(psi_time[i]) + " a.u.",
                    color='black')
                plb.xlabel('x (a.u.)')
                plb.ylabel('psi (arb. u)')
                plt.ylim(ymin=1e-15)
                plt.savefig("figs/Wave_" + str(i).zfill(8) + ".png")
                plt.clf()
