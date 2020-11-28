# ###################################################################
# test plot_barycentric_map
#
# Description
# Plot the barycentric map with an rgb colormap for the states of
# turbulence, with examples.
#
# ###################################################################
# Author: hw
# created: 15. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import numpy as np
from scipy.spatial import Delaunay
import scipy.interpolate as interp

from uncert_ident.visualisation.plotter import *
from uncert_ident.data_handling.flowcase import *
from uncert_ident.methods.geometry import geometry_periodic_hills_lower as geo_ph_low



def colors_to_cmap(colors):
    """
    Yields a matplotlib colormap object
    that,  reproduces the colors in the given array when passed a
    list of N evenly spaced numbers between 0 and 1 (inclusive), where N is the
    first dimension of ``colors``.

    Args:
      colors (ndarray (N,[3|4])): RGBa_array
    Return:
      cmap (matplotlib colormap object): Colormap reproducing input colors,
                                         cmap[i/(N-1)] == colors[i].

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.linspace(0,1,range(len(colors)))
    """
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1), dtype=np.float))/(len(colors) - 1)
    return matplotlib.colors.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] +
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] +
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors))




# Load data
# case_name = 'PH-Breuer-5600'
case_name = 'PH-Breuer-10595'
# case_name = 'PH-Xiao-15'
# case_name1 = "PH-Breuer-5600"
# case_name = 'CBFS-Bentaleb'
# case_name = 'NACA4412-Vinuesa-top-4'
# case_name1 = 'NACA4412-Vinuesa-bottom-4'
# case_name = 'TBL-APG-Bobke-m13'
case_names = [case_name]
# case_names = [case_name, case_name1]

# case_names = ['PH-Breuer-5600', 'PH-Xiao-15', 'CBFS-Bentaleb', 'NACA4412-Vinuesa-top-4', 'TBL-APG-Bobke-m13']


for case_name in case_names:
    case = flowCase(case_name)
    case.get_labels()
    true_fig, _ = case.show_label('anisotropic')
    save("../figures/true_anisotropy_" + case_name + ".pdf")
    plt.close(true_fig)

    data = case.flow_dict
    nx, ny = case.nx, case.ny
    X, Y = data['x'].reshape(nx, ny), data['y'].reshape(nx, ny)
    U, V = data['um'].reshape(nx, ny), data['vm'].reshape(nx, ny)
    vel = np.sqrt(U**2 + V**2).flatten()

    eig1 = data['bij_eig1']
    eig2 = data['bij_eig2']
    eig3 = data['bij_eig3']
    # kbools = data['k'] < 0.001  # Remove low k points, CBFS
    # kbools = data['k'] < 0.0005  # Remove low k points, NACA

    # Define basis points for vertices as (x, y)
    vertex_1c = np.array([1, 0])
    vertex_2c = np.array([0, 0])
    vertex_3c = np.array([0.5, 3**0.5/2])


    # Compute coordinates using the eigenvalues
    c_1c = eig1 - eig2
    c_2c = 2*(eig2 - eig3)
    c_3c = 3*eig3 + 1

    colors = np.vstack([c_1c, c_2c, c_3c]).T
    colors[colors < 0] = 0
    colors[colors > 1] = 1
    # colors[kbools, :] = np.ones(3)  # White
    # colors[kbools, :] = np.array([0, 0, 1])  # Isotropic

    colors = (colors.T / np.max(colors, axis=1)).T  # Normalise


    # Construct triangles for the 2D space
    triang = tri.Triangulation(X.flatten(), Y.flatten())
    cmap_space = colors_to_cmap(colors)
    # triang = tri.Triangulation(X[:, :20].flatten(), Y[:, :20].flatten())
    # cmap_space = colors_to_cmap(colors.reshape(nx, ny, 3)[:, :20])


    # Mask
    # bools = y[triang.triangles].min(axis=1) < geo_ph_low(x[triang.triangles].mean(axis=1))
    bools = vel[triang.triangles] == 0
    bools = [any(b) for b in bools]
    triang.set_mask(bools)




    # Boundaries, seeding points, mask and axis label
    if "PH" in case_name:
        boundaries = get_boundaries(case)
        seeding = np.array([np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 4.0, 4.5]),
                            np.array([1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 2.9, 0.2, 0.4, 0.1, 0.1, 0.1])]).T
        xlabel = "$\dfrac{x}{H}$"
        ylabel = "$\dfrac{y}{H}$"
        title = None
        xlim = [min(X.flatten()) + 0.05, max(X.flatten()) - 0.05]
        ylim = [min(Y.flatten()), max(Y.flatten())]

    elif "NACA" in case_name:
        boundaries = get_boundaries(case)
        if "top" in case_name:
            seeding = np.array([np.array([0.200, 0.200, 0.200, 0.20, 0.200, 0.2, 0.200, 0.20, 0.200, 0.2]),
                                np.array([0.075, 0.075, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3])]).T
        else:
            seeding = np.array([np.array([0.250, 0.250, 0.25, 0.250, 0.25, 0.250, 0.25, 0.250]),
                                -np.array([0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225])]).T
        xlabel = "$\dfrac{x}{c}$"
        ylabel = "$\dfrac{y}{c}$"
        title = None
        xlim = [0.2, 0.97]
        ylim = [-0.03, 0.11]

    elif "TBL" in case_name:
        boundaries = get_boundaries(case)
        seeding = np.array([np.ones(10) * min(X.flatten()),
                            np.linspace(0, 8, 10)]).T
        xlabel = "$x$"
        ylabel = "$y$"
        xlim = [min(X.flatten()) + 0.05, max(X.flatten()) - 0.05]
        ylim = [0, 8]
        title = None

    elif "CBFS" in case_name:
        boundaries = get_boundaries(case)
        seeding = np.array([np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3.00, 3.0, 3.00, 3.00]),
                            np.array([1.1, 1.2, 1.4, 1.6, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.05, 0.15, 0.25, 0.58])]).T
        xlabel = "$\dfrac{x}{H}$"
        ylabel = "$\dfrac{y}{H}$"
        xlim = [-1, 6]
        ylim = [0, 1.4]
        title = None

    else:
        boundaries = get_boundaries(case)
        xlabel = None
        ylabel = None
        title = None
        xlim = [min(X.flatten()) + 0.05, max(X.flatten()) - 0.05]
        ylim = [min(Y.flatten()), max(Y.flatten())]



    # Plot
    if "bottom" not in case_name:
        fig, ax = empty_plot(figwidth=latex_textwidth)

    # Boundaries
    for boundary in boundaries:
        lining(*boundary, linestyle='-', color=cblack,
               # xlim=xlim,
               # ylim=ylim,
               append_to_fig_ax=(fig, ax))



    # Tris with RGB
    ax.tripcolor(triang,
                 np.linspace(0, 1, triang.x.size),
                 edgecolors='none',
                 cmap=cmap_space,
                 shading='gouraud')

    streams(ax, X, Y, U, V, start_points=seeding, color=cblack)

    # Plot limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    set_labels_title(ax, xlabel, ylabel, title)
    ax.set_axis_off()

    save("../figures/title_" + case_name + ".jpg")


show()
close_all()


