# ###################################################################
# Script fig_streamlineas
#
# Description
# Visualise flow with streamlines
#
# ###################################################################
# Author: hw
# created: 27. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np

from uncert_ident.utilities import *
from uncert_ident.visualisation.plotter import *
from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.methods.geometry import get_boundaries


#####################################################################
### Plot
#####################################################################
# Load data
# case_name = 'PH-Breuer-5600'
# case_name = 'PH-Xiao-15'
# case_name = 'CBFS-Bentaleb'
# case_name = 'NACA4412-Vinuesa-bottom-1'
# case_name = 'NACA4412-Vinuesa-top-1'
# case_name = 'NACA0012-Tanarro-top-4'
case_name = 'TBL-APG-Bobke-m13'
case_names = [case_name]

# case_names = ['PH-Breuer-10595', 'PH-Xiao-15', 'CBFS-Bentaleb', 'NACA4412-Vinuesa-top-4', 'TBL-APG-Bobke-b1']


for case_name in case_names:
    case = flowCase(case_name)
    data = case.flow_dict

    nx, ny = case.nx, case.ny
    X, Y = data['x'].reshape(nx, ny), data['y'].reshape(nx, ny)
    U, V = data['um'].reshape(nx, ny), data['vm'].reshape(nx, ny)
    spd = np.sqrt(U**2 + V**2)



    # Boundaries, seeding points, mask and axis label
    if "PH" in case_name:
        boundaries = get_boundaries(case)
        boundaries = list(boundaries)
        # boundaries.append([np.array([0.05, 8.9]), np.array([0.05, 0.05])])
        # boundaries.append([np.array([0.1, 0.1]), np.array([0.05, 2.985])])
        # boundaries.append([np.array([8.9, 8.9]), np.array([0.05, 2.985])])
        # boundaries.append([np.array([0.05, 8.9]), np.array([2.2, 2.2])])
        boundaries.append([np.array([0.1, 0.1]), np.array([0.05, 2.985])])  # PH Xiao 15
        boundaries.append([np.array([0.1, 10.8]), np.array([0.05, 0.05])])  # PH Xiao 15
        boundaries.append([np.array([10.8, 10.8]), np.array([0.05, 2.985])])  # PH Xiao 15
        boundaries.append([np.array([0.1, 10.8]), np.array([2.2, 2.2])])  # PH Xiao 15


        seeding = np.array([np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 4.0, 4.5]),
                            np.array([1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 2.9, 0.2, 0.4, 0.1, 0.1, 0.1])]).T
        mask = spd == 0
        xlabel = "$x/H$"#"$\dfrac{x}{H}$"
        ylabel = "$y/H$"#"$\dfrac{y}{H}$"
        xlim = [min(X.flatten()) + 0.05, max(X.flatten()) - 0.05]
        # ylim = [min(Y.flatten()), max(Y.flatten())]
        ylim = [min(Y.flatten()), 2.2]
        title = None

    elif "NACA" in case_name:
        boundaries = get_boundaries(case)
        boundaries = list(boundaries)
        boundaries.append([np.array([min(X.flatten()) + 0.05, max(X.flatten()) - 0.00]), np.array([min(Y.flatten()) + 0.01, min(Y.flatten()) + 0.01])])
        boundaries.append([np.array([min(X.flatten()) + 0.05, max(X.flatten()) - 0.00]), np.array([max(Y.flatten()) - 0.01, max(Y.flatten()) - 0.01])])
        boundaries.append([np.array([min(X.flatten()) + 0.05, min(X.flatten()) + 0.05]), np.array([min(Y.flatten()) - 0.01, max(Y.flatten()) - 0.01])])
        boundaries.append([np.array([max(X.flatten()) - 0.01, max(X.flatten()) - 0.01]), np.array([min(Y.flatten()) - 0.01, max(Y.flatten()) - 0.01])])

        if "top" in case_name:
            seeding = np.array([np.array([0.200, 0.200, 0.200, 0.20, 0.200, 0.2, 0.200, 0.20, 0.200, 0.2]),
                                np.array([0.075, 0.075, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3])]).T
        else:
            seeding = np.array([np.array([0.250, 0.250, 0.25, 0.250, 0.25, 0.250, 0.25, 0.250]),
                                -np.array([0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225])]).T
        # seeding = np.array([np.ones(10) * min(X.flatten()),
        #                     np.linspace(min(Y.flatten()), max(Y.flatten()), 10)]).T
        mask = spd == 0
        xlabel = "$x/c$"#"$\dfrac{x}{c}$"
        ylabel = "$y/c$"#"$\dfrac{y}{c}$"
        xlim = [min(X.flatten()) + 0.05, max(X.flatten()) - 0.00]
        ylim = [min(Y.flatten()), max(Y.flatten())]
        title = None

    elif "TBL" in case_name:
        boundaries = get_boundaries(case)
        boundaries = list(boundaries)
        boundaries.append([np.array([min(X.flatten()) + 0.10, max(X.flatten()) - 1.00]), np.array([0 + 0.01, 0 + 0.01])])
        boundaries.append([np.array([min(X.flatten()) + 0.10, max(X.flatten()) - 1.00]), np.array([8 - 0.01, 8 - 0.01])])
        boundaries.append([np.array([min(X.flatten()) + 0.10, min(X.flatten()) + 0.10]), np.array([0 - 0.01, 8 - 0.01])])
        boundaries.append([np.array([max(X.flatten()) - 10.00, max(X.flatten()) - 10.00]), np.array([0 - 0.01, 8 - 0.01])])

        # seeding = np.array([np.ones(10) * min(X.flatten()),
        #                     np.linspace(min(Y.flatten()) + 5, max(Y.flatten()), 10)]).T
        seeding = np.array([np.ones(10) * min(X.flatten()),
                            np.linspace(0, 8, 10)]).T
        mask = spd == 0
        xlabel = "$x$"
        ylabel = "$y$"
        xlim = [min(X.flatten()) + 0.05, max(X.flatten()) - 0.05]
        ylim = [0, 8]
        title = None

    elif "CBFS" in case_name:
        boundaries = get_boundaries(case)
        boundaries = list(boundaries)
        boundaries.append([np.array([-1 + 0.05, 6 - 0.05]), np.array([0 + 0.05, 0 + 0.05])])
        boundaries.append([np.array([-1 + 0.05, -1 + 0.05]), np.array([0 + 0.05, 1.4 - 0.05])])
        boundaries.append([np.array([-1 + 0.05, 6 - 0.05]), np.array([1.4 + 0.00, 1.4 + 0.00])])
        boundaries.append([np.array([6 - 0.05, 6 - 0.05]), np.array([0 + 0.05, 1.4 - 0.05])])

        seeding = np.array([np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3.00, 3.0, 3.00, 3.00]),
                            np.array([1.1, 1.2, 1.4, 1.6, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.05, 0.15, 0.25, 0.58])]).T
        mask = spd == 0
        xlabel = "$x/H$"#"$\dfrac{x}{H}$"
        ylabel = "$y/H$"#"$\dfrac{y}{H}$"
        xlim = [-1, 6]
        ylim = [0, 1.4]
        title = None

    else:
        boundaries = get_boundaries(case)
        seeding = np.array([np.ones(10) * min(X.flatten()),
                            np.linspace(min(Y.flatten()), max(Y.flatten()), 10)]).T
        mask = spd == 0
        xlabel = None
        ylabel = None
        xlim = [min(X.flatten()) + 0.05, max(X.flatten()) - 0.05]
        ylim = [min(Y.flatten()), max(Y.flatten())]
        title = None




    # Plot
    fig, ax = empty_plot(figwidth=latex_textwidth*0.3)


    # Boundaries
    # boundaries = list(boundaries)
    # boundaries.append([np.array([0, 9]), np.array([0, 0])])
    # boundaries.append([np.array([0, 0]), np.array([0, 3.035])])
    # boundaries.append([np.array([9, 9]), np.array([0, 3.035])])
    # boundaries.append([[0, 0], [0, 3.035]])
    # boundaries.append([[9, 9], [0, 3.035]])
    for boundary in boundaries:
        lining(*boundary, linestyle='-', color=cblack,
               xlim=xlim,
               ylim=ylim,
               append_to_fig_ax=(fig, ax))


    # Background/mean flow
    spd = np.ma.array(spd, mask=mask)
    contouring(X, Y, spd,
               colorbar=False,
               cbarlabel=r"$\lvert \overline{\mathbf{u}}\rvert$",
               levels=100,
               append_to_fig_ax=(fig, ax))



    # Streamlines
    streams(ax, X, Y, U, V, start_points=seeding)



    # Set limits
    set_labels_title(ax, xlabel, ylabel, title)
    # ax.spines['bottom'].set_color(cblack)
    # ax.spines['top'].set_color(cblack)
    # ax.spines['right'].set_color(cblack)
    # ax.spines['left'].set_color(cblack)
    ax.set_axis_off()
    # ax.set_frame_on(1)


    save("../figures/streamlineas_" + case_name + ".jpg")
    # save("../figures/streamlineas_" + case_name + "_detailed.jpg")



show()
