# ###################################################################
# Script fig_logistic_function
#
# Description
# Visualise the logistic function.
#
# ###################################################################
# Author: hw
# created: 31. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.data_handling.data_import import path_to_raw_data
from uncert_ident.visualisation.plotter import *
from PyFOAM import *



def read_rans_ph_breuer():
    path = "PH-Breuer/rans/case_baseline"
    nx = 120
    ny = 130

    # Cell centred coordinates
    X = getRANSVector("../" + path_to_raw_data + path, "10000", 'C')
    X = X.reshape(3, ny, nx)
    x, y = X[0].T, X[1].T

    # Velocity vector
    U = getRANSVector("../" + path_to_raw_data + path, 10000, 'U')
    U = U.reshape(3, ny, nx)
    um, vm, wm = U[0].T, U[1].T, U[2].T

    # Turbulence data
    k = getRANSScalar("../" + path_to_raw_data + path, 10000, 'k')
    nut = getRANSScalar("../" + path_to_raw_data + path, 10000, 'nut')
    omega = getRANSScalar("../" + path_to_raw_data + path, 10000, 'omega')

    # Save to dict
    rans_data = {
        'x': x,
        'y': y,
        'nx': nx,
        'ny': ny,
        'um': um,
        'vm': vm,
        'k': k,
        'nut': nut,
        'oemga': omega
    }

    return rans_data



#####################################################################
### Plot
#####################################################################
# Get RANS data
# path = "PH-Breuer/rans/case_baseline"
# nx = 120
# ny = 130
#
# X = getRANSVector("../" + path_to_raw_data + path, "10000", 'C')
# X = X.reshape(3, ny, nx)
# x, y = X[0].T, X[1].T
# # scattering(x, y, np.ones_like(x), alpha=0.8, scale=50)
# # show()
#
# k = getRANSScalar("../" + path_to_raw_data + path, 10000, 'k')
# nut = getRANSScalar("../" + path_to_raw_data + path, 10000, 'nut')
# omega = getRANSScalar("../" + path_to_raw_data + path, 10000, 'omega')
# U = getRANSVector("../" + path_to_raw_data + path, 10000, 'U')
# U = U.reshape(3, ny, nx)
# um, vm, wm = U[0].T, U[1].T, U[2].T
# vel = np.sqrt(um**2 + vm**2)


rans_dict = read_rans_ph_breuer()


# Get DNS data
case_name = "PH-Breuer-5600"
case = flowCase(case_name)
case.get_labels()
dns_dict = case.flow_dict



# fig, ax = contouring(x, y, um)

boundaries, xlabel, ylabel, title, xlim, ylim, mask = get_geometry_plot_data(case)

seeding = np.array([np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 4.0, 4.5]),
                    np.array([1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 2.9, 0.2, 0.4, 0.1, 0.1, 0.1])]).T
ylim[1] = 2.2


# Figure
# fig, ax = empty_plot(figwidth=beamer_textwidth*0.9)
fig, ax = case.show_label(label_key="anisotropic", only_positive=True)

# Boundaries, label, title, limits
for boundary in boundaries:
    lining(*boundary,
           linestyle='-',
           lw=3,
           color=cdarkgrey,
           append_to_fig_ax=(fig, ax))
set_labels_title(ax, xlabel, ylabel, title)
set_limits(ax, xlim=xlim, ylim=ylim)


# Streamlines
streams(ax, dns_dict['x'], dns_dict['y'], dns_dict['um'], dns_dict['vm'], start_points=seeding, color=cblack)
streams(ax, rans_dict['x'], rans_dict['y'], rans_dict['um'], rans_dict['vm'], start_points=seeding, color=cred)


legend_elements = [
    plt.Line2D([0], [0], linestyle='-', color=cblack, label=r"\textsf{DNS  - Streamlines}"),
    plt.Line2D([0], [0], linestyle='-', color=cred, label=r"\textsf{RANS - Streamlines}"),
    plt.Line2D([0], [0], linestyle='', lw=0, marker='o', color=cconfusionred, label=r"\textsf{Identification}")
]
ax.legend(handles=legend_elements, loc="upper right")
ax.set_axis_off()



# Profiles
# Interpolate DNS to RANS
# Extract profiles at x = 3, 6


# save("../figures/streamlines_ph_dns.jpg")
# save("../figures/streamlines_ph_rans.jpg")
# save("../figures/streamlines_ph_dns_rans.jpg")
save("../figures/streamlines_ph_dns_rans_label.jpg")
show()



print("nothing")
