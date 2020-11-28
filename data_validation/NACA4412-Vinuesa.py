# ###################################################################
# script Vinuesa_NACA4412_LES
#
# Description
# Create plots to validate imported data against Vinuesa's paper
#
# ###################################################################
# Author: hw
# created: 29. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np

from uncert_ident.data_handling.flowcase import flowCase, compare_integral_quantity, compare_profiles

import uncert_ident.visualisation.plotter as plot



#####################################################################
### Load data
#####################################################################
# Individual case's name
case_names = list()
case_names.append('NACA4412-Vinuesa-top-1')
case_names.append('NACA4412-Vinuesa-top-2')
case_names.append('NACA4412-Vinuesa-top-4')
case_names.append('NACA4412-Vinuesa-top-10')


# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 2, (a) inner-scaled mean velocity profile (b) selected tauij components
compare_profiles('u+', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=[0, 30], xlog=True)
fig, ax = compare_profiles('uu', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=[-0.005, 0.017], xlog=True)
compare_profiles('vv', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=None, xlog=True, append_to_fig_ax=(fig, ax))
compare_profiles('ww', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=None, xlog=True, append_to_fig_ax=(fig, ax))
compare_profiles('uv', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=None, xlog=True, append_to_fig_ax=(fig, ax))

# Fig 5, Evolution of Clauser parameter
compare_integral_quantity('beta', 'x', *hifi_data, ylog=True, xlim=[0.2, 1], ylim=[1e-2, 1e2])
compare_integral_quantity('Reth', 'x', *hifi_data, xlim=[0.2, 1], ylim=[0, 7000])
# compare_integral_quantity('Ret', 'x', *hifi_data, xlim=[0.2, 1], ylim=[0, 900])


# Find threshold for laminar flow comparing to viscous sublayer
# case = hifi_data[0]
# x, y = case.flow_dict['x'], case.flow_dict['y']
# nx, ny = case.nx, case.ny
#
# yp = case.flow_dict['y+']
# idx_lam = yp < 5
#
# idx_lam_crit = case.flow_dict['k'] < 0.002
#
# fig, ax = plot.contouring(x.reshape(nx, ny),
#                           y.reshape(nx, ny),
#                           case.flow_dict['k'].reshape(nx, ny))
# plot.scattering(x[idx_lam], y[idx_lam], np.ones_like(x[idx_lam]), append_to_fig_ax=(fig, ax), cmap='autumn')
# plot.scattering(x[idx_lam_crit], y[idx_lam_crit], np.ones_like(x[idx_lam_crit]), append_to_fig_ax=(fig, ax), cmap='spring')




plot.show()
