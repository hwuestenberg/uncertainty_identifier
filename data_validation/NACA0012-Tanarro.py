# ###################################################################
# script Tanarro_NACA0012_LES
#
# Description
# Create plots to validate imported data against Tanarro's paper
#
# ###################################################################
# Author: hw
# created: 30. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.data_handling.flowcase import flowCase, compare_integral_quantity, compare_profiles

import uncert_ident.visualisation.plotter as plot



#####################################################################
### Load data
#####################################################################
# Individual case's name
case_names = list()
case_names.append('NACA0012-Tanarro-top-4')
case_names.append('NACA4412-Vinuesa-top-4')

# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 2, (a) inner-scaled mean velocity profile (b) selected tauij components
# compare_profiles('u+', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=[0, 30], xlog=True)
# fig, ax = compare_profiles('uu', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=[-0.005, 0.017], xlog=True)
# compare_profiles('vv', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=None, xlog=True, append_to_fig_ax=(fig, ax))
# compare_profiles('ww', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=None, xlog=True, append_to_fig_ax=(fig, ax))
# compare_profiles('uv', 'y+', 0.7, hifi_data[2], xlim=[1, 1e3], ylim=None, xlog=True, append_to_fig_ax=(fig, ax))

# Fig 2, Evolution of Clauser beta
compare_integral_quantity('beta', 'x', *hifi_data, xlim=[0, 1], ylim=[0, 20])
# compare_integral_quantity('beta', 'Ret', *hifi_data, xlim=[0, 400], ylim=[0, 20])

# Fig 4, Evolution of Reynolds number wrt momentum thickness and friction (tau)
compare_integral_quantity('Reth', 'x', *hifi_data, xlim=[0, 1], ylim=[0, 3500])
# compare_integral_quantity('Ret', 'x', *hifi_data, xlim=[0, 1], ylim=[0, 500])

# Fig 6, Inner-scaled mean velocity profiles
compare_profiles('u+', 'y+', 0.4, *hifi_data, xlim=[1, 1e3], ylim=[0, 30], xlog=True)
compare_profiles('u+', 'y+', 0.75, *hifi_data, xlim=[1, 1e3], ylim=[0, 30], xlog=True)


plot.show()