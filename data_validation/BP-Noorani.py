# ###################################################################
# script Noorani_bent_pipe_DNS
#
# Description
# Create plots to validate imported data against Noorani's paper
#
# ###################################################################
# Author: hw
# created: 05. May 2020
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
case_names = []
case_names.append('BP-Noorani-11700-001')
case_names.append('BP-Noorani-11700-01')
# case_names.append('BP-Noorani-11700-03')  # Not usable, see convert_to_mat.py
case_names.append('BP-Noorani-5300-001')


# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 5, mean axial velocity/normalised by u_tau and k
fig, ax = compare_profiles('usm', 'y', 0.0, hifi_data[-1], xlim=[0, 5], var_scale=0.2)
compare_profiles('k', 'y', 0.0, hifi_data[-1], append_to_fig_ax=(fig, ax))

# Fig 10, mean axial velocity
compare_profiles('usm', 'y', 0.0, *hifi_data, xlim=[0, 25])

# Fig 12, turbulent kinetic energy
compare_profiles('k', 'y', 0.0, *hifi_data, xlim=[0, 5])


plot.show()
