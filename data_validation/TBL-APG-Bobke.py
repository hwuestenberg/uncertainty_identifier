# ###################################################################
# script Bobke_FP_APG_LES
#
# Description
# Create plots to validate imported data against Bobke's paper
#
# ###################################################################
# Author: hw
# created: 28. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from uncert_ident.data_handling.flowcase import flowCase, compare_integral_quantity
import uncert_ident.visualisation.plotter as plot



#####################################################################
### Load data
#####################################################################
# Individual case's name
case_names = []
case_names.append('TBL-APG-Bobke-m13')
case_names.append('TBL-APG-Bobke-m16')
case_names.append('TBL-APG-Bobke-m18')
case_names.append('TBL-APG-Bobke-b1')
case_names.append('TBL-APG-Bobke-b2')

# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 1, TKE budget along y+
# hifi_data[0].compare_budgets('k', 'y+')

# Fig 4, Evolution of (a) Ret, (b) Retheta
# compare_integral_quantity('Ret', 'x', *hifi_data, xlim=[0, 2400], ylim=[0, 850])
compare_integral_quantity('Reth', 'x', *hifi_data, xlim=[0, 2400], ylim=[0, 4300])

# Fig 5, (a) Evolution of Cf
compare_integral_quantity('Cf', 'Reth', *hifi_data, xlim=[0, 4300], ylim=[0, 0.008])


# Find laminar flow threshold with y+ < 5
case = hifi_data[0]
x, y = case.flow_dict['x'], case.flow_dict['y']
nx, ny = case.nx, case.ny

yp = case.flow_dict['y+']
idx_lam = yp < 5

idx_lam_crit = case.flow_dict['k'] < 0.002

fig, ax = plot.contouring(x.reshape(nx, ny),
                          y.reshape(nx, ny),
                          case.flow_dict['k'].reshape(nx, ny))
plot.scattering(x[idx_lam_crit], y[idx_lam_crit], np.ones_like(x[idx_lam_crit]), append_to_fig_ax=(fig, ax), cmap='spring')
plot.scattering(x[idx_lam], y[idx_lam], np.ones_like(x[idx_lam]), append_to_fig_ax=(fig, ax), cmap='autumn')


plot.show()
