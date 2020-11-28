# ###################################################################
# script CDC-Laval
#
# Description
# Create plots to validate imported data against Laval's paper
#
# ###################################################################
# Author: hw
# created: 20. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.data_handling.flowcase import flowCase, compare_profiles
from uncert_ident.visualisation.plotter import show


# Individual case's name
case_names = []
case_names.append('CDC-Laval')


# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 4, (a) Cf at lower and upper wall
# fig, ax = hifi_data[0].show_geometry()
# compare_profiles('Cf', 'x', 0, *hifi_data, append_to_fig_ax=(fig, ax))
# compare_profiles('Cf', 'x', 10, *hifi_data, append_to_fig_ax=(fig, ax))

# Fig 4, (b) Reversed flow at lower wall
hifi_data[0].show_flow('um', contour=True, ylim=[0.3, 0.8], xlim=[5.5, 6.8], contour_level=[-0.1, 0, 0.1, 1.0])

show()
