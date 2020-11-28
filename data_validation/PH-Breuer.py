# ###################################################################
# script Breuer_PH_LES_DNS
#
# Description
# Create plots to validate imported data against Breuer's paper
#
# ###################################################################
# Author: hw
# created: 24. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.data_handling.flowcase import flowCase, compare_profiles
from uncert_ident.visualisation.plotter import show, lining, empty_plot
from uncert_ident.utilities import get_profile_data



# Individual case's name
case_names = []
case_names.append('PH-Breuer-10595')
case_names.append('PH-Breuer-5600')
case_names.append('PH-Breuer-2800')
case_names.append('PH-Breuer-1400')
case_names.append('PH-Breuer-700')


# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 21 (b) pressure distribution
# compare_profiles('pm', 'x', 0, *hifi_data)
#
# # Fig 27 (a), (c)
# compare_profiles('um', 'y', 4, *hifi_data, xlim=[-0.2, 1.2], ylim=[-0.1, 3.1])
# compare_profiles('uu', 'y', 4, *hifi_data, xlim=[0, 0.1], ylim=[-0.1, 3.1])
#
# # Fig 23, tiny separation bubble
# hifi_data[3].show_flow('um', contour_level=[-0.1, 0, 0.1, 0.2, 0.3], xlim=[0.55, 0.93], ylim=[0.58, 0.82])
#
# # Fig 24, tiny recirculation bubble
# hifi_data[2].show_flow('um', contour_level=[-0.1, 0, 0.1, 0.2, 0.3], xlim=[6.9, 7.5], ylim=[-0.01, 0.31])
#
# # Fig 29, tke comparison
# hifi_data[0].show_flow('k', contour=False)
# hifi_data[-1].show_flow('k', contour=False)

# Fig 30, lumley triangles with invariants
loc = 4
p_IIb = get_profile_data(hifi_data[0], 'IIb', 'y', loc)
p_IIIb = get_profile_data(hifi_data[0], 'IIIb', 'y', loc)

p2_IIb = get_profile_data(hifi_data[-1], 'IIb', 'y', loc)
p2_IIIb = get_profile_data(hifi_data[-1], 'IIIb', 'y', loc)

fig, ax = empty_plot()
lining(p_IIIb[0][221:-1], p_IIb[0][221:-1], xlim=[-0.03, 0.09], ylim=[0, 0.41], append_to_fig_ax=(fig, ax), linestyle='-s')
lining(p2_IIIb[0][221:-1], p2_IIb[0][221:-1], xlim=[-0.03, 0.09], ylim=[0, 0.41], append_to_fig_ax=(fig, ax))


show()
