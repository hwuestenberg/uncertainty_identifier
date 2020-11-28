# ###################################################################
# script Xiao_PH_DNS
#
# Description
# Create plots to validate imported data against Xiao's paper
#
# ###################################################################
# Author: hw
# created: 12. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.data_handling.flowcase import flowCase, compare_profiles
from uncert_ident.visualisation.plotter import show



# Individual case's name
case_names = list()
case_names.append('PH-Xiao-05')
case_names.append('PH-Xiao-08')
case_names.append('PH-Xiao-10')
case_names.append('PH-Xiao-12')
case_names.append('PH-Xiao-15')

# Instantiate cases
hifi_data = list()
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 2 (b), (c)
hifi_data[-1].show_flow('um', xlim=[0, 5], ylim=[0, 3.036])
hifi_data[0].show_flow('um', xlim=[0, 5], ylim=[0, 3.036])

# Fig 10 (a-f), requires profile data (doesn't work properly) in barycentric map
# 1p0, x/h = 2, 5, 7
hifi_data[0].show_anisotropy()#loc_key='x', loc_value=2)
hifi_data[2].show_anisotropy()#loc_key='x', loc_value=2)
hifi_data[4].show_anisotropy()#loc_key='x', loc_value=2)
# hifi_data[2].show_anisotropy(loc_key='x', loc_value=5)
# hifi_data[2].show_anisotropy(loc_key='x', loc_value=7)
#
# # 1p5, x/h = 2, 5, 7
# hifi_data[-1].show_anisotropy(loc_key='x', loc_value=2)
# hifi_data[-1].show_anisotropy(loc_key='x', loc_value=5)
# hifi_data[-1].show_anisotropy(loc_key='x', loc_value=7)
#

show()
