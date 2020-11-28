# ###################################################################
# script Bentaleb_curved_backwards_facing_step_LES
#
# Description
# Create plots to validate imported data against Bentaleb's paper
#
# ###################################################################
# Author: hw
# created: 13. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.data_handling.flowcase import flowCase, compare_profiles
from uncert_ident.visualisation.plotter import show



# Individual case's name
case_names = []
case_names.append('CBFS-Bentaleb')



# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


#####################################################################
### Figures
#####################################################################
# Fig 19, lumley triangle for x/h = 1, 1.5
hifi_data[0].show_anisotropy(lumley=True, loc_key='x', loc_value=1)
hifi_data[0].show_anisotropy(lumley=True, loc_key='x', loc_value=1.5)

# Recirculation region
hifi_data[0].show_flow('um', xlim=[0.5, 4.6], ylim=[-0.1, 1.2])


show()
