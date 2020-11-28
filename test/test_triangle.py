# ###################################################################
# test lumley_triangle
#
# Description
# Test for the lumley triangle plot.
#
# ###################################################################
# Author: hw
# created: 24. Apr. 2020
# ###################################################################
import matplotlib.pyplot as plt

from uncert_ident.visualisation import plotter as plot
from uncert_ident.data_handling.flowcase import flowCase

# case_name = 'Breuer_PH_LES_DNS_Re10595'
case_name = 'Xiao_PH_DNS_case_1p0_refined_XYZ'
case = flowCase(case_name, dimension=2)

case.show_anisotropy(lumley=False)
case.show_anisotropy(lumley=False, loc_key='x', loc_value=2)

case.show_anisotropy(lumley=True)
case.show_anisotropy(lumley=True, loc_key='x', loc_value=2)

plt.show()
