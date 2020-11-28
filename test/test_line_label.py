# ###################################################################
# test line_label
#
# Description
# Test appending line labels to lining plots.
#
# ###################################################################
# Author: hw
# created: 26. Apr. 2020
# ###################################################################
import matplotlib.pyplot as plt

from uncert_ident.data_handling.flowcase import flowCase, compare_profiles

case_name1 = 'Breuer_PH_LES_DNS_Re10595'
case_name2 = 'Breuer_PH_LES_DNS_Re5600'
case_name3 = 'Breuer_PH_LES_DNS_Re1400'
case1 = flowCase(case_name1, dimension=2)
case2 = flowCase(case_name2, dimension=2)
case3 = flowCase(case_name3, dimension=2)

compare_profiles('um', 'y', 2.0, case1, case2, case3)

plt.show()
