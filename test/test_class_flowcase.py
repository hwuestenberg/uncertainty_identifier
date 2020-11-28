# ###################################################################
# Script test_class_flow_case
#
# Description
# Read flowCase data sets and run all tests.
#
# ###################################################################
# Author: hw
# created: 13. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.data_handling.data_import import find_case_names
from uncert_ident.data_handling.flowcase import flowCase


#####################################################################
### Test
#####################################################################
# Individual case's name
case_names = list()

# case_names.append('PH-Breuer-10595')
# case_names.append('PH-Breuer-5600')
# case_names.append('PH-Breuer-2800')
# case_names.append('PH-Breuer-1400')
# case_names.append('PH-Breuer-700')

# case_names.append('CBFS-Bentaleb')

# case_names.append('TBL-APG-Bobke-b1')
##### case_names.append('TBL-APG-Bobke-b2')
# case_names.append('TBL-APG-Bobke-m13')
# case_names.append('TBL-APG-Bobke-m16')
# case_names.append('TBL-APG-Bobke-m18')

# case_names.append('NACA4412-Vinuesa-top-1')
# case_names.append('NACA4412-Vinuesa-top-2')
# case_names.append('NACA4412-Vinuesa-top-4')
# case_names.append('NACA4412-Vinuesa-top-10')
# case_names.append('NACA4412-Vinuesa-bottom-1')
# case_names.append('NACA4412-Vinuesa-bottom-2')
# case_names.append('NACA4412-Vinuesa-bottom-4')
# case_names.append('NACA4412-Vinuesa-bottom-10')
# case_names.append('NACA0012-Tanarro-top-4')

# case_names.append('PH-Xiao-05')
# case_names.append('PH-Xiao-08')
# case_names.append('PH-Xiao-10')
# case_names.append('PH-Xiao-12')
# case_names.append('PH-Xiao-15')

# case_names.append('PH-Breuer-5600')
# case_names.append('CBFS-Bentaleb')
# case_names.append('TBL-APG-Bobke-m13')
case_names.append('NACA4412-Vinuesa-top-1')
# case_names.append('PH-Xiao-10')

# All available mats
# case_names = find_case_names()


# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


# Compute features and labels
for case in hifi_data:
    case.get_features()
    case.get_labels()
    pass


# Test and display
for case in hifi_data:
    # case.show_geometry()
    case.show_flow('um', contour=True, colorbar=True, contour_level=100)#, contour_level=np.linspace(0, 0.1, 20))
    # case.show_flow('k', contour=True, colorbar=True, contour_level=100)  # , contour_level=np.linspace(0, 0.1, 20))
    # case.show_flow('pm', contour=False, colorbar=True, contour_level=50)  # , contour_level=np.linspace(0, 0.1, 20))
    # case.show_flow('grad_k', contour=True, colorbar=True, contour_level=50)  # , contour_level=np.linspace(0, 0.1, 20))
    # case.show_flow('grad_pm', contour=True, colorbar=True, contour_level=50)  # , contour_level=np.linspace(0, 0.1, 20))
    # case.show_flow('diss_rt', contour=True, colorbar=True)  # , contour_level=np.linspace(0, 0.1, 20))
    # case.show_flow('bij_eig1', contour=False, show_geometry=True)
    # case.show_flow('bij_eig2', contour=False, show_geometry=True)
    # case.show_flow('bij_eig3', contour=False, show_geometry=True)
    # case.show_flow('IIb', contour=False)
    # case.show_features(show_all=True);
    # case.show_features(feature_key='k_eps_Sij')
    # case.show_features(feature_key='grad_pm_stream')
    # case.show_features(feature_key='stream_curv')
    # case.show_features(feature_key='Re_d')
    # case.show_features(feature_key='inv15')
    # case.show_features(feature_key='conv_prod_tke')
    # case.show_label(show_all=True)
    # case.show_label('non_negative')
    # case.show_label('anisotropic')
    # case.show_label('non_linear')
    # case.show_profile('pm', 'x', 1.0)
    pass

# from uncert_ident.methods.features import tke_feature2
# for case in hifi_data:
#     case.feature_dict['tke2'] = tke_feature2(case.flow_dict)
#
# for case in hifi_data:
#     case.show_features("tke2")


plt.show()
#

print("ALL TESTS SUCCESSFUL")


# case = hifi_data[0]
# data = case.flow_dict
#
#
# cond_II = np.array([II > 1/6 for II in data['IIb']])
# cond_eig = np.array([2*(eig1**2 + eig1*eig2 + eig2**2) > 1/6 for eig1, eig2 in zip(data['bij_eig1'], data['bij_eig2'])])
#
