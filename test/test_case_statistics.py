# ###################################################################
# Script test_case_statistics
#
# Description
# Read flowCase data sets and analyse data points, labels, ...
#
# ###################################################################
# Author: hw
# created: 01. Sep. 2020
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

# case_names.append('CDC-Laval')
# case_names.append('CBFS-Bentaleb')

case_names.append('TBL-APG-Bobke-b1')
# case_names.append('TBL-APG-Bobke-b2')
case_names.append('TBL-APG-Bobke-m13')
case_names.append('TBL-APG-Bobke-m16')
case_names.append('TBL-APG-Bobke-m18')

# case_names.append('NACA4412-Vinuesa-top-1')
# case_names.append('NACA4412-Vinuesa-top-2')
# case_names.append('NACA4412-Vinuesa-top-4')
# case_names.append('NACA4412-Vinuesa-top-10')
# case_names.append('NACA4412-Vinuesa-bottom-1')
# case_names.append('NACA4412-Vinuesa-bottom-2')
# case_names.append('NACA4412-Vinuesa-bottom-4')
# case_names.append('NACA4412-Vinuesa-bottom-10')
# case_names.append('NACA0012-Tanarro-top-4')

# case_names.append('BP-Noorani-11700-001')

# case_names.append('PH-Xiao-05')
# case_names.append('PH-Xiao-08')
# case_names.append('PH-Xiao-10')
# case_names.append('PH-Xiao-12')
# case_names.append('PH-Xiao-15')


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


# Get statistics
point_sum = []
n_active_neg_sum = []
n_active_ani_sum = []
for case in hifi_data:
    print(f"\n\nCase name:\t\t\t\t\t{case.case_name}")
    print(f"Number of samples:\t\t\t{case.num_of_points}")

    n_active_neg, n_inactive_neg = np.sort(np.bincount(case.label_dict['non_negative']))
    ratio_neg = np.round(n_active_neg/case.num_of_points, 2)
    n_active_ani, n_inactive_ani = np.sort(np.bincount(case.label_dict['anisotropic']))
    ratio_ani = np.round(n_active_ani/case.num_of_points, 2)
    print(f"Ratio non-negativity:\t\t{ratio_neg}")
    print(f"Ratio anisotropy:\t\t\t{ratio_ani}")

    print(f"Characteristic length:\t\t{case.flow_dict['char_length']}")

    point_sum.append(case.num_of_points)
    n_active_neg_sum.append(n_active_neg)
    n_active_ani_sum.append(n_active_ani)


# Get group stats
total_points = np.sum(point_sum)
total_n_active_neg = np.sum(n_active_neg_sum)
total_n_active_ani = np.sum(n_active_ani_sum)

total_ratio_neg = np.round(total_n_active_neg/total_points, 2)
total_ratio_ani = np.round(total_n_active_ani/total_points, 2)

print(f"\n----------------------------------------------")
print(f"Group stats")
print(f"Total number of samples:\t{total_points}")
print(f"Total ratio non-negativity:\t{total_ratio_neg}")
print(f"Total ratio anisotropy:\t\t{total_ratio_ani}")
