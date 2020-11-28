# ###################################################################
# Script test_read_all_mats
#
# Description
# Read flowCase data sets and run all tests.
#
# ###################################################################
# Author: hw
# created: 20. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.data_handling.data_import import find_case_names
from uncert_ident.data_handling.flowcase import flowCase


#####################################################################
### Test
#####################################################################
# All available (processed) mats
case_names = find_case_names()

# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


# Get data basis' statistics
total_sets = len(hifi_data)
total_num_of_points = 0
for case in hifi_data:
    total_num_of_points += case.num_of_points


# Display stats
print('The data basis contains:')
print('Data sets:\t\t%r' % total_sets)
print('Number of points:\t%r' % total_num_of_points)
