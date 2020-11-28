# ###################################################################
# function compare_flow
#
# Description
# Test the function to load multiple data sets, the data basis, and
# visualise all flows next to each other.
#
# ###################################################################
# Author: hw
# created: 11. Jun. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib.pyplot as plt

from uncert_ident.data_handling.flowcase import flowCase, compare_flow
from uncert_ident.data_handling.data_import import find_case_names

#####################################################################
### Tests
#####################################################################
# Simple grid plot
# for num_of_plots in range(0, 20, 2):
#     # Check required amount of subplots
#     grid_length = int(num_of_plots**0.5)+1
#
#     # Plot all
#     fig = plt.figure()
#     for num in range(num_of_plots):
#         ax = fig.add_subplot(grid_length, grid_length, num+1)
# plt.show()


# Compare flow
case_names = find_case_names()
hifi_data = list()
for case_name in case_names:
    hifi_data.append(flowCase(case_name))

print("No of cases: " + str(len(hifi_data)))

compare_flow('k', *hifi_data)
