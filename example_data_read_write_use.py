# ###################################################################
# script example_data_read_write_use
#
# Description
# Present how to use the data I/O utilities of the uncert_ident
# package.
# Includes example for processing the raw data in inversion/DATA i.e.
# compute gradients and other quantities
# and example on simple visualisation of each data set with the flowCase
# class.
#
# ###################################################################
# Author: hw
# created: 05. Jun. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.data_handling.data_import import find_case_names
from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.visualisation.plotter import show


#####################################################################
### Process raw data
#####################################################################
# Script processes raw data and saves them as .mat under DATA/*/processed/
# Configure which data sets to evaluate in convert_to_mat
import convert_to_mat


#####################################################################
### Read processed data
#####################################################################
# All available mats (Searches for all files in DATA/*/processed/*.mat)
case_names = find_case_names()


# Individual data sets (Follow naming scheme in data_overview)
# case_names = ['PH-Breuer-10595',
#               'PH-Xiao-10',
#               'CDC-Laval'
#               ]


# Instantiate cases
hifi_data = []
for case_name in case_names:
    hifi_data.append(flowCase(case_name))


# Set breakpoint here, to investigate available data structures
print("Breakpoint dummy")


#####################################################################
### Visualise data
#####################################################################
# Plot turbulent kinetic energy
for case in hifi_data:
    case.show_flow('k')

show()
