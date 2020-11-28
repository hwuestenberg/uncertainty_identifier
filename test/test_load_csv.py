# ###################################################################
# script test_load_csv
#
# Description
# Test the function for loading any csv file. Also to test whether a
# file is readable.
#
# ###################################################################
# Author: hw
# created: 14. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np

import uncert_ident.data_handling.data_import as di


#####################################################################
### Test
#####################################################################
path_to_raw = di.path_to_raw_data
folder = 'Tanarro_NACA0012_LES/'
file = 'bottom_boundary_points.dat'

fname = path_to_raw + folder + file

col_names = ['x', 'y']
skip_header = 2
delimiter = ' '

# Load data
data = di.load_csv(fname, col_names=col_names, skip_header=skip_header, delimiter=delimiter)

print('EOF test_load_csv.py')
