# ###################################################################
# script test_save_load_mat
#
# Description
# Test saving and loading mat files with different files, paths, etc.
#
# ###################################################################
# Author: hw
# created: 13. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from os.path import abspath, splitext, basename

from uncert_ident.data_handling.data_import import save_dict_to_mat, load_mat, path_to_processed_data, \
    find_path_to_mat, exist_mat


#####################################################################
### Tests
#####################################################################
# Load mat file
# case_name = 'Breuer_PH_LES_DNS_Re5600'
case_name = 'Vinuesa_NACA4412_LES_top1n'
fname = find_path_to_mat(case_name)
case_name_len = len(case_name)
mat1 = load_mat(fname)

# Save file as test.mat
save_path = '../' + path_to_processed_data + 'test/test.mat'
save_dict_to_mat(save_path, mat1)

# Search test file
if exist_mat('test'):
    print('test.mat found!')
else:
    print('Cannot find test.mat')

