# ###################################################################
# test laminar_criteria
#
# Description
# Test the criteria for laminar flow.
#
# ###################################################################
# Author: hw
# created: 07. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.methods.labels import compute_all_labels
import uncert_ident.visualisation.plotter as plot


#####################################################################
### Test
#####################################################################
# Choose case
case = flowCase('PH-Breuer-10595')
# case = flowCase('PH-Xiao-15')
# case = flowCase('CBFS-Bentaleb')
# case = flowCase('NACA4412-Vinuesa-top-10')
# case = flowCase('TBL-APG-Bobke-m13')


# Choose label
label = 'anisotropic'
# label = 'non_negative'


# Get flow data
dit = case.flow_dict
x, y = dit['x'], dit['y']
nx, ny = dit['nx'], dit['ny']


# Get labels without filter
all_labels_non_laminar = compute_all_labels(dit, laminar_criteria=False)
arr = all_labels_non_laminar[label]
arr = arr.astype(int)
print(f"Number of active labels with laminar points\t{np.bincount(arr).min()}")

# Get labels with filter
all_labels_laminar = compute_all_labels(dit, laminar_criteria=True)
arr_lam = all_labels_laminar[label]
arr_lam = arr_lam.astype(int)
print(f"Number of active labels without laminar points\t{np.bincount(arr_lam).min()}")


# Test laminar criteria
idx = dit['k'] < 0.002
lam = arr[idx]
# sij = dit['Sij'][:,:,idx]
# tij = dit['tauij'][:,:,idx]  # Sij >> tij for "laminar" flow


# Show laminar points
# fig, ax = plot.scattering(x, y, np.ones_like(x), cmap='viridis', scale=10)
# plot.scattering(x[idx], y[idx], lam, cmap='gray', scale=10, append_to_fig_ax=(fig, ax))
# plot.show()



# Compare with/-out criteria
plot.scattering(x, y, arr,
                scale=10,
                title='Without laminar flow criteria',
                xlabel='x/h', ylabel='y/h',
                save='without_laminar_criteria'
                )
plot.scattering(x, y, arr_lam,
                scale=10,
                title='With laminar flow criteria',
                xlabel='x/h', ylabel='y/h',
                save='with_laminar_criteria'
                )
plot.show()


print('EOF test_laminar_criteria')
