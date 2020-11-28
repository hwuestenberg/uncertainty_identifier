# ###################################################################
# function gradient_2d_with_transformation
#
# Description
# Test for the approximation of gradients on orthogonal grids using
# a coordinate transformation. Computes a simple function with
# known derivatives. Compares the exact derivatives to the
# approximation.
#
# ###################################################################
# Author: hw
# created: 30. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np

import uncert_ident.visualisation.plotter as plot

from uncert_ident.methods.gradient import gradient_2d_with_transformation
from uncert_ident.data_handling.flowcase import flowCase


#####################################################################
### Tests
#####################################################################
print("TEST OF GRADIENT COMPUTATION RUNNING...")

# Read any flowcase data
# case_name = 'Breuer_PH_LES_DNS_Re10595'
case_name = 'Noorani_bent_pipe_DNS_R11700k001'

if 'Noorani' in case_name:
    case = flowCase(case_name, dimension=2, coord_sys='polar')
else:
    case = flowCase(case_name, dimension=2, coord_sys='cartesian')

dictionary = case.flow_dict

# Read non-uniform grid coordinates
nx = dictionary['nx']
ny = dictionary['ny']
x = dictionary['x'].reshape(nx, ny)
y = dictionary['y'].reshape(nx, ny)

# Compute test function and exact gradient in both coordinate systems
f = np.sin(x)*np.cos(y)
exact_gradient = np.array([np.cos(x)*np.cos(y),
                           -np.sin(x)*np.sin(y)
                           ]).reshape(2, -1)



# Approximate gradient with transformation
approx_gradient = \
    gradient_2d_with_transformation(f, x, y, coord_sys='polar')

# Evaluate error (exact to approximate)
error = np.ones(nx*ny)
error_x = np.ones(nx*ny)
error_y = np.ones(nx*ny)
for i in range(nx*ny):
    error[i] = np.linalg.norm(exact_gradient[:, i] - approx_gradient[:, i])
    error_x[i] = exact_gradient[0, i] - approx_gradient[0, i]
    error_y[i] = exact_gradient[1, i] - approx_gradient[1, i]

# Plot
print("The mean error is " + str(np.mean(error)))
print("The max error is " + str(np.max(error)))
print("The min error is " + str(np.min(error)))

# plot.scattering(x, y, f, title="f", save='sample_function_sinx_cosy')
# plot.scattering(x, y, exact_gradient[0].reshape(nx, ny),
#                 title="exact")
# plot.scattering(x, y, approx_gradient[0].reshape(nx, ny),
#                 title="approx")

plot.scattering(x, y, error.reshape(nx, ny), title="error",
                colorbar=True)
# plot.scattering(x[3:-3, 1:-1], y[3:-3, 1:-1], error.reshape(nx, ny)[3:-3, 1:-1], title="error",
#                 save='gradient_approximation_error', colorbar=True)
# plot.scattering(x, y, error_x.reshape(nx, ny), title="error x",
#                 save='gradient_approximation_error_x', colorbar=True)
# plot.scattering(x, y, error_y.reshape(nx, ny), title="error y",
#                 save='gradient_approximation_error_y', colorbar=True)

plot.show()


print("TEST OF GRADIENT COMPUTATION COMPLETED")
