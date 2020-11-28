# ###################################################################
# function get_shape
#
# Description
# Test for the approximation of gradients on orthogonal grids using
# a coordinate transformation. Computes a simple function with
# known derivatives. Compares the exact derivatives to the
# approximation.
#
# ###################################################################
# Author: hw
# created: 05. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np

from uncert_ident.utilities import get_shape

#####################################################################
### Tests
#####################################################################

integer = 1
float = 1.0
single_scalar_array = np.array([10])
scalar_array = np.arange(10)

vector = np.arange(3*1).reshape(3, 1)
vector_array = np.arange(3*10).reshape(3, 10)

tensor = np.arange(3*3*1).reshape(3, 3)
tensor_array = np.arange(3*3*10).reshape(3, 3, 10)

list_of_objects = [integer,
                   float,
                   single_scalar_array,
                   scalar_array,
                   vector,
                   vector_array,
                   tensor,
                   tensor_array]

out = list()
for i, obj in enumerate(list_of_objects):
    out.append(get_shape(obj))
    print(out[i])

assert get_shape(integer) == 'scalar'
assert get_shape(float) == 'scalar'
assert get_shape(single_scalar_array) == 'scalar_array'
assert get_shape(scalar_array) == 'scalar_array'
assert get_shape(vector) == 'vector'
assert get_shape(vector_array) == 'vector_array'
assert get_shape(tensor) == 'tensor'
assert get_shape(tensor_array) == 'tensor_array'
