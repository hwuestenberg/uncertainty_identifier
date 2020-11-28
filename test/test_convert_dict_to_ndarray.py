# ###################################################################
# function convert_dict_to_ndarray
#
# Description
# Check the conversion from an dict with equal length ndarrays to an
# ndarray of shape [num_of_arrays, num_of_points].
#
# ###################################################################
# Author: hw
# created: 13. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from uncert_ident.utilities import convert_dict_to_ndarray

#####################################################################
### Tests
#####################################################################

a = {'i': np.array([1, 2, 3, 4, 5]), 'j': np.arange(5), 'k': np.arange(5)*-2}
b = {'p': np.array([15, 33, 23, 55, 85]), 'q': np.arange(5)*100, 'r': np.arange(5)*30}
c = {'x': np.array([15, 33, 23, 55]), 'y': np.arange(5)*100, 'z': np.arange(5)*30}


assert isinstance(convert_dict_to_ndarray(a), np.ndarray), 'Invalid return type: %r' % type(convert_dict_to_ndarray(a))

assert convert_dict_to_ndarray(a).shape == (len(list(a)), len(a['i'])), 'Invalid shape of ndarray.'
assert convert_dict_to_ndarray(a, b).shape == (len(list(a)) + len(list(b)), len(a['i'])), 'Invalid shape of ndarray.'

try:
    convert_dict_to_ndarray(a, b, c)
except AssertionError:
    assert True

print('Test successful')
