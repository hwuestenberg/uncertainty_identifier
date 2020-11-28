# ###################################################################
# module utilities
#
# Description
# Provides handling functions.
#
# ###################################################################
# Author: hw
# created: 24. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from timeit import default_timer as timer
from datetime import datetime

#####################################################################
### Keyword definitions
#####################################################################
GEOMETRY_KW = ['converging_diverging_channel',
               'curved_backwards_facing_step',
               'periodic_hills',
               'wall_mounted_cube',
               'jet_in_cross_flow',
               'flat_plate',
               'naca4412',
               'naca0012',
               'bent_pipe']


FLOWCASE_KW = ['dimension',
               'coord_sys',
               'geometry',
               'geometry_scale',
               'nx',
               'ny']


DATASET_PARAMETERS = ['dimension',
                      'coord_sys',
                      'geometry',
                      'geometry_scale',
                      'char_length',
                      'nu',
                      'rho',
                      'ghost_nx',
                      'block_order',
                      'block_ranges',
                      'subcase_names',
                      'subcase_nxs',
                      'subcase_nys',
                      '']


#####################################################################
### Dictionary keys
#####################################################################
# Cylinder coordinates
CYL_VEL_KEYS = ['urm', 'utm', 'usm']

CYL_RMS_KEYS = ['ur_rms', 'ut_rms', 'us_rms']

CYL_TAU_KEYS = ['urur', 'urut', 'urus',
                'urut', 'utut', 'utus',
                'urus', 'utus', 'usus']

CYL_GRAD_U_KEYS = ['grad_dur_dr', 'grad_dur_dt', 'grad_dur_ds',
                   'grad_dut_dr', 'grad_dut_dt', 'grad_dut_ds',
                   'grad_dus_dr', 'grad_dus_dt', 'grad_dus_ds']

CYL_GRAD_U_NORM_KEYS = ['grad_dur_dr_norm', 'grad_dur_dt_norm', 'grad_dur_ds_norm',
                        'grad_dut_dr_norm', 'grad_dut_dt_norm', 'grad_dut_ds_norm',
                        'grad_dus_dr_norm', 'grad_dus_dt_norm', 'grad_dus_ds_norm']

CYL_GRAD_U2_KEYS = ['grad_dur2_dr', 'grad_dur2_dt', 'grad_dur2_ds',
                    'grad_dut2_dr', 'grad_dut2_dt', 'grad_dut2_ds',
                    'grad_dus2_dr', 'grad_dus2_dt', 'grad_dus2_ds']

CYL_GRAD_KEYS = ['grad_um_cyl', 'grad_um_norm_cyl', 'grad_um2_cyl', 'grad_k_cyl', 'grad_pm_cyl']

CYL_DISS_KEYS = ['diss_urur', 'diss_urut', 'diss_urus',
                 'diss_urut', 'diss_utut', 'diss_utus',
                 'diss_urus', 'diss_utus', 'diss_usus']

CYL_URUR_BUDGET_KEYS = ['prod_urur', 'trb_tsp_urur', 'prs_str_urur', 'prs_dif_urur', 'vsc_dif_urur', 'diss_urur', 'conv_urur', 'balance_urur']
CYL_UTUT_BUDGET_KEYS = ['prod_utut', 'trb_tsp_utut', 'prs_str_utut', 'prs_dif_utut', 'vsc_dif_utut', 'diss_utut', 'conv_utut', 'balance_utut']
CYL_USUS_BUDGET_KEYS = ['prod_usus', 'trb_tsp_usus', 'prs_str_usus', 'prs_dif_usus', 'vsc_dif_usus', 'diss_usus', 'conv_usus', 'balance_usus']
CYL_URUT_BUDGET_KEYS = ['prod_urut', 'trb_tsp_urut', 'prs_str_urut', 'prs_dif_urut', 'vsc_dif_urut', 'diss_urut', 'conv_urut', 'balance_urut']
CYL_URUS_BUDGET_KEYS = ['prod_urus', 'trb_tsp_urus', 'prs_str_urus', 'prs_dif_urus', 'vsc_dif_urus', 'diss_urus', 'conv_urus', 'balance_urus']
CYL_UTUS_BUDGET_KEYS = ['prod_utus', 'trb_tsp_utus', 'prs_str_utus', 'prs_dif_utus', 'vsc_dif_utus', 'diss_utus', 'conv_utus', 'balance_utus']
CYL_ALL_BUDGET_KEYS = [*CYL_URUR_BUDGET_KEYS, *CYL_UTUT_BUDGET_KEYS, *CYL_USUS_BUDGET_KEYS,
                       *CYL_URUT_BUDGET_KEYS, *CYL_URUS_BUDGET_KEYS, *CYL_UTUS_BUDGET_KEYS]

CYL_ALL_KEYS_LISTS = [CYL_VEL_KEYS, CYL_RMS_KEYS, CYL_TAU_KEYS, CYL_GRAD_U_KEYS, CYL_DISS_KEYS, CYL_ALL_BUDGET_KEYS]
CYL_ALL_KEYS = [*CYL_VEL_KEYS, *CYL_RMS_KEYS, *CYL_TAU_KEYS, *CYL_GRAD_U_KEYS, *CYL_DISS_KEYS, *CYL_ALL_BUDGET_KEYS]

# Cartesian coordinates
VEL_KEYS = ['um', 'vm', 'wm']

RMS_KEYS = ['u_rms', 'v_rms', 'w_rms']

TAU_KEYS = ['uu', 'uv', 'uw',
            'uv', 'vv', 'vw',
            'uw', 'vw', 'ww']

GRAD_U_KEYS = ['grad_du_dx', 'grad_du_dy', 'grad_du_dz',
               'grad_dv_dx', 'grad_dv_dy', 'grad_dv_dz',
               'grad_dw_dx', 'grad_dw_dy', 'grad_dw_dz']

GRAD_U_NORM_KEYS = ['grad_du_dx_norm', 'grad_du_dy_norm', 'grad_du_dz_norm',
                    'grad_dv_dx_norm', 'grad_dv_dy_norm', 'grad_dv_dz_norm',
                    'grad_dw_dx_norm', 'grad_dw_dy_norm', 'grad_dw_dz_norm']

GRAD_U2_KEYS = ['grad_du2_dx', 'grad_du2_dy', 'grad_du2_dz',
                 'grad_dv2_dx', 'grad_dv2_dy', 'grad_dv2_dz',
                 'grad_dw2_dx', 'grad_dw2_dy', 'grad_dw2_dz']


GRAD_U_KEYS_STAT_2D = ['grad_du_dx', 'grad_dv_dx', 'grad_dw_dx',
                       'grad_du_dy', 'grad_dv_dy', 'grad_dw_dy']

GRAD_KEYS = ['grad_um', 'grad_um_norm', 'grad_um2', 'grad_k', 'grad_pm']

DISS_KEYS = ['diss_uu', 'diss_uv', 'diss_uw',
             'diss_uv', 'diss_vv', 'diss_vw',
             'diss_uw', 'diss_vw', 'diss_ww']

TKE_BUDGET_KEYS_BL = ['prod_k', 'diss_rt', 'trb_tsp_k', 'vsc_dif_k', 'vel_prs_grd_k', 'conv_k', ]
TKE_BUDGET_KEYS = ['prod_k', 'trb_tsp_k', 'prs_str_k', 'prs_dif_k', 'vsc_dif_k', 'diss_rt', 'conv_k', 'balance_k']

UU_BUDGET_KEYS = ['prod_uu', 'trb_tsp_uu', 'prs_str_uu', 'prs_dif_uu', 'vsc_dif_uu', 'diss_uu', 'conv_uu', 'balance_uu']
VV_BUDGET_KEYS = ['prod_vv', 'trb_tsp_vv', 'prs_str_vv', 'prs_dif_vv', 'vsc_dif_vv', 'diss_vv', 'conv_vv', 'balance_vv']
WW_BUDGET_KEYS = ['prod_ww', 'trb_tsp_ww', 'prs_str_ww', 'prs_dif_ww', 'vsc_dif_ww', 'diss_ww', 'conv_ww', 'balance_ww']
UV_BUDGET_KEYS = ['prod_uv', 'trb_tsp_uv', 'prs_str_uv', 'prs_dif_uv', 'vsc_dif_uv', 'diss_uv', 'conv_uv', 'balance_uv']
UW_BUDGET_KEYS = ['prod_uw', 'trb_tsp_uw', 'prs_str_uw', 'prs_dif_uw', 'vsc_dif_uw', 'diss_uw', 'conv_uw', 'balance_uw']
VW_BUDGET_KEYS = ['prod_vw', 'trb_tsp_vw', 'prs_str_vw', 'prs_dif_vw', 'vsc_dif_vw', 'diss_vw', 'conv_vw', 'balance_vw']

ALL_BUDGET_KEYS = [*UU_BUDGET_KEYS, *VV_BUDGET_KEYS, *WW_BUDGET_KEYS,
                   *UV_BUDGET_KEYS, *UW_BUDGET_KEYS, *VW_BUDGET_KEYS]

ALL_KEYS_LISTS = [VEL_KEYS, RMS_KEYS, TAU_KEYS, GRAD_U_KEYS, DISS_KEYS, ALL_BUDGET_KEYS]
ALL_KEYS = [*VEL_KEYS, *RMS_KEYS, *TAU_KEYS, *GRAD_U_KEYS, *DISS_KEYS, *ALL_BUDGET_KEYS]


# Keys for physical features
PHYSICAL_KEYS = ['qcrit',
                 'tke',
                 'tke2',
                 'Re_d',
                 'k_eps_Sij',
                 'visc_ratio',
                 'orthogonal',
                 # 'conv_prod_tke',
                 # 'tau_ratio',
                 'stream_curv',
                 'grad_pm_stream',
                 'pm_normal_shear_ratio',
                 # 'cubic_nut', excluded non-linearity metric
                 ]
num_of_physical = len(PHYSICAL_KEYS)
num_of_invariants = 47  # Number of invariants in minimal integrity basis for Sij, Wij, Pij, Kij
INVARIANT_KEYS = ["inv{:02d}".format(i) for i in range(num_of_invariants)]#range(len(PHYSICAL_KEYS), len(PHYSICAL_KEYS) + num_of_invariants)]
FEATURE_KEYS = PHYSICAL_KEYS + INVARIANT_KEYS
num_of_features = num_of_physical + num_of_invariants


# Keys for labels
LABEL_KEYS = [
    'non_negative',
    'anisotropic',
    # 'non_linear'
]


# Colors for confusion matrix scatter plots
TRUE_POSITIVE = 1
FALSE_NEGATIVE = 0.7
FALSE_POSITIVE = 0.3
TRUE_NEGATIVE = 0


#####################################################################
### Functions
#####################################################################
def normalise_max_abs(vector):
    """
    Normalise a vector to range [-1, 1] using the
    maximum absolute value along  axis 0.
    :param vector: Any vector of shape [i, n]
    :return: Normalised vector.
    """

    # Check vector shape
    assert len(vector.shape) == 2
    assert vector.shape[0] < vector.shape[1]

    # Normalise
    for i in range(vector.shape[0]):
        maxabs = np.nanmax(np.abs(vector[i]))
        vector[i] = safe_divide(vector[i], maxabs)

    return vector


def feature_to_q_keys(string):
    for i, fkey in enumerate(FEATURE_KEYS):
        string = string.replace(fkey, r"$q_{" + f"{i + 1}" + r"}$")
    return string


def get_shape(value):
    """
    Evaluate the shape of a given quantity and return a string.
    Vector = 1st order tensor.
    Tensor = 2nd order tensor.
    :param value: Arbitrary quantity.
    :return: String specifying the type.
    """

    # Find type
    if isinstance(value, int) or isinstance(value, float):
        return 'scalar'

    # If array find shape
    elif isinstance(value, np.ndarray):
        value_shape = value.shape
        if value_shape[0] == 1:  # Array of single value
            return 'single_scalar_array'
        elif value_shape[0] > 1 and len(value_shape) == 1:  # Array of scalar values
            return 'scalar_array'
        elif value_shape[0] > 1 and value_shape[1] == 1:  # Vector
            return 'vector'
        elif value_shape[0] > 1 and value_shape[0] != value_shape[1] and len(value_shape) == 2:  # Array of vectors
            return 'vector_array'
        elif value_shape[0] > 1 and value_shape[0] == value_shape[1] and len(value_shape) == 2:  # Tensor
            return 'tensor'
        elif value_shape[0] > 1 and value_shape[0] == value_shape[1] and len(value_shape) == 3:  # Array of tensors
            return 'tensor_array'
        else:
            assert False, 'Cannot handle ndarray of shape: %r' % value_shape
    else:
        assert False, 'Cannot handle value of type: %r' % type(value)


def safe_divide(a, b):
    """
    Safe division by zero. Used for normalised velocity gradient.
    Use with caution, no divide-by-zero warning!
    Handles 1D arrays and scalars.

    :return: 0 if b == 0, otherwise a/b
    """

    old_setup = np.seterr(all='raise')  # Numpy raises any error

    shape_a = get_shape(a)
    shape_b = get_shape(b)

    if shape_b == 'single_scalar_array':  # Equivalent for division
        shape_b = 'scalar'


    # Tensor by tensor
    if shape_a == 'tensor' and shape_b == 'tensor':
        ret = np.zeros(a.shape)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                try:
                    ret[i, j] = a[i, j] / b[i, j]
                except ZeroDivisionError:
                    ret[i] = 0  # Instead of nan
                except FloatingPointError:
                    ret[i, j] = 0  # Instead of nan

    # Tensor by scalar
    elif shape_a == 'tensor' and shape_b == 'scalar':
        ret = np.zeros(a.shape)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                try:
                    ret[i, j] = a[i, j] / b
                except ZeroDivisionError:
                    ret[i] = 0  # Instead of nan
                except FloatingPointError:
                    ret[i, j] = 0  # Instead of nan

    # Vector by vector
    elif shape_a == 'vector' and shape_b == 'vector':
        ret = np.zeros(a.shape)
        for i in range(a.shape[0]):
            try:
                ret[i] = a[i] / b[i]
            except ZeroDivisionError:
                ret[i] = 0  # Instead of nan
            except FloatingPointError:
                ret[i] = 0  # Instead of nan

    # Scalar_array by scalar_array
    elif shape_a == 'scalar_array' and shape_b == 'scalar_array':
        ret = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            try:
                ret[i] = a[i] / b[i]
            except ZeroDivisionError:
                ret[i] = 0
            except FloatingPointError:
                ret[i] = 0

    # Scalar_array by scalar
    elif shape_a == 'scalar_array' and shape_b == 'scalar':
        ret = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            try:
                ret[i] = a[i] / b
            except ZeroDivisionError:
                ret[i] = 0
            except FloatingPointError:
                ret[i] = 0

    # Scalar by scalar_array
    elif shape_a == 'scalar' and shape_b == 'scalar_array':
        ret = np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            try:
                ret[i] = a / b[i]
            except ZeroDivisionError:
                ret[i] = 0
            except FloatingPointError:
                ret[i] = 0

    # Scalar by scalar
    elif shape_a == 'scalar' and shape_b == 'scalar':
        try:
            ret = a / b
        except ZeroDivisionError:
            ret = 0
        except FloatingPointError:
            ret = 0

    else:
        assert False, 'ERROR in safe_divide: Cannot divide a by b with ' \
                      'shapes a: ' + str(shape_a) + ' and b: ' + str(shape_b)

    np.seterr(**old_setup)  # Default error settings
    return ret


def convert_dict_to_ndarray(*dictionaries):
    """
    Gather all ndarrays in each dictionary and convert into numpy array.
    Mostly used for features and labels.
    :param dictionaries: Dictionaries with N entries of shape [num_of_points].
    :return: Ndarray of arrays with shape [num_of_arrays, num_of_points].
    """

    array_list = []

    #  Loop all dicts
    for dictionary in dictionaries:
        # Loop all keys
        for key in dictionary.keys():
            # Skip non-ndarray types
            if not isinstance(dictionary[key], np.ndarray):
                continue
            # Append each item to a list
            array_list.append(dictionary[key])

    # Check non-uniform length between arrays
    for item in array_list:
        assert len(item) == len(array_list[0]), 'All arrays must have the same length'

    return np.vstack(array_list)  # .swapaxes(0, 1)


def time_decorator(func):
    def wrapper(*args, **kwargs):
        clock1 = timer()
        func_return = func(*args, **kwargs)
        exec_time = timer() - clock1
        print("{} executed in {}".format(func.__name__, exec_time))
        return func_return
    return wrapper


def get_datetime(return_string=False):
    """
    Return current datetime.
    :param return_string: Returns datetime string as YYYYMMDD_HHMMSS
    :return: yyyy, mm, dd, hh, mm, ss or string.
    """

    dt = datetime.now()
    year = dt.year
    month = dt.month
    day = dt.day

    hour = dt.hour
    minute = dt.minute
    second = dt.second

    l = [year, month, day, hour, minute, second]

    if return_string:
        return "{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}".format(*l)
    else:
        return l


def append_list_till_length(lst, length):
    while len(lst) < length:
        lst.append(lst[0])
    return lst


def get_profile_data(flowcase_object, var_key, coord_key, loc_coord):
    """
    Get values of the requested variable and coordinate for the
    location.
    :param flowcase_object: flowCase type.
    :param var_key: Any flowCase.flow_dict.keys()
    :param coord_key: Either 'x' or 'y'.
    :param loc_coord: Location of the profile.
    :return: List of 2 np.arrays for variable and coordinate.
    """

    # Find indexes in flat data
    loc_idx = find_index_for_location(flowcase_object, coord_key, loc_coord)

    # Extract profile data
    var_data = flowcase_object.flow_dict[var_key][loc_idx]
    coord_data = flowcase_object.flow_dict[coord_key][loc_idx]

    if coord_key == 'x' or coord_key == 'y+':
        return coord_data, var_data
    else:
        return var_data, coord_data


def find_index_for_location(flowcase_object, coord_key, loc_coord):
    """
    Find index of location in x or y that is closest to the requested
    location (loc_coord).
    :param flowcase_object: flowCase type.
    :param coord_key: Varying coordinate.
    :param loc_coord: Location for the fixed coordinate (!=coord_key).
    :return: List of indexes for requested location.
    """

    # For convenience
    nx = flowcase_object.nx
    ny = flowcase_object.ny
    num_of_points = flowcase_object.num_of_points
    flow_dict = flowcase_object.flow_dict

    loc_idx = list()
    idx_list = list()
    # Find indexes of all y closest to loc_coord
    if coord_key == 'x':
        all_loc = flow_dict['y']
        for row in range(nx):
            idx = np.abs(all_loc[row*ny:ny + row*ny] - loc_coord).argmin()
            loc_idx.append(idx + row*ny)
        assert (len(loc_idx) == nx), 'The amount of locations %r does not correspond to the ' \
                                      'number of grid points nx=%r' % (len(loc_idx), nx)

    # Find indexes of all x closest to requested location
    elif coord_key == 'y' or coord_key == 'y+':
        all_loc = flow_dict['x']
        for col in range(ny):
            idx = np.abs(all_loc[col:num_of_points:ny] - loc_coord).argmin()
            idx_list.append(idx)
            loc_idx.append(col + idx*ny)
        assert (len(loc_idx) == ny), 'The amount of locations %r does not correspond to the ' \
                                     'number of grid points ny=%r' % (len(loc_idx), ny)

        # If location is not the end of the data, continue searching (Can be mirrored for x_plots)
        if all([idxs != nx - 1 for idxs in idx_list]):
            loc_idx.reverse()  # Reverse list for properly connected line plots
            for col in range(ny):
                idx2 = np.abs(all_loc[col + (idx_list[col] + 1)*ny:num_of_points:ny] - loc_coord).argmin()
                loc_idx.append(col + (idx_list[col] + idx2 + 1) * ny)



    return loc_idx


def assemble_2nd_order_tensor(dictionary, keys):
    """
    Assemble a 2nd order tensor from its 9 components.
    :param keys: Dict-keys for the 9 components.
    :param dictionary: Requires all 9 terms of the tensor
    according to keys.
    :return: Tensor with shape [3, 3, num_of_points].
    """

    num_of_points = dictionary['nx']*dictionary['ny']
    tensor = np.zeros((3, 3, num_of_points))

    len_keys = len(keys)
    if len_keys > 3:
        num_of_col = int(len(keys)/3)  # Number of columns
        for i in range(3):
            for j in range(num_of_col):
                tensor[i, j] = dictionary[keys[num_of_col*i+j]]

    # Diagonal matrix, Trace only
    elif len_keys == 3:
        tensor[0, 0] = dictionary[keys[0]]
        tensor[1, 1] = dictionary[keys[1]]
        tensor[2, 2] = dictionary[keys[2]]

    return tensor


def convert_cylinder_to_cartesian(dictionary):
    """
    Convert velocity components, their gradients, the reynolds stress
    tensor and its budgets from a cylinder coordinate system to a
    cartesian system.
    :param dictionary: Dictionary of flow data.
    :return: 1: success.
    """

    # Define num of points in x and y direction
    dictionary['nx'] = dictionary['nt']  # Angular number of points
    dictionary['ny'] = dictionary['nr']  # Radial number of points

    # Assemble gradient components into tensor
    if all(cyl_grad_u_key in dictionary for cyl_grad_u_key in CYL_GRAD_U_KEYS) and 'grad_u_cyl' not in dictionary:
        dictionary['grad_u_cyl'] = assemble_2nd_order_tensor(dictionary, CYL_GRAD_U_KEYS)
    elif 'grad_u_cyl' in dictionary:
        pass
    else:
        print('WARNING No gradient data available for conversion cylinder to cartesian coordinates.')

    # Loop all keys in dictionary
    for cyl_key in tuple(dictionary):
        if isinstance(dictionary[cyl_key], (np.ndarray, int, float, str)):
            if cyl_key in CYL_VEL_KEYS:
                velocity_cylinder_to_cartesian(dictionary, cyl_key, verbose=False)
            elif cyl_key == 'grad_u_cyl':
                gradient_cylinder_to_cartesian(dictionary, cyl_key)
            elif cyl_key in CYL_TAU_KEYS:
                tau_components_cylinder_to_cartesian(dictionary, cyl_key, verbose=False)  # Requires off-diagonal elements in tau, for now just RENAMING
            elif cyl_key in CYL_ALL_BUDGET_KEYS:
                budget_tau_components_cylinder_to_cartesian(dictionary, cyl_key, verbose=False)  # Just RENAMING budgets!
            elif cyl_key in CYL_RMS_KEYS:
                rms_velocity_cylinder_to_cartesian(dictionary, cyl_key, verbose=False)  # Just RENAMING rms, requires off-diag tau for conversion
            else:
                pass
        else:
            assert False, 'Unknown variable type in dictionary for key %r with type %r' % (cyl_key, type(dictionary[cyl_key]))

    return 1


def velocity_cylinder_to_cartesian(dictionary, cyl_vel_key, verbose=False):
    """
    Convert velocity components from a cylinder to cartesian
    coordinate system.
    :param verbose: Verbose output to console.
    :param dictionary: Dictionary of flow data.
    :param cyl_vel_key: Dictionary key for the velocity component
    to convert.
    :return: 1:success.
    """

    # For convenience
    t = dictionary['t']
    urm = dictionary['urm']
    utm = dictionary['utm']
    usm = dictionary['usm']

    # Find cartesian key
    key_idx = CYL_VEL_KEYS.index(cyl_vel_key)  # Find index in Key list
    car_vel_key = VEL_KEYS[key_idx]

    # Convert velocity
    if car_vel_key == 'um':
        dictionary[car_vel_key] = urm*np.cos(t) - utm*np.sin(t)
    elif car_vel_key == 'vm':
        dictionary[car_vel_key] = urm*np.sin(t) + utm*np.cos(t)
    elif car_vel_key == 'wm':
        dictionary[car_vel_key] = usm
    else:
        assert False, 'Invalid key mapping from cylinder to cartesian coordinates: %r' % car_vel_key

    # Verbose output
    if verbose:
        print('Velocity component ' + str(cyl_vel_key) + ' converted to ' + str(car_vel_key))

    return 1


def gradient_cylinder_to_cartesian(dictionary, grad_key):
    """
    Convert the gradients wrt cylinder coordinates into gradients
    wrt cartesian and transform the vector components for each
    derivative.
    Nabla_rtz u_rtz > Nabla_xyz u_xyz
    :param grad_key: Key of the gradient to transform.
    :param dictionary: Dictionary of flow data.
    :param grad_key: Key in dictionary for respective gradient
    component.
    :return: 1:success.
    """

    # Transform gradient tensor wrt to cylinder coord into grad wrt to cartesian coordinates
    gradient_tensor_cylinder_to_cartesian(dictionary, grad_key)

    # Transform each gradient from cylinder components to cartesian components
    gradient_vector_components_cylinder_to_cartesian(dictionary)

    return 1


def gradient_tensor_cylinder_to_cartesian(dictionary, grad_key):
    """
    Convert the gradient tensor wrt cylinder coordinates into
    a gradient tensor wrt cartesian coordinates.
    Nabla_xyz = Nabla_rtz * Jac
    :param dictionary: Dictionary of flow data.
    :param grad_key: Key in dictionary for respective gradient.
    :return: 1:success.
    """

    # Check input dictionary keys
    assert len(grad_key) > 1, 'Invalid amount of keys for transformation. Should be 1 key but %r given' % (grad_key)

    # For convenience
    nt = dictionary['nt']
    nr = dictionary['nr']

    tensor = dictionary[grad_key]

    x = dictionary['x']
    y = dictionary['y']
    r = dictionary['r']  # Radius
    # t = dictionary['t']  # Theta
    z = np.zeros(nr*nt)

    # Compute exact Jacobi
    jac11 = safe_divide(x, r)      # dr/dx
    jac12 = safe_divide(y, r)      # dr/dy
    jac21 = safe_divide(-y, r**2)  # dt/dx
    jac22 = safe_divide(x, r**2)   # dt/dy
    jac33 = np.ones(nr*nt)         # dz/dz
    jac = np.array([[jac11, jac12, z],
                    [jac21, jac22, z],
                    [z, z, jac33]])

    # Transform gradient and matrix for linalg methods
    jac = np.moveaxis(jac, 2, 0)
    tensor = np.moveaxis(tensor, 2, 0)

    # Apply coordinate transformation
    new_tensor = np.matmul(jac, tensor)

    # Reverse transformation for linalg-methods
    new_tensor = np.moveaxis(new_tensor, 0, 2)

    # Copy into dictionary
    dictionary['grad_u_cyl_wrt_car'] = new_tensor

    return 1


def gradient_vector_components_cylinder_to_cartesian(dictionary):
    """
    Convert vector components for cylinder coordinates in the velocity
    gradient wrt to cartesian coordinates into vector components for
    cartesian coordinates.
    grad_xyz u_rtz to grad_xyz u_xyz
    :param dictionary: Dictionary of flow data.
    :return: 1:success.
    """

    # For convenience
    urm = dictionary['urm']
    utm = dictionary['utm']

    x = dictionary['x']
    y = dictionary['y']
    r = dictionary['r']
    t = dictionary['t']  # Theta

    a = safe_divide(y, r**2)
    b = safe_divide(x, r**2)

    vel_grad = dictionary['grad_u_cyl_wrt_car']

    dur_dx = vel_grad[0, 0]
    dur_dy = vel_grad[0, 1]
    dur_dz = vel_grad[0, 2]

    dut_dx = vel_grad[1, 0]
    dut_dy = vel_grad[1, 1]
    dut_dz = vel_grad[1, 2]

    dus_dx = vel_grad[2, 0]
    dus_dy = vel_grad[2, 1]
    dus_dz = vel_grad[2, 2]

    # Transform u derivatives components
    du_dx = dur_dx*np.cos(t) - dut_dx*np.sin(t) + a*(urm*np.sin(t) + utm*np.cos(t))
    du_dy = dur_dy*np.cos(t) - dut_dy*np.sin(t) + b*(-urm*np.sin(t) - utm*np.cos(t))
    du_dz = dur_dz*np.cos(t) - dut_dz*np.sin(t)

    # Transform v derivatives components
    dv_dx = dur_dx*np.sin(t) + dut_dx*np.cos(t) + a*(-urm*np.cos(t) + utm*np.sin(t))
    dv_dy = dur_dy*np.sin(t) + dut_dy*np.cos(t) + b*(urm*np.cos(t) - utm*np.sin(t))
    dv_dz = dur_dz*np.sin(t) + dut_dz*np.cos(t)

    # Transform w derivatives components
    dw_dx = dus_dx
    dw_dy = dus_dy
    dw_dz = dus_dz

    # Assemble new gradient tensor
    new_grad = np.array([[du_dx, du_dy, du_dz],
                         [dv_dx, dv_dy, dv_dz],
                         [dw_dx, dw_dy, dw_dz]])

    # Save in dictionary
    dictionary['grad_um'] = new_grad

    return 1


def tau_components_cylinder_to_cartesian(dictionary, cyl_tau_key, verbose=False):
    """
    Convert tau components from a cylinder to cartesian coordinate
    system.
    :param verbose: Verbose output to console.
    :param dictionary: Dictionary of flow data.
    :param cyl_tau_key: Dictionary key for the tau component to
    convert.
    :return: 1:success, -1:No conversion, only rename isotropic stresses.
    """

    # For convenience
    t = dictionary['t']
    urur = dictionary['urur']
    utut = dictionary['utut']
    usus = dictionary['usus']

    # Check for off-diagonal component
    if 'urut' in dictionary:
        urut = dictionary['urut']
    else:
        print('WARNING Conversion of tauij impossible without off-diagonal components. Missing urut')
        print('Variables have only been renamed: urur=uu, utut=vv, usus=ww')
        dictionary['uu'] = urur
        dictionary['vv'] = utut
        dictionary['ww'] = usus
        return -1

    # Find cartesian key
    key_idx = CYL_TAU_KEYS.index(cyl_tau_key)  # Find index in Key list
    car_tau_key = TAU_KEYS[key_idx]

    # Convert tau components
    if car_tau_key == 'uu':
        dictionary[car_tau_key] = urur*np.cos(t)**2 - 2*urut*np.sin(t)*np.cos(t) + utut*np.sin(t)**2
    elif car_tau_key == 'vv':
        dictionary[car_tau_key] = urur*np.sin(t)**2 + 2*urut*np.sin(t)*np.cos(t) + utut*np.cos(t)**2
    elif car_tau_key == 'ww':
        dictionary[car_tau_key] = usus
    elif car_tau_key == 'uv':
        dictionary[car_tau_key] = urur*np.sin(t)*np.cos(t) + urut*(1-2*np.sin(t)**2) - utut*np.sin(t)*np.cos(t)
    elif car_tau_key == 'uw':
        assert False, 'Not implemented yet.'
    elif car_tau_key == 'vw':
        assert False, 'Not implemented yet.'
    else:
        assert False, 'Invalid key mapping from cylinder to cartesian coordinates: %r' % car_tau_key

    # Verbose output
    if verbose:
        print('Tau component ' + str(cyl_tau_key) + ' converted to ' + str(car_tau_key))

    return 1


def budget_tau_components_cylinder_to_cartesian(dictionary, cyl_bud_key, verbose=False):
    """
    Rename tau budgets from cylinder to cartesian.
    :param verbose: Verbose output to console.
    :param dictionary: Dictionary of flow data.
    :param cyl_bud_key: Dictionary key for the tau budget to rename.
    :return: 1:success.
    """

    # Find cartesian key
    key_idx = CYL_ALL_BUDGET_KEYS.index(cyl_bud_key)  # Find index in Key list
    car_bud_key = ALL_BUDGET_KEYS[key_idx]

    # Save same value under new name
    dictionary[car_bud_key] = dictionary[cyl_bud_key]

    # Verbose output
    if verbose:
        print('Tau component ' + str(cyl_bud_key) + ' converted to ' + str(car_bud_key))

    return 1


def rms_velocity_cylinder_to_cartesian(dictionary, cyl_rms_key, verbose=False):
    """
    Rename rms of velocity components.
    :param verbose: Verbose output to console.
    :param dictionary: Dictionary of flow data.
    :param cyl_rms_key: Dictionary key for the rms component to rename.
    :return: 1:success.
    """

    # Find cartesian key
    key_idx = CYL_RMS_KEYS.index(cyl_rms_key)  # Find index in Key list
    car_rms_key = RMS_KEYS[key_idx]

    # Save same value under new name
    dictionary[car_rms_key] = dictionary[cyl_rms_key]

    # Verbose output
    if verbose:
        print('RMS of velocity component ' + str(cyl_rms_key) + ' converted to ' + str(car_rms_key))

    return 1


# Not in use
def gradient_wrt_cylinder_to_wrt_cartesian(dictionary, grad_keys):
    """
    Convert the gradients wrt cylinder coordinates into gradients
    wrt cartesian.
    :param dictionary: Dictionary of flow data.
    :param grad_keys: Key in dictionary for respective gradient
    component.
    :return: 1:success.
    """

    # Check input dictionary keys
    len_keys = len(grad_keys)
    assert len_keys > 3, 'Invalid amount of gradient components. Maximum is 3 but %r were given' % (len_keys)

    # For convenience
    nt = dictionary['nt']
    nr = dictionary['nr']

    # For scalar gradients
    if len_keys == 1:
        gradient = dictionary[grad_keys]

    # Assemble gradient from three keys
    else:
        gradient = np.array([dictionary[grad_keys[0]],
                             dictionary[grad_keys[1]],
                             dictionary[grad_keys[2]]])


    x = dictionary['x']
    y = dictionary['y']
    r = dictionary['r']  # Radius
    # t = dictionary['t']  # Theta
    z = np.zeros(nr*nt)

    # Compute exact Jacobi
    jac11 = safe_divide(x, r)      # dr/dx
    jac12 = safe_divide(y, r)      # dr/dy
    jac21 = safe_divide(-y, r**2)  # dt/dx
    jac22 = safe_divide(x, r**2)   # dt/dy
    jac33 = np.ones(nr*nt)         # dz/dz
    jac = np.array([[jac11, jac12, z],
                    [jac21, jac22, z],
                    [z, z, jac33]])

    # Transform gradient and matrix for linalg methods
    jac = np.moveaxis(jac, 2, 0)
    gradient = np.moveaxis(gradient, 2, 0)
    gradient = gradient[:, :, np.newaxis]

    # Apply coordinate transformation
    trans_gradient = np.matmul(jac, gradient)

    # Reverse transformation for linalg-methods
    trans_gradient = np.swapaxes(trans_gradient.squeeze(), 0, 1)

    return 1


def gradient_wrt_cartesian_to_wrt_cylinder(dictionary, grad_keys):
    """
    Convert the gradients wrt cartesian coordinates into cylindrical.
    :param dictionary: Dictionary of flow data.
    :param grad_keys: Key in dictionary for respective gradient or
    gradient components.
    :return: 1:success.
    """

    # Check input dictionary keys
    len_keys = len(grad_keys)
    assert len_keys > 3, 'Invalid amount of gradient components. Maximum is 3 but %r were given' % (len_keys)

    # Evaluate dictionary key mapping
    if grad_keys in GRAD_U2_KEYS:
        car_keys = GRAD_U2_KEYS
        cyl_keys = CYL_GRAD_U2_KEYS
    elif grad_keys in GRAD_U_NORM_KEYS:
        car_keys = GRAD_U2_KEYS
        cyl_keys = CYL_GRAD_U2_KEYS
    elif grad_keys in GRAD_U_KEYS:
        car_keys = GRAD_U2_KEYS
        cyl_keys = CYL_GRAD_U2_KEYS
    elif grad_keys in GRAD_KEYS:
        car_keys = GRAD_KEYS
        cyl_keys = CYL_GRAD_KEYS
    else:
        assert False, 'Cannot handle given gradient keys: %r' % grad_keys

    # For convenience
    nx = dictionary['nx']
    ny = dictionary['ny']

    # For scalar gradients
    if len_keys == 1:
        gradient = dictionary[grad_keys]

    # Assemble gradient from three keys
    else:
        gradient = np.array([dictionary[grad_keys[0]],
                             dictionary[grad_keys[1]],
                             dictionary[grad_keys[2]]])

    # x = dictionary['x']
    # y = dictionary['y']
    r = dictionary['r']  # Radius
    t = dictionary['t']  # Theta
    z = np.zeros(nx*ny)

    # Compute exact Jacobi
    jac11 = np.cos(t)       # dx/dr
    jac12 = np.sin(t)       # dy/dr
    jac21 = -r*np.sin(t)    # dx/dt
    jac22 = r*np.cos(t)     # dy/dt
    jac33 = np.ones(nx*ny)  # dz/dz
    jac = np.array([[jac11, jac12, z],
                    [jac21, jac22, z],
                    [z, z, jac33]])

    # Transform gradient and matrix for linalg methods
    jac = np.moveaxis(jac, 2, 0)
    gradient = np.moveaxis(gradient, 2, 0)
    gradient = gradient[:, :, np.newaxis]

    # Apply coordinate transformation
    trans_gradient = np.matmul(jac, gradient)

    # Reverse transformation for linalg-methods
    trans_gradient = np.swapaxes(trans_gradient.squeeze(), 0, 1)

    # Map keys from cartesian to cylinder
    for i, grad_key in enumerate(grad_keys):
        grad_key_idx = car_keys.index(grad_key)
        new_key = cyl_keys[grad_key_idx]
        if len_keys == 1:
            dictionary[new_key] = trans_gradient
        elif len_keys > 1:
            dictionary[new_key] = trans_gradient[i]

    return 1





# def cylinder_to_cartesian_keys(cyl_dict):
#     """
#     Map all keys in given dictionary to cartesian. No values are changed.
#     This if for calculation purposes only.
#     :param cyl_dict: Dictionary of flow data with cylinder coordinates, velocityies, budgets.
#     :return: 1: success.
#     """
#
#     # Loop all cylinder-coordinate-based variables that need renaming
#     for cyl_key in tuple(cyl_dict):
#         if isinstance(cyl_dict[cyl_key], np.ndarray) or isinstance(cyl_dict[cyl_key], int) or isinstance(cyl_dict[cyl_key], float):
#             if cyl_key in CYL_ALL_KEYS:
#                 # Find index in Key list
#                 cyl_key_idx = CYL_ALL_KEYS.index(cyl_key)
#                 car_key = ALL_KEYS[cyl_key_idx]
#
#                 # Save with cartesian name
#                 cyl_dict[car_key] = cyl_dict[cyl_key]
#
#                 # Remove cylinder coordinate variable
#                 del cyl_dict[cyl_key]
#             else:
#                 pass
#         else:
#             assert False, 'Unknown variable type in dictionary for key %r with type %r' % (cyl_key, type(cyl_dict[cyl_key]))
#
#     return 1


# Transforms cartesian gradients into cylinder gradients, not working atm
# def gradient_cylinder_to_cartesian(dictionary, grad_keys):
#     """
#     Convert the gradients wrt cylinder coordinates into cartesian.
#     :param dictionary: Dictionary of flow data.
#     :param grad_keys: Key in dictionary for respective gradient or
#     gradient components.
#     :return: 1:success.
#     """
#
#     # Check input dictionary keys
#     len_keys = len(grad_keys)
#     assert len_keys > 3, 'Invalid amount of gradient components. Maximum is 3 but %r were given' % (len_keys)
#
#     # Evaluate dictionary key mapping
#     if grad_keys in GRAD_U2_KEYS:
#         car_keys = GRAD_U2_KEYS
#         cyl_keys = CYL_GRAD_U2_KEYS
#     elif grad_keys in GRAD_U_NORM_KEYS:
#         car_keys = GRAD_U2_KEYS
#         cyl_keys = CYL_GRAD_U2_KEYS
#     elif grad_keys in GRAD_U_KEYS:
#         car_keys = GRAD_U2_KEYS
#         cyl_keys = CYL_GRAD_U2_KEYS
#     elif grad_keys in GRAD_KEYS:
#         car_keys = GRAD_KEYS
#         cyl_keys = CYL_GRAD_KEYS
#     else:
#         assert False, 'Cannot handle given gradient keys: %r' % grad_keys
#
#     # For convenience
#     nt = dictionary['nt']
#     nr = dictionary['nr']
#
#     # For scalar gradients
#     if len_keys == 1:
#         gradient = dictionary[grad_keys]
#
#     # Assemble gradient from three keys
#     else:
#         gradient = np.array([dictionary[grad_keys[0]],
#                              dictionary[grad_keys[1]],
#                              dictionary[grad_keys[2]]])
#
#
#     x = dictionary['x']
#     y = dictionary['y']
#     r = dictionary['r']  # Radius
#     # t = dictionary['t']  # Theta
#     z = np.zeros(nr*nt)
#
#     # Compute exact Jacobi
#     jac11 = safe_divide(x, r)      # dr/dx
#     jac12 = safe_divide(y, r)      # dr/dy
#     jac21 = safe_divide(-y, r**2)  # dt/dx
#     jac22 = safe_divide(x, r**2)   # dt/dy
#     jac33 = np.ones(nr*nt)         # dz/dz
#     jac = np.array([[jac11, jac12, z],
#                     [jac21, jac22, z],
#                     [z, z, jac33]])
#
#     # Transform gradient and matrix for linalg methods
#     jac = np.moveaxis(jac, 2, 0)
#     gradient = np.moveaxis(gradient, 2, 0)
#     gradient = gradient[:, :, np.newaxis]
#
#     # Apply coordinate transformation
#     trans_gradient = np.matmul(jac, gradient)
#
#     # Reverse transformation for linalg-methods
#     trans_gradient = np.swapaxes(trans_gradient.squeeze(), 0, 1)
#
#     # Map keys from cylinder to cartesian
#     for i, grad_key in enumerate(grad_keys):
#         grad_key_idx = cyl_keys.index(grad_key)
#         new_key = car_keys[grad_key_idx]
#         if len_keys == 1:
#             dictionary[new_key] = trans_gradient
#         elif len_keys > 1:
#             dictionary[new_key] = trans_gradient[i]
#
#     return 1


# def get_grid_line_data(data_dict, data_key, coord_key, grid_idx):
#     """
#     Read all data points along the given coordinate for the specified location on the grid.
#     :param data_key: Key for the requested data.
#     :param data_dict: Dictionary of flow data.
#     :param coord_key: Key for the coordinate.
#     :param grid_idx: Location in terms of grid points.
#     :return: Ndarray of requested data.
#     """
#
#     # For convenience
#     nx = data_dict['nx']
#     ny = data_dict['ny']
#
#     # Find index in flat array of requested data and return coordinate and variable data
#     if coord_key == 'x':
#         loc_idx = [grid_idx + i*ny for i in range(nx)]
#         line_data = data_dict[data_key][loc_idx]
#         coord_data = data_dict[coord_key][loc_idx]
#         return coord_data, line_data
#
#     elif coord_key == 'y':
#         loc_idx = [i + grid_idx*nx for i in range(ny)]
#         line_data = data_dict[data_key][loc_idx]
#         coord_data = data_dict[coord_key][loc_idx]
#         return line_data, coord_data
#
#     else:
#         assert False, 'Invalid coord_key'
