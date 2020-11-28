# ###################################################################
# module gradient
#
# Description
# The module supplies functions to compute the gradient of scalar
# or vector quantities on a two-dimensional grid.
# The method is based on a coordinate transformation from a given
# orthogonal grid onto a uniform grid.
#
# ###################################################################
# Author: hw
# created: 07. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import sys

from uncert_ident.utilities import safe_divide, \
    VEL_KEYS, GRAD_U_KEYS, GRAD_U2_KEYS, GRAD_U_NORM_KEYS, \
    CYL_VEL_KEYS


#####################################################################
### Functions
#####################################################################
def gradient_2d_with_transformation(var_2d, x_2d, y_2d, coord_sys='cartesian'):
    """
    Compute gradient of var_2d with respect to cartesian
    coordinates (x, y) using a coordinate transformation onto a
    uniform grid with coordinates (xi, eta).

    :param coord_sys: Activate alterations for specific coordinate
    systems.
    :param var_2d: 2D matrix of differentiable variable
    :param x_2d: 2D matrix of horizontal coordinates for each grid point
    :param y_2d: "" vertical ""
    :return: List of two 2D matrices with gradients wrt x, y
    """

    # Manipulate data for 2nd order accurate derivatives at theta=2pi
    if coord_sys == 'cylinder':
        x_2d, y_2d, var_2d = add_cylinder_data(x_2d, y_2d, var_2d)

    # Determine gradients wrt uniform grid
    dx_dxi, dx_deta = np.gradient(x_2d)  # x_2d[i][j]: i=physically horizontal direction, j=physically vertical direction
    dy_dxi, dy_deta = np.gradient(y_2d)
    du_dxi, du_deta = np.gradient(var_2d)

    # Construct stacked gradient and jacobi matrix
    grad_u_xi_eta = np.array([du_dxi.flatten(),
                              du_deta.flatten()])
    jac = np.array([[dx_dxi.flatten(), dx_deta.flatten()],
                    [dy_dxi.flatten(), dy_deta.flatten()]])

    # Modify for linalg-methods
    jac = np.moveaxis(jac, 2, 0)
    grad_u_xi_eta = np.moveaxis(grad_u_xi_eta, 1, 0)
    grad_u_xi_eta = grad_u_xi_eta[:, :, np.newaxis]

    # Invert jacobi, matrix multiply with gradient
    inv_jac = np.linalg.inv(jac)
    grad_u_x_y = np.matmul(inv_jac.swapaxes(1, 2), grad_u_xi_eta)

    # Reverse modification for linalg-methods
    grad_u_x_y = np.swapaxes(grad_u_x_y.squeeze(), 0, 1)

    if coord_sys == 'cylinder':
        grad_u_x_y = remove_cylinder_data(x_2d.shape, grad_u_x_y)

    return grad_u_x_y


def velocity_gradient(data_dict, coord_sys='cartesian', exponent=1, normalise=False):
    """
    Compute 2D gradient of mean velocity field.
    :param coord_sys: Respective coordinate system for the gradients.
    :param data_dict: Dictionary of flow data. Requires mean
    velocity components and grid coordinates.
    :param exponent: Optional choice for gradient of quadratic or
    cubic velocity.
    :param normalise: Optional normalisation with velocity magnitude.
    :return: 2D Gradient of mean velocity with shape [3, 3, nx, ny]
    """

    # Set keys wrt coordinate system
    vel_keys = VEL_KEYS if coord_sys == 'cartesian' else CYL_VEL_KEYS

    # Check if third velocity component is available
    if vel_keys[-1] in data_dict:
        pass
    else:
        vel_keys = vel_keys[:-1]

    # For convenience
    nx = data_dict['nx']
    ny = data_dict['ny']
    num_of_points = nx*ny

    # Initialise arrays
    tensor_grad_vel = np.zeros((3, 3, num_of_points))
    norm_factor = np.array([1])

    # Normalisation with velocity magnitude (norm of u)
    if normalise:
        norm_factor = 0
        for vel_key in vel_keys:
            norm_factor += data_dict[vel_key]**2
        norm_factor = norm_factor**0.5

    # Compute gradients for each velocity component
    for i, key in enumerate(vel_keys):
        tensor_grad_vel[i, 0:2, :] = \
            gradient_2d_with_transformation(safe_divide(data_dict[key]**exponent, norm_factor).reshape(nx, ny),
                                            data_dict['x'].reshape(nx, ny),
                                            data_dict['y'].reshape(nx, ny))

    # Requires definition of grad_keys for respective
    # gradient (normalised, with exponent, nominal)
    # Save individual gradients in dictionary
    # for i in range(3):
    #     for j in range(3):
    #         key = grad_keys[3*i+j]
    #         data_dict[key] = tensor_grad_vel[i, j, :]

    return tensor_grad_vel


def scalar_gradient(data_dict, scalar_key, coord_sys='cartesian'):
    """
    Compute gradient of scalar field in 2D.
    :param coord_sys: Respective coordinate system for the gradients.
    :param data_dict: Dictionary of flow data. Requires
    scalar field corresponding to scalar_key and grid coordinates.
    :param scalar_key: Key in data_dict for scalar field.
    :return: 1: success.
    """

    assert(scalar_key in data_dict), "data_dict does not contain scalar_key: %r" % scalar_key

    # Consider coordinate system
    nx = data_dict['nx'] if coord_sys == 'cartesian' else data_dict['nt']
    ny = data_dict['ny'] if coord_sys == 'cartesian' else data_dict['nr']

    num_of_points = nx*ny
    gradient = np.zeros((3, num_of_points))

    gradient[0:2, :] = gradient_2d_with_transformation(data_dict[scalar_key].reshape(nx, ny),
                                                       data_dict['x'].reshape(nx, ny),
                                                       data_dict['y'].reshape(nx, ny))
    # Save gradient in dictionary directly
    # gradient_key = 'grad_' + scalar_key
    # data_dict[gradient_key] = gradient

    # Conversion to cylinder coordinates, not used
    # if coord_sys == 'cylinder':
    #     gradient_cartesian_to_cylinder(data_dict, gradient_key)

    return gradient


def add_cylinder_data(x_matrix, y_matrix, var_matrix):
    """
    Correct gradient accuracy for cylinder coordinates by adding the
    0 degree data at the end of the matrices and 359 degree at the
    beginning.
    :param var_matrix: 2D matrix of variable data.
    :param x_matrix: 2D matrix of x-coordinates.
    :param y_matrix: 2D matrix of y-coordinates.
    :return: Corrected x- and y-matrix.
    """

    # Generate new matrices with increased size
    new_x_matrix = np.zeros((x_matrix.shape[0] + 2, x_matrix.shape[1]))
    new_y_matrix = np.zeros((x_matrix.shape[0] + 2, x_matrix.shape[1]))
    new_var_matrix = np.zeros((x_matrix.shape[0] + 2, x_matrix.shape[1]))

    # Add 359 degree to beginning and 0 degree to end
    new_x_matrix[0, :] = x_matrix[-1, :]  # 359 degrees
    new_x_matrix[-1, :] = x_matrix[0, :]  # 0 degree
    new_x_matrix[1:-1, :] = x_matrix  # Full data

    # Same for y coordinates
    new_y_matrix[0, :] = y_matrix[-1, :]  # 359 degrees
    new_y_matrix[-1, :] = y_matrix[0, :]  # 0 degree
    new_y_matrix[1:-1, :] = y_matrix  # Full data

    # And for variable
    new_var_matrix[0, :] = var_matrix[-1, :]  # 359 degrees
    new_var_matrix[-1, :] = var_matrix[0, :]  # 0 degree
    new_var_matrix[1:-1, :] = var_matrix  # Full data

    return new_x_matrix, new_y_matrix, new_var_matrix


def remove_cylinder_data(x_shape, gradient):
    """
    Reverse changes from add_cylinder_data. Removes the doubled data
    at beginning and end of array for gradients.
    :param x_shape: Shape of 2D matrices.
    :param gradient: Flat gradient vector.
    :return: Corrected gradient.
    """

    # Remove additional data at beginning and end of array
    gradient = gradient.reshape(2, *x_shape)
    new_gradient = gradient[:, 1:-1, :]
    new_gradient = new_gradient.reshape(2, -1)  # Transform to flat vector

    return new_gradient




# VELOCITY GRADIENT FOR COMPUTATION OF CYLINDRICAL GRADIENTS
# def velocity_gradient(data_dict, coord_sys, exponent=1, normalise=False):
#     """
#     Compute 2D gradient of mean velocity field.
#     :param coord_sys: Respective coordinate system for the gradients.
#     :param data_dict: Dictionary of flow data. Requires mean
#     velocity components and grid coordinates.
#     :param exponent: Optional choice for gradient of quadratic or
#     cubic velocity.
#     :param normalise: Optional normalisation with velocity magnitude.
#     :return: 2D Gradient of mean velocity with shape [3, 3, nx, ny]
#     """
#
#     # Set keys wrt coordinate system
#     vel_keys = VEL_KEYS if coord_sys == 'cartesian' else CYL_VEL_KEYS
#
#     # Set keys wrt type of gradient
#     if normalise:
#         grad_keys = GRAD_U_NORM_KEYS
#     elif exponent == 2:
#         grad_keys = GRAD_U2_KEYS
#     else:
#         grad_keys = GRAD_U_KEYS
#
#     # Check if third velocity component is available
#     if vel_keys[-1] in data_dict:
#         pass
#     else:
#         vel_keys = vel_keys[:-1]
#
#     # For convenience
#     nx = data_dict['nx']
#     ny = data_dict['ny']
#     num_of_points = nx*ny
#
#     # Initialise arrays
#     tensor_grad_vel = np.zeros((3, 3, num_of_points))
#     norm_factor = np.array([1])
#
#     # Normalisation with velocity magnitude (norm of u)
#     if normalise:
#         norm_factor = 0
#         for vel_key in vel_keys:
#             norm_factor += data_dict[vel_key]**2
#         norm_factor = norm_factor**0.5
#
#     # Compute gradients for each velocity component
#     for i, key in enumerate(vel_keys):
#         tensor_grad_vel[i, 0:2, :] = \
#             gradient_2d_with_transformation(safe_divide(data_dict[key]**exponent, norm_factor).reshape(nx, ny),
#                                             data_dict['x'].reshape(nx, ny),
#                                             data_dict['y'].reshape(nx, ny))
#
#     # Save individual gradients in dictionary
#     for i in range(3):
#         for j in range(3):
#             key = grad_keys[3*i+j]
#             data_dict[key] = tensor_grad_vel[i, j, :]
#
#     # Transform into gradients wrt cylinder coordinates
#     if coord_sys == 'cylinder':
#         gradient_cartesian_to_cylinder(data_dict, grad_keys[0:3])  # du
#         gradient_cartesian_to_cylinder(data_dict, grad_keys[3:6])  # dv
#         gradient_cartesian_to_cylinder(data_dict, grad_keys[6:])  # dw
#
#     return 1
