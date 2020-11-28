# ###################################################################
# module features
#
# Description
# Methods for physical features and invariants based on
# minimum integrity basis for mean-strain- and -rotation-rate tensor,
# pseudotensor for pressure gradient and pseudotensor for tke gradient.
#
# ###################################################################
# Author: hw
# created: 07. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from numpy import trace, dot, cross, abs, identity
from numpy.linalg import norm

from uncert_ident.utilities import safe_divide, PHYSICAL_KEYS, time_decorator, normalise_max_abs, num_of_invariants
from uncert_ident.methods.geometry import get_lower_boundary_y

#####################################################################
### Functions
#####################################################################
# Convenience function (really convenient?)
def n_vector_l2_norm(vector):
    """
    Compute L2 norm for n vectors with 3 dimensions.

    :param vector: Array of shape [3, nx*ny].
    :return: L2 norm for each vector.
    """

    return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5


def n_matrix_fro_norm(matrix):
    """
    Compute Frobenius norm for n matrices with 3x3 dimensions.

    :param matrix: Array of shape [3, 3, nx*ny].
    :return: Frobenius norm for each matrix.
    """

    num_of_points = matrix.shape[-1]
    norm = np.zeros(num_of_points)

    for i in range(num_of_points):
        norm[i] = (trace(dot(matrix[:, :, i],  matrix[:, :, i])))**0.5

    return norm


# Invariants
def construct_antisymmetric_grad_p_grad_k(data_dict):
    """
    Construct antisymmetric pseudotensors for gradient
    of pressure and turbulent kinetic energy.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    # For convenience
    num_of_points = data_dict['num_of_points'][0]
    grad_pm = data_dict['grad_pm']
    grad_k = data_dict['grad_k']

    # Pseudotensor = -I x grad
    negID = -identity(3) + 0
    Pij = np.zeros((3, 3, num_of_points))
    Kij = np.zeros((3, 3, num_of_points))
    for i in range(num_of_points):
        Pij[:, :, i] = cross(negID, grad_pm[:, i])
        Kij[:, :, i] = cross(negID, grad_k[:, i])

    # Safe back into dictionary
    data_dict['Pij'] = Pij
    data_dict['Kij'] = Kij

    return 1


def normalise_Sij_Wij_Pij_Kij(data_dict):
    """
    Normalise the Tensors according to Wu et al. (2018), table 1.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    # Check availability of tensors
    assert data_dict.keys() >= {'Sij', 'Wij', 'Pij', 'Kij'}, 'Tensors could not be found, ' \
                                                             'call construct_antisymmetric_grad_p_grad_k()'

    # For convenience
    num_of_points = data_dict['num_of_points'][0]

    Sij = data_dict['Sij']
    Wij = data_dict['Wij']
    Pij = data_dict['Pij']
    Kij = data_dict['Kij']

    tke = data_dict['k']
    eps = data_dict['diss_rt']
    rho = data_dict['rho']
    try:
        vel = np.array([
            data_dict['um'],
            data_dict['vm'],
            data_dict['wm']
        ])
    except KeyError:
        vel = np.array([
            data_dict['um'],
            data_dict['vm'],
            np.zeros_like(data_dict['um'])
        ])
    grad_um = data_dict['grad_um']

    # Normalise each matrix
    for i in range(num_of_points):
        s = Sij[:, :, i]
        w = Wij[:, :, i]
        p = Pij[:, :, i]
        k = Kij[:, :, i]
        rho_mat_deriv = rho * norm(dot(grad_um[:, :, i], vel[:, i]))  # Norm of material derivative multiplied by rho

        Sij[:, :, i] = safe_divide(s, abs(s) + abs(safe_divide(eps[i], tke[i])))
        Wij[:, :, i] = safe_divide(w, abs(w) + abs(safe_divide(eps[i], tke[i])))#>Pope vs. norm(w))>Wu et al.
        Pij[:, :, i] = safe_divide(p, abs(p) + abs(rho_mat_deriv))
        Kij[:, :, i] = safe_divide(k, abs(k) + abs(safe_divide(eps[i], np.sqrt(tke[i]))))

    # Safe back into dictionary
    data_dict['Sij_norm'] = Sij
    data_dict['Wij_norm'] = Wij
    data_dict['Pij_norm'] = Pij
    data_dict['Kij_norm'] = Kij

    return 1


def get_inv(data_dict):
    """
    Compute minimal integrity basis for the four normalised
    tensors Sij, Wij, Pij, Kij according to Handbook of Fluid Dynamics
    p. A-37. Also, see Wu et al. (2018).
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    # Check avialability of normalised tensors
    assert data_dict.keys() >= {'Sij_norm', 'Wij_norm', 'Pij_norm', 'Kij_norm'}

    # For convenience
    num_of_points = data_dict['num_of_points'][0]

    S = data_dict['Sij_norm']
    W = data_dict['Wij_norm']
    P = data_dict['Pij_norm']
    K = data_dict['Kij_norm']

    # Compute each possible invariant for ns=1, na=3
    inv = np.zeros((num_of_invariants, num_of_points))
    for i in range(num_of_points):
        s = S[:, :, i]
        w = W[:, :, i]
        p = P[:, :, i]
        k = K[:, :, i]

        # num_of_symmetric, num_of_antisymmetric
        # 1,0
        # inv[0, :] = trace(s) == 0
        inv[0, i] = trace(dot(s, s))
        inv[1, i] = trace(dot(s, dot(s, s)))

        # 0,1
        inv[2, i] = trace(dot(w, w))
        inv[3, i] = trace(dot(p, p))
        inv[4, i] = trace(dot(k, k))

        # 0,2
        inv[5, i] = trace(dot(w, p))
        inv[6, i] = trace(dot(w, k))
        inv[7, i] = trace(dot(p, k))

        # 0,3
        inv[8, i] = trace(dot(w, dot(p, k)))

        # 1,1
        inv[9, i] = trace(dot(w, dot(w, s)))
        inv[10, i] = trace(dot(w, dot(w, dot(s, s))))
        inv[11, i] = trace(dot(w, dot(w, dot(s, dot(w, dot(s, s))))))

        inv[12, i] = trace(dot(p, dot(p, s)))
        inv[13, i] = trace(dot(p, dot(p, dot(s, s))))
        inv[14, i] = trace(dot(p, dot(p, dot(s, dot(p, dot(s, s))))))

        inv[15, i] = trace(dot(k, dot(k, s)))
        inv[16, i] = trace(dot(k, dot(k, dot(s, s))))
        inv[17, i] = trace(dot(k, dot(k, dot(s, dot(k, dot(s, s))))))

        # 1,2
        inv[18, i] = trace(dot(w, dot(p, s)))
        inv[19, i] = trace(dot(w, dot(p, dot(s, s))))
        inv[20, i] = trace(dot(w, dot(w, dot(p, s))))
        inv[21, i] = trace(dot(p, dot(p, dot(w, s))))
        inv[22, i] = trace(dot(w, dot(w, dot(p, dot(s, s)))))
        inv[23, i] = trace(dot(p, dot(p, dot(w, dot(s, s)))))
        inv[24, i] = trace(dot(w, dot(w, dot(s, dot(p, dot(s, s))))))
        inv[25, i] = trace(dot(p, dot(p, dot(s, dot(w, dot(s, s))))))

        inv[26, i] = trace(dot(w, dot(k, s)))
        inv[27, i] = trace(dot(w, dot(k, dot(s, s))))
        inv[28, i] = trace(dot(w, dot(w, dot(k, s))))
        inv[29, i] = trace(dot(k, dot(k, dot(w, s))))
        inv[30, i] = trace(dot(w, dot(w, dot(k, dot(s, s)))))
        inv[31, i] = trace(dot(k, dot(k, dot(w, dot(s, s)))))
        inv[32, i] = trace(dot(w, dot(w, dot(s, dot(k, dot(s, s))))))
        inv[33, i] = trace(dot(k, dot(k, dot(s, dot(w, dot(s, s))))))

        inv[34, i] = trace(dot(p, dot(k, s)))
        inv[35, i] = trace(dot(p, dot(k, dot(s, s))))
        inv[36, i] = trace(dot(p, dot(p, dot(k, s))))
        inv[37, i] = trace(dot(k, dot(k, dot(p, s))))
        inv[38, i] = trace(dot(p, dot(p, dot(k, dot(s, s)))))
        inv[39, i] = trace(dot(k, dot(k, dot(p, dot(s, s)))))
        inv[40, i] = trace(dot(p, dot(p, dot(s, dot(k, dot(s, s))))))
        inv[41, i] = trace(dot(k, dot(k, dot(s, dot(p, dot(s, s))))))

        # 1,3
        inv[42, i] = trace(dot(w, dot(p, dot(k, s))))
        inv[43, i] = trace(dot(w, dot(k, dot(p, s))))
        inv[44, i] = trace(dot(w, dot(p, dot(k, dot(s, s)))))
        inv[45, i] = trace(dot(w, dot(k, dot(p, dot(s, s)))))
        inv[46, i] = trace(dot(w, dot(p, dot(s, dot(k, dot(s, s))))))

    return inv


def compute_minimal_integrity_basis_Sij_Wij_Pij_Kij(data_dict):
    """
    Compute all invariants for the minimal integrity basis of the
    2nd order tensors Sij, Wij, Pij, Kij (see also get_inv()).
    :param data_dict: Dictionary of flow data.
    :return: Array of invariants with shape [num_of_features, num_of_points].
    """

    # Check avialability of tensors
    assert data_dict.keys() >= {'Sij', 'Wij', 'grad_pm', 'grad_k', 'k', 'diss_rt', 'rho', 'grad_um', 'um', 'vm'}

    # compute normalised features
    construct_antisymmetric_grad_p_grad_k(data_dict)
    normalise_Sij_Wij_Pij_Kij(data_dict)
    inv = get_inv(data_dict)
    inv = normalise_max_abs(inv)

    # Transform invariants' array into dict
    inv_dict = dict()
    for i in range(num_of_invariants):
        key = "inv{:02d}".format(i)
        inv_dict[key] = inv[i]

    return inv_dict


# Physical features
def q_criterion(data_dict, normalise=True):
    """
    Compute Q-Criterion. Equation from Wang et al. (2017).

    :param data_dict: Dictionary of flow data. Requires
    mean-strain and rotation-rate tensors.
    :param normalise: Optional normalisation for features.
    :return: Vector of Q-Criterion.
    """

    num_of_points = data_dict['nx']*data_dict['ny']
    raw_feature = np.zeros(num_of_points)
    norm_factor = np.zeros(num_of_points)

    # For convenience
    Sij = data_dict['Sij']
    Wij = data_dict['Wij']

    for i in range(num_of_points):
        norm_factor[i] = trace(dot(Sij[:, :, i], Sij[:, :, i]))
        raw_feature[i] = 0.5*(trace(dot(Wij[:, :, i], Wij[:, :, i])) -
                              norm_factor[i])

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def tke_feature(data_dict, normalise=True):
    """
    Compute normalised turbulent kinetic energy as feature.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires turbulent
    kinetic energy and mean velocity components.
    :param normalise: Optional normalisation for features.
    :return: Flat matrix of turbulent kinetic energy.
    """

    # For convenience
    k = data_dict['k']
    Sij = data_dict['Sij']
    nu = data_dict['nu'][0]
    # um = data_dict['um']
    # vm = data_dict['vm']
    #
    # if 'wm' in data_dict:
    #     wm = data_dict['wm']
    # else:
    #     wm = 0

    raw_feature = k

    # norm_factor = 0.5*(um**2 + vm**2 + wm**2)  Not Galilean invariant
    norm_factor = nu * np.sqrt([
        trace(dot(Sij[:, :, i], Sij[:, :, i])) for i in range(Sij.shape[2])
    ])

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def tke_feature2(data_dict, normalise=True):
    """
    Compute normalised turbulent kinetic energy as feature.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires turbulent
    kinetic energy and mean velocity components.
    :param normalise: Optional normalisation for features.
    :return: Flat matrix of turbulent kinetic energy.
    """

    # For convenience
    k = data_dict['k']
    Sij = data_dict['Sij']
    nu = data_dict['nu'][0]
    um = data_dict['um']
    vm = data_dict['vm']

    if 'wm' in data_dict:
        wm = data_dict['wm']
    else:
        wm = 0

    raw_feature = k

    norm_factor = 0.5*(um**2 + vm**2 + wm**2)
    # norm_factor = nu * np.sqrt([
    #     trace(dot(Sij[:, :, i], Sij[:, :, i])) for i in range(Sij.shape[2])
    # ])

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def distance_nearest_wall(data_dict):
    """
    Compute distance to the nearest wall. Assume walls at
    indices [i, 0] and [i, -1], if reshaped(nx, ny).

    :param data_dict: Dictionary of flow data. Requires
    x- and y- coordinates.
    :return: Distance to nearest wall.
    """

    # For convenience
    nx = data_dict['nx']
    ny = data_dict['ny']
    x = data_dict['x'].reshape((nx, ny))
    y = data_dict['y'].reshape((nx, ny))
    d = np.zeros(x.shape)
    geo = data_dict['geometry']

    # Compare distance to lower and upper wall
    for i in range(nx):
        for j in range(ny):
            dx_lower = np.abs(x[i, j] - x[:, 0])
            dx_upper = np.abs(x[i, j] - x[:, -1])

            # dy_lower = np.abs(y[i, j] - y[:, 0])
            dy_lower = np.abs(y[i, j] - get_lower_boundary_y(data_dict, x[i, j]))
            dy_upper = np.abs(y[i, j] - y[:, -1])

            distance_lower = np.sqrt(dx_lower**2 + dy_lower**2)
            distance_upper = np.sqrt(dx_upper**2 + dy_upper**2)

            d[i, j] = np.min([np.min(distance_lower), np.min(distance_upper)])

    return d.reshape(-1)


def reynolds_number_wall_distance(data_dict, limiter=True):
    """
    Compute wall-distance Reynolds number.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires
    turbulent kinetic energy and distance to nearest wall.
    :param limiter: Limit the output i.e. min(Re_d, 2).
    :return: Wall-distance Reynolds number.
    """

    num_of_points = data_dict['nx']*data_dict['ny']

    # For convenience
    k = data_dict['k']
    geo = data_dict['geometry']
    d_wall = distance_nearest_wall(data_dict)
    nu = data_dict['nu']
    nu = np.mean(nu) if isinstance(nu, np.ndarray) else nu

    Re_d = np.sqrt(k)*d_wall/50/nu
    if limiter:
        Re_d = [2 if Re_i > 2 else Re_i for Re_i in Re_d]
        if geo in ['curved_backwards_facing_step', 'naca4412', 'naca0012', 'flat_plate']:
            Re_d = [2 if tke < 1e-3 else Re_i for Re_i, tke in zip(Re_d, k)]
        # Re_d = [2 if Re_d[i] > 2 else Re_d[i] for i in range(num_of_points)]

    return np.array(Re_d)


def pressure_gradient_along_streamline(data_dict, normalise=True):
    """
    Compute pressure gradient along a streamline.
    Equation from Wang et al. (2017).

    :param data_dict: Dictionary of flow data. Requires
    mean velocity vector and mean pressure gradient.
    :param normalise: Optional normalisation for features.
    :return: Vector of pressure gradient along a streamline.
    """

    # For convenience
    um = data_dict['um']
    vm = data_dict['vm']

    if 'wm' in data_dict:
        wm = data_dict['wm']
    else:
        wm = 0

    grad_pm = data_dict['grad_pm']

    raw_feature = (um * grad_pm[0] +
                   vm * grad_pm[1] +
                   wm * grad_pm[2])

    norm_grad_pm = n_vector_l2_norm(grad_pm)
    norm_factor = norm_grad_pm*(um**2 + vm**2 + wm**2)

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def turbulent_time_scale_to_mean_strain_rate(data_dict, normalise=True):
    """
    Compute ratio of turbulent kinetic energy to dissipation rate.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires isotropic
    reynolds stresses, the  dissipation rate and the mean-strain-rate
    tensor.
    :param normalise: Optional normalisation for features.
    :return: Vector of ratio.
    """

    # For convenience
    k = data_dict['k']
    diss_rt = data_dict['diss_rt']
    Sij = data_dict['Sij']

    raw_feature = safe_divide(k, diss_rt)

    norm_factor = safe_divide(1, n_matrix_fro_norm(Sij))

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def viscosity_ratio(data_dict, normalise=True):
    """
    Compute ratio of eddy to molecular viscosity.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires turbulent
    eddy viscosity.
    :param normalise: Optional normalisation for features.
    :return: Vector of viscosity ratio.
    """

    # For convenience
    nut = data_dict['nut']
    nu = data_dict['nu']
    nu = np.mean(nu) if isinstance(nu, np.ndarray) else nu

    raw_feature = nut

    norm_factor = 100*nu

    if normalise:
        return raw_feature / (np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def pressure_normal_to_shear_stress(data_dict, normalise=True):
    """
    Compute ratio of pressure normal stress to normal shear
    stresses. Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires pressure
    gradient and gradient of squared velocity.
    :param normalise: Optional normalisation for features.
    :return: Vector of ratio.
    """

    # For convenience
    grad_pm = data_dict['grad_pm']
    grad_um2 = data_dict['grad_um2']
    rho = data_dict['rho']

    raw_feature = n_vector_l2_norm(grad_pm)

    norm_factor = 0.5*rho*(grad_um2[0, 0] +
                               grad_um2[1, 1] +
                               grad_um2[2, 2])

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def orthogonality_velocity_with_gradient(data_dict, normalise=True):
    """
    Compute scalar product of local velocity along streamline and its
    gradient i.e. 0 if orthogonal. Equation from Gorle (2014).

    :param data_dict: Dictionary of flow data. Requires mean velocity
    and its gradient.
    :param normalise: Optional normalisation for features.
    :return: Vector of ratio.
    """

    num_of_points = data_dict['nx']*data_dict['ny']
    norm_factor = np.zeros(num_of_points)

    # For convenience
    um = data_dict['um']
    vm = data_dict['vm']

    if 'wm' in data_dict:
        wm = data_dict['wm']
    else:
        wm = 0

    grad_um = data_dict['grad_um']

    raw_feature = um**2*grad_um[0, 0, :] + \
                  um*vm*grad_um[0, 1, :] + \
                  um*wm*grad_um[0, 2, :] + \
                  vm*um*grad_um[1, 0, :] + \
                  vm**2*grad_um[1, 1, :] + \
                  vm*wm*grad_um[1, 2, :] + \
                  wm*um*grad_um[2, 0, :] + \
                  wm*vm*grad_um[2, 1, :] + \
                  wm**2*grad_um[2, 2, :]

    norm_factor = ((um**2 + vm**2 + wm**2)*(
            (um*grad_um[0, 0, :] +
            vm*grad_um[1, 0, :] +
            wm*grad_um[2, 0, :])**2 +
            (um*grad_um[0, 1, :] +
            vm*grad_um[1, 1, :] +
            wm*grad_um[2, 1, :])**2 +
            (um*grad_um[0, 2, :] +
            vm*grad_um[1, 2, :] +
            wm*grad_um[2, 2, :])**2
                                            ))**0.5

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def convection_to_production_tke(data_dict, normalise=True):
    """
    Compute ratio of convection to production of turbulent kinetic
    energy. Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires turbulent
    kinetic energy gradient, reynolds stress tensor, mean velocity and
    mean-strain-rate tensor.
    :param normalise: Optional normalisation for features.
    :return: Vector of ratio.
    """

    num_of_points = data_dict['nx']*data_dict['ny']
    norm_factor = np.zeros(num_of_points)

    # For convenience
    um = data_dict['um']
    vm = data_dict['vm']

    if 'wm' in data_dict:
        wm = data_dict['wm']
    else:
        wm = 0

    grad_k = data_dict['grad_k']
    tauij = data_dict['tauij']
    Sij = data_dict['Sij']

    raw_feature = um*grad_k[0] + \
                  vm*grad_k[1] + \
                  wm*grad_k[2]

    for i in range(num_of_points):
        norm_factor[i] = abs(trace(dot(tauij[:, :, i],
                                          Sij[:, :, i])))

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def total_to_normal_reynolds_stress(data_dict, normalise=True):
    """
    Compute ratio of total to normal reynolds stress.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires reynolds
    stress tensor.
    :param normalise: Optional normalisation for features. If False
    returns total reynolds stress.
    :return: Vector of ratio.
    """

    # For convenience
    tauij = data_dict['tauij']
    k = data_dict['k']

    raw_feature = n_matrix_fro_norm(tauij)

    norm_factor = k

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def cubic_eddy_viscosity_comparison(data_dict, normalise=True):
    """
    Compute feature for the comparison of eddy viscosity models.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires CEVM reynolds
    stress, linear reynolds stress and mean-strain-rate tensor.
    :param normalise: Optional normalisation for features.
    :return: Vector of ratio.
    """

    num_of_points = data_dict['nx']*data_dict['ny']
    norm_factor = np.zeros(num_of_points)
    raw_feature = np.zeros(num_of_points)

    # For convenience
    Sij = data_dict['Sij']
    tauij = data_dict['tauij']
    tauij_cubic = data_dict['tauij_cubic']

    for i in range(num_of_points):
        raw_feature[i] = trace(dot(Sij[:, :, i],
                                   tauij[:, :, i] - tauij_cubic[:, :, i]))
        norm_factor[i] = 2*trace(dot(Sij[:, :, i], tauij[:, :, i]))

    if normalise:
        return safe_divide(raw_feature, np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


def streamline_curvature(data_dict, normalise=True):
    """
    Compute streamline curvature. Equation from Wang et al. (2017).

    :param data_dict: Dictionary of flow data. Requires normalised
    mean velocity gradient, mean velocity and characteristic length.
    :param normalise: Optional normalisation for features.
    :return: Vector of ratio.
    """

    # For convenience
    um = data_dict['um']
    vm = data_dict['vm']

    if 'wm' in data_dict:
        wm = data_dict['wm']
    else:
        wm = 0

    grad_um_norm = data_dict['grad_um_norm']
    char_length = data_dict['char_length']

    x_component = um*grad_um_norm[0, 0, :] + \
                  vm*grad_um_norm[0, 1, :] + \
                  wm*grad_um_norm[0, 2, :]

    y_component = um*grad_um_norm[1, 0, :] + \
                  vm*grad_um_norm[1, 1, :] + \
                  wm*grad_um_norm[1, 2, :]

    z_component = um*grad_um_norm[2, 0, :] + \
                  vm*grad_um_norm[2, 1, :] + \
                  wm*grad_um_norm[2, 2, :]

    raw_feature = safe_divide((x_component**2 + y_component**2 + z_component**2)**0.5,
                              (um**2 + vm**2 + wm**2)**0.5)

    norm_factor = 1/char_length

    if normalise:
        return raw_feature / (np.abs(raw_feature) + np.abs(norm_factor))
    else:
        return raw_feature


# List of function handles for physical features, order must correspond to PHYSICAL_KEYS
PHYSICAL_FUNS = [
    q_criterion,
    tke_feature,
    tke_feature2,
    reynolds_number_wall_distance,
    turbulent_time_scale_to_mean_strain_rate,
    viscosity_ratio,
    orthogonality_velocity_with_gradient,
    # convection_to_production_tke,
    # total_to_normal_reynolds_stress,
    # cubic_eddy_viscosity_comparison,
    streamline_curvature,
    pressure_gradient_along_streamline,
    pressure_normal_to_shear_stress
]


def compute_physical_features(data_dict):
    """
    Calls each function for a physical features that is appended to
    PHYSICAL_FUN and saves it with corresponding PHYSICAL_KEY.
    :param data_dict: Dictionary of flow data.
    :return: Dictionary with physical features.
    """

    phy_feat_dict = dict()
    for fun, key in zip(PHYSICAL_FUNS, PHYSICAL_KEYS):
        # If pressure gradient is unknown, skip dependent features
        if key == 'grad_pm_stream' or key == 'pm_normal_shear_ratio':
            if 'grad_pm' in data_dict:
                phy_feat_dict[key] = fun(data_dict)
            else:
                pass
        else:
            phy_feat_dict[key] = fun(data_dict)

    test_physical_features(phy_feat_dict)

    return phy_feat_dict


@time_decorator
def compute_all_features(data_dict):
    """
    Call handling functions to compute either physical features according to
    Ling & Templeton (2015) and Wang (2017) or invariants according to
    Ling (2016a) and Wang (2017).
    :param data_dict: Dictionary of flow data.
    :return: Dictionary of all features.
    """

    physical_features = compute_physical_features(data_dict)
    invariant_features = compute_minimal_integrity_basis_Sij_Wij_Pij_Kij(data_dict)

    # Merge features into common dict
    all_features = dict()
    all_features.update(physical_features)
    all_features.update(invariant_features)

    # Test features
    test_all_features(all_features)

    return all_features


#####################################################################
### Tests
#####################################################################
def test_all_features(feature_dict, verbose=False):
    """
    Test all features to be within the normalised range [-1, 1].
    :param feature_dict: Dictionary of features.
    :param verbose: Optional, output range of each feature.
    :return: 1:success.
    """

    assert test_physical_features(feature_dict, verbose=verbose)
    assert test_invariant_features(feature_dict, verbose=verbose)

    return 1


def test_physical_features(data_dict, verbose=False):
    """
    Check physical features wrt normalisation. All features, except
    wall-distance Reynolds number, should be within [-1, 1].

    :param verbose: Print each features range.
    :param data_dict: Dictionary of flow data. Should contain features
    defined with PHYSICAL_KEYS.
    :return: 1:success. Print to console, if normalisation is exceeded.
    """

    # Loop all physical features,
    # check normalised limits within [-1, 1] and
    # inform if value reaches beyond limits
    for key in PHYSICAL_KEYS:
        if key in data_dict:
            maximum = np.round(np.max(data_dict[key]), 3)
            minimum = np.round(np.min(data_dict[key]), 3)
            if key == 'Re_d':
                if maximum > 2 or minimum < 0:
                    print("WARNING")
                    print("Feature " + str(key) + " exceeds limits.")
                    print(str(key) + " ranges from " + str(minimum) +
                          " to " + str(maximum) + "\n")

            elif maximum > 1 or minimum < -1:
                print("WARNING")
                print("Feature " + str(key) + " exceeds limits.")
                print(str(key) + " ranges from " + str(minimum) +
                      " to " + str(maximum) + "\n")

            elif verbose:
                print("Feature " + str(key) + " ranges from " +
                      str(minimum) + " to " + str(maximum) + "\n")

            else:
                pass

        else:
            print("WARNING")
            print("Feature " + str(key) + " not defined.\n")

    return 1


def test_invariant_features(data_dict, verbose=False):
    """
    Check invariant features for normalised range of [-1, 1].
    :param verbose: Print each features range.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    # Loop all invariants,
    # check normalised limits within [-1, 1] and
    # inform if value reaches beyond limits
    for key in ["inv{:02d}".format(i) for i in range(num_of_invariants)]:
        if key in data_dict:
            maximum = np.round(np.max(data_dict[key]), 3)
            minimum = np.round(np.min(data_dict[key]), 3)
            if maximum > 1 or minimum < -1:
                print("WARNING")
                print("Feature " + str(key) + " exceeds limits.")
                print(str(key) + " ranges from " + str(minimum) +
                      " to " + str(maximum) + "\n")

            elif verbose:
                print("Feature " + str(key) + " ranges from " +
                      str(minimum) + " to " + str(maximum) + "\n")

            else:
                pass

        else:
            print("WARNING")
            print("Feature " + str(key) + " not defined.\n")

    return 1

