# ###################################################################
# module turbulence_properties
#
# Description
# Methods for the computation of quantities that describe and define
# turbulent motion in fluid flow.
#
# ###################################################################
# Author: hw
# created: 07. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from numpy import trace, dot
from numpy.linalg import eigvalsh
from scipy import optimize


from uncert_ident.utilities import safe_divide, time_decorator, \
    assemble_2nd_order_tensor, convert_cylinder_to_cartesian, \
    VEL_KEYS, RMS_KEYS, TAU_KEYS, GRAD_U_KEYS, GRAD_U_NORM_KEYS, GRAD_U2_KEYS, GRAD_U_KEYS_STAT_2D, DISS_KEYS,\
    CYL_VEL_KEYS, CYL_RMS_KEYS, CYL_TAU_KEYS, CYL_GRAD_U_KEYS, CYL_GRAD_U_NORM_KEYS, CYL_GRAD_U2_KEYS
from uncert_ident.methods import gradient as grad


#####################################################################
### Constants
#####################################################################
C_MU = 0.09  # According to Ling (k-eps)


#####################################################################
### Functions
#####################################################################
# Handling functions
def check_coordinate_system(data_dict, coord_sys):
    """
    Transform from any coordinate system into a cartesian system.
    :param data_dict: Dictionary of flow data.
    :param coord_sys: String according to below if-conditions.
    :return: 1:success.
    """

    # Check coordinate systems
    if coord_sys == 'cylinder':
        convert_cylinder_to_cartesian(data_dict)
        pass
    elif coord_sys == 'cartesian':
        pass
    else:
        assert False, 'Can only handle coordinate systems \"cartesian\" or \"polar\". Invalid coord_sys: %r' % (coord_sys)

    return 1


def get_reynolds_stress_tensor(data_dict, coord_sys='cartesian'):
    """
    Evaluate reynolds stress components and construct the
    2nd order tensor.
    :param data_dict: Dictionary of flow data.
    :param coord_sys: Choose keys for corresponding coordinate system.
    :return: 1: success.
    """

    # Computation with cylinder components possible
    tau_keys = TAU_KEYS if coord_sys == 'cartesian' else CYL_TAU_KEYS

    # Filter low/noise turbulent kinetic energy
    # data_dict['k']


    # All reynolds stress components given
    if all(tau_key in data_dict for tau_key in tau_keys):
        data_dict['tauij'] = assemble_2nd_order_tensor(data_dict, tau_keys)

    # Only diagonal components, eventually some of-diagonals too
    elif all(diag_key in data_dict for diag_key in tau_keys[::4]):
        data_dict['tauij'] = reynolds_stress_tensor(data_dict)

    # No tau components, but rms known
    elif all(rms_key in data_dict for rms_key in RMS_KEYS):
        data_dict['uu'], \
        data_dict['vv'], \
        data_dict['ww'] = isotropic_stress_from_rms(data_dict)
        data_dict['tauij'] = reynolds_stress_tensor(data_dict)

    else:
        assert False, 'Could not find tau components in data_dict'

    return 1


def get_turbulent_kinetic_energy(data_dict, coord_sys='cartesian'):
    """
    Evaluate turbulent kinetic energy.
    :param data_dict: Dictionary of flow data.
    :param coord_sys: Choose keys for corresponding coordinate system.
    :return: 1: success.
    """

    # Computation with cylinder components possible
    tau_keys = TAU_KEYS if coord_sys == 'cartesian' else CYL_TAU_KEYS
    tau_keys = tau_keys[::4]  # Only isotropic components

    if 'k' in data_dict:
        pass
    else:
        data_dict['k'] = turbulent_kinetic_energy(data_dict, tau_keys)

    data_dict['grad_k'] = grad.scalar_gradient(data_dict, 'k', coord_sys)

    return 1


def get_velocity_gradient(data_dict, coord_sys='cartesian'):
    """
    Evaluate velocity gradients.
    :param data_dict: Dictionary of flow data.
    :param coord_sys: Choose keys for corresponding coordinate system.
    :return: 1: success.
    """

    # Find respective dictionary keys
    if coord_sys == 'cylinder':
        grad_u_keys = CYL_GRAD_U_KEYS
        # grad_u_key = 'grad_u_cyl'
        # grad_u_norm_keys = CYL_GRAD_U_NORM_KEYS
        # grad_u_norm_key = 'grad_u_norm_cyl'
        # grad_u2_keys = CYL_GRAD_U2_KEYS
        # grad_u2_key = 'grad_u2_cyl'
    else:
        grad_u_keys = GRAD_U_KEYS
        # grad_u_key = 'grad_um'
        # grad_u_norm_keys = GRAD_U_NORM_KEYS
        # grad_u_norm_key = 'grad_u_norm'
        # grad_u2_keys = GRAD_U2_KEYS
        # grad_u2_key = 'grad_u2'

    # All Gradient components given
    if all(gradient_key in data_dict for gradient_key in grad_u_keys):
        data_dict['grad_um'] = assemble_2nd_order_tensor(data_dict, grad_u_keys)

    # Statistically 2D flow (Does not work with cylinder coordinates atm)
    elif all(gradient_key in data_dict for gradient_key in GRAD_U_KEYS_STAT_2D):
        data_dict['grad_um'] = assemble_2nd_order_tensor(data_dict, GRAD_U_KEYS_STAT_2D)

    # Approximate gradient of u
    elif 'grad_um' not in data_dict:
        data_dict['grad_um'] = grad.velocity_gradient(data_dict, coord_sys)
        # data_dict[grad_u_key] = assemble_2nd_order_tensor(data_dict, grad_u_keys)
    else:
        pass

    # Approximate gradient of u/norm(u)
    data_dict['grad_um_norm'] = grad.velocity_gradient(data_dict, coord_sys, normalise=True)
    # data_dict[grad_u_norm_key] = assemble_2nd_order_tensor(data_dict, grad_u_norm_keys)

    # Approximate gradient of u**2
    data_dict['grad_um2'] = grad.velocity_gradient(data_dict, coord_sys, exponent=2)
    # data_dict[grad_u2_key] = assemble_2nd_order_tensor(data_dict, grad_u2_keys)

    return 1


def get_pressure_quantities(data_dict, coord_sys='cartesian'):
    """
    Compute the pressure gradient in cartesian coordinates, if mean
    pressure field is known.
    :param data_dict: Dictionary of flow data.
    :param coord_sys: Choose keys for corresponding coordinate system.
    :return: 1:success.
    """

    # Check for mean pressure field
    if 'pm' in data_dict:
        data_dict['grad_pm'] = grad.scalar_gradient(data_dict, 'pm', coord_sys=coord_sys)
    else:  #TODO Pressure reconstruction with OpenFOAM
        print('WARNING in get_pressure_quantities: No mean pressure found, hence no pressure gradient.')

    if 'p_rms' in data_dict:
        data_dict['pp'] = pressure_fluctuation_from_rms(data_dict)

    return 1


def get_strain_rotation_rate_tensor(data_dict):
    """
    Compute the mean-strain and -rotation-rate tensor from the
    velocity gradient.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    data_dict['Sij'], \
    data_dict['Wij'] = strain_rotation_rate(data_dict)

    # Investigation on mass conversation and accuracy of data:
    nx = data_dict['nx']
    ny = data_dict['ny']
    data_dict['trSij'] = np.zeros(nx*ny)
    for i in range(nx*ny):
        data_dict['trSij'][i] = trace(data_dict['Sij'][:, :, i])

    return 1


def get_linear_eddy_viscosity(data_dict):
    """
    Determine the turbulent eddy viscosity based on Boussinesq's
    hypothesis i.e. a linear eddy viscosity model.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    data_dict['nut'] = turbulent_eddy_viscosity(data_dict)

    # Investigation on mass conversation i.e. tr(Sij)=0?
    data_dict['nut2'] = turbulent_eddy_viscosity2(data_dict)
    data_dict['diff_nut'] = data_dict['nut']-data_dict['nut2']

    return 1


def get_dissipation_rate(data_dict):
    """
    Ideally assemble dissipation tensor and compute dissipation rate
    with its trace.
    If dissipation rate is known just continue.
    Else use k-omg model for epsilon.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    # Complete dissipation tensor available
    if all(dissipation_key in data_dict for dissipation_key in DISS_KEYS):
        data_dict['diss'] = assemble_2nd_order_tensor(data_dict, DISS_KEYS)
        data_dict['diss_rt'] = dissipation_rate(data_dict)

    # Only isotropic components available
    elif all(dissipation_key in data_dict for dissipation_key in DISS_KEYS[::4]):
        data_dict['diss'] = assemble_2nd_order_tensor(data_dict, DISS_KEYS[::4])
        data_dict['diss_rt'] = dissipation_rate(data_dict)
    elif 'diss_rt' in data_dict:
        pass
    else:
        data_dict['diss_rt'] = modeled_dissipation_rate(data_dict)

    return 1


def get_anisotropy(data_dict):
    """
    Compute anisotropy tensor, normalised tensor, invariants of the
    normalised tensor and coordinates in lumley's triangle.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    # Tensors
    data_dict['aij'] = anisotropy_tensor(data_dict)
    data_dict['bij'] = anisotropy_tensor(data_dict, normalise=True)

    # Eigenvalues
    data_dict['bij_eig1'], \
    data_dict['bij_eig2'], \
    data_dict['bij_eig3'] = anisotropy_eigenvalues(data_dict, normalise=True)

    # Invariants
    data_dict['IIb'] = anisotropy_2nd_invariant(data_dict)
    data_dict['IIIb'] = anisotropy_3rd_invariant(data_dict)

    # Lumley triangle coordinates
    data_dict['bij_eta'], \
    data_dict['bij_xi'] = anisotropy_invariants_to_eta_xi(data_dict)

    return 1


def get_near_wall_quantities(data_dict):
    """
    Compute near-wall quantities for turbulent boundary layers.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    # Check for kinematic viscosity and friction velocity
    if 'nu' in data_dict and 'u_tau' in data_dict:
        data_dict['delta_nu'] = viscous_length_scale(data_dict)
        data_dict['y+'] = y_plus(data_dict, offset=data_dict['wall_offset'])
        data_dict['u+'] = u_plus(data_dict)

    # Check for 99% BL thickness and friction reynolds number
    elif 'delta99' in data_dict and 'Re_tau' in data_dict:
        data_dict['delta_nu'] = viscous_length_scale(data_dict)
        data_dict['y+'] = y_plus(data_dict, offset=data_dict['wall_offset'])
        print('No viscosity or friction velocity found in data.')

    # Check for friction velocity only
    elif 'u_tau' in data_dict:
        data_dict['u+'] = u_plus(data_dict)
        print('No viscosity nu found in data.')

    else:
        print('No near-wall quantities could be computed.')

    return 1


@time_decorator
def get_cubic_eddy_viscosity(data_dict):
    """
    Compute a cubic eddy viscosity and the modelled reynolds stress
    tensor according to the model of Craft et al.
    :param data_dict: Dictionary of flow data.
    :return: 1:success.
    """

    data_dict['nut_cubic'] = cubic_eddy_viscosity(data_dict)
    data_dict['tauij_cubic'] = reynolds_stress_cubic_eddy_viscosity_model(data_dict)

    return 1


# Turbulence functions
def isotropic_stress_from_rms(data_dict, coord_sys='cartesian'):
    """
    Get the diagonal components of tau (tauii) from rms values.
    :param coord_sys: Set coordinate system.
    :param data_dict: Dictionary of flow data.
    :return: Tuple of 3 components for success.
    """

    # For convenience
    if coord_sys == 'polar':
        u_rms = data_dict['ur_rms']
        v_rms = data_dict['ut_rms']
    else:
        u_rms = data_dict['u_rms']
        v_rms = data_dict['v_rms']

    # Check for w component
    if 'w_rms' in data_dict and 'wm' in data_dict:
        if coord_sys == 'polar':
            w_rms = data_dict['us_rms']
        else:
            w_rms = data_dict['w_rms']
    else:
        w_rms = 0

    # Extract tauii
    uu = u_rms**2
    vv = v_rms**2
    ww = w_rms**2

    return uu, vv, ww


def pressure_fluctuation_from_rms(data_dict):
    """
    Get the mean-squared of the fluctuating pressure from rms data.
    :param data_dict: Dictionary of flow data.
    :return: pp.
    """
    # Check for fluctuating pressure
    if 'p_rms' in data_dict:
        p_rms = data_dict['p_rms']
    else:
        p_rms = 0

    # Extract pp
    pp = p_rms**2

    return pp


def turbulent_kinetic_energy(data_dict, tau_keys):
    """
    Compute turbulent kinetic energy. Equation from Pope (2000).
    :param tau_keys: Keys in dictionary for isotropic reynolds stresses.
    :param data_dict: Dictionary of flow data. Requires isotropic
    reynolds stresses.
    :return: Turbulent kinetic energy.
    """

    # Isotropic stresses
    k = 0.5*(data_dict[tau_keys[0]] +
             data_dict[tau_keys[1]] +
             data_dict[tau_keys[2]])
    k[k < 1e-10] = 0

    return k


def reynolds_stress_tensor(data_dict):
    """
    Compute reynolds stress tensor.

    :param data_dict: Dictionary of flow data. Requires
    isotropic reynolds stresses.
    :return: Reynolds stress tensor.
    """

    num_of_points = data_dict['nx']*data_dict['ny']
    tau = np.zeros((3, 3, num_of_points))

    tau[0, 0, :] = data_dict['uu']
    tau[1, 1, :] = data_dict['vv']
    tau[2, 2, :] = data_dict['ww']

    for key, value in data_dict.items():
        if key == 'uv':
            tau[0, 1, :] = value
            tau[1, 0, :] = value

        if key == 'uw':
            tau[0, 2, :] = value
            tau[2, 0, :] = value

        if key == 'vw':
            tau[1, 2, :] = value
            tau[2, 1, :] = value

    return tau


def modeled_dissipation_rate(data_dict):
    """
    Compute dissipation rate. Equation from Wilcox (2006).

    :param data_dict: Dictionary of flow data. Requires isotropic
    reynolds stresses and eddy viscosity.
    :return: Dissipation rate.
    """

    return C_MU*data_dict['k']**2*data_dict['nut']


def dissipation_rate(data_dict):
    """
    Compute the dissipation rate from trace of dissipation
    tensor according to Kolmogorov's hypothesis of local isotropic
    dissipation at smallest scales.
    :param data_dict: Requires 6 terms of the dissipation tensor
    with keys defined according to DISS_KEYS.
    :return: Dissipation rate.
    """

    diss_rt = 0.5 * (data_dict['diss'][0, 0] +
                     data_dict['diss'][1, 1] +
                     data_dict['diss'][2, 2])

    return diss_rt


def turbulent_eddy_viscosity(data_dict):
    """
    Extract turbulent eddy viscosity from high-fidelity data.
    Equation from Ling and Templeton (2015).
    :param data_dict: Dictionary of flow data. Requires turbulent
    kinetic energy, reynolds stress tensor and mean strain-rate
    tensor.
    :return: Vector of turbulent eddy viscosity.
    """

    # For convenience
    num_of_points = data_dict['nx']*data_dict['ny']

    tauij = data_dict['tauij']
    Sij = data_dict['Sij']
    k = data_dict['k']

    nu_t_term1 = np.zeros(num_of_points)
    nu_t_term2 = np.zeros(num_of_points)
    nu_t_term3 = np.zeros(num_of_points)

    for i in range(num_of_points):
        nu_t_term1[i] = -1*trace(dot(tauij[:, :, i], Sij[:, :, i]))
        nu_t_term2[i] = 2/3*k[i]*trace(Sij[:, :, i])
        nu_t_term3[i] = 2*trace(dot(Sij[:, :, i], Sij[:, :, i]))

    return grad.safe_divide((nu_t_term1 + nu_t_term2), nu_t_term3)


def turbulent_eddy_viscosity2(data_dict):
    """
    Compute turbulent eddy viscosity. Equation from Ling and
    Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires turbulent
    kinetic energy, reynolds stress tensor and mean strain-rate
    tensor.
    :return: Vector of turbulent eddy viscosity.
    """

    # For convenience
    num_of_points = data_dict['nx']*data_dict['ny']

    tauij = data_dict['tauij']
    Sij = data_dict['Sij']

    nu_t_term1 = np.zeros(num_of_points)
    nu_t_term3 = np.zeros(num_of_points)

    for i in range(num_of_points):
        nu_t_term1[i] = -1*trace(dot(tauij[:, :, i], Sij[:, :, i]))
        nu_t_term3[i] = 2*trace(dot(Sij[:, :, i], Sij[:, :, i]))

    return grad.safe_divide(nu_t_term1, nu_t_term3)


def strain_rotation_rate(data_dict):
    """
    Compute mean-strain and -rotation-rate tensor. Equation from
    Pope (2000).

    :param data_dict: Dictionary of flow data. Requires gradient of
    averaged velocity.
    :return: Mean-strain-rate (Sij) and mean-rotation-rate (Wij)
    tensor with shape [3, 3, nx*ny].
    """

    # For convenience
    grad_um = data_dict['grad_um']

    num_of_points = data_dict['nx']*data_dict['ny']
    Sij = np.zeros((3, 3, num_of_points))
    Wij = np.zeros((3, 3, num_of_points))

    for i in range(num_of_points):
        Sij[:, :, i] = 0.5 * (grad_um[:, :, i] +
                              np.transpose(grad_um[:, :, i]))
        Wij[:, :, i] = 0.5 * (grad_um[:, :, i] -
                              np.transpose(grad_um[:, :, i]))

    return Sij, Wij


def anisotropy_tensor(data_dict, normalise=False):
    """
    Compute the anisotropic part aij of the reynolds stress
    tensor tauij.
    :param normalise: Optionally normalise by 2k, i.e. bij.
    :param data_dict: Dictionary of flow data.
    :return: 2nd order tensor aij of shape [3, 3, nx*ny].
    """

    # For convenience
    k = data_dict['k']
    tauij = data_dict['tauij']

    num_of_points = data_dict['nx']*data_dict['ny']
    dij = np.diag((1, 1, 1))

    # Check for off-diagonal elements in tauij
    if not check_tau_anisotropy(data_dict):
        print('WARNING Only isotropic stress known for anisotropy.')

    # Compute bij if normalised tensor is requested
    if normalise:
        # Check whether aij is known
        try:
            aij = data_dict['aij']
        except KeyError:
            aij = np.zeros((3, 3, num_of_points))
            for i in range(num_of_points):
                aij[:, :, i] = tauij[:, :, i] - 2/3*k[i]*dij
        # Compute bij from aij
        bij = np.zeros((3, 3, num_of_points))
        for i in range(num_of_points):
            bij[:, :, i] = safe_divide(aij[:, :, i], 2*k[i])
        return bij

    else:
        aij = np.zeros((3, 3, num_of_points))
        for i in range(num_of_points):
            aij[:, :, i] = tauij[:, :, i] - 2/3*k[i]*dij
        return aij


def anisotropy_eigenvalues(data_dict, normalise=False):
    """
    Compute the eigenvalues of the (normalised) anisotropy tensor.
    :param data_dict: Dictionary of flow data.
    :param normalise: If true, compute eigenvalues of bij.
    :return: Eigenvalues lambda1, lambda2, lambda3.
    """

    # Moveaxis for linalg-requested shape
    if normalise:
        eig = eigvalsh(np.moveaxis(data_dict['bij'], 2, 0))
    else:
        eig = eigvalsh(np.moveaxis(data_dict['aij'], 2, 0))

    eig = np.sort(eig, axis=1)  # Sort ascending
    eig = eig[:, ::-1]  # Reverse to descending
    eig = np.moveaxis(eig, 0, 1)

    return eig


def anisotropy_2nd_invariant(data_dict):
    """
    Compute second invariant IIb of the anisotropy tensor.
    Equation based on Pope (2000).

    :param data_dict: Dictionary of flow data. Requires normalised
    anisotropy tensor bij.
    :return: Second invariant of bij.
    """

    # For convenience
    num_of_points = data_dict['nx']*data_dict['ny']
    bij = data_dict['bij']

    # Compute second invariant
    IIb = np.zeros(num_of_points)
    for i in range(num_of_points):
        IIb[i] = -2 * (0.5*(np.trace(bij[:, :, i])**2 - np.trace(np.dot(bij[:, :, i], bij[:, :, i]))))

    # Alternative from eigenvalues
    # eig1 = data_dict['bij_eig1']
    # eig2 = data_dict['bij_eig2']
    # eig3 = data_dict['bij_eig3']
    # IIb = eig1*eig2 + eig2*eig3 + eig1*eig3

    # Change sign of negative zeros
    IIb[IIb == 0] = 0

    return IIb


def anisotropy_3rd_invariant(data_dict):
    """
    Compute third invariant IIIb of the anisotropy tensor.
    Equation based on Pope (2000).

    :param data_dict: Dictionary of flow data. Requires normalised
    anisotropy tensor bij.
    :return: Third invariant of bij.
    """

    # For convenience
    num_of_points = data_dict['nx']*data_dict['ny']
    bij = data_dict['bij']


    # Compute third invariant
    # IIIb = np.zeros(num_of_points)
    IIIb = 3*np.linalg.det(np.moveaxis(bij, 2, 0))

    # Alternative from eigenvalues
    # eig1 = data_dict['bij_eig1']
    # eig2 = data_dict['bij_eig2']
    # eig3 = data_dict['bij_eig3']
    # IIIb = eig1*eig2*eig3

    return IIIb


def anisotropy_invariants_to_eta_xi(data_dict):
    """
    Compute convenient coordinates for lumley's triangle.
    Equation based on Pope (2000).

    :param data_dict: Dictionary of flow data. Requires invariants of
    bij.
    :return: Both coordinates for invariants.
    """

    # Compute coordinates
    eta = np.sqrt(data_dict['IIb']/6)
    xi = np.cbrt(data_dict['IIIb']/6)

    return eta, xi


def check_tau_anisotropy(data_dict):
    """
    Check for off-diagonal components in reynolds stress tensor.
    :param data_dict: Dictionary of flow data.
    :return: 1:off-diagonal elements found. 0:No off-diagonals.
    """

    off_diag_keys = ['uv', 'uw', 'vw']

    if any([key in data_dict for key in off_diag_keys]):
        return 1
    else:
        return 0


def viscous_length_scale(data_dict):
    """
    Compute the length scale of the near-wall region in a turbulent
    boundary layer.
    :param data_dict: Dictionary of flow data.
    :return: Viscous length scale.
    """
    try:
        return data_dict['nu']/data_dict['u_tau']
    except KeyError:
        try:
            return data_dict['delta99'] / data_dict['Re_tau']
        except KeyError:
            print('Could not compute viscous length scale. -1 return instead.')
            return -1


def y_plus(data_dict, offset=0):
    """
    Compute the normalised near-wall distance y+.
    :param data_dict: Dictionary of flow data.
    :param offset: Offset in wall-normal direction, if y[0] != 0
    :return: Normalised near-wall distance.
    """

    # For convenience
    nx = data_dict['nx']
    ny = data_dict['ny']
    y = data_dict['y'].reshape(nx, ny)
    delta_nu = data_dict['delta_nu']

    # Enforce offset to be an ndarray
    if isinstance(offset, int) or isinstance(offset, float):
        offset = np.zeros(nx) + offset

    yp = np.zeros((nx, ny))
    for i in range(nx):
        yp[i, :] = (y[i, :] - offset[i])/delta_nu[i]

    return yp.ravel()


def u_plus(data_dict):
    """
    Compute the normalised near-wall velocity u+ i.e. mean velocity
    normalised by friction velocity.
    :param data_dict: Dictionary of flow data.
    :return: Normalised near-wall distance.
    """

    # For convenience
    nx = data_dict['nx']
    ny = data_dict['ny']
    um = data_dict['um'].reshape(nx, ny)
    u_tau = data_dict['u_tau']

    up = np.zeros((nx, ny))
    for i in range(nx):
        up[i, :] = um[i, :]/u_tau[i]

    return up.ravel()


def reynolds_stress_cubic_eddy_viscosity_model(data_dict):
    """
    Compute reynolds stress tauij based on a cubic eddy viscosity
    model. The particular model applied here is based on
    Craft's model, but with modifications by Ling & Templeton.
    Tensor version without multiplied Sij.
    Equation from Ling & Templeton (2015).

    :param data_dict: Dictionary of flow data. Requires turbulent
    kinetic energy, eddy viscosity,
    mean-strain- and -rotation-rate tensor.
    :return: Reynolds stress tensor for cubic eddy viscosity model.
    """

    num_of_points = data_dict['nx']*data_dict['ny']
    tauij_cubic = np.zeros((3, 3, num_of_points))

    # For convenience
    k = data_dict['k']
    Sij = data_dict['Sij']
    Wij = data_dict['Wij']
    nut_cubic = data_dict['nut_cubic']
    dij = np.diag((1, 1, 1))

    for i in range(num_of_points):
        # Linear tauij
        linear_stress = 2/3*k[i]*dij - 2*nut_cubic[i]*Sij[:, :, i]

        # Terms with nut**2
        c1_term = -0.4*(dot(Sij[:, :, i], Sij[:, :, i]) -
                        1/3*dij*trace(dot(Sij[:, :, i], Sij[:, :, i])))
        c2_term = 0.4*(dot(Wij[:, :, i], Sij[:, :, i]) +
                       np.transpose(dot(Wij[:, :, i], Sij[:, :, i])))
        c3_term = 1.04*(dot(Wij[:, :, i], Wij[:, :, i]) -
                        1/3*trace(dot(Wij[:, :, i], Wij[:, :, i]))*dij)
        nu2_terms = c1_term + c2_term + c3_term

        # Terms with nut**3
        c4_term = -80*(dot(Sij[:, :, i], dot(Sij[:, :, i], Wij[:, :, i])) +
                       np.transpose(dot(Sij[:, :, i], dot(Wij[:, :, i], Sij[:, :, i]))))
        c6_term = -40*Sij[:, :, i]*trace(dot(Sij[:, :, i], Sij[:, :, i]))
        c7_term = 40*Sij[:, :, i]*trace(dot(Wij[:, :, i], Wij[:, :, i]))
        nu3_terms = c4_term + c6_term + c7_term

        # Resulting tauij_cubic
        tauij_cubic[:, :, i] = linear_stress + \
                               safe_divide(nut_cubic[i]**2, C_MU*k[i])*nu2_terms + \
                               safe_divide(nut_cubic[i]**3, k[i]**2)*nu3_terms

    return tauij_cubic


def scalar_cubic_eddy_viscosity_model(nut, Sij, Wij, k, aijSij=0):
    """
    Compute anisotropic stress aij based on cubic eddy viscosity
    model. The particular model applied here is based on
    Craft's model, but with modifications by Ling & Templeton.
    Sij is applied to the equation to extract eddy viscoisty.
    Equation from Ling & Templeton (2015).

    :param nut: Eddy viscosity.
    :param Sij: Mean-strain-rate tensor with shape [3, 3].
    :param Wij: Mean-rotation-rate tensor with shape [3, 3].
    :param k: Turbulent kinetic energy.
    :param aijSij: Optional contracted anisotropic stress (aij Sij), if
    function is used for residual computation.
    LHS = left-hand side.
    :return: Either contracted anisotropic stress with mean-strain-rate
    tensor (aij Sij) or the residual of a cost function for a given
    eddy viscosity.
    """

    # Term with nut**1
    linear_term = -2*nut*trace(dot(Sij, Sij))

    # Terms with nut**2
    c1_term = -0.4*(trace(dot(Sij, dot(Sij, Sij))) -
                    1/3*trace(Sij)*trace(dot(Sij, Sij)))
    c2_term = 0.4*(trace(dot(Sij, dot(Wij, Sij))) +
                   trace(dot(Sij, np.transpose(dot(Wij, Sij)))))
    c3_term = 1.04*(trace(dot(Sij, dot(Wij, Wij))) -
                    1/3*trace(dot(Wij, Wij))*trace(Sij))
    nu2_terms = c1_term + c2_term + c3_term

    # Terms with nut**3
    c4_term = -80*(trace(dot(Sij, dot(Sij, dot(Sij, Wij)))) +
                   trace(dot(Sij, np.transpose(dot(Sij, dot(Wij, Sij))))))
    c6_term = -40*(trace(dot(Sij, Sij))*trace(dot(Sij, Sij)))
    c7_term = 40*trace(dot(Sij, Sij))*trace(dot(Wij, Wij))
    nu3_terms = c4_term + c6_term + c7_term


    cost_fun = linear_term + \
               safe_divide(nut**2, C_MU*k)*nu2_terms + \
               safe_divide(nut**3, k**2)*nu3_terms - \
               aijSij

    return cost_fun


def derivative_scalar_cubic_eddy_viscosity_model(nut, Sij, Wij, k, aijSij=0):
    """
    Derivative of the contracted model of craft wrt the eddy viscosity.
    :param nut: Eddy viscosity.
    :param Sij: Mean-strain-rate tensor with shape [3, 3].
    :param Wij: Mean-rotation-rate tensor with shape [3, 3].
    :param k: Turbulent kinetic energy.
    :param aijSij: Vanishes due to derivative.
    :return: Either contracted anisotropic stress with mean-strain-rate
    tensor (aij Sij) or the residual of a cost function for a given
    eddy viscosity.
    """

    # Terms without nut
    constant = -2*trace(dot(Sij, Sij))

    # Terms with nut
    c1_term = -0.4*(trace(dot(Sij, dot(Sij, Sij))) -
                    1/3*trace(Sij)*trace(dot(Sij, Sij)))
    c2_term = 0.4*(trace(dot(Sij, dot(Wij, Sij))) +
                   trace(dot(Sij, np.transpose(dot(Wij, Sij)))))
    c3_term = 1.04*(trace(dot(Sij, dot(Wij, Wij))) -
                    1/3*trace(dot(Wij, Wij))*trace(Sij))
    nu_terms = c1_term + c2_term + c3_term

    # Terms with nut**2
    c4_term = -80*(trace(dot(Sij, dot(Sij, dot(Sij, Wij)))) +
                   trace(dot(Sij, np.transpose(dot(Sij, dot(Wij, Sij))))))
    c6_term = -40*(trace(dot(Sij, Sij))*trace(dot(Sij, Sij)))
    c7_term = 40*trace(dot(Sij, Sij))*trace(dot(Wij, Wij))
    nu2_terms = c4_term + c6_term + c7_term


    cost_fun = constant + \
               safe_divide(2*nut, C_MU*k)*nu_terms + \
               safe_divide(3*nut**2, k**2)*nu2_terms

    return cost_fun


def second_derivative_scalar_cubic_eddy_viscosity_model(nut, Sij, Wij, k, aijSij=0):
    """
    Derivative of the contracted model of craft wrt the eddy viscosity.
    :param nut: Eddy viscosity.
    :param Sij: Mean-strain-rate tensor with shape [3, 3].
    :param Wij: Mean-rotation-rate tensor with shape [3, 3].
    :param k: Turbulent kinetic energy.
    :param aijSij: Vanishes due to derivative.
    :return: Either contracted anisotropic stress with mean-strain-rate
    tensor (aij Sij) or the residual of a cost function for a given
    eddy viscosity.
    """

    # Terms without nut
    c1_term = -0.4*(trace(dot(Sij, dot(Sij, Sij))) -
                    1/3*trace(Sij)*trace(dot(Sij, Sij)))
    c2_term = 0.4*(trace(dot(Sij, dot(Wij, Sij))) +
                   trace(dot(Sij, np.transpose(dot(Wij, Sij)))))
    c3_term = 1.04*(trace(dot(Sij, dot(Wij, Wij))) -
                    1/3*trace(dot(Wij, Wij))*trace(Sij))
    constant = c1_term + c2_term + c3_term

    # Terms with nut
    c4_term = -80*(trace(dot(Sij, dot(Sij, dot(Sij, Wij)))) +
                   trace(dot(Sij, np.transpose(dot(Sij, dot(Wij, Sij))))))
    c6_term = -40*(trace(dot(Sij, Sij))*trace(dot(Sij, Sij)))
    c7_term = 40*trace(dot(Sij, Sij))*trace(dot(Wij, Wij))
    nu_terms = c4_term + c6_term + c7_term


    cost_fun = constant + \
               safe_divide(2, C_MU*k)*constant + \
               safe_divide(6*nut, k**2)*nu_terms

    return cost_fun


def cubic_eddy_viscosity(data_dict):
    """
    The cubic eddy viscosity model (CEVM) is solved for the eddy viscosity.
    The model corresponds to Ling & Templeton's modification of
    Craft's original model.
    :param data_dict: Dictionary of flow data. Requires reynolds
    stress tensor, mean-strain- and -rotation-rate tensor and turbulent
    kinetic energy.
    :return: Eddy viscosity using the given model.
    """

    num_of_points = data_dict['nx']*data_dict['ny']
    nut_cubic = np.zeros(num_of_points)

    # For convenience
    k = data_dict['k']
    Sij = data_dict['Sij']
    Wij = data_dict['Wij']
    nut = data_dict['nut']
    tauij = data_dict['tauij']

    # Solve Craft's cubic eddy viscosity model by minimisation
    for i in range(num_of_points):
        nut0 = nut[i]  # Initial value = linear eddy viscosity

        # Contraction of aij with Sij
        aijSij = trace(dot(tauij[:, :, i], Sij[:, :, i])) - \
                 2/3*k[i]*trace(Sij[:, :, i])

        # Additional arguments for the model
        args_cubic = (Sij[:, :, i],
                      Wij[:, :, i],
                      k[i],
                      aijSij)

        # nut = 0 > nut_cubic = 0
        if nut0 == 0:
            nut_cubic[i] = 0
        else:
            nut_cubic[i] = optimize.newton(scalar_cubic_eddy_viscosity_model,
                                           nut0, args=args_cubic,
                                           fprime=derivative_scalar_cubic_eddy_viscosity_model,
                                           fprime2=second_derivative_scalar_cubic_eddy_viscosity_model,
                                           maxiter=50
                                           )

    return nut_cubic


def turbulence_properties(flow_data_dict, coord_sys='cartesian'):
    """
    Compute a number of quantities relevant for turbulent flow.
    :param flow_data_dict: Dictionary of flow data.
    :param coord_sys: Option if data is based on a different coordinate system.
    :return: 1: success.
    """

    # Transform all data to cartesian coordinates
    check_coordinate_system(flow_data_dict, coord_sys)

    # Reynolds stress tensor
    get_reynolds_stress_tensor(flow_data_dict)

    # Turbulent kinetic energy and gradient
    get_turbulent_kinetic_energy(flow_data_dict)

    # Velocity gradients
    get_velocity_gradient(flow_data_dict, coord_sys)

    # Pressure gradient
    get_pressure_quantities(flow_data_dict, coord_sys)

    # Mean-strain and -rotation-rate tensor
    get_strain_rotation_rate_tensor(flow_data_dict)

    # Linear eddy viscosity
    get_linear_eddy_viscosity(flow_data_dict)

    # Dissipation tensor and rate
    get_dissipation_rate(flow_data_dict)

    # Anisotropy tensor
    get_anisotropy(flow_data_dict)

    # Near-wall quantities
    get_near_wall_quantities(flow_data_dict)

    # Cubic eddy viscosity model (Massive computational costs, only for non-linearity metric)
    # get_cubic_eddy_viscosity(flow_data_dict)

    return 1





# OLD FUNCTION
# def get_velocity_gradient(data_dict, coord_sys='cartesian'):
#     """
#     Evaluate velocity gradients.
#     :param data_dict: Dictionary of flow data.
#     :param coord_sys: Choose keys for corresponding coordinate system.
#     :return: 1: success.
#     """
#
#     # Find respective dictionary keys
#     if coord_sys == 'cylinder':
#         grad_u_keys = CYL_GRAD_U_KEYS
#         grad_u_key = 'grad_u_cyl'
#         grad_u_norm_keys = CYL_GRAD_U_NORM_KEYS
#         grad_u_norm_key = 'grad_u_norm_cyl'
#         grad_u2_keys = CYL_GRAD_U2_KEYS
#         grad_u2_key = 'grad_u2_cyl'
#     else:
#         grad_u_keys = GRAD_U_KEYS
#         grad_u_key = 'grad_um'
#         grad_u_norm_keys = GRAD_U_NORM_KEYS
#         grad_u_norm_key = 'grad_u_norm'
#         grad_u2_keys = GRAD_U2_KEYS
#         grad_u2_key = 'grad_u2'
#
#     # All Gradient components given
#     if all(gradient_key in data_dict for gradient_key in grad_u_keys):
#         data_dict['grad_um'] = assemble_2nd_order_tensor(data_dict, grad_u_keys)
#
#     # Statistically 2D flow (Does not work with cylinder coordinates atm)
#     elif all(gradient_key in data_dict for gradient_key in GRAD_U_KEYS_STAT_2D):
#         data_dict['grad_um'] = assemble_2nd_order_tensor(data_dict, GRAD_U_KEYS_STAT_2D)
#
#     # Approximate gradient
#     else:
#         grad.velocity_gradient(data_dict, coord_sys)
#         data_dict[grad_u_key] = assemble_2nd_order_tensor(data_dict, grad_u_keys)
#
#     # Approximate gradient of u/norm(u)
#     grad.velocity_gradient(data_dict, coord_sys, exponent=2)
#     data_dict[grad_u_norm_key] = assemble_2nd_order_tensor(data_dict, grad_u_norm_keys)
#
#     # Approximate gradient of u**2
#     grad.velocity_gradient(data_dict, coord_sys, normalise=True)
#     data_dict[grad_u2_key] = assemble_2nd_order_tensor(data_dict, grad_u2_keys)
#
#     return 1
