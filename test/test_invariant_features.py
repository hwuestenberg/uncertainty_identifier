# ###################################################################
# Script test_class_flow_case
#
# Description
# Load any flow data and compute invariants of a minimal integrity
# basis for given tensors.
#
# ###################################################################
# Author: hw
# created: 21. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from numpy import dot, trace, abs, identity, cross
from numpy.linalg import norm

from sklearn.preprocessing import maxabs_scale

from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.utilities import time_decorator, safe_divide


def construct_antisymmetric_grad_p_grad_k(data_dict):
    num_of_points = data_dict['num_of_points'][0]
    grad_pm = data_dict['grad_pm']
    grad_k = data_dict['grad_k']

    negID = -identity(3) + 0
    Pij = np.zeros((3, 3, num_of_points))
    Kij = np.zeros((3, 3, num_of_points))
    for i in range(num_of_points):
        Pij[:, :, i] = cross(negID, grad_pm[:, i])
        Kij[:, :, i] = cross(negID, grad_k[:, i])

    data_dict['Pij'] = Pij
    data_dict['Kij'] = Kij

    return 1


def normalise_Sij_Wij_Pij_Kij(data_dict):
    assert data_dict.keys() >= {'Sij', 'Wij', 'Pij', 'Kij'}
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

    for i in range(case.num_of_points):
        s = Sij[:, :, i]
        w = Wij[:, :, i]
        p = Pij[:, :, i]
        k = Kij[:, :, i]
        rho_mat_deriv = rho * norm(dot(grad_um[:, :, i], vel[:, i]))  # Norm of material derivative multiplied by rho

        Sij[:, :, i] = safe_divide(s, abs(s) + abs(safe_divide(eps[i], tke[i])))
        Wij[:, :, i] = safe_divide(w, abs(w) + abs(safe_divide(eps[i], tke[i])))#norm(w))
        Pij[:, :, i] = safe_divide(p, abs(p) + abs(rho_mat_deriv))
        Kij[:, :, i] = safe_divide(k, abs(k) + abs(safe_divide(eps[i], np.sqrt(tke[i]))))

    data_dict['Sij_norm'] = Sij
    data_dict['Wij_norm'] = Wij
    data_dict['Pij_norm'] = Pij
    data_dict['Kij_norm'] = Kij

    return 1


@time_decorator
def get_inv(data_dict):
    assert data_dict.keys() >= {'Sij_norm', 'Wij_norm', 'Pij_norm', 'Kij_norm'}
    S = data_dict['Sij_norm']
    W = data_dict['Wij_norm']
    P = data_dict['Pij_norm']
    K = data_dict['Kij_norm']

    inv = np.zeros((47, case.num_of_points))
    for i in range(case.num_of_points):
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


def normalise_max_abs(vector):
    assert len(vector.shape) == 2
    assert vector.shape[0] < vector.shape[1]

    for i in range(vector.shape[0]):
        maxabs = np.nanmax(abs(vector[i]))
        vector[i] = safe_divide(vector[i], maxabs)

    return vector


#####################################################################
### Test
#####################################################################
# Load data
# case = flowCase('PH-Breuer-10595')
case = flowCase('TBL-APG-Bobke-b2')


# Build anti-symmetric pseudotensors Pij, Kij
construct_antisymmetric_grad_p_grad_k(case.flow_dict)


# Normalise all tensors
normalise_Sij_Wij_Pij_Kij(case.flow_dict)


# Compute invariants by hand
inv = get_inv(case.flow_dict)
inv_norm = inv.copy()

# Normalise invariants
inv_norm = normalise_max_abs(inv_norm)




# Safe invariants to flow_dict (Only for visualisation, safe to feature_dict!)
for i in range(47):
    key = "inv" + str(i)
    case.flow_dict[key] = inv_norm[i, :]


# Plot invariants
for i in range(47):
    key = "inv" + str(i)
    case.show_flow(key)





