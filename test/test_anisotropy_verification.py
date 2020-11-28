# ###################################################################
# Script test_class_flow_case
#
# Description
# Verify anisotropy calculation.
#
# ###################################################################
# Author: hw
# created: 17. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.utilities import safe_divide
from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.methods.turbulence_properties import anisotropy_tensor, anisotropy_eigenvalues
from uncert_ident.visualisation.plotter import lining, scattering, empty_plot, show



def comparison_rounded(a, b, precision=8):
    return (np.round(a, precision) == np.round(b, precision)).all()



#####################################################################
### Test
#####################################################################
# Get PH-Breuer-10595 as test object
case = flowCase('PH-Breuer-10595')
# case.get_features()
# case.get_labels()


# Get data
k = case.flow_dict['k']
tauij = case.flow_dict['tauij']
print("Shape of tauij: {:}".format(tauij.shape))
length = tauij.shape[-1]


# Compute anisotropy with turbulence_properties
num_of_points = case.nx*case.ny
dij = np.diag((1, 1, 1))

aij = np.zeros((3, 3, num_of_points))
for i in range(num_of_points):
    aij[:, :, i] = tauij[:, :, i] - 2/3*k[i]*dij
# Compute bij from aij
bij = np.zeros((3, 3, num_of_points))
for i in range(num_of_points):
    bij[:, :, i] = safe_divide(aij[:, :, i], 2*k[i])

# Get eigenvalues of bij
eig = np.linalg.eigvalsh(np.moveaxis(case.flow_dict['bij'], 2, 0))
eig = np.sort(eig, axis=1)
eig = eig[:, ::-1]
eig = np.moveaxis(eig, 0, 1)
eig1, eig2, eig3 = eig

# Get invariants of bij
inv1 = (eig1 + eig2 + eig3)
inv2 = -2*(eig1*eig2 + eig2*eig3 + eig1*eig3)
inv3 = 3*(eig1*eig2*eig3)

# By matrix
IIb = np.zeros(length)
for i in range(length):
    IIb[i] = -2 * (0.5*(np.trace(bij[:, :, i])**2 - np.trace(np.dot(bij[:, :, i], bij[:, :, i]))))
IIIb = 3*np.linalg.det(np.moveaxis(bij, 2, 0))






# Compute anisotropy with reference methods (from M. Kaandorp)
tauAni = np.zeros_like(tauij)
for i in range(length):
    tauAni[:, :, i] = tauij[:, :, i] / (2*k[i]) - np.diag([1/3., 1/3., 1/3.])

    nanbools = np.isnan(tauAni[:, :, i])
    # if nanbools.any():
    #     print(tauAni[:, :, i])
    #     print(k[i])
    # Remove zeros
    tauAni[:, :, np.argwhere(k == 0.0).reshape(-1)] = 0


eigVal = np.zeros([3, length])
for i in range(length):
    a, b = np.linalg.eig(tauAni[:, :, i])
    eigVal[:, i] = sorted(a, reverse=True)

II = np.zeros(length)
III = np.zeros(length)
for i in range(length):
    II[i] = 2 * (eigVal[0, i] ** 2 + eigVal[0, i] * eigVal[1, i] + eigVal[1, i] ** 2)
    III[i] = -3 * eigVal[0, i] * eigVal[1, i] * (eigVal[0, i] + eigVal[1, i])


assert comparison_rounded(bij, tauAni)
assert comparison_rounded(eig, eigVal)
assert comparison_rounded(inv1, np.zeros_like(inv1))
assert comparison_rounded(IIb, II)
assert comparison_rounded(IIIb, III)
assert comparison_rounded(IIb, inv2)
assert comparison_rounded(IIIb, inv3)


print('All assertions succeeded')


# # Test lumley triangle
# # Plot realizability limits
# limit_linestyle = '-k'
#
# # Left limit
# fig, ax = lining([0, -1 / 6], [0, 1 / 6], xlim=[-1 / 5, 1 / 2.8], ylim=[-0.01, 1 / 2.8], linestyle=limit_linestyle)
# # Right limit
# lining([0, 1 / 3], [0, 1 / 3], append_to_fig_ax=(fig, ax), linestyle=limit_linestyle)
#
# # Upper limit
# xi_range = np.linspace(-1 / 6, 1 / 3, 20)
# eta_lim = (1 / 27 + 2 * xi_range ** 3) ** 0.5
# lining(xi_range, eta_lim, append_to_fig_ax=(fig, ax), linestyle=limit_linestyle)
#
#
# # Anisotropy metric limit
# xi = np.linspace(-1/6, 1/6, 1000)
# eta = np.ones_like(xi)*1/6
# scattering(xi, eta, np.ones_like(xi),
#            append_to_fig_ax=(fig, ax),
#            alpha=0.4,
#            scale=5,
#            xlabel=r'$\xi$', ylabel=r'$\eta$',
#            )
#
# # Show area of true values
# xixi, etaeta = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
# bools = np.logical_and(
#     np.logical_and(
#         etaeta > 1/6, xixi > -1/3),
#     xixi < 1/3)
# scattering(xixi[bools], etaeta[bools], np.ones_like(xixi[bools]),
#            append_to_fig_ax=(fig, ax),
#            alpha=0.4,
#            scale=5,
#            xlabel=r'$\xi$', ylabel=r'$\eta$',
#            )


# Test lumley triangle
# Plot realizability limits
limit_linestyle = '-k'

# Left limit
fig, ax = lining([0, -1 / 6], [0, 1 / 6], xlim=[-1 / 5, 1 / 2.8], ylim=[-0.01, 1 / 2.8], linestyle=limit_linestyle)
# Right limit
lining([0, 1 / 3], [0, 1 / 3], append_to_fig_ax=(fig, ax), linestyle=limit_linestyle)

# Upper limit
xi_range = np.linspace(-1 / 6, 1 / 3, 20)
eta_lim = (1 / 27 + 2 * xi_range ** 3) ** 0.5
lining(xi_range, eta_lim, append_to_fig_ax=(fig, ax), linestyle=limit_linestyle)


# Anisotropy metric limit
xi = np.linspace(-1/6, 1/6, 1000)
eta = np.ones_like(xi)*1/6
scattering(xi, eta, np.ones_like(xi),
           append_to_fig_ax=(fig, ax),
           alpha=0.4,
           scale=5,
           xlabel=r'$\xi$', ylabel=r'$\eta$',
           )

# Show area of true values
xixi, etaeta = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
bools = np.logical_and(
    np.logical_and(
        etaeta > 1/6, xixi > -1/3),
    xixi < 1/3)
scattering(xixi[bools], etaeta[bools], np.ones_like(xixi[bools]),
           append_to_fig_ax=(fig, ax),
           alpha=0.4,
           scale=5,
           xlabel=r'$\xi$', ylabel=r'$\eta$',
           )





# # Test barycentric map
# # Define vertices of the triangle (x, y)
# vertex_1c = np.array([1, 0])
# vertex_2c = np.array([0, 0])
# vertex_3c = np.array([0.5, 3 ** 0.5 / 2])
#
# # Define limits ([x1, x2], [y1, y2])
# lower = [np.array([vertex_1c[0], vertex_2c[0]]), np.array([vertex_1c[1], vertex_2c[1]])]
# left = [np.array([vertex_2c[0], vertex_3c[0]]), np.array([vertex_2c[1], vertex_3c[1]])]
# right = [np.array([vertex_1c[0], vertex_3c[0]]), np.array([vertex_1c[1], vertex_3c[1]])]
#
# # Plot boundaries, demarcate realizable triangle
# fig, ax = empty_plot()
# limit_linestyle = '-k'  # Solid black line
# lining(*lower, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax), xlim=[-0.1, 1.1], ylim=[-0.1, 1.1])
# lining(*left, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax))
# lining(*right, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax))
#
#
# # Create grid of values
# ei1, ei2, ei3 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
# INV2 = 2*(ei1**2 + ei1*ei2 + ei2**2)
# INV3 = -3*ei1*ei2*(ei1 + ei2)
#
# bools = np.logical_and(
#     np.logical_and(
#         INV2 > 1/6, INV3 > -1/36),
#     INV3 < 2/9)
#
# eig1 = ei1[bools]
# eig2 = ei2[bools]
# eig3 = ei3[bools]
#
#
# # Compute coordinates using the eigenvalues
# c_1c = eig1 - eig2
# c_2c = 2 * (eig2 - eig3)
# c_3c = 3 * eig3 + 1
#
# # Compute barycentric coordinates
# x = c_1c * vertex_1c[0] + c_2c * vertex_2c[0] + c_3c * vertex_3c[0]
# y = c_1c * vertex_1c[1] + c_2c * vertex_2c[1] + c_3c * vertex_3c[1]
#
# # Plot all points into the triangle
# lining(x, y, linestyle='o', append_to_fig_ax=(fig, ax))





show()
