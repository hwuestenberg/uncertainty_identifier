# ###################################################################
# test plot_barycentric_map
#
# Description
# Plot the barycentric map with labels for the states of turbulence
# with example.
#
# ###################################################################
# Author: hw
# created: 15. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib.pyplot as plt
import numpy as np

from uncert_ident.visualisation.plotter import *
from uncert_ident.data_handling.flowcase import *


# Define basis points for vertices as (x, y)
vertex_1c = np.array([1, 0])
vertex_2c = np.array([0, 0])
vertex_3c = np.array([0.5, 3**0.5/2])

eig1 = 1/3
eig2 = 0
eig3 = -1/3

c_1c = eig1 - eig2
c_2c = 2*(eig2 - eig3)
c_3c = 3*eig3 + 1

plane_strain_2c_x = c_1c*vertex_1c[0] + c_2c*vertex_2c[0] + c_3c*vertex_3c[0]
plane_strain_2c_y = c_1c*vertex_1c[1] + c_2c*vertex_2c[1] + c_3c*vertex_3c[1]
vertex_plane_strain_2c = np.array([plane_strain_2c_x, plane_strain_2c_y])


# Define limits ([x1, x2], [y1, y2])
boundary = [np.array([vertex_3c[0], vertex_2c[0], vertex_1c[0], vertex_3c[0]]), np.array([vertex_3c[1], vertex_2c[1], vertex_1c[1], vertex_3c[1]])]
plane_strain = [np.array([vertex_plane_strain_2c[0], vertex_3c[0]]), np.array([vertex_plane_strain_2c[1], vertex_3c[1]])]
# lower = [np.array([vertex_1c[0], vertex_2c[0]]), np.array([vertex_1c[1], vertex_2c[1]])]
# left = [np.array([vertex_2c[0], vertex_3c[0]]), np.array([vertex_2c[1], vertex_3c[1]])]
# right = [np.array([vertex_1c[0], vertex_3c[0]]), np.array([vertex_1c[1], vertex_3c[1]])]



# Plot boundaries, demarcate realizable triangle
fig, ax = empty_plot(figwidth=latex_textwidth/2)
limit_linestyle = '-k'  # Solid black line
lining(*boundary, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax), xlim=[-0.1, 1.1], ylim=[-0.1, 1.1])
lining(*plane_strain, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax))
# lining(*lower, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax), xlim=[-0.1, 1.1], ylim=[-0.1, 1.1])
# lining(*left, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax))
# lining(*right, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax))



# Plot limiting states
point_1C = np.array([1.02, -0.02])
text_1C = "1C"
angle_1C = 0
point_2C = np.array([-0.09, -0.02])
text_2C = "2C"
angle_2C = 0
point_3C = np.array([1/2-0.03, 3**0.5/2+0.02])
text_3C = "3C"
angle_3C = 0

# Plot intermediate states
point_exp = np.array([0.6, 0.73+0.01])
text_exp = "Axisym. expansion"
angle_exp = -np.arccos(1/3**0.5)*180/np.pi+8
point_con = np.array([0.1-0.04, 0.2-0.05])
text_con = "Axisym. contraction"
angle_con = np.arccos(1/3**0.5)*180/np.pi-8

# Plot plane strain
point_ps = np.array([0.35-0.02, 0.07])
text_ps = "Plane-strain"
angle_ps = 72


points = [point_1C, point_2C, point_3C, point_exp, point_con, point_ps]
texts = [text_1C, text_2C, text_3C, text_exp, text_con, text_ps]
angles = [angle_1C, angle_2C, angle_3C, angle_exp, angle_con, angle_ps]


for point, text, angle in zip(points, texts, angles):
    # trans_angle = plt.gca().transData.transform_angles(np.array((angle,)), point.reshape((1, 2)))[0]
    ax.text(*point, text, rotation=angle, rotation_mode='anchor')

ax.set_xlim([-0.1, 1])
ax.set_ylim([-0.1, 1])

plt.axis("off")
plt.savefig("../figures/barycentric_map_triangle_turbulence_states.pdf", format='pdf', bbox_inches='tight')


show()

