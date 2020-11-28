# ###################################################################
# Script fig_knn_principle
#
# Description
# Visualise the principle of a kNN density estimator.
#
# ###################################################################
# Author: hw
# created: 26. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.utilities import *
from uncert_ident.visualisation.plotter import *
from uncert_ident.data_handling.flowcase import flowCase



#####################################################################
### Plot
#####################################################################
# Define points
pi = np.array([0.5, 0.5])  # Centre
pk = np.array([0.6, 0.45])  # k=1 Nearest neighbour

delta = max(abs(pk[0] - pi[0]), abs(pk[1] - pi[1]))
eps = 2*delta

p1 = np.array([0.55, 0.8])
p2 = np.array([0.43, 0.9])
p3 = np.array([0.9, 0.15])
p10 = np.array([0.45, 0.2])

p4 = np.array([0.1, 0.55])
p5 = np.array([0.8, 0.43])

p6 = np.array([0.3, 0.75])
p7 = np.array([0.2, 0.85])

p8 = np.array([0.8, 0.9])
p9 = np.array([0.77, 0.88])
p14 = np.array([0.65, 0.3])

p11 = np.array([0.1, 0.2])
p12 = np.array([0.25, 0.1])
p13 = np.array([0.35, 0.3])

points_x = [pi[0], pk[0], p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0], p13[0], p14[0]]
points_y = [pi[1], pk[1], p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1], p10[1], p11[1], p12[1], p13[1], p14[1]]

iss = []
for i, x, y in zip((np.arange(len(points_x[2:]))+2).tolist(), points_x[2:], points_y[2:]):
    if pi[0]-delta <= x <= pi[0]+delta:
        pass
    elif pi[1]-delta <= y <= pi[1]+delta:
        pass
    else:
        iss.append(i)

points = np.array([points_x, points_y])
inactive = points[:, iss]
points = points[:, ~np.isin(np.arange(points.shape[1]), iss)]


# Points
fig, ax = empty_plot(figwidth=latex_textwidth/3, ratio="square")
# lining(*points, linestyle='ok', append_to_fig_ax=(fig, ax))
# lining(*inactive, linestyle='none', marker='o', append_to_fig_ax=(fig, ax))
# ax.plot(*inactive, marker='o', markerfacecolor="none", markeredgecolor='k', linewidth=0)
ax.scatter(*points, s=20, facecolors='k', edgecolors='k')
ax.scatter(*inactive, s=20, facecolors='none', edgecolors='k', linewidth=1.2)
ax.text(pi[0], pi[1]+0.06, r"$z_i$", ha="center", va="center")
ax.text(pk[0]+0.06, pk[1]+0.04, r"$z_k$", ha="center", va="center")


# kNN area
line_xl = [[pi[0]+delta, pi[0]+delta], [0, 1.1]]
line_xr = [[pi[0]-delta, pi[0]-delta], [0, 1.1]]
line_yu = [[0, 1.1], [pi[1]+delta, pi[1]+delta]]
line_yl = [[0, 1.1], [pi[1]-delta, pi[1]-delta]]
lining(*line_xl, linestyle='--k', marker=None, append_to_fig_ax=(fig, ax))
lining(*line_xr, linestyle='--k', marker=None, append_to_fig_ax=(fig, ax))
lining(*line_yu, linestyle='--k', marker=None, append_to_fig_ax=(fig, ax))
lining(*line_yl, linestyle='--k', marker=None, append_to_fig_ax=(fig, ax))


# Arrows
ax.annotate("", xy=(pi[0]-delta, 0.05),
            xytext=(pi[0]+delta, 0.05),
            arrowprops=dict(
                arrowstyle="<->",
                color='k',
                lw=1.5,
                ls='-'))
ax.text(pi[0], 0.09, r"$2\Delta_i$", ha="center", va="baseline")

ax.annotate("", xy=(0.95, pi[1]-delta),
            xytext=(0.95, pi[1]+delta),
            arrowprops=dict(
                arrowstyle="<->",
                color='k',
                lw=1.5,
                ls='-'))
ax.text(0.87, pi[1], r"$2\Delta_i$", ha="center", va="center", rotation=0)

# X-axis
ax.annotate("", xy=(-0.01, 0.00),
            xytext=(1.01, 0.00),
            arrowprops=dict(
                arrowstyle="<-",
                color='k',
                lw=1.5,
                ls='-'))
ax.text(0.93, 0.05, r"$q$", ha="center", va="center")

# Y-axis
ax.annotate("", xy=(0.00, -0.01),
            xytext=(0.00, 1.01),
            arrowprops=dict(
                arrowstyle="<-",
                color='k',
                lw=1.5,
                ls='-'))
ax.text(0.05, 0.95, r"$y$", ha="center", va="center")


# Turn off axis, set limits
ax.set_axis_off()
ax.set_xlim([-0.01, 1])
ax.set_ylim([-0.01, 1])


save("./figures/knn_distance.pdf")
show()
