# ###################################################################
# Script fig_algorithms
#
# Description
#
# ###################################################################
# Author: hw
# created: 04. Sep. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from uncert_ident.visualisation.plotter import *



#####################################################################
### Plot
#####################################################################
# Define grid
lower = 0
upper = 1
y = np.linspace(upper, lower, 4)
# x = np.array([0.4, 1, 0.4, 1])
x = np.array([0.4])
dy = abs(y[0] - y[1])

# Plot box
fig, ax = empty_plot(figwidth=beamer_textwidth)

# ax.annotate(r".   .       All data sets $S_i$    .       .", xy=(0.5, 0.9), xycoords="data",
#             va="center", ha="center",
#             bbox=dict(boxstyle="square", fc="w"))

# i = 0
# for y, x in product(y, x):
#     if i in [0, 5, 10, 15]:
#         fc = corange
#     else:
#         fc = clightyellow
#     ax.annotate(rf"$S_{i%4+1}$", xy=(x, y), xycoords="data",
#                 va="center", ha="center",
#                 bbox=dict(boxstyle="square", fc=fc))
#     i += 1
textss = [
    ["", "", "Adapt $w$", "Linear $M(x)$"],
    ["Build candidates $b = \{x_1^2, x_1^2 x_2^2, \ldots\}$", "Discover active candidates", "Adapt active $w$", "Linear $M(b)$"],
]
for xi, texts in zip(x, textss):
    for yi, text in zip(y, texts):
        ax.annotate(fr"{text}", xy=(xi, yi), xycoords="data",
                    va="center", ha="center",
                    bbox=dict(boxstyle="square", fc=cwhite))
        if text == "" or text == textss[0][2]:
            pass
        else:
            ax.annotate("", xy=[xi, yi+dy*0.2], xytext=[xi, yi+dy*0.8], arrowprops=dict(fc='k', ec='k', arrowstyle="simple", lw=1.5))

# ax.annotate(r"Train", xy=(0.86, 0.55), xycoords="data",
#             va="center", ha="center",
#             bbox=dict(boxstyle="square", fc=clightyellow))

ax.set_axis_off()

save("./figures/algorithms_lr.pdf")
show()
