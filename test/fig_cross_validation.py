# ###################################################################
# Script fig_cross_validation
#
# Description
# Visualise the cross-validation splits.
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
lower = 0.1
upper = 0.7
x = np.linspace(lower+0.2, upper, 4)
y = np.linspace(upper, lower, 4)


# Plot box
fig, ax = empty_plot(figwidth=latex_textwidth/2)

ax.annotate(r".   .       All data sets $S_i$    .       .", xy=(0.5, 0.9), xycoords="data",
            va="center", ha="center",
            bbox=dict(boxstyle="square", fc="w"))

i = 0
for y, x in product(y, x):
    if i in [0, 5, 10, 15]:
        fc = corange
    else:
        fc = clightyellow
    ax.annotate(rf"$S_{i%4+1}$", xy=(x, y), xycoords="data",
                va="center", ha="center",
                bbox=dict(boxstyle="square", fc=fc))
    i += 1

ax.annotate(r"Test", xy=(0.85, 0.7), xycoords="data",
            va="center", ha="center",
            bbox=dict(boxstyle="square", fc=corange))
ax.annotate(r"Train", xy=(0.86, 0.55), xycoords="data",
            va="center", ha="center",
            bbox=dict(boxstyle="square", fc=clightyellow))

ax.set_axis_off()

save("./figures/cross_validation.pdf")
show()
