import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from uncert_ident.utilities import TRUE_NEGATIVE, TRUE_POSITIVE, FALSE_NEGATIVE, FALSE_POSITIVE
from uncert_ident.visualisation.plotter import empty_plot, lining, scattering, \
    latex_textwidth, cblack, cgrey, cwhite, save, all_colors


fig, ax = empty_plot(figwidth=latex_textwidth)
lining([0, 1], [0, 1],
       linestyle='-k',
       append_to_fig_ax=(fig, ax))


points = [[0, 1],
          [1, 1],
          [0, 0],
          [0.44, 0.77],
          [0.22, 0.55]]

marker = cycle(['o', 'P', 's', '.', '.'])
# fillstyles = ["full"] * 5
# colors = cycle(all_colors[:5])
for point, mark in zip(points, marker):

    lining(*point,
           color=cblack,
           xlim=[0.0, 1.0],
           ylim=[0.0, 1.0],
           marker=mark,
           linestyle='-',
           markersize=20,
           xlabel="False-positive rate",
           ylabel="True-positive rate",
           # zorder=2.5,
           append_to_fig_ax=(fig, ax))

ax.text(points[4][0]+0.03, points[4][1]-0.01, "A", fontsize=11)
ax.text(points[3][0]+0.03, points[3][1]-0.01, "B", fontsize=11)


ax.set_aspect("equal")

save("../figures/" + "roc_reference.pdf")
plt.show()
