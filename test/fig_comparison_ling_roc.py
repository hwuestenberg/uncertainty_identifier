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


points = [
    [0.019, 0.265],     # Wüst LogReg II
    # [0.949, 1.000],     # Wüst LogReg nut
    [0.078, 0.876],     # Wüst SpaRTA II
    # [0.331, 0.768],     # Wüst SpaRTA nut
    # [0.021, 0.417],     # Wüst Best-LogReg II
    # [0.606, 0.874],     # Wüst Best-LogReg nut
    # [0.047, 0.875],     # Wüst Best-SpaRTA II
    # [0.149, 0.877],     # Wüst Best-SpaRTA nut
    # [0.065, 0.786],     # Ling RF II
    # [0.093, 0.572],     # Ling RF nut
]

legend_elements = []
marker = cycle(['s', '^', '^', '^', 'o', 'o'])
fillstyles = ["full", "none", "full", "none", "full", "none"]
colors = cycle([all_colors[1], all_colors[3]])
labels = ["Logistic Regression", "SpaRTA", "SpaRTA - interpretable", "SpaRTA - interpretable", "SpaRTA - complex", "SpaRTA - complex"]
for point, mark, color, fill, label in zip(points, marker, colors, fillstyles, labels):
    lining(*point,
           color=color,
           xlim=[0.0, 1.0],
           ylim=[0.0, 1.0],
           marker=mark,
           linestyle='-',
           markersize=10,
           fillstyle=fill,
           xlabel="False-positive rate",
           ylabel="True-positive rate",
           # zorder=2.5,
           append_to_fig_ax=(fig, ax))
    legend_elements.append(
        plt.Line2D([0], [0], marker=mark, linestyle='-', markerfacecoloralt=cblack, markersize=10,
                   markeredgecolor=cblack, markeredgewidth=1, color=color, label=label, lw=0,
                   fillstyle=fill)
    )
# ax.text(points[4][0]+0.03, points[4][1]-0.01, "A", fontsize=11)
# ax.text(points[3][0]+0.03, points[3][1]-0.01, "B", fontsize=11)
ax.set_aspect("equal")
ax.legend(handles=[legend_elements[0], legend_elements[1]], loc="lower right")
# ax.legend(handles=[legend_elements[0], legend_elements[2], legend_elements[4]], loc="lower right")
ax.annotate("", xy=[0.35, 0.65], xytext=[0.5, 0.5], arrowprops=dict(fc='k', ec='k', arrowstyle="simple", lw=1.5))


# save("../figures/" + "roc_algorithmic_comparison.pdf")
save("../figures/" + "roc_algorithmic_performance_no_best.pdf")
plt.show()

