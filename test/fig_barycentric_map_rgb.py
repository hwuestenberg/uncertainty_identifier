# ###################################################################
# test plot_barycentric_map
#
# Description
# Plot the barycentric map with an rgb colormap for the states of
# turbulence, with examples.
#
# ###################################################################
# Author: hw
# created: 15. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import numpy as np
from scipy.spatial import Delaunay

from uncert_ident.visualisation.plotter import *
from uncert_ident.data_handling.flowcase import *
from uncert_ident.methods.geometry import geometry_periodic_hills_lower as geo_ph_low

iX = 1
iY = 0
class BarycentricColormap:

    def __init__(self):
        # Vertices of output triangle
        self.xv = np.array([[1, 0],
                            [0, 0],
                            [.5, np.sqrt(3) / 2]])
        xv = self.xv
        self.Tinv = np.linalg.inv(
            np.array([[xv[0, iX] - xv[2, iX], xv[1, iX] - xv[2, iX]],
                      [xv[0, iY] - xv[2, iY], xv[1, iY] - xv[2, iY]]]))

    def bary2cartesian(self, lam):
        """
        Convert barycentric coordinates (normalized) ``lam`` (ndarray (N,3)), to
        Cartesian coordiates ``x`` (ndarray (N,2)).
        """
        return np.einsum('ij,jk', lam, self.xv)

    def cartesian2bary(self, x):
        """
        Convert Cartesian coordiates ``x`` (ndarray (N,2)), to barycentric
        coordinates (normalized) ``lam`` (ndarray (N,3)).
        """
        lam = np.zeros((x.shape[0], 3))
        lam[:, :2] = np.einsum('ij,kj->ki', self.Tinv, x - self.xv[2])
        lam[:, 2] = 1. - lam[:, 0] - lam[:, 1]
        return lam

    def trigrid(self, n=10):
        """Uniform grid on triangle in barycentric coordinates."""
        lam = []
        for lam1 in range(n):
            for lam2 in range(n - lam1):
                lam3 = n - lam1 - lam2
                lam.append([lam1, lam2, lam3])
        return np.array(lam) / float(n)

    def randomgrid(self, n):
        lam = np.random.random((n, 3))
        return self.normalize(lam)

    def normalize(self, lam):
        """Normalize Barycentric coordinates to 1."""
        return (lam.T / np.sum(lam, axis=1)).T


def colors_to_cmap(colors):
    """
    Yields a matplotlib colormap object
    that,  reproduces the colors in the given array when passed a
    list of N evenly spaced numbers between 0 and 1 (inclusive), where N is the
    first dimension of ``colors``.

    Args:
      colors (ndarray (N,[3|4])): RGBa_array
    Return:
      cmap (matplotlib colormap object): Colormap reproducing input colors,
                                         cmap[i/(N-1)] == colors[i].

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.linspace(0,1,range(len(colors)))
    """
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1), dtype=np.float))/(len(colors) - 1)
    return matplotlib.colors.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] +
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] +
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors))


case = flowCase('PH-Breuer-10595')
# case = flowCase('PH-Xiao-05')
# case = flowCase('CBFS-Bentaleb')
# case = flowCase('NACA4412-Vinuesa-top-10')
# case = flowCase('TBL-APG-Bobke-m13')

data = case.flow_dict
x = data['x']
y = data['y']

eig1 = data['bij_eig1']
eig2 = data['bij_eig2']
eig3 = data['bij_eig3']

# Define basis points for vertices as (x, y)
vertex_1c = np.array([1, 0])
vertex_2c = np.array([0, 0])
vertex_3c = np.array([0.5, 3**0.5/2])

# Compute coordinates using the eigenvalues
c_1c = eig1 - eig2
c_2c = 2*(eig2 - eig3)
c_3c = 3*eig3 + 1
colors = np.vstack([c_1c, c_2c, c_3c]).T
colors = (colors.T / np.max(colors, axis=1)).T  # Normalise


# Construct triangles for the 2D space
triang = tri.Triangulation(x, y)

# Define cmap
cmap_space = colors_to_cmap(colors)

# Mask off unwanted triangles.
# For PH: y < y_wall(x)
bools = y[triang.triangles].min(axis=1) < geo_ph_low(x[triang.triangles].mean(axis=1))
triang.set_mask(bools)




fig, ax = empty_plot()
# Plot boundaries
boundaries = get_boundaries(case)
for boundary in boundaries:
    plot.lining(*boundary, linestyle="-k", append_to_fig_ax=(fig, ax))

# ax.set_aspect('equal')
ax.set_xlim([x.min()+0.1, x.max()-0.1])
ax.set_ylim([y.min(), y.max()])
ax.set_xlabel(r"$x/H$")
ax.set_ylabel(r"$y/H$")
# ax.set_title("Turbulence states with RGB colormap")
# ax.triplot(triang, lw=0.5, color='white')
ax.tripcolor(triang,
             np.linspace(0, 1, case.num_of_points),
             edgecolors='none',
             cmap=cmap_space,
             shading='gouraud')

save("../figures/barycentric_map_rgb_example_ph.jpg")
# plt.savefig("../figures/barycentric_map_rgb_example_ph.pdf", format='pdf', bbox_inches='tight')

show()
plt.close()


#
#
#
# barymap = BarycentricColormap()
#
# lamlegend  = barymap.trigrid(1000)
# xlegend    = barymap.bary2cartesian(lamlegend)
# trilegend  = Delaunay(xlegend)
#
# lamlegend = (lamlegend.T / np.max(lamlegend, axis=1)).T  # Normalise
# cmap_legend = colors_to_cmap(lamlegend)
#
# fig2, ax = empty_plot(figwidth=latex_textwidth/2)
# ax.tripcolor(xlegend[:,0], xlegend[:,1], trilegend.simplices,
#              np.linspace(0,1,xlegend.shape[0]),
#              edgecolors='none', cmap=cmap_legend, shading='gouraud')
#
#
# # Add text
# point_1C = np.array([1.02, -0.02])
# text_1C = "1C"
# angle_1C = 0
# point_2C = np.array([-0.09, -0.02])
# text_2C = "2C"
# angle_2C = 0
# point_3C = np.array([1/2-0.03, 3**0.5/2+0.02])
# text_3C = "3C"
# angle_3C = 0
#
# points = [point_1C, point_2C, point_3C]
# texts = [text_1C, text_2C, text_3C]
# angles = [angle_1C, angle_2C, angle_3C]
#
# for point, text, angle in zip(points, texts, angles):
#     # trans_angle = plt.gca().transData.transform_angles(np.array((angle,)), point.reshape((1, 2)))[0]
#     ax.text(*point, text, rotation=angle, rotation_mode='anchor')
# ax.set_xlim([-0.1, 1])
# ax.set_ylim([-0.1, 1])
#
# plt.axis("off")
#
#
#
# save("../figures/barycentric_map_triangle_rgb.jpg")
# # plt.savefig("../figures/barycentric_map_triangle_rgb.pdf", format='pdf', bbox_inches='tight')
#
# show()
# #
