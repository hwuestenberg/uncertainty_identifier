# ###################################################################
# test plot_barycentric_anisotropy_metric
#
# Description
# Identify Ling's limit for anisotropy in barycentric coordinates.
#
# ###################################################################
# Author: hw
# created: 22. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.spatial import Delaunay

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


from uncert_ident.visualisation.plotter import *
from uncert_ident.data_handling.flowcase import *


def fII(C1c, C2c):
    return 2/3 * C1c**2 + 1/3 * C1c * C2c + 1/6 * C2c**2


def fIII(C1c, C2c):
    return 2/9 * C1c**3 + 1/6 * C1c**2 * C2c - 1/12 * C1c * C2c**2 - 1/36 * C2c**3


def define_vertices():
    # Define basis points for vertices as (x, y)
    vertex_1c = np.array([1, 0])
    vertex_2c = np.array([0, 0])
    vertex_3c = np.array([0.5, 3 ** 0.5 / 2])

    return [vertex_1c, vertex_2c, vertex_3c]


def c_to_xy(c1, c2, c3):
    vertex1, vertex2, vertex3 = define_vertices()

    X = c1*vertex1[0] + c2*vertex2[0] + c3*vertex3[0]
    Y = c1*vertex1[1] + c2*vertex2[1] + c3*vertex3[1]

    return X, Y


def draw_triangle(fig, ax, vertextext=True, planestrain=False, intermediate_states=False):
    vertex1, vertex2, vertex3 = define_vertices()

    # Boundaries
    boundary = [np.array([vertex3[0], vertex2[0], vertex1[0], vertex3[0]]),
                np.array([vertex3[1], vertex2[1], vertex1[1], vertex3[1]])]

    lining(*boundary, linestyle='-', color=cblack, append_to_fig_ax=(fig, ax))



    # Plane-strain line
    if planestrain:
        eig1 = 1 / 3
        eig2 = 0
        eig3 = -1 / 3

        c_1c = eig1 - eig2
        c_2c = 2 * (eig2 - eig3)
        c_3c = 3 * eig3 + 1

        plane_strain_2c_x = c_1c * vertex1[0] + c_2c * vertex2[0] + c_3c * vertex3[0]
        plane_strain_2c_y = c_1c * vertex1[1] + c_2c * vertex2[1] + c_3c * vertex3[1]
        vertex_plane_strain_2c = np.array([plane_strain_2c_x, plane_strain_2c_y])
        plane_strain = [np.array([vertex_plane_strain_2c[0], vertex3[0]]),
                        np.array([vertex_plane_strain_2c[1], vertex3[1]])]

        lining(*plane_strain, linestyle='-', color=cblack, append_to_fig_ax=(fig, ax))


    # Add vertices' text
    if vertextext:
        point_1C = np.array([1.02, -0.02])
        text_1C = "1C"
        angle_1C = 0
        point_2C = np.array([-0.09, -0.02])
        text_2C = "2C"
        angle_2C = 0
        point_3C = np.array([1 / 2 - 0.03, 3 ** 0.5 / 2 + 0.02])
        text_3C = "3C"
        angle_3C = 0

        points = [point_1C, point_2C, point_3C]
        texts = [text_1C, text_2C, text_3C]
        angles = [angle_1C, angle_2C, angle_3C]

        for point, text, angle in zip(points, texts, angles):
            # trans_angle = plt.gca().transData.transform_angles(np.array((angle,)), point.reshape((1, 2)))[0]
            ax.text(*point, text, rotation=angle, rotation_mode='anchor', fontsize=11)



    if intermediate_states:
        # Plot intermediate states
        point_exp = np.array([0.6, 0.73 + 0.01])
        text_exp = "Axisym. expansion"
        angle_exp = -np.arccos(1 / 3 ** 0.5) * 180 / np.pi + 8
        point_con = np.array([0.1 - 0.04, 0.2 - 0.05])
        text_con = "Axisym. contraction"
        angle_con = np.arccos(1 / 3 ** 0.5) * 180 / np.pi - 8

        # Plot plane strain
        point_ps = np.array([0.35 - 0.02, 0.07])
        text_ps = "Plane-strain"
        angle_ps = 72

        points = [point_exp, point_con, point_ps]
        texts = [text_exp, text_con, text_ps]
        angles = [angle_exp, angle_con, angle_ps]

        for point, text, angle in zip(points, texts, angles):
            # trans_angle = plt.gca().transData.transform_angles(np.array((angle,)), point.reshape((1, 2)))[0]
            ax.text(*point, text, rotation=angle, rotation_mode='anchor')

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([-0.1, 1])

    return 1




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


# Generate grid on triangle
res = 200
low = 0
high = 1
c_1c, c_2c, c_3c = np.meshgrid(np.linspace(low, high, res),
                               np.linspace(low, high, res),
                               np.linspace(low, high, res))
c_1c = c_1c.reshape(-1)
c_2c = c_2c.reshape(-1)
c_3c = c_3c.reshape(-1)


# Enforce realizability
bools = np.vstack([c_1c, c_2c, c_3c]).sum(axis=0) <= 1.0
cs = np.vstack([c_1c, c_2c, c_3c])[:, bools]


# Delete duplicates
# x, y = c_to_xy(cs[0], cs[1], cs[2])
# xy = np.vstack([x, y])
# unixy, uidx = np.unique(xy, axis=1, return_index=True)  # Wrong!
# cs = cs[:, uidx]

c_1c, c_2c, c_3c = cs[0], cs[1], cs[2]  # Full triangle, no duplicates


# Get invariants (according to Banerjee)
II = 2/3 * c_1c**2 + \
     1/3 * c_1c * c_2c + \
     1/6 * c_2c**2
III = 2/9 * c_1c**3 + \
      1/6 * c_1c**2 * c_2c - \
      1/12 * c_1c * c_2c**2 - \
      1/36 * c_2c**3


# Check realizability
# lower = 3/2*(4/3*III)**(2/3)
# upper = 2*III + 2/9
# real = np.logical_and(II > lower, II < upper)
# II = II[real]
# III = III[real]
# scattering(III, II, np.ones_like(II))
#
# cs = cs[:, real]
# c_1c, c_2c, c_3c = cs[0], cs[1], cs[2]  # Full triangle, enforce II-realizability
# x, y = c_to_xy(cs[0], cs[1], cs[2])
# scattering(x, y, np.ones_like(x))
# show()
# exit()



# Check Ling's condition
idx_II = II > 1/6
# idx_II = ~idx_II


# Double check with eigenvalues
eig1 = 2*c_1c/3 + c_2c/6
eig2 = c_2c/6 - c_1c/3
eig3 = -c_1c/3 - c_2c/3
idx_eig = (2*(eig1**2 + eig1*eig2 + eig2**2)) > 1/6

assert np.all(idx_II == idx_eig)

# Convert to 2D coordinates
x, y = c_to_xy(c_1c, c_2c, c_3c)


# Exclude non-condition points
# II = II[idx_II]
# III = III[idx_II]
eig1 = eig1[idx_II]
eig2 = eig2[idx_II]
# c_1c = c_1c[idx_II]
# c_2c = c_2c[idx_II]
# c_3c = c_3c[idx_II]
X = x[idx_II]
Y = y[idx_II]



# Compare to Lumley triangle
# scattering(eig2, eig1, np.ones_like(eig1))  # Eigenvalue triangle
# scattering(III, II, np.ones_like(II))  # Invariant triangle
# eta = np.sqrt(II/6)
# xi = np.cbrt(III/6)
# eta = np.sqrt(1/3*(eig1**2 + eig1*eig2 + eig2**2))
# xi = np.cbrt(-1/2*eig1*eig2*(eig1 + eig2))
# lumley_triangle(xi, eta)


# Prepare RGB triangle
barymap = BarycentricColormap()
lamlegend  = barymap.trigrid(100)
xlegend    = barymap.bary2cartesian(lamlegend)

trilegend  = tri.Triangulation(xlegend[:, 0], xlegend[:, 1])
lamlegend = (lamlegend.T / np.max(lamlegend, axis=1)).T  # Normalise

cmap_legend = colors_to_cmap(lamlegend)



# Setup plot
fig, ax = empty_plot(figwidth=latex_textwidth*0.55)



# Full RGB triangle
ax.tripcolor(trilegend,
             np.linspace(0, 1, xlegend.shape[0]),
             edgecolors='none',
             cmap=cmap_legend,
             shading='gouraud')

# # Blank/mask
xy = np.vstack([x, y])
# scattering(xy[0], xy[1], np.ones_like(xy[0]), scale=20, alpha=0.1)
unique_xy = np.unique(xy, axis=1)
# scattering(unique_xy[0], unique_xy[1], np.ones_like(unique_xy[0]), scale=20, alpha=0.1)

XY = np.vstack([X, Y])
# scattering(XY[0], XY[1], np.ones_like(XY[0]), scale=20, alpha=0.1)
unique_XY = np.unique(XY, axis=1)
# scattering(unique_XY[0], unique_XY[1], np.ones_like(unique_XY[0]), scale=20, alpha=0.1)
#
xx = x[~idx_II]
yy = y[~idx_II]
xxyy = np.vstack([xx, yy])
# scattering(xx, yy, np.ones_like(xx), scale=20, alpha=0.1)
unique_xxyy = np.unique(xxyy, axis=1)
# scattering(unique_xxyy[0], unique_xxyy[1], np.ones_like(unique_xxyy[0]), scale=20, alpha=0.1)

#
# booleans = []
# for point in unique_XY.T:
#     booleans.append(all([point[0] == other_point[0] and point[1] == other_point[1] for other_point in unique_xxyy.T]))

# scattering(x[~idx_II], y[~idx_II], np.ones_like(x[~idx_II]), scale=10, append_to_fig_ax=(fig, ax), colorbar=False, color=cblack, alpha=0.1)
scattering(x[idx_II], y[idx_II], np.ones_like(x[idx_II]), scale=20, append_to_fig_ax=(fig, ax), colorbar=False, color=cwhite, alpha=1)



# scattering(x[~idx_II], y[~idx_II], np.ones_like(x[~idx_II]), scale=50, append_to_fig_ax=(fig, ax), colorbar=False, color=cblack, alpha=0.2)

# Add Triangle boundary
draw_triangle(fig, ax)#, True, True, True)
ax.annotate("$II > 1/6$", xy=[0.57, 0.11], xytext=[0.57, 0.11], bbox=dict(boxstyle="square,pad=0.3", fc=cwhite, ec=cblack, lw=2))#, arrowprops=dict(fc='k', ec='k', arrowstyle="simple", lw=1.5))

close_all()
fig, ax = empty_plot(figwidth=latex_textwidth*0.55)
ax.set_xlim([-0.1, 1])
ax.set_ylim([-0.1, 1])

plt.axis("off")
# save("../figures/anisotropy_beamer_metric.jpg")
save("../figures/anisotropy_beamer_white.jpg")
# save("../figures/anisotropy_beamer.jpg")
show()
plt.close()






