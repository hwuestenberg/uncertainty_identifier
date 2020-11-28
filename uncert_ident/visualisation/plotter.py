# ###################################################################
# module plotter
#
# Description
# Provides functions for simplified creation of plots.
#
# ###################################################################
# Author: hw
# created: 26. Feb. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import numpy as np
from scipy.interpolate import griddata
import sympy as sy

from sklearn.metrics import plot_confusion_matrix

from uncert_ident.data_handling.data_import import check_create_directory
from uncert_ident.methods.geometry import get_boundaries, get_lower_boundary_y


# Plot layout and latex config
try:
    plt.style.use("./uncert_ident/visualisation/LaTurbuleX.mplstyle")
except OSError:
    plt.style.use("../uncert_ident/visualisation/LaTurbuleX.mplstyle")
plt.rc('text.latex', preamble=r"\usepackage{mathtools}\usepackage[cm]{sfmath}")

doc_textwidth = 464.02107
beamer_textwidth = 324.36168
latex_textwidth = beamer_textwidth  # in pt, use \the\textwidth in LaTeX



# non_11_size = 9
# plt.rc('font', size=non_11_size)          # controls default text sizes
# plt.rc('axes', titlesize=non_11_size)     # fontsize of the axes title
# plt.rc('axes', labelsize=non_11_size)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=non_11_size)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=non_11_size)    # fontsize of the tick labels



# Default directory
save_dir = "./figures/"



# Colors (Use sequential map for greyscale compatibility)
# general_cmap = plt.cm.inferno
general_cmap = plt.cm.hot
confusion_cmap = plt.cm.RdYlBu
grey_cmap = plt.cm.Greys
bg_cmap = plt.cm.viridis

confusion_colors = confusion_cmap(np.linspace(0, 1, 100))
cconfusionred = confusion_colors[99]
cconfusionblue = confusion_colors[0]

general_colors = general_cmap(np.linspace(0, 1, 100))
# cblack = general_colors[0]
cdarkred = general_colors[20]
cred = general_colors[38]
corange = general_colors[60]
cyellow = general_colors[72]
clightyellow = general_colors[87]

bg_colors = bg_cmap(np.linspace(0, 1, 100))
cgreen = bg_colors[80]
cblue = bg_colors[25]
cpurple = bg_colors[0]

grey_colors = grey_cmap(np.linspace(0, 1, 100))
cwhite = grey_colors[0]
cgrey = grey_colors[50]
cdarkgrey = grey_colors[70]
cblack = grey_colors[-1]

all_colors = [cdarkred, cred, corange, cyellow, clightyellow, cblack, cgrey, cwhite]


# Line2D objects
line2d = Line2D
lss = ['-', ':', '--', '-.']






#####################################################################
### Functions
#####################################################################
def set_size(width=latex_textwidth, fraction=1, square=False):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    square: bool
            Width == Height

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if square:
        ratio = 1
    else:
        golden_ratio = (5 ** 0.5 - 1) / 2
        ratio = golden_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    return fig_width_in, fig_height_in


def empty_plot(no_axis=None, figwidth=0., ratio="golden", plot_grid=None):
    """
    Empty plot that is filled by subsequent calls to below plotter functions.
    :param figwidth: Width of the figure in inch.
    :param no_axis: Option to not return an axis on figure
    :return: Figure and axes.subplot objects.
    """
    if figwidth:
        if ratio == "golden":
            figsize = set_size(figwidth)
        elif ratio == "square":
            figsize = set_size(figwidth, square=True)
        else:
            assert False, f"Invalid ratio for figure size:\t{ratio}"
    else:
        figsize = set_size()

    fig = plt.figure(figsize=figsize)
    if not no_axis:
        ax = fig.add_subplot()
        return fig, ax

    return fig


def test_cmap(cmap):
    scattering(np.arange(0, 100), np.zeros(100), np.arange(100), color=cmap(np.linspace(0, 1, 100)), scale=100)


def fmt_sci(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def fmt_dec(x, pos):
    return f"{x:1.1f}"




def check_append_to_fig_ax(append_to_fig_ax):
    """
    Check whether a new figure and axis needs to be generated or append
    to given figure and axis.
    :param append_to_fig_ax: Tuple with figure and axis object.
    :return: Tuple of figure and axis.
    """

    #  Create new figure and axis or append to given
    if isinstance(append_to_fig_ax, tuple) and isinstance(append_to_fig_ax[0], bool):
        fig, ax = empty_plot()
    elif isinstance(append_to_fig_ax, tuple) and isinstance(append_to_fig_ax[0], plt.Figure) and isinstance(append_to_fig_ax[1], plt.Subplot):
        fig = append_to_fig_ax[0]
        ax = append_to_fig_ax[1]
    else:
        assert False, 'Cannot handle given figure and axes type: %r' % (type(append_to_fig_ax))

    return fig, ax


def safe_replace(string):
    """
    LaTeX formatting interprets underscores as math mode and
    throws errors. Replace underscores, if string was given.
    :param string: Any string.
    :return: String without underscores or None/
    """

    if not string:
        return string


    if "$" not in string:
        try:
            string = string.replace('_', ' ')
        except AttributeError:
            string = None

    return string


def set_limits(axis, xlim, ylim):
    """
    Set limits for an axis.
    :param axis: Axis object.
    :param xlim: List of lower and upper limit in x.
    :param ylim: List of lower and upper limit in y.
    :return: 1: success.
    """

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

    return 1


def set_labels_title(axis, xlabel, ylabel, title):
    """
    Set strings for a given axis.
    :param axis: Axis object.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    :return: 1: success.
    """

    xlabel = safe_replace(xlabel)  # LaTeX correction
    axis.set_xlabel(xlabel)

    ylabel = safe_replace(ylabel)  # LaTeX correction
    axis.set_ylabel(ylabel)

    title = safe_replace(title)  # LaTeX correction
    axis.set_title(title)

    return 1


def lining(x, y,
           linestyle='', alpha=1.0, lw=1,
           title=None, xlabel='Abscissa', ylabel='Ordinate',
           xlim=None,
           ylim=None,
           scale_x=1,
           scale_y=1,
           line_label=None,
           sname=None,
           append_to_fig_ax=(False, False),
           xlog=False,
           ylog=False,
           marker=None,
           markerfacecoloralt=None,
           markersize=5,
           markeredgecolor=cblack,
           markeredgewidth=1,
           color=cblack,
           grid=None,
           fillstyle="full"):
    """
    Two-dimensional plot of Y against X.

    :param scale_y: Scale the y-value by factor.
    :param scale_x: Scale the x-value by factor.
    :param xlog: Choose logarithmic scaling on abcissa.
    :param ylog: Choose logarithmic scaling on ordinate.
    :param append_to_fig_ax: Plot on an existing figure and axis.
    :param sname: Filename for saving.
    :param alpha: Transparency of the line.
    :param line_label: Label for the line plot.
    :param ylim: Optional y-axis limits as tuple.
    :param xlim: Optional x-axis limits as tuple.
    :param title: Optional title.
    :param ylabel: Optional y-axis label.
    :param xlabel: Optional x-axis label.
    :param x: Abscissa.
    :param y: Ordinate.
    :param linestyle: Brief linestyle.
    :return: Tuple of figure and axis object.
    """

    #  Create new figure and axis or append to given
    fig, ax = check_append_to_fig_ax(append_to_fig_ax)

    # Correct line_label (for LaTeX)
    line_label = safe_replace(line_label)

    # Plot line
    x = scale_x*x
    y = scale_y*y
    plot1, = ax.plot(x, y, linestyle,
                     lw=lw,
                     label=line_label,
                     alpha=alpha,
                     fillstyle=fillstyle,
                     marker=marker,
                     markerfacecoloralt=markerfacecoloralt,
                     markersize=markersize,
                     markeredgecolor=markeredgecolor,
                     markeredgewidth=markeredgewidth,
                     color=color)

    # Set limits
    if xlim or ylim:
        set_limits(ax, xlim, ylim)


    # Set labels and title  
    set_labels_title(ax, xlabel, ylabel, title)

    # Set axis scaling
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    # Update legend
    if line_label:
        ax.legend(loc='best')

    # Set grid
    if grid:
        ax.grid(color='k', linestyle='-', linewidth=0.5)

    # Save figure
    if sname:
        save(sname + ".pdf")

    return fig, ax


def multi_lining(*xy_data, linestyle='', xlabel='Abscissa', ylabel='Ordinate', title=None, xlim=None, ylim=None, line_label=None, sname=None):
    """
    Two-dimensional plot of Y against X.

    :param line_label: Label for the line plot.
    :param ylim: Optional y-axis limits as tuple.
    :param xlim: Optional x-axis limits as tuple.
    :param title: Optional title.
    :param ylabel: Optional y-axis label.
    :param xlabel: Optional x-axis label.
    :param xy_data: Tuple of pairs of np.array types
    :param linestyle: Shortform linestyle.
    :return: void.
    """
    #  Create and setup figure and axis
    fig = plt.figure()
    ax = fig.add_subplot()

    # Check inputs
    xy_len = len(xy_data)
    assert(np.mod(xy_len, 2) == 0), "Input to multi_lining must contain pairs of x and y data, instead odd length: %r" % xy_len

    num_of_lines = int(xy_len/2)
    line_label_len = len(line_label)
    assert(num_of_lines == line_label_len), "Each line must have one label, number of lines %r does not correspond to number of labels %r" % (num_of_lines, line_label_len)

    #  Plot data
    plots = list()
    for i in range(0, xy_len, 2):
        plots.append(
            ax.plot(xy_data[i], xy_data[i+1], linestyle, label=line_label[int(i/2)])[0]
        )

    # Add text
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add legend
    ax.legend(handles=plots, loc='best')

    if sname:
        save(sname + ".pdf")

    return


def contouring(x_2d,
               y_2d,
               var_2d,
               filled=True,
               levels=10,
               colorbar=True,
               cbarlabel=None,
               cmap=general_cmap,
               title=None,
               xlabel='Abcissa',
               ylabel='Ordinate',
               xlim=None,
               ylim=None,
               sname=None,
               append_to_fig_ax=(False, False)):
    """
    Creates a filled contour plot or, optionally, contour lines.
    :param colorbar: Optional deactivation of colorbar.
    :param append_to_fig_ax: Plot on an existing figure and axis.
    :param sname: Filename for saving.
    :param ylim: Optional y-axis limits as tuple.
    :param xlim: Optional x-axis limits as tuple.
    :param ylabel: Optional y-axis label.
    :param xlabel: Optional x-axis label.
    :param title: Optional title.
    :param levels: Number of or specified levels for contours.
    :param x_2d: Two-dimensional x-coordinates (meshgrid).
    :param y_2d: Two-dimensional y-coordinates (meshgrid).
    :param var_2d: Two-dimensional scalar.
    :param filled: Optional choice of contour lines (=False).
    :return: void.
    """

    #  Create and setup figure and axis
    if isinstance(append_to_fig_ax, tuple) and isinstance(append_to_fig_ax[0], bool):
        fig, ax = empty_plot()
    elif isinstance(append_to_fig_ax, tuple) and isinstance(append_to_fig_ax[0], plt.Figure) and isinstance(append_to_fig_ax[1], plt.Subplot):
        fig = append_to_fig_ax[0]
        ax = append_to_fig_ax[1]
    else:
        assert False, 'Cannot handle given figure and axes type: %r' % (type(append_to_fig_ax))

    #  Plot data
    if filled:
        plot1 = ax.contourf(x_2d, y_2d, var_2d, levels=levels, cmap=cmap, corner_mask=True)
    else:
        plot1 = ax.contour(x_2d, y_2d, var_2d, levels=levels, cmap=cmap)

    # Set limits
    set_limits(ax, xlim, ylim)

    # Set label and title
    set_labels_title(ax, xlabel, ylabel, title)

    if colorbar:
        cbar = fig.colorbar(plot1, format=plt.FuncFormatter(fmt_dec))
        if cbarlabel:
            cbar.ax.set_xlabel(cbarlabel, ha="center", va="center")

    # Save figure
    if sname:
        save(sname + ".pdf")

    return fig, ax


def scattering(x_2d,
               y_2d,
               var_2d,
               scale=1,
               alpha=1.0,
               color=None,
               cmap=general_cmap,
               marker='o',
               zorder=1,
               title=None,
               xlabel='Abscissa',
               ylabel='Ordinate',
               xlim=None,
               ylim=None,
               labels_color=[],
               labels_size=[],
               legend_transparency=1.0,
               colorbar=None,
               sname=None,
               append_to_fig_ax=(False, False)):
    """
    Creates a scatter plot.
    :param colorbar: Optional colorbar.
    :param legend_transparency: Transparency of the legend box and text.
    :param append_to_fig_ax: Plot on an existing figure and axis.
    :param sname: Filename for saving.
    :param ylim: Optional y-axis limits as tuple.
    :param xlim: Optional x-axis limits as tuple.
    :param cmap: Colormap for the coloring of points.
    :param alpha: Transparency of points.
    :param labels_size: Optional labels for the size of dots.
    :param labels_color: Optional labels for the color of dots.
    :param title: Optional title.
    :param ylabel: Optional y-axis label.
    :param xlabel: Optional x-axis label.
    :param scale: Optional scaling of the dot size.
    :param color: Optional color, if var_2d is not used.
    :param x_2d: Two-dimensional x-coordinates (meshgrid).
    :param y_2d: Two-dimensional y-coordinates (meshgrid).
    :param var_2d: Two-dimensional scalar.
    :return: void.
    """

    #  Create and setup figure and axis
    fig, ax = check_append_to_fig_ax(append_to_fig_ax)

    # Define color explicitly or with var_2d
    if isinstance(color, (np.ndarray, list, tuple)):
        plot1 = ax.scatter(x_2d, y_2d, s=scale*np.ones_like(var_2d), color=color, cmap=cmap, alpha=alpha, marker=marker, zorder=zorder)
    else:
        plot1 = ax.scatter(x_2d, y_2d, s=scale*np.ones_like(var_2d), c=var_2d, cmap=cmap, alpha=alpha, marker=marker, zorder=zorder)

    # Set limits
    set_limits(ax, xlim, ylim)

    # Set label and title
    set_labels_title(ax, xlabel, ylabel, title)

    # Set colorbar
    if colorbar:
        fig.colorbar(plot1)

    # Legend for dot color
    if labels_color:
        handles, labels = plot1.legend_elements(prop="colors")
        if len(labels_color) >= len(labels):
            labels = labels_color
        legend_color = ax.legend(handles, labels, loc='upper right', title=None, framealpha=legend_transparency)
        ax.add_artist(legend_color)

    # Legend for dot size
    if labels_size:
        handles, labels = plot1.legend_elements(prop="sizes")
        if len(labels_size) == len(labels):
            labels = labels_size
        legend_size = ax.legend(handles, labels, loc='lower right', title=None, framealpha=legend_transparency)
        ax.add_artist(legend_size)

    # Save figure
    if sname:
        save(sname + ".pdf")

    return fig, ax


def baring(pos, heights, width,
           color, xticklabels,
           xticklabel_bottom_pos=None,
           title=None,
           xlabel='Abscissa',
           ylabel='Magnitude',
           xlim=None,
           ylim=None,
           xlog=None,
           ylog=None,
           sname=None,
           append_to_fig_ax=(False, False)):
    #  Create and setup figure and axis
    fig, ax = check_append_to_fig_ax(append_to_fig_ax)

    ax.bar(pos, heights, width=width, color=color)

    # Set limits
    set_limits(ax, xlim, ylim)

    # Set label and title
    set_labels_title(ax, xlabel, ylabel, title)

    # Set xticks position and label
    if xticklabels:
        ax.set_xticks(pos)
        ax.set_xticklabels(xticklabels, rotation=0)

    # Adjust position for labels
    if xticklabel_bottom_pos:
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l, xticklabel_bottom_pos, w, h])

    # Set axis scaling
    if xlog:
        ax.set_autoscalex_on(True)
        ax.set_xscale('log')
    if ylog:
        ax.set_autoscaley_on(True)
        ax.set_yscale('log')

    # Save figure
    if sname:
        save(sname + ".pdf")

    return 1


def lumley_triangle(xi, eta, sname=None):
    """
    Print a lumley triangle visualising the anisotropy of the turbulence
    at a given grid point.
    :param sname: Name with which a pdf of the plot is saved.
    :param xi: Xi coordinate(s) in the triangle.
    :param eta: Eta coordinate(s) in the triangle.
    :return: void.
    """

    # Plot realizability limits
    limit_linestyle = '-k'

    # Left limit
    fig, ax = lining([0, -1/6], [0, 1/6], xlim=[-1/5, 1/2.8], ylim=[-0.01, 1/2.8], linestyle=limit_linestyle)
    # Right limit
    lining([0, 1/3], [0, 1/3], append_to_fig_ax=(fig, ax), linestyle=limit_linestyle)

    # Upper limit
    xi_range = np.linspace(-1/6, 1/3, 20)
    eta_lim = (1/27 + 2*xi_range**3)**0.5
    lining(xi_range, eta_lim, append_to_fig_ax=(fig, ax), linestyle=limit_linestyle)

    # Plot given data
    scattering(xi, eta, np.ones_like(xi),
               append_to_fig_ax=(fig, ax),
               alpha=0.4,
               scale=5,
               xlabel=r'$\xi$', ylabel=r'$\eta$',
               sname=sname)


def barycentric_map(eig1, eig2, eig3, sname=None, linestyle='x'):
    """
    Visualise the anisotropy using its eigenvalues mapped onto
    barycentric coordinates in a equilateral triangle. The triangle
    depicts 1C, 2C or 3C (component) turbulence.
    :param eig1: 1st eigenvalue of aij/bij.
    :param eig2: 2nd eigenvalue of aij/bij.
    :param eig3: 3rd eigenvalue of aij/bij.
    :param sname: Name with which the plot is saved as pdf.
    :param linestyle: Style of the plotted line inside the triangle.
    :return: 1: success.
    """
    # Define basis points for vertices as (x, y)
    vertex_1c = np.array([1, 0])
    vertex_2c = np.array([0, 0])
    vertex_3c = np.array([0.5, 3**0.5/2])

    # Define limits ([x1, x2], [y1, y2])
    lower = [np.array([vertex_1c[0], vertex_2c[0]]), np.array([vertex_1c[1], vertex_2c[1]])]
    left = [np.array([vertex_2c[0], vertex_3c[0]]), np.array([vertex_2c[1], vertex_3c[1]])]
    right = [np.array([vertex_1c[0], vertex_3c[0]]), np.array([vertex_1c[1], vertex_3c[1]])]

    # Plot boundaries, demarcate realizable triangle
    fig, ax = empty_plot()
    limit_linestyle = '-k'  # Solid black line
    lining(*lower, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax), xlim=[-0.1, 1.1], ylim=[-0.1, 1.1])
    lining(*left, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax))
    lining(*right, linestyle=limit_linestyle, append_to_fig_ax=(fig, ax))

    # Compute coordinates using the eigenvalues
    c_1c = eig1 - eig2
    c_2c = 2*(eig2 - eig3)
    c_3c = 3*eig3 + 1

    # Compute barycentric coordinates
    x = c_1c*vertex_1c[0] + c_2c*vertex_2c[0] + c_3c*vertex_3c[0]
    y = c_1c*vertex_1c[1] + c_2c*vertex_2c[1] + c_3c*vertex_3c[1]

    # Plot limiting states
    point_1C = np.array([1.02, -0.02])
    text_1C = "1C"
    angle_1C = 0
    point_2C = np.array([-0.07, -0.02])
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
        ax.text(*point, text, rotation=angle, rotation_mode='anchor', fontsize=16)
    plt.axis("off")


    # Plot all points into the triangle
    lining(x, y, linestyle=linestyle, alpha=0.3, sname=sname, append_to_fig_ax=(fig, ax))


def physical_confusion(case, confusion, show_background=False, zoom_data=None, sname=None):
    """
    Plot the confusion matrix results on the physical domain of
    a flowCase for qualitative evaluation.
    :param show_background: Show transparent contour of mean flow.
    :param data_dict: flow_dict from a flowCase object.
    :param confusion: Results from confusion function.
    :return: 1:success.
    """

    # Get physical data
    data_dict = case.flow_dict
    stat_x, stat_y = data_dict['x'], data_dict['y']
    nx, ny = data_dict['nx'], data_dict['ny']


    # Plot setup
    xs, ys, confusions = [], [], []
    zorders = [3, 2.9, 2.7, 2.8]
    labels = [r"\textsf{True-Positive}", r"\textsf{True-Negative}", r"\textsf{False-Positive}", r"\textsf{False-Negative}"]
    #["True-Positive", "True-Negative", "False-Positive", "False-Negative"]
    # labels = ["Richtig-Positiv", "Richtig-Negativ", "Falsch-Positiv", "Falsch-Negativ"]
    label_values = [1, 0, 0.3, 0.7]

    # Get boundaries and limits
    boundaries, xlabel, ylabel, title, xlim, ylim, mask, seeding = get_geometry_plot_data(case, return_seeding=True)
    stat_x = stat_x[~mask]
    stat_y = stat_y[~mask]
    confusion = confusion[~mask]

    if zoom_data:
        xlims = [xlim] + zoom_data[0]
        ylims = [ylim] + zoom_data[1]
        fracs = [0.9] + [0.49]*len(zoom_data[0])
        zname_adds = ["boxes"] + zoom_data[2]
        scales = [10] + zoom_data[3]
        legend_locs = [None] + zoom_data[4]
        xlabels = [xlabel] + zoom_data[5]
        ylabels = [ylabel] + zoom_data[6]
    else:
        xlims = [xlim]
        ylims = [ylim]
        fracs = [0.9]
        zname_adds = [""]
        scales = [10]
        legend_locs = ['upper right']
        xlabels = [xlabel]
        ylabels = [ylabel]

    # Produce regular plot and zoomed sections
    for frac, xlim, ylim, zname_add, scale, leg_loc, xlabel, ylabel in zip(fracs, xlims, ylims, zname_adds, scales, legend_locs, xlabels, ylabels):
        # Create figure
        fig, ax = empty_plot(figwidth=latex_textwidth*frac)


        # Find values corresponding to each label
        for mark in label_values:
            idx = np.where(confusion == mark)
            xs.append(stat_x[idx])
            ys.append(stat_y[idx])
            confusions.append(confusion[idx])

        # Setup colors
        clrs = confusion_cmap(label_values)

        # Scatter plot on physical domain
        for x, y, confusion, zorder in zip(xs, ys, confusions, zorders):
            ax.scatter(x, y, s=scale, c=confusion, cmap=confusion_cmap, vmin=0, vmax=1, zorder=zorder)


        # Boundaries, label, title, limits
        for boundary in boundaries:
            lining(*boundary,
                   linestyle='-',
                   color=cblack,
                   append_to_fig_ax=(fig, ax))
        set_labels_title(ax, xlabel, ylabel, title)
        set_limits(ax, xlim=xlim, ylim=ylim)


        # Legend
        legend_elements = list()
        if leg_loc:
            for clr, label in zip(clrs, labels):
                legend_elements.append(Line2D([0], [0], marker='o', color=clr, markersize=4, markerfacecolor=clr, label=label, linestyle=''))
            ax.legend(handles=legend_elements, loc=leg_loc)


        # Background flow
        if show_background:
            nx, ny = case.nx, case.ny
            X, Y = case.flow_dict['x'].reshape(nx, ny), case.flow_dict['y'].reshape(nx, ny)
            U, V = case.flow_dict['um'].reshape(nx, ny), case.flow_dict['vm'].reshape(nx, ny)
            if "NACA" in case.case_name:
                pass
            else:
                streams(ax, X, Y, U, V, start_points=seeding, color=cgrey)
            pass

        # Zoom boxes
        if len(xlims) > 1 and zname_add == "boxes":
            for xi, yi in zip(xlims[1:], ylims[1:]):
                # Create a Rectangle patch
                rect = patches.Rectangle((xi[0], yi[0]), abs(xi[1]-xi[0]), abs(yi[1] - yi[0]), linewidth=2.0, edgecolor=cblack, facecolor='none', zorder=10)

                # Add the patch to the Axes
                ax.add_patch(rect)

        # Save
        if sname:
            save(sname + "_" + zname_add + ".jpg")

    return 1


def physical_decision(case, decision, show_background=True, sname=None):
    """
    Plot the confusion matrix results on the physical domain of
    a flowCase for qualitative evaluation.
    :param show_background: Show transparent contour of mean flow.
    :param data_dict: flow_dict from a flowCase object.
    :param decision: Result from smooth prediction.
    :return: 1:success.
    """

    # Get physical data
    data_dict = case.flow_dict
    stat_x, stat_y = data_dict['x'], data_dict['y']
    nx, ny = data_dict['nx'], data_dict['ny']


    # Normalise with Softmax
    decision = np.c_[-decision, decision]
    expo = np.exp(decision)
    soft_deci = expo/expo.sum(axis=1).reshape(-1, 1)
    soft_deci = soft_deci[:, 1]


    # Create figure
    fig, ax = empty_plot(figwidth=latex_textwidth)
    boundaries, xlabel, ylabel, title, xlim, ylim, mask = get_geometry_plot_data(case)


    # Probability prediction
    soft_deci[mask] = np.nan
    contouring(stat_x.reshape(nx, ny),
               stat_y.reshape(nx, ny),
               soft_deci.reshape(nx, ny),
               cmap=general_cmap,
               levels=np.linspace(0, 1, 10),
               colorbar=True,
               append_to_fig_ax=(fig, ax))


    # Boundaries, label, title, limits
    for boundary in boundaries:
        lining(*boundary,
               linestyle='-',
               color=cblack,
               append_to_fig_ax=(fig, ax))
    set_labels_title(ax, xlabel, ylabel, title)
    set_limits(ax, xlim=xlim, ylim=ylim)


    # Background flow
    if show_background:
        ax.contourf(stat_x.reshape(nx, ny),
                    stat_y.reshape(nx, ny),
                    data_dict['um'].reshape(nx, ny),
                    cmap='viridis',
                    alpha=0.4)

    # Legend
    # legend_elements = list()
    # for clr, label in zip(clrs, labels):
    #     legend_elements.append(Line2D([0], [0], marker='o', color=clr, markersize=4, markerfacecolor=clr, label=label))
    # ax.legend(handles=legend_elements, loc=0)

    # Save
    if sname:
        save(sname + ".jpg")

    return 1


def get_geometry_plot_data(case, return_seeding=None):
    case_name = case.case_name
    x = case.flow_dict['x']
    y = case.flow_dict['y']

    if "PH" in case_name:
        boundaries = get_boundaries(case)
        seeding = np.array([np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 4.0, 4.5]),
                            np.array([1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 2.9, 0.2, 0.4, 0.1, 0.1, 0.1])]).T
        # xlabel = "$\dfrac{x}{H}$"
        # ylabel = "$\dfrac{y}{H}$"
        xlabel = "$x/H$"
        ylabel = "$y/H$"
        title = None
        xlim = [min(x) + 0.05, max(x) - 0.05]
        ylim = [min(y)-0.01, max(y)+0.01]
        mask = y < get_lower_boundary_y(case, x)

    elif "NACA" in case_name:
        boundaries = get_boundaries(case)
        if "top" in case_name:
            seeding = np.array([np.array([0.200, 0.200, 0.200, 0.20, 0.200, 0.2, 0.200, 0.20, 0.200, 0.2]),
                                np.array([0.075, 0.075, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3])]).T
        else:
            seeding = np.array([np.array([0.250, 0.250, 0.25, 0.250, 0.25, 0.250, 0.25, 0.250]),
                                -np.array([0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225])]).T
        # xlabel = "$\dfrac{x}{c}$"
        # ylabel = "$\dfrac{y}{c}$"
        xlabel = "$x/c$"
        ylabel = "$y/c$"
        title = None
        # xlim = [0.2, 0.97]
        # ylim = [0.01, 0.11]
        xlim = [min(x), max(x)]
        ylim = [min(y), max(y)]
        mask = np.zeros_like(x, dtype=bool)

    elif "TBL" in case_name:
        boundaries = get_boundaries(case)
        seeding = np.array([np.zeros(10)+300,
                            np.linspace(0, 8, 10)]).T
        xlabel = "$x$"
        ylabel = "$y$"
        xlim = [min(x) + 0.05, max(x) - 0.05]
        ylim = [0, 8]
        title = None
        mask = y < get_lower_boundary_y(case, x)

    elif "CBFS" in case_name:
        boundaries = get_boundaries(case)
        seeding = np.array([np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3.00, 3.0, 3.00, 3.00]),
                            np.array([1.1, 1.2, 1.4, 1.6, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.05, 0.15, 0.25, 0.58])]).T
        # xlabel = "$\dfrac{x}{H}$"
        # ylabel = "$\dfrac{y}{H}$"
        xlabel = "$x/H$"
        ylabel = "$y/H$"
        xlim = [-1, 6]
        ylim = [0, 1.4]
        title = None
        mask = y < get_lower_boundary_y(case, x)

    else:
        boundaries = get_boundaries(case)
        seeding = None
        xlabel = None
        ylabel = None
        title = None
        xlim = [min(x) + 0.05, max(x) - 0.05]
        ylim = [min(y), max(y)]
        mask = y < get_lower_boundary_y(case, x)

    if return_seeding:
        return boundaries, xlabel, ylabel, title, xlim, ylim, mask, seeding
    else:
        return boundaries, xlabel, ylabel, title, xlim, ylim, mask


def write_model_to_latex(term):
    if term=='const':
        tmp = sy.sympify(term.replace('**1.0', '').replace('np.cos', 'cos').replace('np.sin', 'sin'))
    elif term == "":
        tmp = r"\text{empty}"
    else:
        tmp = sy.sympify(term.replace('**1.0', '').replace('np.cos', 'cos').replace('np.sin', 'sin').replace('const',
                                                                                                              '1'))
    model_str = '$' + sy.latex(tmp) + '$'
    return model_str


def get_labels(lib):
    return np.array([write_model_to_latex(np.array(lib)[i]) for i in range(len(lib))])


def model_matrix(model_mat, library, gs, ax_ind=2):
    labels = get_labels(library)

    # cmap = get_colormap(which_colormap='c')
    cmap = general_cmap
    cmap.set_bad(color='white')

    ax = plt.subplot(gs[ax_ind])
    model_mat = np.ma.masked_where(model_mat == 0.0, model_mat)
    #    ax2.set_xticks([])#ax2.set_xticks(np.arange(0,len(MSE_vec),1))#
    #    ax0.set_xticklabels(all_labels, rotation=90)
    ax.tick_params(which='minor', length=0)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xticks(np.arange(0, len(labels), 1))
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)

    ax.set_yticklabels(np.arange(1, model_mat.shape[0] + 1))
    ax.set_yticks(np.arange(0, model_mat.shape[0], 1))
    ax.set_yticks(np.arange(-.5, model_mat.shape[0], 1), minor=True)

    if model_mat.shape[0] > 15:
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)

    ax.set_ylabel('model index $i$')
    r = np.max(abs(model_mat))
    # tmp = model_mat[ind_sort_y, :]
    im = ax.imshow(model_mat[:, :], cmap=cmap, aspect='auto', origin='lower', vmin=-r, vmax=r)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    return im


def model_matrix_with_score(df_models, candidate_library, indexes, sname=None):
    model_mat = np.stack(df_models['coef_'][indexes].values)
    num_models = model_mat.shape[0]
    num_active_cand = model_mat.shape[1]

    # Sort according to complexity
    ind_sort = np.array([np.count_nonzero(model_mat[::-1, :][:, i]) for i in range(num_active_cand)]).argsort()[::-1]

    plt.figure(figsize=(8, 15))
    gs = gridspec.GridSpec(2, 2, width_ratios=[10, 1.5], height_ratios=[1, 15])
    gs.update(wspace=0.05, hspace=0.05)

    # Plot model matrix
    im = model_matrix(model_mat[:, ind_sort], candidate_library[ind_sort], gs)

    # Plot colorbar
    ax0 = plt.subplot(gs[0])
    cb = plt.colorbar(im, cax=ax0, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    # Plot MSE values
    ax3 = plt.subplot(gs[3])
    ax3.plot(df_models['f1_test'][indexes].values, np.arange(0, num_models), 'o', label='')
    xticks = ax3.get_xticks()
    ax3.set_xticks([xticks[0], xticks[-1]], minor=True)
    if num_models > 15:
        ax3.set_yticks(np.arange(0, num_models, 1))
    ax3.set_ylim([-0.5, num_models-0.5])
    ax3.set_yticklabels([])
    ax3.set_xlabel('F1-measure')

    # Save
    if sname:
        save(sname + ".pdf")

    return 1


def precision_recall_plot(precision, recall, thresholds, sname=None):
    """
    Plot the precision-recall curve to evaluate a classifier's quality.
    :param precision: Return of scikit-learn's precision_recall_curve().
    :param recall: Return of scikit-learn's precision_recall_curve().
    :param thresholds: Return of scikit-learn's precision_recall_curve().
    :param sname: Savename or None.
    :return: 1:success.
    """
    fig, ax = lining(recall,
                     precision,
                     xlabel='Recall',
                     ylabel='Precision',
                     title='Precision-Recall curve')
    zero_thresh = np.argmin(np.abs(thresholds))  # Find estimator's decision threshold
    lining(recall[zero_thresh],
           precision[zero_thresh],
           xlabel='Recall',
           ylabel='Precision',
           title='Precision-Recall curve',
           xlim=[0., 1.1],
           ylim=[0., 1.1],
           linestyle='o',
           append_to_fig_ax=(fig, ax),
           sname=sname + '_PRcurve' if sname else None,
           )

    return 1


def input_correlation(inputs, labels):
    """
    Create a grid of scatter plots which highlight the correlations
    of marker in feature space.

    :param inputs: Array of inputs/features for each coordinate
    with shape [num_of_coordinates, num_of_features].
    :param labels: labels for each coordinate with shape [num_of_coordinates].
    :return: void.
    """
    dim = inputs.shape[1]

    fig = plt.figure()
    spec = gridspec.GridSpec(nrows=dim-1, ncols=dim-1)

    for i in range(dim-1):
        for j in range(dim-1):
            if j > i:
                continue
            ax = fig.add_subplot(spec[i, j])
            ax.scatter(inputs[:, j], inputs[:, i+1], s=5*np.ones_like(labels), c=labels, alpha=0.3)
            if j == 0:
                ax.set_ylabel("$\lambda " + str(i+2) + "$")
            if i == j:
                ax.set_title("$\lambda_" + str(j+1) + "$")

    return


def confusion_matrix(classifier, inputs, true_labels, normalise=False, labels=('False', 'True'), title=None, cmap=general_cmap, sname=None):
    """
    Confusion matrix predicted with inputs and compared to true labels.

    :param classifier: Instance of a scikit classifier.
    :param inputs: Features for the prediction.
    :param true_labels: Labels for evaluation of predictions.
    :param normalise: Normalise number of TP, TN, FP, FN.
    :param labels: Name for 'Negative' and 'Positive' label in
    this order.
    :param title: Title of the plot.
    :param cmap: Colormap.
    :param sname: sname a pdf of the figure.
    :return: void.
    """

    #  Create figure and confusion matrix
    plot1 = plot_confusion_matrix(classifier, inputs, true_labels,
                                   normalize=normalise, cmap=cmap,
                                   display_labels=labels)

    if title:
        plot1.ax_.set_title(title)

    if sname:
        save(sname + ".pdf")

    return


def streams(ax, xx, yy, u, v, density=10, base_map=False, start_points=None, color=cgrey):
    # Create uniform grid
    x = np.linspace(xx.min(), xx.max(), 1000)
    y = np.linspace(yy.min(), yy.max(), 1000)

    xi, yi = np.meshgrid(x, y)

    # Flatten
    px = xx.flatten()
    py = yy.flatten()
    pu = u.flatten()
    pv = v.flatten()
    speed = np.sqrt(u**2 + v**2).reshape(xx.shape)
    pspeed = speed.flatten()

    # Interpolate onto uniform grid
    gu = griddata((px, py), pu, (xi, yi), fill_value=0, method='linear')
    gv = griddata((px, py), pv, (xi, yi), fill_value=0, method='linear')
    gspeed = griddata((px, py), pspeed, (xi, yi), fill_value=0)

    # Others
    lw = 6*gspeed/np.nanmax(gspeed)
    if base_map:
        xx, yy = ax(xx, yy)
        xi, yi = ax(xi, yi)

    c = ax.streamplot(x, y, gu, gv, color=color.tolist(), arrowstyle='->', density=density, linewidth=0.8, start_points=start_points, zorder=5)


def show():
    plt.show()


def close_all():
    plt.close('all')


def save(fname):
    check_create_directory(fname)
    if "pdf" in fname:
        plt.savefig(fname, format='pdf', bbox_inches='tight')
    elif "jpg" in fname:
        plt.savefig(fname, format='jpg', bbox_inches='tight', dpi=600)
    elif "png" in fname:
        plt.savefig(fname, format='png', bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight')


def update_line_legend(ax, line_label):
    # Get legend handles and labels
    legend = ax.legend()
    lines = legend.get_lines()
    texts = legend.get_texts()
    labels = [text.get_text() for text in texts]

    assert(len(lines) == len(labels)), 'Invalid number of line2D handles for amount of labels'

    # Replace empty label with new one
    labels[-1] = line_label.replace('_', ' ')

    # Update legend
    ax.legend(lines, labels, loc='best')

    return 1



# def streamlines(x, y, u, v):
#     fig = plt.figure()
#     ax = fig.add_subplot()
#
#     ax.streamplot(x, y, u, v)
#
#     return
