# ###################################################################
# class flowCase
#
# Description
# Implement a class for simple handling of each data set.
#
# ###################################################################
# Author: hw
# created: 13. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from os.path import basename, splitext
import numpy as np

from uncert_ident.utilities import get_profile_data, FLOWCASE_KW, convert_dict_to_ndarray

from uncert_ident.data_handling.data_import import find_path_to_mat, load_mat, save_dict_to_mat, exist_mat, path_to_processed_data

from uncert_ident.methods.geometry import get_boundaries
from uncert_ident.methods.features import compute_all_features
from uncert_ident.methods.labels import compute_all_labels

from uncert_ident.visualisation import plotter as plot


#####################################################################
### Class
#####################################################################
class flowCase:

    def __init__(self, case_name, get_features=False, get_labels=False, verbose=True):
        """
        Load pre-processed data for given flowCase from .mat file.
        Give dimensionality of data.
        :param case_name: Name of .mat file for case.
        """

        # Load pre-processed data
        assert(type(case_name)) is str, "case_name is not a string: %r" % case_name
        if verbose:
            print("Loading data for %r" % case_name)
        path_to_mat = find_path_to_mat(case_name)
        self.flow_dict = load_mat(path_to_mat)
        self.case_name = basename(case_name)  # Remove path

        # Check availability of parameters
        for flowcase_kw in FLOWCASE_KW:
            assert flowcase_kw in self.flow_dict, 'Could not find %r in dictionary of mat file. Cannot read file without %r parameter.' % (flowcase_kw, flowcase_kw)

        # Find dimension of data
        assert(type(self.flow_dict['dimension'])) is int or float, "dimension is not int or float: %r" % self.flow_dict['dimension']
        self.dimension = int(self.flow_dict['dimension'])

        # Find coordinate system type
        assert(isinstance(self.flow_dict['coord_sys'], str)), 'Can only handle coordinate systems \"cartesian\" or \"polar\". Invalid coord_sys: %r' % self.flow_dict['coord_sys']
        self.coord_sys = self.flow_dict['coord_sys']

        # Find geometry
        assert (isinstance(self.flow_dict['geometry'], str)), 'Invalid geometry: %r' % self.flow_dict['geometry']
        self.geometry = self.flow_dict['geometry']

        # Find geometry scaling
        assert (type(self.flow_dict['geometry_scale'])) is int or float, "geoemtry_scale is not int or float: %r" % self.flow_dict['geometry_scale']
        self.geometry_scale = float(self.flow_dict['geometry_scale'])

        # Read properties
        assert (self.flow_dict['nx'] >= 1), "nx is invalid: %r" % self.flow_dict['nx']
        self.nx = int(self.flow_dict['nx'])
        assert (self.flow_dict['ny'] >= 1), "ny is invalid: %r" % self.flow_dict['ny']
        self.ny = int(self.flow_dict['ny'])
        self.num_of_points = int(self.nx*self.ny)

        # Features and labels
        self.feature_dict = dict()
        if get_features:
            self.get_features(verbose=verbose)
        self.label_dict = dict()
        if get_labels:
            self.get_labels(verbose=verbose)


    def get_features(self, recompute=False, verbose=True):
        case_name = self.case_name

        # Check for feature file
        if exist_mat(case_name + '_features') and not recompute:
            path_to_mat = find_path_to_mat(case_name + '_features')
            self.feature_dict = load_mat(path_to_mat)
            if verbose:
                print('Loaded ' + case_name + '_features')

        # If not available, compute features
        else:
            if verbose:
                print('Computing features for ' + case_name)
            self.feature_dict = compute_all_features(self.flow_dict)
            path_to_mat = find_path_to_mat(case_name)
            save_dict_to_mat(path_to_mat[:-4] + '_features', self.feature_dict)


    def get_labels(self, recompute=False, verbose=True):
        case_name = self.case_name

        # Check for label file
        if exist_mat(case_name + '_labels') and not recompute:
            path_to_mat = find_path_to_mat(case_name + '_labels')
            self.label_dict = load_mat(path_to_mat)
            if verbose:
                print('Loaded ' + case_name + '_labels')

        # If not available, compute labels
        else:
            if verbose:
                print('Computing labels for ' + case_name)
            self.label_dict = compute_all_labels(self.flow_dict)
            path_to_mat = find_path_to_mat(case_name)
            save_dict_to_mat(path_to_mat[:-4] + '_labels', self.label_dict)


    def show_geometry(self):
        """
        Plot the geometry for the given case using a Line2D.
        :return: figure and axis object for the plot.
        """

        # Get points on the boundaries
        boundaries = get_boundaries(self)

        # Configure the plot
        xlim = [np.min(self.flow_dict['x']), np.max(self.flow_dict['x'])]
        ylim = [np.min(self.flow_dict['y']), np.max(self.flow_dict['y'])]
        linestyle = '-k'  # Solid black line

        # Plot boundaries
        fig, ax = plot.empty_plot()
        for boundary in boundaries:
            plot.lining(*boundary, linestyle=linestyle, xlim=xlim, ylim=ylim, append_to_fig_ax=(fig, ax))

        return fig, ax


    def show_flow(self, var_key, contour=True, contour_level=10, xlim=None, ylim=None, colorbar=False, show_geometry=True):
        """
        Display the requested variable in the physical coordinates.
        :param show_geometry: Print outline of the boundary.
        :param contour_level: Set amount of or specific levels.
        :param contour: Option for contour plot instead of scatter.
        :param ylim: List with shape [min_y, max_y]
        :param xlim: List with shape [min_x, max_x]
        :param var_key: Key in flow_dict for flow quantity.
        :return: void.
        """

        assert(var_key in self.flow_dict), "var_key is not a valid key for flow data: %r" % var_key

        title = var_key + ' in ' + self.case_name

        if self.flow_dict[var_key].shape[0] == 3:
            var = self.flow_dict[var_key][0]  # x-component of vector
        else:
            var = self.flow_dict[var_key]

        # 2D Geometry outline
        if show_geometry:
            fig, ax = self.show_geometry()
        else:
            fig, ax = plot.empty_plot()

        # Contour plot
        if contour:
            plot.contouring(self.flow_dict['x'].reshape(self.nx, self.ny),
                            self.flow_dict['y'].reshape(self.nx, self.ny),
                            var.reshape(self.nx, self.ny),
                            xlabel='x', ylabel='y',
                            title=title,
                            levels=contour_level,
                            colorbar=colorbar,
                            xlim=xlim, ylim=ylim,
                            append_to_fig_ax=(fig, ax))

        # Scatter plot
        else:
            plot.scattering(self.flow_dict['x'],
                            self.flow_dict['y'],
                            var,
                            scale=20, alpha=0.3,
                            colorbar=colorbar,
                            xlim=xlim, ylim=ylim,
                            xlabel='x', ylabel='y',
                            title=title,
                            append_to_fig_ax=(fig, ax))


    def show_label(self, label_key='non_negative', show_all=False, show_geometry=True, show_background=False, labelpos='center', only_positive=False, zoom_box=False):
        """
        Display the labels in the physical coordinates.
        :param show_geometry: Print outline of the boundary.
        :param label_key: Key for the label (see utilities.py)
        :param show_all: Display all labels.
        :return: void.
        """

        if show_all:
            for key in self.label_dict:

                # Create figure
                fig, ax = plot.empty_plot(figwidth=plot.latex_textwidth)
                boundaries, xlabel, ylabel, title, xlim, ylim, mask, seeding = plot.get_geometry_plot_data(self, return_seeding=True)

                plot.scattering(self.flow_dict['x'][~mask],
                                self.flow_dict['y'][~mask],
                                self.label_dict[key][~mask],
                                scale=10, alpha=1.0,
                                title=key,
                                cmap=plot.confusion_cmap,
                                append_to_fig_ax=(fig, ax))


                # 2D Geometry outline
                if show_geometry:
                    for boundary in boundaries:
                        plot.lining(*boundary,
                                    linestyle='-',
                                    color=plot.cblack,
                                    append_to_fig_ax=(fig, ax))
                    plot.set_labels_title(ax, xlabel, ylabel, title)
                    plot.set_limits(ax, xlim=xlim, ylim=ylim)
                else:
                    fig, ax = plot.empty_plot(figwidth=plot.latex_textwidth)



                # Background flow
                if show_background:
                    nx, ny = self.nx, self.ny
                    X, Y = self.flow_dict['x'].reshape(nx, ny), self.flow_dict['y'].reshape(nx, ny)
                    U, V = self.flow_dict['um'].reshape(nx, ny), self.flow_dict['vm'].reshape(nx, ny)
                    if "NACA" in self.case_name:
                        pass
                    else:
                        plot.streams(ax, X, Y, U, V, start_points=seeding, color=plot.cblack)
                    pass

        else:
            assert (label_key in self.label_dict), "label_key is not a valid key for labels: %r" % label_key


            # Create figure
            fig, ax = plot.empty_plot(figwidth=plot.latex_textwidth*0.9)
            boundaries, xlabel, ylabel, title, xlim, ylim, mask, seeding = plot.get_geometry_plot_data(self, return_seeding=True)


            # plot.scattering(self.flow_dict['x'][~mask],
            #                 self.flow_dict['y'][~mask],
            #                 self.label_dict[label_key][~mask],
            #                 scale=10,
            #                 alpha=1.0,
            #                 cmap=plot.confusion_cmap,
            #                 append_to_fig_ax=(fig, ax))
            X, Y, LABEL = self.flow_dict['x'][~mask], self.flow_dict['y'][~mask], self.label_dict[label_key][~mask]
            positive = LABEL == 1
            negative = LABEL == 0
            xs = [X[positive], X[negative]]
            ys = [Y[positive], Y[negative]]
            labels = [LABEL[positive], LABEL[negative]]
            zorders = [3, 2.9, 2.7, 2.8]
            if only_positive:
                xs = [xs[0]]
                ys = [ys[0]]
                labels = [labels[0]]
            for x, y, label, zorder in zip(xs, ys, labels, zorders):
                ax.scatter(x, y, s=10, c=label, cmap=plot.confusion_cmap, vmin=0, vmax=1, zorder=zorder)

            # Background flow
            if show_background:
                nx, ny = self.nx, self.ny
                X, Y = self.flow_dict['x'].reshape(nx, ny), self.flow_dict['y'].reshape(nx, ny)
                U, V = self.flow_dict['um'].reshape(nx, ny), self.flow_dict['vm'].reshape(nx, ny)
                if "NACA" in self.case_name:
                    pass
                else:
                    plot.streams(ax, X, Y, U, V, start_points=seeding, color=plot.cgrey)
                pass

            # 2D Geometry outline
            if show_geometry:
                for boundary in boundaries:
                    plot.lining(*boundary,
                                linestyle='-',
                                color=plot.cblack,
                                append_to_fig_ax=(fig, ax))
                plot.set_labels_title(ax, xlabel, ylabel, title)
                plot.set_limits(ax, xlim=xlim, ylim=ylim)
            else:
                fig, ax = plot.empty_plot(figwidth=plot.latex_textwidth)

            # Legend
            labels = [r"\textsf{True-Positive}", r"\textsf{True-Negative}", r"\textsf{False-Positive}", r"\textsf{False-Negative}"]
            # labels = ["Richtig-Positiv", "Richtig-Negativ", "Falsch-Positiv", "Falsch-Negativ"]
            label_values = [1, 0, 0.3, 0.7]

            cmap = plot.confusion_cmap
            clrs = cmap(label_values)

            legend_elements = list()
            for clr, label in zip(clrs[:2], labels[:2]):
                legend_elements.append(
                    plot.line2d([0], [0], marker='o', linestyle='', color=clr, markersize=4, markerfacecolor=clr, label=label))
            ax.legend(handles=legend_elements, loc=labelpos)


            # Zoom box
            if isinstance(zoom_box, list):
                for xi, yi in zip(zoom_box[0], zoom_box[1]):
                    # Create a Rectangle patch
                    rect = plot.patches.Rectangle((xi[0], yi[0]), abs(xi[1] - xi[0]), abs(yi[1] - yi[0]), linewidth=2.0,
                                                  edgecolor=plot.cblack, facecolor='none', zorder=10)
                    ax.add_patch(rect)

        return fig, ax


    def show_features(self, feature_key='tke', show_all=False, show_geometry=True):
        """
        Display the feature in the physical domain.
        :param show_geometry: Print outline of the boundary.
        :param feature_key: Key for the feature (see utilties.py and features.py)
        :param show_all: Display all features (WARNING >50 plots).
        :return: 1:success.
        """

        # Plot all features
        if show_all:
            for key in self.feature_dict:
                # 2D Geometry outline
                if show_geometry:
                    fig, ax = self.show_geometry()
                else:
                    fig, ax = plot.empty_plot()

                plot.scattering(self.flow_dict['x'],
                                self.flow_dict['y'],
                                self.feature_dict[key],
                                scale=5, alpha=1.0,
                                xlabel='x', ylabel='y', title=key,
                                colorbar=True,
                                append_to_fig_ax=(fig, ax))
        # Plot requested feature
        else:
            assert (feature_key in self.feature_dict), "feature_key is not a valid key for features: %r" % feature_key

            # 2D Geometry outline
            if show_geometry:
                fig, ax = self.show_geometry()
            else:
                fig, ax = plot.empty_plot()

            plot.scattering(self.flow_dict['x'],
                            self.flow_dict['y'],
                            self.feature_dict[feature_key],
                            scale=5, alpha=1.0,
                            xlabel='x/h', ylabel='y/h', title=feature_key,
                            cmap='viridis',
                            colorbar=True,
                            append_to_fig_ax=(fig, ax))

        return 1


    def show_profile(self, var_key, coord_key, loc_list):
        """
        Display profile along coordinate for any variable
        :param coord_key: Varying coordinate x or y.
        :param loc_list: Location of fixed coordinate.
        :param var_key: Variable to display on the profile.
        :return: void.
        """

        # Horizontal profile
        if coord_key == 'x':
            xlabel = coord_key
            ylabel = var_key

        # Vertical profile
        elif coord_key == 'y':
            xlabel = var_key
            ylabel = coord_key

        else:
            xlabel = 'error'
            ylabel = 'error'
            assert (coord_key == 'x' or coord_key == 'y'), 'Invalid key for coordinates, ' \
                                                           'must be x or y instead: %r' % coord_key

        # Get data for requested profile
        profile_data = get_profile_data(self, var_key, coord_key, loc_list)

        # Plot the profile
        plot.lining(*profile_data, xlabel=xlabel, ylabel=ylabel, title=self.case_name,
                    line_label=var_key + ' at y = ' + str(loc_list))
        return


    def show_integral_quantity(self, var_key, coord_key, xlog=False, ylog=False, xlim=None, ylim=None):
        """
        Plot an integral quantity along the given coordinate.
        :param ylim: List with shape [min_y, max_y]
        :param xlim: List with shape [min_x, max_x]
        :param ylog: Option for log-scaled y-axis.
        :param xlog: Option for log-scaled x-axis.
        :param var_key: Key for an integral quantity.
        :param coord_key: Key for any coordinate.
        :return: 1: success.
        """

        # For convenience
        coord = self.flow_dict[coord_key]
        var = self.flow_dict[var_key]
        nx = self.flow_dict['nx']
        ny = self.flow_dict['ny']

        if coord.shape == var.shape:
            plot.lining(coord, var,
                        xlog=xlog, ylog=ylog,
                        xlim=xlim, ylim=ylim,
                        xlabel=coord_key, ylabel=var_key,
                        line_label=None)
            return 1

        elif coord_key == 'x':
            plot.lining(coord[::ny], var,
                        xlog=xlog, ylog=ylog,
                        xlim=xlim, ylim=ylim,
                        xlabel=coord_key, ylabel=var_key,
                        line_label=None)
            return 1

        elif coord_key == 'y':
            plot.lining(coord[:ny], var,
                        xlog=xlog, ylog=ylog,
                        xlim=xlim, ylim=ylim,
                        xlabel=coord_key, ylabel=var_key,
                        line_label=None)
            return 1

        else:
            assert False, 'Cannot print variable and coordinate: %r and %r' % (var_key, coord_key)


    def show_anisotropy(self, sname=None, lumley=False, loc_key=None, loc_value=None):
        """
        Plot an barycentric mapping of the anisotropy tensor into an
        equilateral triangle. Optionally plot the Lumley triangle.
        :param loc_value: Value of the location e.g. x-coordinate.
        :param loc_key: Key of the location variable.
        :param lumley: Option for lumley triangle.
        :param save: Option to save the file under given string.
        :return: void.
        """

        # Check input arguments for profile data
        if loc_key and loc_value is None:
            assert loc_value, 'Only coordinate key given, missing location value (loc_value).'
        elif not loc_key and loc_value or loc_value == 0:
            assert loc_key, 'Only location value given, missing coordinate key (loc_coord).'

        # If location is requested, get and plot profile data
        elif loc_key and loc_value or loc_value == 0:
            # Index in profile data
            if loc_key == 'x':
                loc_idx = 1
            elif loc_key == 'y':
                loc_idx = 0
            else:
                assert False, 'Cannot handle given loc_key.'

            if lumley:
                profile_xi = get_profile_data(self, 'bij_xi', loc_key, loc_value)
                profile_eta = get_profile_data(self, 'bij_eta', loc_key, loc_value)
                plot.lumley_triangle(profile_xi[loc_idx],
                                     profile_eta[loc_idx],
                                     sname=sname)
            else:
                profile_eig1 = get_profile_data(self, 'bij_eig1', loc_key, loc_value)
                profile_eig2 = get_profile_data(self, 'bij_eig2', loc_key, loc_value)
                profile_eig3 = get_profile_data(self, 'bij_eig3', loc_key, loc_value)
                plot.barycentric_map(profile_eig1[loc_idx],
                                     profile_eig2[loc_idx],
                                     profile_eig3[loc_idx],
                                     sname=sname)

        # Plot all data points
        else:
            if lumley:
                plot.lumley_triangle(self.flow_dict['bij_xi'],
                                     self.flow_dict['bij_eta'],
                                     sname=sname)
            else:
                plot.barycentric_map(self.flow_dict['bij_eig1'],
                                     self.flow_dict['bij_eig2'],
                                     self.flow_dict['bij_eig3'],
                                     sname=sname)


    # def show_budgets(self, budget_key, coord_key, xlim=None, ylim=None):
    #     """
    #     Plot all budget terms for the tke in BL flow, tke or any tauij
    #     element along any coordinate.
    #     :param budget_key: Key for the budgets of interest e.g. k, uu, ..
    #     :param coord_key: Key for the coordinate.
    #     :param xlim: Limits of plot in x.
    #     :param ylim: Limits of plot in y.
    #     :return: 1: success, void: error.
    #     """
    #
    #     fig, ax = plot.empty_plot()
    #
    #     # Keys for k and tauij
    #     tau_k_keys = ['k', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw']
    #
    #     # Find all budget keys for chosen budget
    #     for i, tau_k_key in enumerate(tau_k_keys):
    #         if tau_k_key == 'k' and self.case_name[:-5] == 'Bobke_FP_APG_LES':
    #             budget_keys = ALL_BUDGET_KEYS[i]
    #             break
    #         elif budget_key == 'k':
    #             budget_keys = ALL_BUDGET_KEYS[i+1]
    #             break
    #         # All tauij entries (uu, vv, ..)
    #         else:
    #             budget_keys = ALL_BUDGET_KEYS[i+1]
    #             break
    #
    #     # Plot budgets along coord_key
    #     for key in budget_keys:
    #         if key == 'diss_k':
    #             key = 'diss_rt'
    #         plot.lining(self.flow_dict[key], self.flow_dict[coord_key],
    #                     xlim=xlim, ylim=ylim,
    #                     ylabel=budget_key + ' budgets', xlabel=coord_key,
    #                     line_label=key,
    #                     append_to_fig_ax=(fig, ax))
    #
    #     return 1


#####################################################################
### Functions
#####################################################################
def compare_profiles(var_key, coord_key, loc, *cases, var_scale=1, xlim=None, ylim=None, xlog=False, ylog=False, append_to_fig_ax=(False, False)):
    """
    Compare a profile-plot setup between different flowCases.
    :param append_to_fig_ax: Append plot to another figure and axes.
    :param var_scale: Scaling factor for the variable.
    :param ylog: Optional logarithmic scale for y-axis.
    :param xlog: Optional logarithmic scale for x-axis.
    :param ylim: List with shape [min_y, max_y]
    :param xlim: List with shape [min_x, max_x]
    :param var_key: Any key within flowCase.flow_dict.
    :param coord_key: Varying coordinate.
    :param loc: Fixed location.
    :param cases: All flowCase objects.
    :return: Tuple of figure and axes.
    """

    # Generate new plot or append to given
    fig, ax = plot.check_append_to_fig_ax(append_to_fig_ax)

    for case in cases:
        profile_data = get_profile_data(case, var_key, coord_key, loc)

        # Horizontal profile
        if coord_key == 'x' or coord_key == 'y+':
            xlabel = coord_key
            ylabel = var_key
            scale_x = 1
            scale_y = var_scale

            # Vertical profile
        elif coord_key == 'y':
            xlabel = var_key
            ylabel = coord_key
            scale_x = var_scale
            scale_y = 1

        else:
            xlabel = 'error'
            ylabel = 'error'
            scale_x = 1
            scale_y = 1
            assert (coord_key == 'x' or coord_key == 'y'), 'Invalid key for coordinates, ' \
                                                           'must be x or y instead: %r' % coord_key

        plot.lining(*profile_data,
                    append_to_fig_ax=(fig, ax),
                    xlim=xlim, ylim=ylim,
                    xlog=xlog, ylog=ylog,
                    scale_x=scale_x, scale_y=scale_y,
                    xlabel=xlabel, ylabel=ylabel,
                    line_label=case.case_name)

    return fig, ax


def compare_integral_quantity(var_key, coord_key, *cases, xlog=False, ylog=False, xlim=None, ylim=None):
    """
    Compare an integral quantity along the given coordinate.
    :param ylim: List with shape [min_y, max_y]
    :param xlim: List with shape [min_x, max_x]
    :param ylog: Option for log-scaled y-axis.
    :param xlog: Option for log-scaled x-axis.
    :param var_key: Key for an integral quantity.
    :param coord_key: Key for any coordinate.
    :return: 1: success.
    """

    # Generate figure and subplot
    fig, ax = plot.empty_plot()

    for case in cases:
        # For convenience
        coord = case.flow_dict[coord_key]
        var = case.flow_dict[var_key]
        nx = case.flow_dict['nx']
        ny = case.flow_dict['ny']

        if coord.shape == var.shape:
            plot.lining(coord, var,
                        xlog=xlog, ylog=ylog,
                        xlim=xlim, ylim=ylim,
                        xlabel=coord_key, ylabel=var_key,
                        line_label=case.case_name,
                        append_to_fig_ax=(fig, ax))

        elif coord_key == 'x':
            plot.lining(coord[::ny], var,
                        xlog=xlog, ylog=ylog,
                        xlim=xlim, ylim=ylim,
                        xlabel=coord_key, ylabel=var_key,
                        line_label=case.case_name,
                        append_to_fig_ax=(fig, ax))

        elif coord_key == 'y':
            plot.lining(coord[:ny], var,
                        xlog=xlog, ylog=ylog,
                        xlim=xlim, ylim=ylim,
                        xlabel=coord_key, ylabel=var_key,
                        line_label=case.case_name,
                        append_to_fig_ax=(fig, ax))

        else:
            assert False, 'Cannot print variable and coordinate: %r and %r' % (var_key, coord_key)

    return 1


def compare_flow(var_key, *cases):
    """
    Compare distinct flow cases for a given variable on a
    grid of plots.
    :param var_key: Key of variable.
    :param cases: flowCase instances.
    :return: 1:success.
    """

    # Check valid key
    assert var_key in cases[0].flow_dict, "Invalid variable key given: %r" % var_key


    # Check required amount of subplots
    num_of_plots = len(cases)
    grid_length = int(num_of_plots ** 0.5) + 1


    # Create figure
    fig = plot.empty_plot(no_axis=True)

    # Add Title
    ax = fig.add_subplot(111)
    ax.set_title('Data basis with %r cases' % num_of_plots)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Plot individual cases
    for i, case in enumerate(cases):
        ax = fig.add_subplot(grid_length, grid_length, i + 1)  # Add plot to grid
        ax.get_xaxis().set_visible(False)  # No axes
        ax.get_yaxis().set_visible(False)

        x = case.flow_dict['x']
        y = case.flow_dict['y']
        nx = case.flow_dict['nx']
        ny = case.flow_dict['ny']
        var = case.flow_dict[var_key]

        plot.contouring(x.reshape(nx, ny),
                        y.reshape(nx, ny),
                        var.reshape(nx, ny),
                        xlabel='', ylabel='',
                        colorbar=False,
                        append_to_fig_ax=(fig, ax))
    plot.show()

    return 1





# def test_init_case(flowcase_object):
#     """
#     Test initialisation of a flow case object.
#     :param flowcase_object: class flowcase_object type object.
#     :return: void.
#     """
#
#     # Check properties
#     assert(flowcase_object.num_of_points >= 1), "num_of_points is invalid: %r" % flowcase_object.num_of_points
#
#     return 1


# def test_flow_case(flowcase_object):
#     """
#     Run all tests on a flow case object.
#     :param flowcase_object: class flowcase_object type object.
#     :return: 1
#     """
#
#     print("BEGIN TEST flow case " + str(flowcase_object.case_name))
#
#     test_init_case(flowcase_object)
#     # test_features(flowcase_object)
#     # test_labels(flowcase_object)
#
#     print("COMPLETED TEST flow case " + str(flowcase_object.case_name))
