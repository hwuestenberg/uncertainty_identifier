# ###################################################################
# Script test_ph_hill_geometry
#
# Description
# Create the hill geometry for the periodic hills test case.
#
# ###################################################################
# Author: hw
# created: 08. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import sys
import numpy as np
import scipy.interpolate as inter

from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.data_handling.data_import import load_csv, path_to_raw_data
from uncert_ident.visualisation.plotter import lining, show, empty_plot, scattering
from uncert_ident.methods.geometry import geometry_periodic_hills_lower, geometry_curved_backwards_facing_step_lower, geometry_naca_profile


#####################################################################
### Periodic hill
#####################################################################
# fig, ax = empty_plot()
# # Reference geometries from openFOAM RANS case
# fname = '../' + path_to_raw_data + 'Xiao_PH_DNS/case_1p0_refined_XYZ/dns-data/hills_geo.dat'
# geo_1p0 = load_csv(fname, ['x', 'y', 'z'], skip_header=1, delimiter=' ')
#
# fname = '../' + path_to_raw_data + 'Xiao_PH_DNS/case_1p5/dns-data/hills_geo.dat'
# geo_1p5 = load_csv(fname, ['x', 'y', 'z'], skip_header=1, delimiter=' ')
#
# fname = '../' + path_to_raw_data + 'Xiao_PH_DNS/case_0p5/dns-data/hills_geo.dat'
# geo_0p5 = load_csv(fname, ['x', 'y', 'z'], skip_header=1, delimiter=' ')
#
#
# # Plot reference
# lining(geo_1p0['x'], geo_1p0['y'], append_to_fig_ax=(fig, ax), xlim=[-9.1, 18.1], ylim=[-1.1, 2.1], line_label='Geo 1p0', linestyle='--')
# lining(geo_1p5['x'], geo_1p5['y'], append_to_fig_ax=(fig, ax), line_label='Geo 1p5', linestyle='--')
# lining(geo_0p5['x'], geo_0p5['y'], append_to_fig_ax=(fig, ax), line_label='Geo 0p5', linestyle='--')
#
#
# # Given (normalised) points
# points = np.array([[0, 9, 14, 20, 30, 40, 54], [28, 27, 24, 19, 11, 4, 0]])/28
# x_all = np.linspace(-9, 18, 1000)
#
#
# # Plot parameterised geometries
# lining(x_all, geometry_periodic_hills_lower(x_all, factor=1.0), line_label='hills 1p0', append_to_fig_ax=(fig, ax))
# lining(x_all, geometry_periodic_hills_lower(x_all, factor=1.5), line_label='hills 1p5', append_to_fig_ax=(fig, ax))
# lining(x_all, geometry_periodic_hills_lower(x_all, factor=0.5), line_label='hills 0p5', append_to_fig_ax=(fig, ax))
# lining(*points, linestyle='rx', append_to_fig_ax=(fig, ax), line_label='Points ref')


#####################################################################
### Bent pipe
#####################################################################
# fig, ax = empty_plot()
#
# # Get data
# case_name = 'Noorani_bent_pipe_DNS_R11700k01'
# case = flowCase(case_name)
#
# x = case.flow_dict['x']
# y = case.flow_dict['y']
#
# # Plot reference
# scattering(x, y, case.flow_dict['k'], append_to_fig_ax=(fig, ax))
#
# # Compute boundary coordinates
# radius = 1
# theta = np.linspace(0, 2*np.pi, 360)
#
# x = radius*np.cos(theta)
# y = radius*np.sin(theta)
#
# assert isinstance(x, np.ndarray), 'Wrong type in x: %r' % type(x)
# assert isinstance(y, np.ndarray), 'Wrong type in y: %r' % type(y)
# outer = tuple([x, y])
#
# # Plot boundary
# lining(*outer, append_to_fig_ax=(fig, ax))




#####################################################################
### Flat plate
#####################################################################
# fig, ax = empty_plot()
#
# # Get data
# case_name = 'Bobke_FP_APG_LES_m13n'
# case = flowCase(case_name)
#
# x = case.flow_dict['x']
# y = case.flow_dict['y']
#
# # Plot reference
# scattering(x, y, case.flow_dict['k'], append_to_fig_ax=(fig, ax))
#
# num_of_points = 100
# x_max = 3000
# x = np.linspace(0, x_max, num_of_points)
# y = np.zeros(num_of_points)
#
# lower = tuple([x, y])
# lining(*lower, append_to_fig_ax=(fig, ax))




#####################################################################
### Converging-diverging channel
#####################################################################
# fig, ax = empty_plot()
#
# # Get data
# case_name = 'Laval_converging_diverging_channel_DNS'
# case = flowCase(case_name)
#
# x = case.flow_dict['x']
# y = case.flow_dict['y']
#
# x_min = np.min(x)
# x_max = np.max(x)
# y_min = np.min(y)
# y_max = np.max(y)
#
# # Plot reference
# scattering(x, y, case.flow_dict['k'], append_to_fig_ax=(fig, ax))
#
#
# # Interpolate lower boundary from given mesh points
# lower_points = load_csv('../' + path_to_raw_data + case_name + '/lower_boundary_points.dat', col_names=['x', 'y'], delimiter=' ', skip_header=2)
# lower_fun = inter.interp1d(lower_points['x'], lower_points['y'])
# lower = (x, lower_fun(x))
#
# # Boundaries depend on min and max coordinates
# upper = (np.array([x_min, x_max]), np.array([y_max, y_max]))
# inlet = (np.array([x_min, x_min]), np.array([y_min, y_max]))
# outlet = (np.array([x_max, x_max]), np.array([y_min, y_max]))
#
# boundary = tuple([upper,
#                   lower,
#                   inlet,
#                   outlet])
#
# # Plot boundaries
# for b in boundary:
#     lining(*b, append_to_fig_ax=(fig, ax))




#####################################################################
### Curved-backwards-facing step
#####################################################################
# fig, ax = empty_plot()
#
# # Get data
# case_name = 'Bentaleb_curved_backwards_facing_step_LES'
# case = flowCase(case_name)
#
# x = case.flow_dict['x']
# y = case.flow_dict['y']
#
# x_min = np.min(x)
# x_max = np.max(x)
# y_min = np.min(y)
# y_max = np.max(y)
#
# # Plot reference
# scattering(x, y, case.flow_dict['k'], append_to_fig_ax=(fig, ax))
#
#
# # Lower boundary according to paper
# lower = (x, geometry_curved_backwards_facing_step_lower(x))
#
# # Boundaries depend on min and max coordinates
# upper = (np.array([x_min, x_max]), np.array([y_max, y_max]))
# inlet = (np.array([x_min, x_min]), np.array([1.0, y_max]))
# outlet = (np.array([x_max, x_max]), np.array([y_min, y_max]))
#
# boundary = tuple([upper,
#                   lower,
#                   inlet,
#                   outlet])
#
# # Plot boundaries
# for b in boundary:
#     lining(*b, append_to_fig_ax=(fig, ax))


#####################################################################
### NACA4412
#####################################################################
fig, ax = empty_plot()

# Get data
# case_name = 'Vinuesa_NACA4412_LES_top4n'
case_name = 'Tanarro_NACA0012_LES_top4n12'
case = flowCase(case_name)

x = case.flow_dict['x']
y = case.flow_dict['y']

xa = case.flow_dict['xa']
ya = case.flow_dict['ya']

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

# Plot reference
scattering(x, y, np.arange(2600), append_to_fig_ax=(fig, ax))

ref = (xa, ya)
lining(*ref, append_to_fig_ax=(fig, ax), linestyle='rx')

# Compute naca profile
x = np.linspace(0, 1, 1000)

# upper = geometry_naca_profile(x, 4, 4, 12, 'upper')
# lower = geometry_naca_profile(x, 4, 4, 12, 'lower')

upper = geometry_naca_profile(x, 0, 0, 12, 'upper')
lower = geometry_naca_profile(x, 0, 0, 12, 'lower')

boundary = tuple([upper,
                  lower])

# Plot boundaries
for b in boundary:
    lining(*b, append_to_fig_ax=(fig, ax), linestyle='k--')

show()




print('EOF test_geometry')
