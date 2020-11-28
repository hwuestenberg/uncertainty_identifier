# ###################################################################
# module geomerty
#
# Description
# Provide functions that compute the boundary coordinates for each
# flow case.
#
# ###################################################################
# Author: hw
# created: 13. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from scipy.interpolate import interp1d

from uncert_ident.data_handling.data_import import load_csv, path_to_raw_data


#####################################################################
### Functions
#####################################################################
def get_boundaries(flowcase_object):
    """
    Get the boundaries for the geometry of the given flowCase object.
    :param flowcase_object: Object of class flowCase or flowCase.flow_dict.
    :return: Tuple of boundaries (i.e.tuple of tuple of ndarrays with x,y-coordinates).
    """

    # Get case data
    if isinstance(flowcase_object, dict):
        x = flowcase_object['x']
        y = flowcase_object['y']
        geometry = flowcase_object['geometry']
        geometry_scale = flowcase_object['geometry_scale']
    else:
        x = flowcase_object.flow_dict['x']
        y = flowcase_object.flow_dict['y']
        geometry = flowcase_object.geometry
        geometry_scale = flowcase_object.geometry_scale


    # Choose function for boundaries according to geometry
    if geometry == 'converging_diverging_channel':
        boundaries = geometry_converging_diverging_channel(x, y)
    elif geometry == 'curved_backwards_facing_step':
        boundaries = geometry_curved_backwards_facing_step(x, y)
    elif geometry == 'periodic_hills':
        boundaries = geometry_periodic_hills(x, y, geometry_scale)
    # elif geometry == 'wall_mounted_cube':
    #     boundaries = geometry_wall_mounted_cube()
    # elif geometry == 'jet_in_cross_flow':
    #     boundaries = geometry_jet_in_cross_flow()
    elif geometry == 'flat_plate':
        boundaries = geometry_flat_plate(x)
    elif geometry == 'naca4412':
        x = np.linspace(0, 1)
        boundaries = geometry_naca4412(x)
    elif geometry == 'naca0012':
        x = np.linspace(0, 1)
        boundaries = geometry_naca0012(x)
    elif geometry == 'bent_pipe':
        boundaries = geometry_bent_pipe(x, y)
    else:
        assert False, 'Invalid geometry: %r' % geometry

    # Split into individual boundaries for simplicity
    return boundaries


def get_lower_boundary_y(flowcase_object, x):
    # Get case data
    if isinstance(flowcase_object, dict):
        geometry = flowcase_object['geometry']
        geometry_scale = flowcase_object['geometry_scale']
        geo_y = flowcase_object['y']
    else:
        geometry = flowcase_object.geometry
        geometry_scale = flowcase_object.geometry_scale
        geo_y = flowcase_object.flow_dict['y']


    # Choose function for boundaries according to geometry
    if geometry == 'converging_diverging_channel':
        y = geometry_converging_diverging_channel_lower(x)
    elif geometry == 'curved_backwards_facing_step':
        y = geometry_curved_backwards_facing_step_lower(x)
    elif geometry == 'periodic_hills':
        y = geometry_periodic_hills_lower(x, geometry_scale)
    elif geometry == 'flat_plate':
        y = 0
    elif geometry == 'naca4412':
        if any(geo_y > 0):
            y = geometry_naca_profile(x, 4, 4, 12, 'top')
        else:
            y = geometry_naca_profile(x, 4, 4, 12, 'bottom')
    elif geometry == 'naca0012':
        if any(geo_y > 0):
            y = geometry_naca_profile(x, 0, 0, 12, 'top')
        else:
            y = geometry_naca_profile(x, 0, 0, 12, 'bottom')
    else:
        assert False, 'Invalid geometry: %r' % geometry

    return y


def geometry_converging_diverging_channel(x_coords, y_coords):
    """
    Create a tuple of boundary coordinates for the
    converging-diverging channel case.
    :param y_coords: Coordinates in vertical direction.
    :param x_coords: Coordinates in horizontal direction.
    :return: Boundary = Tuple( Tuple(ndarray, ndarray), ...)
    """

    # Limits of the domain
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Boundaries from min and max coordinates
    upper = (np.array([x_min, x_max]), np.array([y_max, y_max]))
    all_x = np.linspace(x_min, x_max, 1000)
    lower = (all_x, geometry_converging_diverging_channel_lower(all_x))
    inlet = (np.array([x_min, x_min]), np.array([y_min, y_max]))
    outlet = (np.array([x_max, x_max]), np.array([y_min, y_max]))

    # Assemble tuple of all points
    boundary = tuple([upper,
                      lower,
                      inlet,
                      outlet])

    return boundary


def geometry_converging_diverging_channel_lower(x_coords):
    """
    Interpolate y coordinates on lower boundary for any x_coordinates
    from given mesh points.
    :param x_coords: Coordinates on the horizontal axis.
    :return: tuple of x- and y-ndarray of coordinates.
    """

    # Interpolate lower boundary from given mesh points
    lower_points = load_csv(path_to_raw_data + 'CDC-Laval/data/' + '/lower_boundary_points.dat',
                            col_names=['x', 'y'],
                            delimiter=' ', skip_header=2)
    lower_fun = interp1d(lower_points['x'], lower_points['y'], kind='linear')

    return lower_fun(x_coords)


def geometry_periodic_hills(x_coords, y_coords, factor=1):
    """
    Create a tuple of 2D points for all boundaries in the
    periodic hills case.
    :param factor: Parameterised geometry.
    :param y_coords: Coordinates in vertical direction.
    :param x_coords: Coordinates in horizontal direction.
    :return: Boundary = Tuple( Tuple(ndarray, ndarray), ...)
    """

    # Parameterised geometry
    x_max = 3.858 * factor + 5.142
    x_min = np.min(x_coords)
    y_max = np.max(y_coords)
    y_min = np.min(y_coords)

    # Upper boundary
    upper = [np.array([x_min, x_max]), np.array([y_max, y_max])]
    all_x = np.linspace(x_min, x_max, 1000)
    lower = (all_x, geometry_periodic_hills_lower(all_x, factor))
    inlet = (np.array([x_min, x_min]), np.array([1, y_max]))
    outlet = (np.array([x_max, x_max]), np.array([1, y_max]))

    # Assemble tuple of all points
    boundary = tuple([upper,
                      lower,
                      inlet,
                      outlet])

    return boundary


def geometry_periodic_hills_lower(x_coords, factor=1):
    """
    Compute the periodic hills geometry similar to Mellen et al 2000 or
    Xiao et al. 2020. But with dimensional x = x*28. Otherwise
    the geometry is wrong.
    :param factor: Factor for stretching/bulging of the geometry
    :param x_coords: Array of non-dimensional x-coordinates to be
    computed.
    :return: Non-dimensional y-coordinates.
    """

    # Loop requires iterable > Convert int and float to ndarray
    if isinstance(x_coords, int) or isinstance(x_coords, float):
        x_coords = np.array([x_coords])
    elif isinstance(x_coords, np.ndarray):
        pass
    else:
        assert False, 'Invalid input type: %r' % (type(x_coords))

    # Set geometry
    tot_len = 3.858 * factor + 5.142
    hill_len = 1.929 * factor
    pos_x_1 = 0.321
    pos_x_2 = 0.500
    pos_x_3 = 0.714
    pos_x_4 = 1.071
    pos_x_5 = 1.429

    # Loop over all given x_coords
    y = np.zeros_like(x_coords)
    for i, x in enumerate(x_coords):

        # Give mirrored and shifted coordinates for second hill
        if tot_len - hill_len <= x <= tot_len:
            shift = tot_len
        else:
            shift = 0

        # Shift and mirror polynomial, if x @ windward hill
        x = np.abs(x / factor - shift / factor)

        # Use polynomial depending on position
        if 0 <= x <= pos_x_1:
            x = x * 28  # Dimensionalise with original hill height to fix polynomial
            A = 1.000e+0
            B = 0.000e+0
            C = 2.420e-4
            D = -7.588e-5
            y[i] = np.min((np.ones_like(x), np.array(
                A + B * x + C * x ** 2 + D * x ** 3)), axis=0)

        elif pos_x_1 < x <= pos_x_2:
            x = x * 28  # Dimensionalise with original hill height to fix polynomial
            A = 8.955e-1
            B = 3.483e-2
            C = -3.628e-3
            D = 6.749e-5
            y[i] = A + B * x + C * x ** 2 + D * x ** 3

        elif pos_x_2 < x <= pos_x_3:
            x = x * 28  # Dimensionalise with original hill height to fix polynomial
            A = 9.213e-1
            B = 2.931e-2
            C = -3.234e-3
            D = 5.809e-5
            y[i] = A + B * x + C * x ** 2 + D * x ** 3

        elif pos_x_3 < x <= pos_x_4:
            x = x * 28  # Dimensionalise with original hill height to fix polynomial
            A = 1.445e+0
            B = -4.927e-2
            C = 6.950e-4
            D = -7.394e-6
            y[i] = A + B * x + C * x ** 2 + D * x ** 3

        elif pos_x_4 < x <= pos_x_5:
            x = x * 28  # Dimensionalise with original hill height to fix polynomial
            A = 6.402e-1
            B = 3.123e-2
            C = -1.988e-3
            D = 2.242e-5
            y[i] = A + B * x + C * x ** 2 + D * x ** 3

        elif pos_x_5 < x <= hill_len / factor:
            x = x * 28  # Dimensionalise with original hill height to fix polynomial
            A = 2.014e+0
            B = -7.180e-2
            C = 5.875e-4
            D = 9.553e-7
            y[i] = np.max((np.zeros_like(x), np.array(
                A + B * x + C * x ** 2 + D * x ** 3)), axis=0)

        # Set to zero beyond the hill
        else:
            y[i] = 0.0

    return y


def geometry_curved_backwards_facing_step(x_coords, y_coords):
    """
    Create a tuple of boundary coordinates for the
    curved backwards facing step case.
    :param y_coords: Coordinates in vertical direction.
    :param x_coords: Coordinates in horizontal direction.
    :return: Boundary = Tuple( Tuple(ndarray, ndarray), ...)
    """

    # Limits of the domain
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Lower boundary according to paper
    all_x = np.linspace(x_min, x_max, 10000)
    lower = (all_x, geometry_curved_backwards_facing_step_lower(all_x))

    # Boundaries depend on min and max coordinates
    upper = (np.array([x_min, x_max]), np.array([y_max, y_max]))
    inlet = (np.array([x_min, x_min]), np.array([1.0, y_max]))
    outlet = (np.array([x_max, x_max]), np.array([y_min, y_max]))

    boundary = tuple([upper,
                      lower,
                      inlet,
                      outlet])

    return boundary


def geometry_curved_backwards_facing_step_lower(x_coords):
    """
    Compute y-coordinates for given x-coordinates according to
    Bentaleb et al.
    :param x_coords: Array of non-dimensional x-coordinates to be
    computed.
    :return: Non-dimensional y-coordinates.
    """

    # Loop requires iterable > Convert int and float to ndarray
    if isinstance(x_coords, int) or isinstance(x_coords, float):
        x_coords = np.array([x_coords])
    elif isinstance(x_coords, np.ndarray):
        pass
    else:
        assert False, 'Invalid input type: %r' % (type(x_coords))

    # Set geometry/parameters
    r1 = 4.030174603
    r2 = 0.33396825
    x2 = 3.44888889
    y2 = 1.93587302
    pos_x_1 = 2.29936508
    pos_x_2 = 2.83492064
    pos_x_3 = 2.93650794

    # Loop all x_coords
    y = np.zeros_like(x_coords)
    for i, x in enumerate(x_coords):

        # Piece-wise equation
        if x < 0:
            y[i] = 1
        elif 0 <= x <= pos_x_1:
            y[i] = np.min((np.ones_like(x), np.array(
                1 - r1 + np.sqrt(r1 ** 2 - x ** 2))), axis=0)

        elif pos_x_1 < x <= pos_x_2:
            y[i] = y2 - np.sqrt(r1 ** 2 / 4 - (x2 - x) ** 2)

        elif pos_x_2 < x <= pos_x_3:
            y[i] = r2 - np.sqrt(r2 ** 2 - (pos_x_3 - x) ** 2)

        # Set to zero beyond the hill
        else:
            y[i] = 0.0

    return y


def geometry_flat_plate(x_coords):
    """
    Generate boundary coordinates of the flat plate.
    :param x_coords: Coordinates in x-direction
    :return: Boundary = Tuple( Tuple(ndarray, ndarray), ...)
    """

    # Limits of the domain
    x_max = np.max(x_coords)
    x_min = np.min(x_coords)

    # x and y coordinates of the lower boundary i.e. the plate
    all_x = np.linspace(x_min, x_max, 1000)
    y = np.zeros(all_x.shape[0])

    lower = tuple([all_x, y])

    boundary = tuple([lower])

    return boundary


def geometry_naca4412(x_coords):
    """
    Compute boundaries of the NACA 4412 airfoil.
    :return: Boundary = Tuple( Tuple(ndarray, ndarray), ...)
    """

    # Limits of the domain
    x_max = np.max(x_coords)
    x_min = np.min(x_coords)

    # Upper and lower boundary of the profile
    all_x = np.linspace(x_min, x_max, 1000)
    top = geometry_naca_profile(all_x, 4, 4, 12, 'top')
    bottom = geometry_naca_profile(all_x, 4, 4, 12, 'bottom')

    boundary = tuple([top,
                      bottom])

    return boundary


def geometry_naca0012(x_coords):
    """
    Compute boundaries of the NACA 0012 airfoil.
    :return: Boundary = Tuple( Tuple(ndarray, ndarray), ...)
    """

    # Limits of the domain
    x_max = np.max(x_coords)
    x_min = np.min(x_coords)

    # Upper and lower boundary of the profile
    all_x = np.linspace(x_min, x_max, 1000)
    top = geometry_naca_profile(all_x, 0, 0, 12, 'top')
    bottom = geometry_naca_profile(all_x, 0, 0, 12, 'bottom')

    boundary = tuple([top,
                      bottom])

    return boundary


def geometry_naca_profile(x_coords, m, p, t, surface):
    """
    Compute NACA 4-digit profile. According to airfoiltools.com.
    :param x_coords: Horizontal coordinates.
    :param m: Maximum camber times 100.
    :param p: Position of maximum camber times 10.
    :param t: Thickness times 100.
    :param surface: Top or bottom surface.
    :return: y-coordinates of the surface.
    """

    # For convenience
    if isinstance(x_coords, (int, float)):
        x = np.array([x_coords])
    else:
        x = x_coords

    # Correct parameters
    m = m/100
    p = p/10
    t = t/100

    # Camber coordinates
    yc = np.array([m/p**2*(2*p*xc - xc**2) if 0 <= xc < p else m/(1 - p)**2*(1 - 2*p + 2*p*xc - xc**2) for xc in x])
    dyc = np.array([2*m/p**2*(p - xc) if 0 <= xc < p else 2*m/(1 - p)**2*(p - xc) for xc in x])

    # Thickness distribution
    yt = t/0.2*(0.2969*x**0.5 - 0.126*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)

    theta = np.arctan(dyc)
    if surface == 'top' or surface == 'upper':
        points = ([x - yt*np.sin(theta), yc + yt*np.cos(theta)])
    elif surface == 'bottom' or surface == 'lower':
        points = ([x + yt*np.sin(theta), yc - yt*np.cos(theta)])
    else:
        assert False, 'Invalid surface parameter: %r' % surface

    return points


def geometry_bent_pipe(x_coords, y_coords):
    """
    Compute the outer coordinates for given point(s) x, y.
    :param x_coords: Position on x-axis.
    :param y_coords: Position on y-axis.
    :return: Tuple of tuple of boundary coordinates.
    """

    # Limits of the domain
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Compute x and y of outer geometry
    outer_radius = 1
    # theta = np.arctan(safe_divide(y_coords, x_coords))
    theta = np.linspace(0, 2*np.pi, 360)

    x = outer_radius*np.cos(theta)
    y = outer_radius*np.sin(theta)

    outer = tuple([x, y])
    boundary = tuple([outer])

    return boundary

