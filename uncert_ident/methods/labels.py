# ###################################################################
# module labels
#
# Description
# Methods for the evaluation of error metrics according to Ling &
# Templeton (2015).
#
# ###################################################################
# Author: hw
# created: 07. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
from scipy.optimize import newton, root_scalar

from uncert_ident.utilities import safe_divide, LABEL_KEYS, time_decorator


#####################################################################
### Functions
#####################################################################
def non_negativity_label(data_dict):
    """
    Evaluate non-negativity labels based on error metric by Ling and
    Templeton (2015). A negative eddy viscosity is an error and hence
    results in active label (=1).
    :param data_dict: Dictionary of flow data. Requires
    eddy viscosity.
    :return: Vector of non-negativity labels = {0, 1}.
    """

    return np.array([component < 0 for component in data_dict['nut']])


def anisotropy_label(data_dict):
    """
    Evaluate anisotropy labels based on error metric by Ling and
    Templeton (2015). An invariant, i.e. -2*II2, greater than 1/6
    corresponds to two-component turbulence and results in an active
    label (=1).
    :param data_dict: Dictionary of flow data. Requires
    2nd invariant of anisotropy tensor.
    :return: Vector of anisotropy labels = {0, 1}.
    """

    # Condition by invariant or eigenvalue
    cond_II = np.array([II > 1 / 6 for II in data_dict['IIb']])
    # cond_eig = np.array([2 * (eig1 ** 2 + eig1 * eig2 + eig2 ** 2) > 1 / 6
    #                      for eig1, eig2 in zip(data_dict['bij_eig1'], data_dict['bij_eig2'])])

    return cond_II


def non_linearity(data_dict):
    """
    Compute normalised ratio of cubic and linear eddy viscosity, see
    Ling & Templeton (2015), eq. (7).
    :param data_dict: Dictionary of flow data.
    :return: Ndarray of shape [n_points].
    """

    # For convenience
    lin_nut = data_dict['nut']
    cub_nut = data_dict['nut_cubic']

    diff_nut = np.abs(cub_nut - lin_nut)
    sum_nut = np.abs(cub_nut) + np.abs(lin_nut)
    non_linearity_ratio = safe_divide(diff_nut, sum_nut)

    return non_linearity_ratio


def non_linearity_objective_function(threshold, non_linearity_ratio):
    """
    Objective function for the optimisation that finds an appropriate
    threshold for the non-linearity label.
    :param threshold: Threshold for in-/active label decision.
    :param non_linearity_ratio: Ratio from function non_linearity.
    :return: Difference in average of active labels to Ling's 0.2.
    """

    # Determine labels with given threshold
    label = np.array([component > threshold for component in non_linearity_ratio])

    # Average labels
    avg_label = np.mean(label)

    # Difference to Ling's average of 0.2
    diff_in_avg = avg_label - 0.2

    return diff_in_avg


def non_linearity_label(data_dict):
    """
    Evaluate non-linearity labels based on error metric by Ling and
    Templeton (2015). The label is active (=1) when the normalised
    difference between Craft's cubic eddy viscosity and a linear eddy
    viscosity reach beyond a threshold. The threshold is set, so that
    20% of labels are active for any flow case. (Ling chose 0.15)
    An optimisation problem is solved to find a corresponding value
    for a given flow case.
    :param data_dict: Dictionary of flow data. Requires linear and
    cubic eddy viscosity.
    :return: Vector of non-linearity labels = {0, 1}.
    """

    # Compute normalised difference for nut
    non_linearity_ratio = non_linearity(data_dict)

    # Optimise to satisfy Ling's condition of 20% active labels
    initial_threshold = np.mean(non_linearity_ratio)  # According to Ling's paper
    try:
        threshold = newton(non_linearity_objective_function,
                           initial_threshold,
                           args=[non_linearity_ratio])
    except RuntimeError:
        threshold = initial_threshold

    non_linearity_labels = np.array([component > threshold
                                     for component in non_linearity_ratio])

    return non_linearity_labels


# Collect all label-computing functions
LABEL_FUNS = [non_negativity_label,
              anisotropy_label,
              # non_linearity_label
              ]


@time_decorator
def compute_all_labels(data_dict, laminar_criteria=True):
    """
    Call subfunction for each label and return a list of label arrays.
    :param laminar_criteria: Option to not deactivate laminar points.
    :param data_dict: Dictionary of flow data.
    :return: Dictionary of ndarrays. Arrays of shape [n_points].
    """

    all_labels = dict()

    for fun, key in zip(LABEL_FUNS, LABEL_KEYS):
        all_labels[key] = fun(data_dict)

        if laminar_criteria:
            # Deactivate laminar points
            all_labels[key][data_dict['k'] < 0.001] = False


    test_labels(all_labels)

    return all_labels




def test_labels(label_dict):
    """
    Test whether all labels are either active (True) or
    inactive (False).
    :param label_dict: Dictionary of all labels.
    :return: Success 1, Failure -1.
    """

    for label_key in LABEL_KEYS:
        for label in label_dict[label_key]:
            assert(label == True or label == False), "Invalid label found for " \
                                                     "label type " + str(label_key) + ": %r" % label
            return -1
    return 1
