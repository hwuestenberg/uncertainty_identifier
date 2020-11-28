# ###################################################################
# module classification
#
# Description
# Methods for handling and evaluation of Machine learning classifier.
#
# ###################################################################
# Author: hw
# created: 07. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import pandas as pd

from itertools import combinations

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, average_precision_score
from sklearn.feature_selection import mutual_info_classif

from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.data_handling.data_import import find_case_names, load_model
from uncert_ident.visualisation.plotter import baring, general_cmap, show
from uncert_ident.utilities import TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE, \
    PHYSICAL_KEYS, num_of_physical, INVARIANT_KEYS, num_of_invariants, FEATURE_KEYS, num_of_features, \
    LABEL_KEYS
from uncert_ident.visualisation.plotter import precision_recall_plot, physical_confusion, physical_decision


#####################################################################
### Functions
#####################################################################
def get_databasis_frames(case_name=[None], get_features=False, get_labels=False):
    """
    Generate dataframes for the databasis for easy data handling.
    :param get_features: Optional feature frame.
    :param get_labels: Optional label frame.
    :return: List of databasis-frame and optionally features, labels
    """

    # Collect return frames
    frames = list()

    # Read given case_name or use all cases
    if isinstance(case_name, list) and not isinstance(case_name[0], type(None)):
        case_names = case_name
    elif isinstance(case_name, str):
        case_names = [case_name]
    else:
        case_names = find_case_names()

    # Load given case names
    cases = list()
    for case_name in case_names:
        cases.append(flowCase(case_name, get_features=True, get_labels=True, verbose=False))

    # Collect data from all cases into lists
    nams, npts, dims, geos, feat, labl = list(), list(), list(), list(), list(), list()
    for case in cases:
        nams.append(case.case_name)
        npts.append(case.num_of_points)
        dims.append(case.dimension)
        geos.append(case.geometry)
        case.feature_dict.update({'case': case.case_name})
        case.label_dict.update({'case': case.case_name})
        feat.append(pd.DataFrame(case.feature_dict))
        labl.append(pd.DataFrame(case.label_dict))
        assert len(feat[-1]) == len(labl[-1]), f"Invalid num of points for feature and labels for {nams[-1]}"


    # Create metadata (databasis) frame
    frames.append(
        pd.DataFrame({
            "names": nams,
            "dimension": dims,
            "geometry": geos,
            "num_of_points": npts
        }))

    # Create feature frame
    if get_features:
        frames.append(
            pd.concat(feat)
        )

    # Create label frame
    if get_labels:
        frames.append(
            pd.concat(labl)
        )

    return frames


def get_sample_group_names(dataset):
    """
    Setup the list of cases for a defined test scenario.
    :param dataset: String for a test scenario.
    :return: list of hold-out and test data.
    """

    if dataset == 'sep':
        cases = [
            "PH-Breuer",
            "PH-Xiao",
            "CBFS-Bentaleb",
        ]
    elif dataset == 'ph':
        cases = [
            "PH-Breuer",
            "PH-Xiao",
        ]
    elif dataset == 'pg':
        cases = [
            "NACA",
            "TBL-APG",
        ]
    else:
        cases = [
            "PH-Breuer",
            "CBFS-Bentaleb",
            "NACA",
            "TBL-APG",
            "PH-Xiao",
        ]

    return cases


def get_sample_cases(group_names):
    if not isinstance(group_names, list):
        group_names = list([group_names])  # enforce list

    group_sample_cases = []
    list_all = find_case_names()
    for group_name in group_names:
        bools = [group_name in a for a in list_all]
        group_sample_cases.append(np.array(list_all)[bools].tolist())

    return group_sample_cases


def get_group_samples(scenario, feature_keys, label_index, sample_size=10000, group_names=None, undersampling=False):
    """
    Get subsampled data for each geometry.
    :param undersampling: Class balancing with undersampling.
    :param scenario: Choose test scenario, see get_sample_group_names.
    :param feature_keys: Keys for features in case.feature_dict.
    :param label_index: Index for label in case.label_dict.
    :param sample_size: Number of sampels created from a geometry.
    :return: List of samples with [feat, labl, groups]
    """

    if group_names:
        pass  # Use provided group_names

    # Or get names from scenario
    else:
        group_names = get_sample_group_names(scenario)

    # Get name of each case from group_names
    group_sample_cases = get_sample_cases(group_names)

    # Get subsampled data
    samples = []
    for i, sample_cases in enumerate(group_sample_cases):
        samples.append(
            subsample_data_group(sample_cases,
                                 feature_keys,
                                 label_index,
                                 sample_size=sample_size,
                                 undersampling=undersampling))
        local_sample_size = len(samples[i][1])
        samples[i].append([group_names[i]] * int(local_sample_size))

    return samples


def get_test_scenario(dataset):
    """
    Setup the list of cases for a defined test scenario.
    :param dataset: String for a test scenario.
    :return: list of hold-out and test data.
    """

    # Separation data only
    if dataset == 'sep':
        train = [
            'PH-Breuer-700',
            'PH-Breuer-1400',
            'PH-Breuer-2800',
            'PH-Breuer-5600',
            'PH-Breuer-10595',
            'PH-Xiao-08',
            'PH-Xiao-12',
            'PH-Xiao-10',
            'PH-Xiao-15',
        ]
        # hold_out = [
        #     'CBFS-Bentaleb',
        #     'TBL-APG-Bobke-b2',
        #     'TBL-APG-Bobke-b1',
        #     'TBL-APG-Bobke-m13',
        #     'TBL-APG-Bobke-m16',
        #     'TBL-APG-Bobke-m18',
        #     'CDC-Laval',
        #     # 'PH-Breuer-1400',
        #     'PH-Breuer-10595',
        #     # 'PH-Breuer-2800',
        #     # 'PH-Breuer-700',
        #     # 'PH-Breuer-5600',
        #     'NACA4412-Vinuesa-top-4',
        #     'NACA4412-Vinuesa-bottom-10',
        #     'NACA4412-Vinuesa-bottom-4',
        #     'NACA4412-Vinuesa-bottom-1',
        #     'NACA4412-Vinuesa-top-1',
        #     'NACA4412-Vinuesa-bottom-2',
        #     'NACA4412-Vinuesa-top-10',
        #     'NACA4412-Vinuesa-top-2',
        #     # 'PH-Xiao-12',
        #     'PH-Xiao-15',
        #     # 'PH-Xiao-08',
        #     # 'PH-Xiao-05',
        #     # 'PH-Xiao-10',
        #     'NACA0012-Tanarro-top-4'
        # ]
        test = [
            'CBFS-Bentaleb',    # Extrapolation (Different geometry)
            'PH-Xiao-05',       # Extrapolation (Steep hill geometry)
        ]


    # Pressure gradient data only
    elif dataset == 'pg':
        train = [
            'TBL-APG-Bobke-b1',
            'TBL-APG-Bobke-b2',
            'TBL-APG-Bobke-m13',
            'TBL-APG-Bobke-m16',
            'NACA4412-Vinuesa-bottom-10',
            'NACA4412-Vinuesa-bottom-4',
            'NACA4412-Vinuesa-bottom-2',
            'NACA4412-Vinuesa-bottom-1',
            'NACA4412-Vinuesa-top-10',
            'NACA4412-Vinuesa-top-2',
            'NACA4412-Vinuesa-top-1',
        ]
        # hold_out = [
        #     'CBFS-Bentaleb',
        #     'TBL-APG-Bobke-b2',
        #     # 'TBL-APG-Bobke-b1',
        #     # 'TBL-APG-Bobke-m13',
        #     # 'TBL-APG-Bobke-m16',
        #     # 'TBL-APG-Bobke-m18',
        #     'CDC-Laval',
        #     'PH-Breuer-1400',
        #     'PH-Breuer-10595',
        #     'PH-Breuer-2800',
        #     'PH-Breuer-700',
        #     'PH-Breuer-5600',
        #     # 'NACA4412-Vinuesa-top-4',
        #     # 'NACA4412-Vinuesa-bottom-10',
        #     # 'NACA4412-Vinuesa-bottom-4',
        #     # 'NACA4412-Vinuesa-bottom-1',
        #     # 'NACA4412-Vinuesa-top-1',
        #     # 'NACA4412-Vinuesa-bottom-2',
        #     # 'NACA4412-Vinuesa-top-10',
        #     # 'NACA4412-Vinuesa-top-2',
        #     'PH-Xiao-12',
        #     'PH-Xiao-15',
        #     'PH-Xiao-08',
        #     'PH-Xiao-05',
        #     'PH-Xiao-10',
        #     'NACA0012-Tanarro-top-4'
        # ]
        test = [
            'TBL-APG-Bobke-m18',        # Extrapolation (PG)
            'NACA0012-Tanarro-top-4',   # Extrapolation (Different wing profile)
            'NACA4412-Vinuesa-top-4',   # Interpolation (Intermediate Re)
        ]


    # All data
    elif dataset == 'all':
        train = [
            'TBL-APG-Bobke-b1',
            'TBL-APG-Bobke-b2',
            'TBL-APG-Bobke-m13',
            'TBL-APG-Bobke-m16',
            'PH-Breuer-700',
            'PH-Breuer-1400',
            'PH-Breuer-2800',
            'PH-Breuer-5600',
            'PH-Breuer-10595',
            'NACA4412-Vinuesa-bottom-10',
            'NACA4412-Vinuesa-bottom-4',
            'NACA4412-Vinuesa-bottom-2',
            'NACA4412-Vinuesa-bottom-1',
            'NACA4412-Vinuesa-top-10',
            'NACA4412-Vinuesa-top-4',
            'NACA4412-Vinuesa-top-2',
            'NACA4412-Vinuesa-top-1',
            'PH-Xiao-08',
            'PH-Xiao-12',
            'PH-Xiao-10',
            'PH-Xiao-15',
        ]
        # hold_out = [
        #     'CBFS-Bentaleb',
        #     'TBL-APG-Bobke-b2',
        #     # 'TBL-APG-Bobke-b1',
        #     # 'TBL-APG-Bobke-m13',
        #     'TBL-APG-Bobke-m16',
        #     # 'TBL-APG-Bobke-m18',
        #     'CDC-Laval',
        #     # 'PH-Breuer-1400',
        #     'PH-Breuer-10595',
        #     # 'PH-Breuer-2800',
        #     # 'PH-Breuer-700',
        #     # 'PH-Breuer-5600',
        #     # 'NACA4412-Vinuesa-top-4',
        #     # 'NACA4412-Vinuesa-bottom-10',
        #     # 'NACA4412-Vinuesa-bottom-4',
        #     # 'NACA4412-Vinuesa-bottom-1',
        #     # 'NACA4412-Vinuesa-top-1',
        #     # 'NACA4412-Vinuesa-bottom-2',
        #     # 'NACA4412-Vinuesa-top-10',
        #     # 'NACA4412-Vinuesa-top-2',
        #     # 'PH-Xiao-12',
        #     # 'PH-Xiao-15',
        #     # 'PH-Xiao-08',
        #     # 'PH-Xiao-05',
        #     # 'PH-Xiao-10',
        #     'NACA0012-Tanarro-top-4'
        # ]
        test = [
            'CBFS-Bentaleb',            # Extrapolation (Different geometry)
            'PH-Xiao-05',               # Extrapolation (Steeper hill geometry)
            'TBL-APG-Bobke-m18',        # Extrapolation (Strongest clauser pressure parameter over Re_tau)
            'NACA0012-Tanarro-top-4',   # Extrapolation (Different wing profile)
        ]


    else:
        assert False, "Invalid data set definition: {:s}".format(dataset)

    assert isinstance(train, list), 'hold_out_cases needs to be a list type'
    assert isinstance(test, list), 'test_cases needs to be a list type'

    return train, test


def get_scenario_set_variants(scenario):
    sset_names = get_sample_group_names(scenario)
    train_name_sets = [list(train) for train in list(combinations(sset_names, r=len(sset_names)-1))]
    test_names = list()
    for train_name_set in train_name_sets:
        for sset_name in sset_names:
            if sset_name not in train_name_set:
                test_names.append(sset_name)
            else:
                pass
    group_names = [train_list + [test_list] for train_list, test_list in zip(train_name_sets, test_names)]

    return group_names


def get_data(dataset_config):
    """
    Load all data and select hold_out and test data for given configuration.
    :param dataset_config: String for configuration (sep, pg or all)
    :return: List of hold_out and test cases and dataframes for features and labels.
    """
    # Define test scenario
    list_train, list_test = get_test_scenario(dataset_config)

    # Get dataframes
    df_info, df_X, df_y = get_databasis_frames(get_features=True, get_labels=True)

    return list_train, list_test, df_X, df_y


def get_groups(df_X, list_cases):
    """
    Get groups for LOGO-Cross validation.
    :param df_X: Dataframe with features.
    :param list_cases: List of hold-out data.
    :return: List of case_names for each index.
    """

    # Get group-wise data, remove test data
    grps = df_X.loc[:, 'case'].to_list()  # Get list of names with correct indexes
    grps = [case for case in grps if case in list_cases]  # Remove hold out data

    return grps


def get_n_groups(grps, list_cases):
    """
    Get number of distinct groups in grps.
    :param list_cases: List of hold-out data.
    :param grps: List of case_names ordered by data sets.
    :return: Number of groups.
    """
    # Sample weights, inverse frequency of sample points (1/n_samples)
    num_of_elements_per_group = [grps.count(name) for name in find_case_names() if name in list_cases]
    n_grps = len(num_of_elements_per_group)
    # s_weights = np.array([1 / num_of_elements_per_group[j] for j, num in enumerate(num_of_elements_per_group)
    #                       for i in range(num)])
    # C = s_weights.min()  # Adapt regularisation to order of sample weights

    return n_grps


def get_class_weights(list_cases, label_index, verbose=False):
    """
    Compute balanced class weights for given labels.
    :param verbose: Print result.
    :param list_cases: List of all cases.
    :param label_index: Index in label array.
    :return: Array of active and inactive class weights.
    """

    _, df_y = get_databasis_frames(get_labels=True)
    label = df_y.loc[df_y['case'].isin(list_cases), LABEL_KEYS[label_index]].to_numpy()
    cws = label.shape[-1] / 2 / np.bincount(label)
    dict_cws = dict(zip(np.unique(label).tolist(), cws))

    if verbose:
        bins = np.bincount(label)
        neg = bins[0] / np.sum(bins)
        pos = bins[1] / np.sum(bins)
        print("Total number of points:\t{0}".format(np.sum(bins)))
        print("Percent of Negatives:\t{:3.2f}".format(neg))
        print("Percent of Positives:\t{:3.2f}".format(pos))
        print("Class weights are: {0}".format(dict_cws))

    return dict_cws


def get_test_data(case_name, feature_keys, label_index):
    """
    Get all features and labels for cases specified in list_cases.
    :param case_name: Case name.
    :param feature_keys: List of keys for features.
    :param label_index: Integer for label.
    :return: Numpy array for features and labels.
    """

    # Load data
    df_data, df_X, df_y = get_databasis_frames(case_name, True, True)

    # Extract features and labels
    X = get_feature_for_list(df_X, [case_name], feature_keys)
    y = get_label_for_list(df_y, [case_name], label_index)

    return X, y


def get_feat_labl_for_list(df_X, df_y, list_cases, feature_keys, label_index, feature_only=False, label_only=False):
    """
    Get all features and labels for cases specified in list_cases.
    :param label_only: Return only labels
    :param feature_only: Return only features.
    :param df_X: Dataframe of features.
    :param df_y: Dataframe of labels.
    :param list_cases: List of case_names.
    :param feature_keys: List of keys for features.
    :param label_index: Integer for label.
    :return: Numpy array for features and labels.
    """

    # Extract features and labels
    X = get_feature_for_list(df_X, list_cases, feature_keys)
    y = get_label_for_list(df_y, list_cases, label_index)

    if feature_only:
        return X
    elif label_only:
        return y
    else:
        return X, y


def get_feature_for_list(df_X, list_cases, feature_keys):
    """
    Get all features for cases specified in list_cases.
    :param df_X: Dataframe of features.
    :param list_cases: List of case_names.
    :param feature_keys: List of keys for features.
    :return: Numpy array for features and labels.
    """
    # Extract features on all points
    X = df_X.loc[df_X['case'].isin(list_cases), feature_keys].to_numpy()

    return X


def get_label_for_list(df_y, list_cases, label_index):
    """
    Get all features and labels for cases specified in list_cases.
    :param df_y: Dataframe of labels.
    :param list_cases: List of case_names.
    :param label_index: Integer for label.
    :return: Numpy array for features and labels.
    """
    # Extract labels
    y = df_y.loc[df_y['case'].isin(list_cases), LABEL_KEYS[label_index]].to_numpy()

    return y


def test_train_split_group_tsc_feature(lib_data, list_train, list_test, list_groups):
    """
    Groupwise splitting of features for TSC where features are in lib_data.
    :param lib_data: Numpy array of features evaluated for each sample.
    :param list_train: List of case_names for training.
    :param list_test: List of case_names for testing.
    :param list_groups: List of case_names in sample order.
    :return: Arrays for train and test features.
    """
    # Number of samples for scenario
    num_of_samples = len(list_groups)

    # Groupwise splitting
    train_bools = np.isin(np.array(list_groups), np.array(list_train))
    test_bools = np.isin(np.array(list_groups), np.array(list_test))
    X_train = lib_data[train_bools]
    X_test = lib_data[test_bools]
    assert X_train.shape[0] + X_test.shape[0] == num_of_samples

    return X_train, X_test


def test_train_split_group_tsc_label(array_y, list_train, list_test, list_groups):
    """
    Groupwise splitting of features for TSC where features are in lib_data.
    :param array_y: Numpy array of labels.
    :param list_train: List of case_names for training.
    :param list_test: List of case_names for testing.
    :param list_groups: List of case_names in sample order.
    :return: Arrays for train and test features.
    """
    # Number of samples for scenario
    num_of_samples = len(list_groups)

    # Groupwise splitting
    train_bools = np.isin(np.array(list_groups), np.array(list_train))
    test_bools = np.isin(np.array(list_groups), np.array(list_test))
    y_train = array_y[train_bools]
    y_test = array_y[test_bools]
    assert y_train.shape[0] + y_test.shape[0] == num_of_samples

    return y_train, y_test


def test_train_split_group_label(df_y, list_train, list_test, list_groups, label_index):
    """
    Groupwise splitting of labels into train and test data.
    :param df_y: Dataframe of labels.
    :param list_train: List of case_names for training.
    :param list_test: List of case_names for testing.
    :param list_groups: List of case_names in sample order.
    :param label_index: Index for the label/marker/metric.
    :return: Arrays for train and test labels.
    """
    # Number of samples for scenario
    num_of_samples = len(list_groups)

    # Split labels
    y_train = get_label_for_list(df_y, list_train, label_index)
    y_test = get_label_for_list(df_y, list_test, label_index)
    assert y_train.shape[0] + y_test.shape[0] == num_of_samples

    return y_train, y_test


def test_train_split_group_feature(df_X, list_train, list_test, list_groups, feature_keys):
    """
    Groupwise splitting of labels into train and test data.
    :param df_X: Dataframe of features.
    :param list_train: List of case_names for training.
    :param list_test: List of case_names for testing.
    :param list_groups: List of case_names in sample order.
    :param feature_keys: Keys for features.
    :return: Arrays for train and test labels.
    """
    # Number of samples for scenario
    num_of_samples = len(list_groups)

    # Split labels
    X_train = get_feature_for_list(df_X, list_train, feature_keys)
    X_test = get_feature_for_list(df_X, list_test, feature_keys)
    assert X_train.shape[0] + X_test.shape[0] == num_of_samples

    return X_train, X_test


def test_train_split_feat_labl(df_X, df_y, list_train, list_test, feature_keys, label_index, sample_size=10000, verbose=False):
    """
    Split each of features and labels into train and test data and transform into numpy arrays
    for scikit-learn functions. Uses the lists of train and test data defined in get_data().
    :param verbose: Print percent of test data.
    :param df_X: Dataframe of features.
    :param df_y: Dataframe of labels.
    :param list_train: List of hold-out data.
    :param list_test: List of test data.
    :param feature_keys: List of features to be used.
    :param label_index: Index for error metric to be used.
    :return: Train/test features and labels as numpy arrays.
    """
    # Get groups
    list_groups = get_groups(df_X, list_train + list_test)

    # Splitting
    X_train, X_test = test_train_split_group_feature(df_X, list_train, list_test, list_groups, feature_keys)
    y_train, y_test = test_train_split_group_label(df_y, list_train, list_test, list_groups, label_index)

    a_train, b_train = subsample_data(list_train, feature_keys, label_index, sample_size=sample_size)
    a_test, b_test = subsample_data(list_test, feature_keys, label_index, sample_size=sample_size)

    # Print percent of test data
    if verbose:
        print("Percentage of test data: {:3.2f}".format(
            X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])))

    return X_train, X_test, y_train, y_test


def test_train_split_feat_labl_subsample(list_train, list_test, feature_keys, label_index, sample_size=10000, verbose=False):
    """
    Split each of features and labels into train and test data and transform into numpy arrays
    for scikit-learn functions. Uses the lists of train and test data defined in get_data().
    :param verbose: Print percent of test data.
    :param list_train: List of hold-out data.
    :param list_test: List of test data.
    :param feature_keys: List of features to be used.
    :param label_index: Index for error metric to be used.
    :return: Train/test features and labels as numpy arrays.
    """
    # Subsample and split
    X_train, y_train, groups_train = subsample_data(list_train, feature_keys, label_index, sample_size=sample_size)
    X_test, y_test, groups_test = subsample_data(list_test, feature_keys, label_index, sample_size=sample_size)
    groups = [groups_train, groups_test]

    # Print percent of test data
    if verbose:
        print("Percentage of test data: {:3.2f}".format(
            X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])))

    return X_train, X_test, y_train, y_test, groups


def subsample_data(list_all, feature_keys, label_index, sample_size=10000):
    """
    Subsample the given data sets with sample size points.
    Random sampling wihtout replacement.
    :param list_all: List of flowCases.
    :param feature_keys: Keys for features.
    :param label_index: Index of the label.
    :param sample_size: No of points to randomly sample.
    :return: feature array, label_array
    """
    # sample_size = int(1e4)
    # label_index = 0
    # feature_keys = FEATURE_KEYS
    label = LABEL_KEYS[label_index]


    # Initiate arrays for all features and labels
    all_feat = np.array([])
    all_labl = np.array([])
    groups = []

    # Loop
    for case in list_all:
        s_size = int(sample_size/2)
        df_data, df_feat, df_labl = get_databasis_frames(case, True, True)


        # Exclude zero feature points
        idx_zeros = df_feat.index[df_feat.loc[:, :'inv46'].eq(0.0).all(axis=1)]
        # print(f"{len(idx_zeros)} zeros excluded")
        df_feat.drop(df_feat.index[idx_zeros])
        df_labl.drop(df_feat.index[idx_zeros])


        # N_inactive, N_active for label
        n0, n1 = np.bincount(df_labl.loc[df_labl['case'] == case, label])
        n_minority = np.min([n0, n1])
        if n_minority < s_size:
            # print(f"WARNING\tNot enough points in data set {case}\tn_minority {n_minority} < {s_size} samples\t{n_minority*2} points will be used.")
            s_size = n_minority


        # Get indexes
        bools0 = df_labl[label] == 0
        bools1 = df_labl[label] == 1
        idx0 = df_labl.index[bools0].to_list()
        idx1 = df_labl.index[bools1].to_list()
        assert len(idx0) == n0, "Invalid bincount operation: idx0 {idx0} != {n0} n0"
        assert len(idx1) == n1, "Invalid bincount operation: idx1 {idx1} != {n1} n1"


        # Random sampling without replacement
        sample_idx0 = np.random.choice(idx0, s_size, replace=False)
        sample_idx1 = np.random.choice(idx1, s_size, replace=False)
        assert all(np.isin(sample_idx0, idx0)), "Invalid sampling for idx0!"
        assert all(np.isin(sample_idx1, idx1)), "Invalid sampling for idx1!"


        # Get sampled features and labels
        feat0 = df_feat.iloc[sample_idx0][feature_keys].to_numpy()
        labl0 = df_labl.iloc[sample_idx0][label].to_numpy()
        feat1 = df_feat.iloc[sample_idx1][feature_keys].to_numpy()
        labl1 = df_labl.iloc[sample_idx1][label].to_numpy()


        # Merge active/inactive arrays
        feat = np.concatenate([feat0, feat1], axis=0)
        labl = np.concatenate([labl0, labl1], axis=0)



        # Merge into all arrays
        if all_feat.shape == (0,):
            all_feat = feat
            all_labl = labl
            groups = groups + [case] * s_size*2
        else:
            all_feat = np.concatenate([all_feat, feat], axis=0)
            all_labl = np.concatenate([all_labl, labl], axis=0)
            groups = groups + [case] * s_size*2

        del df_data, df_feat, df_labl

    return all_feat, all_labl, groups


def subsample_data_group(list_all, feature_keys, label_index, sample_size=10000, undersampling=False):
    """
    Subsample the given data sets with sample size points.
    Random sampling wihtout replacement.
    :param undersampling: Class balancing with undersampling.
    :param list_all: List of flowCases.
    :param feature_keys: Keys for features.
    :param label_index: Index of the label.
    :param sample_size: No of points to randomly sample.
    :return: feature array, label_array
    """
    # sample_size = int(1e4)
    # label_index = 0
    # feature_keys = FEATURE_KEYS
    label = LABEL_KEYS[label_index]


    # Load data
    df_data, df_feat, df_labl = get_databasis_frames(list_all, True, True)


    # Check total number of samples
    num_of_samples = len(df_labl)
    if sample_size > num_of_samples:
        print(f"Sample size is larger than available samples: sample_size {int(sample_size)} > {num_of_samples} number of samples)")
        print(f"Sample size reduced to number of samples:\t{num_of_samples}")
        sample_size = int(num_of_samples)


    # Correct index
    df_feat.index = np.arange(len(df_feat))
    df_labl.index = np.arange(len(df_labl))


    # Exclude zero feature points
    idx_zeros = df_feat.index[df_feat.loc[:, :'inv46'].eq(0.0).all(axis=1)]
    # print(f"{len(idx_zeros)} zeros excluded")
    df_feat.drop(df_feat.index[idx_zeros])
    df_labl.drop(df_feat.index[idx_zeros])


    # Balance classes by sampling equal amounts from majority and minority class
    if undersampling:
        # Sample twice (majority + minority) -> half sample size
        s_size = int(sample_size / 2)

        # N_inactive, N_active for label
        n0, n1 = np.bincount(df_labl.loc[np.isin(df_labl['case'].to_list(), list_all), label])
        n_minority = np.min([n0, n1])
        if n_minority < s_size:
            print(f"WARNING\tNot enough points in data set {list_all}\tn_minority {n_minority} < {s_size} samples\t{n_minority*2} points will be used.")
            s_size = n_minority


        # Get indexes for active and inactive marker
        bools0 = df_labl[label] == 0
        bools1 = df_labl[label] == 1
        idx0 = df_labl.index[bools0].to_list()
        idx1 = df_labl.index[bools1].to_list()
        assert len(idx0) == n0, "Invalid bincount operation: idx0 {idx0} != {n0} n0"
        assert len(idx1) == n1, "Invalid bincount operation: idx1 {idx1} != {n1} n1"


        # Random sampling without replacement
        sample_idx0 = np.random.choice(idx0, s_size, replace=False)
        sample_idx1 = np.random.choice(idx1, s_size, replace=False)
        assert all(np.isin(sample_idx0, idx0)), "Invalid sampling for idx0!"
        assert all(np.isin(sample_idx1, idx1)), "Invalid sampling for idx1!"

        # Get sampled features and labels
        feat0 = df_feat.iloc[sample_idx0][feature_keys].to_numpy()
        labl0 = df_labl.iloc[sample_idx0][label].to_numpy()
        feat1 = df_feat.iloc[sample_idx1][feature_keys].to_numpy()
        labl1 = df_labl.iloc[sample_idx1][label].to_numpy()


        # Merge active/inactive arrays
        feat = np.concatenate([feat0, feat1], axis=0)
        labl = np.concatenate([labl0, labl1], axis=0)


    # Subsample (without class balancing)
    else:
        s_size = int(sample_size)
        all_idx = df_labl.index.to_list()
        sample_idx = np.random.choice(all_idx, s_size, replace=False)
        assert all(np.isin(sample_idx, all_idx)), "Invalid SUBsampling!"

        # Get sampled features and labels
        feat = df_feat.iloc[sample_idx][feature_keys].to_numpy()
        labl = df_labl.iloc[sample_idx][label].to_numpy()


    del df_data, df_feat, df_labl

    return [feat, labl]


def get_config_from_filename(filename):
    """
    Identify case configuration from filename.
    :param filename: Path to file and name.
    :return: List of model_constructor, scenario, label_name, label_index.
    """
    # Define possible scenarios and labels
    algorithms = ['logReg', 'tsc']
    scenarios = ['sep', 'pg', 'all']
    labels = ['non_negative', 'anisotropic', 'non_linear']
    ret = []

    # Search each algorithm, scenario and label
    for algo in algorithms:
        if filename.find(algo + "_") + 1:
            ret.append(get_identifier_constructor(algo))
    for scnr in scenarios:
        if filename.find("_" + scnr + "_") + 1:
            ret.append(scnr)
    for labl in labels:
        if filename.find("_" + labl + "_") + 1:
            ret.append(labl)
            ret.append({'non_negative': 0, 'anisotropic': 1, 'non_linear': 2}[labl])

    # ret = [constructor, scenario, label_name, label_index]
    return ret


def get_identifier_constructor(algorithm):
    """
    Get scikit-learn model constructor for a given filename abbreviation.
    :param algorithm: Abbreviation for an algorithm in filename.
    :return:
    """
    # Find constructor from algorithm
    if algorithm == 'logReg':
        model_constructor = LogisticRegression
    elif algorithm == 'tsc':
        model_constructor = RidgeClassifier
    else:
        assert False, f"Invalid algorithm in filename:\t{algorithm}"

    return model_constructor


def predict_case(identifier, prediction_case_name, df_X, df_y, feature_keys, label_index, sname=None):
    """
    Predict on physical domain for given flow case.
    :param feature_keys: Keys for features to use.
    :param sname: Savename or None.
    :param label_index: Choice of label.
    :param identifier: sklearn-estimator, fitted to data.
    :param prediction_case_name: Any case_name.
    :param df_X: Dataframe of features.
    :param df_y: Dataframe of labels.
    :return: 1:success.
    """

    # Build confusion matrix from predicted and true labels
    pred_feat = df_X.loc[df_X['case'] == prediction_case_name, feature_keys].to_numpy()
    pred_labl = identifier.predict(pred_feat)
    true_labl = df_y.loc[df_y['case'] == prediction_case_name, LABEL_KEYS[label_index]].to_numpy()
    confusion = confusion_matrix(pred_labl, true_labl)
    decision = identifier.decision_function(pred_feat)

    # Get test case physical data
    pred_data = flowCase(prediction_case_name)

    # Plot confusion on physical domain
    physical_confusion(pred_data, confusion, show_background=False, sname=sname + '_confusion' if sname else None)
    physical_decision(pred_data, decision, show_background=False, sname=sname + '_decision' if sname else None)

    return 1


def predict_case_tsc(identifier, prediction_case_name, candidate_lib_data, df_y, list_groups, label_index, sname=None):
    """
    Predict on physical domain for given flow case.
    :param label_index: Index for label/marker.
    :param list_groups: List of case_names for given scenario.
    :param df_y: Dataframe of labels.
    :param candidate_lib_data: Array of evaluated candidates for each sample.
    :param sname: Savename or None.
    :param identifier: sklearn-estimator, fitted to data.
    :param prediction_case_name: Any case_name.
    :return: 1:success.
    """

    # Get predicted and true labels
    test_bools = np.isin(np.array(list_groups), np.array([prediction_case_name]))
    pred_feature = candidate_lib_data[test_bools]  # Which features are available in model?
    pred_label = identifier.predict(pred_feature)
    true_label = get_label_for_list(df_y, [prediction_case_name], label_index)

    # Get confusion matrix
    confusion = confusion_matrix(pred_label, true_label)
    decision = identifier.decision_function(pred_feature)

    # Get test case physical data
    pred_data = flowCase(prediction_case_name)

    # Plot confusion on physical domain
    physical_confusion(pred_data, confusion, show_background=False, sname=sname + '_confusion' if sname else None)
    physical_decision(pred_data, decision, show_background=False, sname=sname + '_decision' if sname else None)

    return 1


def predict_case_raw(identifier, case_name, feature_keys, label_index, zoom_data=None, sname=None):
    """
    Predict on physical domain for given flow case.
    :param feature_keys: Keys for features to use.
    :param sname: Savename or None.
    :param label_index: Choice of label.
    :param identifier: sklearn-estimator, fitted to data.
    :param case_name: Any case_name.
    :param df_X: Dataframe of features.
    :param df_y: Dataframe of labels.
    :return: 1:success.
    """

    # Load test data

    df_data, df_X, df_y = get_databasis_frames(case_name, True, True)

    # Build confusion matrix from predicted and true labels
    pred_feat = df_X.loc[df_X['case'] == case_name, feature_keys].to_numpy()
    pred_labl = identifier.predict(pred_feat)
    true_labl = df_y.loc[df_y['case'] == case_name, LABEL_KEYS[label_index]].to_numpy()
    confusion = confusion_matrix(pred_labl, true_labl)
    decision = identifier.decision_function(pred_feat)

    # Get test case physical data
    pred_data = flowCase(case_name)

    # Plot confusion on physical domain
    physical_confusion(pred_data, confusion, show_background=True, zoom_data=zoom_data, sname=sname + '_confusion' if sname else None)
    physical_decision(pred_data, decision, show_background=False, sname=sname + '_decision' if sname else None)

    return 1


def predict_case_raw_tsc(identifier, case_name, test_features, test_label, zoom_data=None, sname=None):
    """
    Predict on physical domain for given flow case.
    :param sname: Savename or None.
    :param identifier: sklearn-estimator, fitted to data.
    :param case_name: Any case_name.
    :return: 1:success.
    """
    # Get prediction
    pred_labl = identifier.predict(test_features)

    # Build confusion matrix from predicted and true labels
    confusion = confusion_matrix(pred_labl, test_label)
    decision = identifier.decision_function(test_features)

    # Get test case physical data
    pred_data = flowCase(case_name)

    # Plot confusion on physical domain
    physical_confusion(pred_data, confusion, show_background=True, zoom_data=zoom_data, sname=sname + '_confusion' if sname else None)
    physical_decision(pred_data, decision, show_background=False, sname=sname + '_decision' if sname else None)

    return 1


def quantitative_evaluation(identifier, X_test, y_test, filename):
    """
    Print classification report and save to csv.
    :param identifier: Scikit-learn classifier.
    :param X_test: Test features.
    :param y_test: Test label.
    :param filename: Filename for saving.
    :return: 1:success.
    """
    y_predicted = identifier.predict(X_test)
    print(classification_report(y_test, y_predicted, target_names=['Inactive', 'Active']))
    print("Average precision: {:3.2f}".format(average_precision_score(y_test, identifier.decision_function(X_test))))

    # Create Dataframe, transpose and save to csv
    rprt = classification_report(y_test, y_predicted, target_names=['Inactive', 'Active'], output_dict=True)
    df_rprt = pd.DataFrame(rprt).transpose()
    df_rprt.to_csv("results/{:s}_cls_rprt.csv".format(filename))

    return 1


def qualitative_evaluation(identifier, list_test, df_X, df_y, feature_keys, label_index, filename):
    """
    Show prediction on all test cases visually with confusion matrix (discrete) or decision function (smooth).
    :param identifier: Scikit-learn classifier.
    :param list_test: List of case_names for testing.
    :param df_X: Dataframe of features.
    :param df_y: Dataframe of labels
    :param feature_keys: Keys for features.
    :param label_index: Index for label.
    :param filename: Name for saving plots.
    :return: 1:success.
    """
    # Loop all test_cases
    for case in list_test:
        predict_case(identifier, case, df_X, df_y, feature_keys, label_index,
                     sname=filename + '_' + 'qualitative' + '_' + case)

    return 1


def qualitative_evaluation_tsc(identifier, list_test, candidate_lib_data, df_y, list_groups, label_index, filename):
    """
    Show prediction on all test cases visually with confusion matrix (discrete) or decision function (smooth).
    :param identifier: Scikit-learn classifier.
    :param list_test: List of case_names for testing.
    :param candidate_lib_data: Array of evaluated candidates for each sample.
    :param df_y: Dataframe of labels.
    :param list_groups: List of case_names for given scenario.
    :param label_index: Index for label/marker.
    :param filename: Name for saving plots.
    :return: 1:success.
    """
    # Loop all test_cases
    for case in list_test:
        predict_case_tsc(identifier, case, candidate_lib_data, df_y, list_groups, label_index,
                         sname=filename + '_' + 'qualitative' + '_' + case)

    return 1


def evaluate_model(filename):
    """
    Evaluate a trained model for a given configuration.
    :param filename: Name of a model file in "models/".
    :return: 1:success.
    """

    # Check filename
    if filename.find("_model"):
        filename = filename[:-6]

    # Setup
    scenario, label_name, label_index = get_config_from_filename(filename)
    feature_keys = FEATURE_KEYS

    # Load data
    df_info, df_X, df_y = get_databasis_frames(get_features=True, get_labels=True)
    list_train, list_test = get_test_scenario(scenario)
    X_train, X_test, y_train, y_test = test_train_split_feat_labl(df_X, df_y, list_train, list_test, feature_keys, label_index)


    # Load identifier
    identifier = load_model(filename + "_" + "model")


    # Model complexity and structure
    show_model(identifier, sname=filename + '_' + 'model_struct')


    # Qualitative evaluation
    qualitative_evaluation(identifier, list_test, df_X, df_y, feature_keys, label_index, filename)


    # Quantitative evaluation (Print and save classification report)
    quantitative_evaluation(identifier, X_test, y_test, filename)


    # Save Precision-Recall curve
    prc, rcl, thrshlds = precision_recall_curve(y_test, identifier.decision_function(X_test))
    precision_recall_plot(prc, rcl, thrshlds, sname=filename + "_" + "PRcurve")

    return 1


def feature_selection(scenario, label_index, sample_size=1e5, k=3, n_feat=None, create_bar_plot=None):
    """
    Identify mutual information for each feature,
    bar plot the MI and return a sorted list of feature keys.
    :param label_index: Index of error metric.
    :param scenario: Test scenario.
    :param n_feat: Number of features to consider for plot and return.
    :param k: k-neareast neighbour for MI estimation.
    :return: List of feature keys sorted by MI.
    """
    print("Running feature selection...")

    # No feature selection required, if all features are used.
    if not n_feat:
        print("All features are used")
        return FEATURE_KEYS


    # Sample features and labels
    sampls = get_group_samples(scenario, FEATURE_KEYS, label_index, int(sample_size), undersampling=True)
    feat = np.vstack([s[0] for s in sampls])
    labl = np.hstack([s[1] for s in sampls])


    # Evaluate mutual information
    mi = mutual_info_classif(feat, labl, n_neighbors=k)
    selected_feature_keys = [fkeys for _, fkeys in sorted(zip(mi, FEATURE_KEYS), key=lambda pair: pair[0], reverse=True)]
    mi = np.sort(mi)[::-1]  # Sort ascending

    # Map fkeys to q_i with np.where(fkey == FEATURE_KEYS)

    # Select only subset of features
    if n_feat:
        mi = mi[:n_feat]
        selected_feature_keys = selected_feature_keys[:n_feat]

    if create_bar_plot:
        baring(pos=np.arange(mi.shape[0]) + 1,
               heights=mi,
               width=0.5,
               xticklabels=[sfk.replace('_', '') for sfk in selected_feature_keys],
               ylabel='Mutual Information',
               color=general_cmap(np.linspace(0, 1))[25],
               sname=None,
               )

    print(f"Selected features:\t{selected_feature_keys}")

    return selected_feature_keys


def cross_validation(identifier, df_X, df_y, list_train, feature_keys, label_index):
    """
    Manual cross-validation. Gives approximately the same result as scikit-learn cross_validate.
    Slower than scikit-learn!
    :param identifier: Scikit-learn identifier.
    :param df_X: Dataframe of features.
    :param df_y: Dataframe of labels.
    :param list_train: List of training cases.
    :param feature_keys: Keys for features.
    :param label_index: Index for labels.
    :return: Dictionary of test and train scores.
    """
    test_score = []
    train_score = []
    for test_case in list_train:
        # LeaveOneGroupOut splits
        subtrain_cases_bool = np.array(list_train) != np.array(test_case)
        subtrain_cases = np.array(list_train)[subtrain_cases_bool].tolist()
        feat_subtrain, feat_test, labl_subtrain, labl_test = test_train_split_feat_labl(df_X, df_y,
                                                                                        subtrain_cases, [test_case],
                                                                                        feature_keys, label_index)

        # Instantiate new classifier and fit
        tmp_idf = clone(identifier)
        tmp_idf.fit(feat_subtrain, labl_subtrain)

        # Scoring
        train_score.append(
            f1_score(labl_subtrain, tmp_idf.predict(feat_subtrain))
        )
        test_score.append(
            f1_score(labl_test, tmp_idf.predict(feat_test))
        )

    result = dict({'test_score': test_score, 'train_score': train_score})

    return result


def bias_variance_analysis(identifier, df_X, df_y, list_all, feature_keys, label_index):
    """
    Bias-Variance analysis for a learning curve.
    :param identifier: Scikit-learn identifier.
    :param df_X: Dataframe of features.
    :param df_y: Dataframe of labels.
    :param list_all: List of all cases in data basis.
    :param feature_keys: Keys for features.
    :param label_index: Index for labels.
    :return: Dictionary of test and train scores.
    """
    test_score = []
    train_score = []
    num_of_points = []
    num_of_datasets = []
    for j, valid_case in enumerate(list_all):
        print("Cross-validation for case {0:01d} out of {1:01d}".format(j + 1, len(list_all)))
        # LeaveOneGroupOut splits
        train_bool = np.array(list_all) != np.array(valid_case)
        list_train = np.array(list_all)[train_bool].tolist()

        # Incrementally use more training data for fitting the identifier, static validation data from LOGO-CV
        for i in range(len(list_train)):
            print("Fitting with {0:01d} training sets out of {1:01d}".format(i + 1, len(list_train)))
            # Randomly select subset of training data
            random_select = np.random.choice(range(len(list_train)), i + 1, replace=False)
            sublist_train = np.array(list_train)[random_select].tolist()

            # Subtrain/test splits
            feat_subtrain, feat_valid, labl_subtrain, labl_valid = test_train_split_feat_labl(df_X, df_y,
                                                                                              sublist_train,
                                                                                              [valid_case], feature_keys,
                                                                                              label_index)

            # Get number of points and datasets
            num_of_datasets.append(i + 1)
            num_of_points.append(labl_subtrain.shape[0])

            # Fit new identifier
            tmp_idf = clone(identifier)
            # tmp_idf.fit(feat_subtrain, labl_subtrain)
            tmp_idf.fit(np.array([[0, 1, 1, 0, 1, 1, 0, 0], [0, -1, -1, 0, -1, -1, 0, 0]]).T, np.array([0, 1, 1, 0, 1, 1, 0, 0]))


            # Scoring
            train_score.append(
                -i
                # f1_score(labl_subtrain, tmp_idf.predict(feat_subtrain))
            )
            test_score.append(
                -i
                # f1_score(labl_valid, tmp_idf.predict(feat_valid))
            )

    # Learning-curve-like output
    # n_datasets = np.arange(len(list_all))+1
    # train_scores = np.array(train_score).reshape(len(list_all), len(list_all) - 1)
    # test_scores = np.array(test_score).reshape(len(list_all), len(list_all) - 1)

    # Full results
    result = dict(
        {'test_score': test_score,
         'train_score': train_score,
         'num_of_points': num_of_points,
         'num_of_datasets': num_of_datasets,
         }
    )

    return result#, train_scores, test_scores, n_datasets


def confusion_matrix(predicted_labels, true_labels, return_list=True):
    """
    Evaluate confusion matrix for predicted labels.

    :param predicted_labels: Labels predicted by for the given case.
    :param true_labels: True labels for the case.
    :return: Confusion matrix as vector with 1=TP, 2=TN, 3=FN, 4-FP.
    """
    num_of_cells = predicted_labels.shape[-1]
    confusion = np.zeros(num_of_cells)

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(num_of_cells):
        if predicted_labels[i] == true_labels[i]:
            if predicted_labels[i]:
                confusion[i] = TRUE_POSITIVE
                tp += 1
            else:
                confusion[i] = TRUE_NEGATIVE
                tn += 1

        elif predicted_labels[i] != true_labels[i]:
            if predicted_labels[i]:
                confusion[i] = FALSE_POSITIVE
                fp += 1
            else:
                confusion[i] = FALSE_NEGATIVE
                fn += 1

    if return_list:
        return confusion
    else:
        return tp, fp, tn, fn


def matthews_correlation_coefficient(TP, TN, FP, FN):
    numerator = TP*TN - FP*FN
    denominator = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    return numerator/denominator


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def show_model(identifier, print_text=False, sname=None):
    """
    Print formatted model coefficients with respective feature name.
    :param sname: Save figure.
    :param print_text: Print features and coefficients to console.
    :param identifier: sklearn-estimator for linear model.
    :return: 1: success, -1:unknown number of features.
    """

    if isinstance(identifier.coef_, np.ndarray):
        coefs = identifier.coef_.reshape(-1)
        len_coef = len(coefs)
    else:
        coefs = identifier.coef_
        len_coef = 1

    if len_coef == num_of_features:
        num = num_of_features
        keys = FEATURE_KEYS
    elif len_coef == num_of_physical:
        num = num_of_physical
        keys = PHYSICAL_KEYS
    elif len_coef == num_of_invariants:
        num = num_of_invariants
        keys = INVARIANT_KEYS
    else:
        print("WARNING in show_model\nUnknown number of features")
        num = len_coef
        keys = [str(i) for i in range(len_coef)]

    if print_text:
        print('Coefficients: ')
        for coef, key in zip(coefs, keys):
            print('{:10.2f} {:s}'.format(coef, key))

        print('Magnitude of coefficients: {:10.4f}'.format(np.linalg.norm(coefs, ord=1)))

    else:
        # Bar config
        bar_width = 0.9
        bar_pos = [i for i in range(num)]

        # Absolute coefficients and normalised colour
        coefs = np.abs(coefs)
        colours = general_cmap(coefs/np.max(coefs))

        baring(bar_pos,
               coefs,
               bar_width,
               colours,
               keys,
               xticklabel_bottom_pos=0.2,
               ylog=False,
               sname=sname)

    return 1


def show_cv_result(result, score_str, sname=None):
    """
    Show results and best estimator of a
    cross-validation.
    :param sname: Save figure.
    :param score_str: String for the scoring in use.
    :param result: Return value of CV.
    :return: 1:success.
    """

    # Get best estimator and score
    df = pd.DataFrame(result)
    loc_string = 'test_' + score_str
    best_identifier = df.loc[df[loc_string].argmax(), 'estimator']
    best_score = df.loc[df[loc_string].argmax(), loc_string]

    # Print overall result
    print(df)
    print("Mean {:s}: {:3.2f}".format(loc_string, df[loc_string].mean()))
    print("Best {:s} is: {:3.2f}".format(loc_string, best_score))

    # Print estimator with score
    print('\nBEST IDENTIFIER')
    # show_model(best_identifier, print_text=True)


    return 1


#####################################################################
### Tests
#####################################################################
def verify_classifier(classifier, inputs, true_labels, verbose=None):
    """
    Evaluate the performance of a classifier.

    :param classifier: Scikit classifier type.
    :param inputs: Inputs for learning and predictions.
    :param true_labels: True labels.
    :return: void.
    """
    prediction = classifier.predict(inputs)
    confusion = confusion_matrix(prediction, true_labels)
    TPs = np.count_nonzero(confusion == TRUE_POSITIVE)
    TNs = np.count_nonzero(confusion == TRUE_NEGATIVE)
    FNs = np.count_nonzero(confusion == FALSE_NEGATIVE)
    FPs = np.count_nonzero(confusion == FALSE_POSITIVE)

    total_error = (FNs + FPs) / confusion.shape[-1]
    TP_rate = TPs/(TPs + FNs)  # Specificity
    TN_rate = TNs/(TNs + FPs)  # Sensitivity
    FP_rate = FPs/(TNs + FPs)
    FN_rate = FNs/(TPs + FNs)

    CA_error = (FP_rate + FN_rate)/2
    #  score = classifier.score(inputs, true_labels)  # Score = 1 - total error
    MCC = matthews_correlation_coefficient(TPs, TNs, FPs, FNs)

    if verbose:
        print("Total error: \t" + str(total_error))
        print("CA error: \t\t" + str(CA_error))
        print("MCC: \t\t\t" + str(MCC))
        print()

        print("TP rate: \t\t" + str(TP_rate))
        print("TN rate: \t\t" + str(TN_rate))

        print("FP rate: \t\t" + str(FP_rate))
        print("FN rate: \t\t" + str(FN_rate))
        print('\n')

    return {'total_error': total_error, 'TP_rate': TP_rate, 'TN_rate': TN_rate, 'FP_rate': FP_rate, 'FN_rate': FN_rate, 'CA_error': CA_error, 'MCC': MCC}
