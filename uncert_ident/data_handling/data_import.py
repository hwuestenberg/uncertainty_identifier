# ###################################################################
# module data_import
#
# Description
# Load/import data from mat or csv files. Return dictionaries.
#
# ###################################################################
# Author: hw
# created: 03. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import sys
from os.path import basename, isfile, splitext, abspath, dirname
from pathlib import Path
from glob import glob

import numpy as np
from scipy.interpolate import griddata
import scipy.io as sio
from PyFOAM import *

from uncert_ident.utilities import FLOWCASE_KW

sys.path.append("..")

#####################################################################
### Constants
#####################################################################
path_to_raw_data = './../../../DATA/'
path_to_processed_data = './../../../DATA/'


#####################################################################
### Functions
#####################################################################
def convert_item_to_1darray(dictionary):
    """
    Only ndarrays can be saved in mat file. Hence, transform all
    integer and floats to 1x1 ndarrays.
    :param dictionary: Dictionary to save to mat file.
    :return: 1:success.
    """

    for key in dictionary.keys():
        item = dictionary[key]
        if isinstance(item, int) or isinstance(item, float):
            dictionary[key] = np.array([item])
        elif isinstance(item, str):
            dictionary[key] = np.array([item])
        elif isinstance(item, np.ndarray):
            pass
        elif isinstance(item, list) and not isinstance(item[0], list):
            pass
        elif isinstance(item, type(None)):
            dictionary[key] = np.array([item])
        elif isinstance(item, dict):
            names = [str(k) for k in item]
            formats = [type(item[k]) for k in item]
            dictionary[key] = np.array(list(item.items()), dtype=dict(names=names, formats=formats))
        else:
            print('ERROR in convert_int_to_1darray: '
                  'Could not handle dictionary element with key: %r' % key)


    return 1


def convert_1darray_to_item(data_dict, keys):
    """
    Convert 1x1 ndarrays to previous value type for all given keys.
    For exmaple, used for scalars that are used as indexes (nx, ny).
    :param data_dict: Dictionary read from mat file.
    :param keys: Keys of 1x1 ndarrays to transform to integers.
    :return: 1:success.
    """


    for key in keys:
        if key in data_dict:
            item_type = type(data_dict[key][0])
            if np.issubdtype(item_type, np.integer):
                data_dict[key] = int(data_dict[key])
            elif np.issubdtype(item_type, np.floating):
                data_dict[key] = float(data_dict[key])
            elif np.issubdtype(item_type, np.character):
                data_dict[key] = str(data_dict[key][0])
        else:
            pass

    return 1


def check_mat_extension(fname, add_extension=True):
    """
    Check for mat file specifier and optionally add the file specifier.
    :param fname: Path or filename.
    :param add_extension: Option to add an mat specifier.
    :return: fname with/-out mat specifier.
    """

    path, specifier = splitext(fname)

    # Case: File specifier is .mat
    if specifier == '.mat':
        if add_extension:
            return path + '.mat'
        else:
            return path
    # Case: No file specifier (.mat)
    else:
        if add_extension:
            return path + '.mat'
        else:
            return path


def check_path_location():
    """
    Check current directory and adapt path if required.
    :return: Addition to path.
    """

    directory = basename(abspath('.'))
    if directory in ['test', 'validation']:
        path_add = '../'
    else:
        path_add = ''

    return path_add


def check_create_directory(fname):
    """
    Check if directory and all parent directories exist and create
    them if they dont.
    :param fname: Path to a file.
    :return:
    """

    path_to_lowest_directory = Path(fname).parent
    path_to_lowest_directory.mkdir(parents=True, exist_ok=True)

    return 1


def find_results():
    """
    Find all result directories in ./results/*
    :return: List of directory names.
    """
    path_to_results = glob('./results/*')
    assert len(path_to_results), "No results found in ./results/"

    result_names = [path.replace('./results/', '') for path in path_to_results]
    # result_names = [path.replace('', '') for path in path_to_results]

    return sorted(result_names)


def find_all_mats(return_filename=False):
    """
    Search for available mat-files in path_to_processed_data.
    :param return_filename: Return only filename without paths or
    extension.
    :return: list of filenames without .mat extension.
    """

    # Adapt path if script is not run in main directory
    path_add = check_path_location()

    # Find all mat files
    all_path_to_mat = glob(path_add + path_to_processed_data + '*/processed/*.mat')

    # If no file found, exit
    if len(all_path_to_mat) == 0:
        sys.exit('ERROR in find_path_to_mat: '
                 'No mat file found in \"' + str(path_to_processed_data) + '\"')

    if return_filename:
        return [splitext(basename(path))[0] for path in all_path_to_mat]
    else:
        return all_path_to_mat


def find_path_to_mat(searched_mat_name):
    """
    Search for the given mat-file in path_to_processed_data.
    :param searched_mat_name: String with filename.
    :return: Path to filename.
    """

    assert isinstance(searched_mat_name, str), \
        "searched_mat_name is not a string: %r" % searched_mat_name

    # Eventually add .mat extension
    check_mat_extension(searched_mat_name)

    # Adapt path if test script is run
    path_add = check_path_location()

    # Find all mat files
    all_path_to_mat = glob(path_add + path_to_processed_data + '*/processed/*.mat')
    for path_to_mat in all_path_to_mat:
        assert (isfile(path_to_mat)), \
            "Invalid path to file found: %r" % path_to_mat

        # Extract basename without extension
        mat_name = splitext(basename(path_to_mat))[0]
        if mat_name == searched_mat_name:
            return path_to_mat

    # If no file found, exit
    sys.exit('ERROR in find_path_to_mat: '
             'No file found for given \"' + str(searched_mat_name) + '\"')


def find_case_names():
    """
    Search for available mat-files in path_to_processed_data.
    features and labels.
    :return: list of filenames without .mat extension.
    """

    # Adapt path if script is not run in main directory
    path_add = check_path_location()

    # Find all mat files
    all_path_to_mat = glob(path_add + path_to_processed_data + '*/processed/*.mat')
    all_path_to_mat = [path for path in all_path_to_mat if '_features' not in path and '_labels' not in path]  # Remove features and labels files

    # If no file found, exit
    if len(all_path_to_mat) == 0:
        sys.exit('ERROR in find_path_to_mat: '
                 'No mat file found in \"' + str(path_to_processed_data) + '\"')

    # Validate each path to a mat file
    else:
        all_case_names = list()
        for path_to_mat in all_path_to_mat:
            assert (isfile(path_to_mat)), \
                "Invalid path to file found: %r" % path_to_mat
            case_name = splitext(basename(path_to_mat))[0]
            all_case_names.append(case_name)

    return all_case_names


def find_all_raw_files():
    """
    Find all raw files in path_to_raw_data.
    :return: List of paths to found files.
    """

    # Adapt path if script is not run in main directory
    path_add = check_path_location()

    # Find all csv, dat or mat files in raw (subfolders)
    wild_paths = ['*/', '*/*/', '*/*/*/', '*/*/*/*/', '*/*/*/*/*/']  # Slight overkill, but to be sure
    all_files = list()
    for wild_path in wild_paths:
        all_files += glob(path_add + path_to_raw_data + wild_path + '*.csv')
        all_files += glob(path_add + path_to_raw_data + wild_path + '*.dat')
        all_files += glob(path_add + path_to_raw_data + wild_path + '*.mat')

    # Remove path, only return filename with extension
    for i, file in enumerate(all_files):
        all_files[i] = basename(file)

    return all_files


def exist_file(filename):
    """
    Check if filename exist in
    :param filename:
    :return: 1:file exits.
    """

    all_files = find_all_raw_files()
    filename = basename(filename)  # Remove path from filename
    if filename in all_files:
        return 1
    else:
        return 0


def exist_mat(mat_name):
    """
    Check whether a given mat file exists. Searches for mat-files in
    path_to_processed_data.
    :return: 1:file exists, 0:file not found.
    """

    mat_name = check_mat_extension(mat_name, add_extension=False)

    all_mats = find_all_mats(return_filename=True)
    if mat_name in all_mats:
        return 1
    else:
        return 0


def exist_case(case_name):
    """
    Check whether a given case file exists. Searches for case_name in
    path_to_processed_data.
    :return: 1:file exists, 0:file not found.
    """

    case_name = check_mat_extension(case_name, add_extension=False)

    all_files = find_case_names()
    if case_name in all_files:
        return 1
    else:
        return 0


def save_dict_to_mat(fname, save_dict):
    """
    Save dictionary into mat file.
    :param fname: Path to where the mat file should be saved.
    :param save_dict: Dictionary that will be saved.
    :return: 1:success.
    """

    convert_item_to_1darray(save_dict)
    fname = check_mat_extension(fname, add_extension=True)
    check_create_directory(fname)
    sio.savemat(fname, mdict=save_dict)

    assert test_save_mat(fname, save_dict), 'ERROR in save_dict_to_mat: ' \
                                            'Keys in saved and loaded dictionary are incompatible.' \
                                            'Check saved mat file!'

    print('File ' + basename(fname) + ' saved successfully')

    return 1


def load_mat(fname, kth_data=False):
    """
    Load data from mat file and return as dict.
    :param fname: Path to mat file.
    :param kth_data: Option for kth mat files.
    :return: Dictionary of data in mat file.
    """

    fname = check_mat_extension(fname, add_extension=True)
    data = sio.matlab.loadmat(fname)

    # Delte matlab elements
    del data['__header__']
    del data['__version__']
    del data['__globals__']

    # Works for KTH data sets
    if kth_data:
        data_dict = dict()
        for case_key in list(data):
            case = data[case_key]
            case_dict = dict()

            for var_key in case.dtype.fields.keys():
                var = case[var_key]
                var_ndarray = np.array([var[0][entry] for entry in range(var.shape[1])]).squeeze()
                case_dict[var_key] = var_ndarray

            data_dict[case_key] = case_dict

    # Handle file in defined file format
    else:
        data_dict = data
        for key in data_dict.keys():
            if data_dict[key].shape[0] == 1:
                data_dict[key] = data_dict[key].reshape(-1)
        convert_1darray_to_item(data_dict, FLOWCASE_KW)

    return data_dict


def test_save_mat(fname, save_dict):
    """
    Ensure equality of the saved mat file and the same mat file
    when loaded with load_mat.
    :param fname: mat filename.
    :param save_dict: Dictionary that was saved with save_dict_to_mat.
    :return: 1:success.
    """

    test_dict = load_mat(fname)
    convert_1darray_to_item(save_dict, FLOWCASE_KW)  # Loaded state for save_dict

    # Compare keys of dictionaries
    if save_dict.keys() == test_dict.keys():
        for key in save_dict.keys():
            if isinstance(save_dict[key], int):
                if save_dict[key] == test_dict[key]:
                    continue
                else:
                    sys.exit('ERROR in save_dict_to_mat: '
                             'Saved and loaded dictionary are incompatible in key ' + str(key))
            elif isinstance(save_dict[key], str):
                if save_dict[key] == test_dict[key]:
                    continue
                else:
                    sys.exit('ERROR in save_dict_to_mat: '
                             'Saved and loaded dictionary are incompatible in key ' + str(key))
            elif isinstance(save_dict[key], np.ndarray):
                if all(save_dict[key].flatten() == test_dict[key].flatten()):
                    continue
                else:
                    sys.exit('ERROR in save_dict_to_mat: '
                             'Saved and loaded dictionary are incompatible in key ' + str(key))
    else:
        return -1

    return 1


def load_csv(fname, col_names, skip_header=0, newline_char=None, delimiter=','):
    """
    Load csv file and save as dictionary.
    :param delimiter: Delimiter used in csv file.
    :param newline_char: Optional newline character used in csv file.
    :param fname: Path to file.
    :param skip_header: Number of rows above the data. These are skipped.
    :param col_names: Name for each column.
    :return: Dictionary with entry for each column
    """

    # Adapt path if test script is run
    path_add = check_path_location()

    # Load data from csv
    file = np.genfromtxt(path_add + fname, delimiter=delimiter, skip_header=skip_header, comments=newline_char, names=col_names)

    # Write data into dictionary
    load_dict = dict()

    # For single variable files
    if file.dtype == None:
        load_dict[col_names] = file

    # For files with multiple variables
    else:
        for key in file.dtype.fields.keys():
            load_dict[key] = file[key]

    return load_dict


def save_model(identifier, filename, feature_keys=None, label_index=None, test_set=None, candidate_library=None):
    """
    Save an identifier to file.
    :param identifier: LogisticRegression identifier.
    :param filename: Path to file and filename.
    :return: 1:success.
    """
    # Transfer idf attributes to dict
    idf_dict = identifier.__dict__


    # Cannot preserve all types (e.g. NoneType, dict, ..)
    for key in list(idf_dict):
        item = idf_dict[key]
        if not isinstance(item, (int, bool, float, str, np.ndarray)):
            del idf_dict[key]

    # Further model information
    if feature_keys:
        idf_dict['feature_keys'] = feature_keys

    if not isinstance(label_index, type(None)):  # Could also be zero
        idf_dict['label_index'] = label_index

    if test_set:
        idf_dict['test_set'] = test_set

    if isinstance(candidate_library, list):
        idf_dict['candidate_library'] = candidate_library

    # Write dict to mat
    assert save_dict_to_mat(filename, idf_dict)

    return 1


def load_model(identifier_constructor, filename):
    """
    Load identifier configuration to dictionary.
    Does not preserve random_state, l1_ratio and class_weight!
    :param filename: path to file and filename.
    :return: Sklearn.LogisticRegression instance.
    """
    # Load model config from data
    idf_dict = load_mat(filename)

    # Adjustment due to dict conversion
    idf_dict['coef_'] = idf_dict['coef_'].reshape(1, -1)

    # Remove spaces in feature keys
    if 'feature_keys' in idf_dict:
        idf_dict['feature_keys'] = [fkey.replace(' ', '') for fkey in idf_dict['feature_keys']]

    if 'label_index' in idf_dict:
        idf_dict['label_index'] = idf_dict['label_index'][0]  # Unpack array

    # Remove spaces in candidate library
    if 'candidate_library' in idf_dict:
        idf_dict['candidate_library'] = [b.replace(' ', '') for b in idf_dict['candidate_library']]

    # Reconstruct classifier from dict
    identifier = identifier_constructor()
    identifier.fit(np.array([0, 1, 2, 3]).reshape(-1, 1), np.array([0, 0, 1, 1]))  # Fake fit for model setup
    for key in idf_dict.keys():
        identifier.__setattr__(key, idf_dict[key])

    return identifier


def interpolate_on_structured_grid(dns_dict, struct_grid):
    """
    Interpolate from block-structured grid onto structured curvilinear grid.
    :param dns_dict: Dictionary of flow case data. Requires x- and y- coordinate
    with keys 'x' and 'y'.
    :param struct_grid: x- and y-coordinates of a structured grid.
    :return: Dictionary with new coordinates and interpolated quantities.
    """

    for var_key in list(dns_dict):
        if var_key == 'x' or var_key == 'y':
            continue
        dns_dict[var_key] = griddata((dns_dict['x'], dns_dict['y']), dns_dict[var_key],
                                     (struct_grid[0], struct_grid[1]))

    dns_dict['x'] = struct_grid[0]
    dns_dict['y'] = struct_grid[1]

    return dns_dict


def read_rans_ph_breuer():
    path = "PH-Breuer/rans/case_baseline"
    nx = 120
    ny = 130

    # Cell centred coordinates
    X = getRANSVector(path_to_raw_data + path, "10000", 'C')
    X = X.reshape(3, ny, nx)
    x, y = X[0].T, X[1].T

    # Velocity vector
    U = getRANSVector(path_to_raw_data + path, 10000, 'U')
    U = U.reshape(3, ny, nx)
    um, vm, wm = U[0].T, U[1].T, U[2].T

    # Turbulence data
    k = getRANSScalar(path_to_raw_data + path, 10000, 'k').reshape(ny, nx).T.flatten()
    nut = getRANSScalar(path_to_raw_data + path, 10000, 'nut').reshape(ny, nx).T.flatten()
    omega = getRANSScalar(path_to_raw_data + path, 10000, 'omega').reshape(ny, nx).T.flatten()

    # Save to dict
    rans_data = {
        'x': x,
        'y': y,
        'nx': nx,
        'ny': ny,
        'um': um,
        'vm': vm,
        'k': k,
        'nut': nut,
        'oemga': omega
    }

    return rans_data

