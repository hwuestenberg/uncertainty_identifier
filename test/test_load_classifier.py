# ###################################################################
# test load_classifier
#
# Description
# Reconstruct a classifier from loaded coefficient and use for
# predictions.
#
# ###################################################################
# Author: hw
# created: 31. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import pandas as pd

from uncert_ident.data_handling.data_import import save_model, load_model

from sklearn.linear_model import LogisticRegression as logReg
import turbosparsereg.dataprocessing.dataprocessor as dataproc


#####################################################################
### Functions
#####################################################################
def build_library_features(operations, active, feature_array, feature_keys):
    """
    Build library for SpaRTA model discovery.
    :param feature_array: Array of features.
    :param operations: Operations list.
    :param active: Actives list.
    :param feature_keys: Feature keys.
    :return: Candidate strings and evaluated candidates.
    """

    # Create dict with {feature_key: feature_array[n_samples]}
    var_dict = dict()
    for j, key in enumerate(operations['var']):
        if key != 'const':
            var_dict[key] = feature_array[:, j-1]
    var_dict['const'] = np.ones(len(var_dict[feature_keys[0]]))


    # Create library of all cases
    buildlib = dataproc.BuildLibrary(operations)
    B = buildlib.build_B_library(var_dict, active)
    # print(f"n_features:\t{B.shape}\tExpected size of B_data:\t{B.shape[0]*len(var_dict[feat_keys[0]])*8/1e9:3.2f}Gb")
    # assert False, "Debug size of B and B_data"


    # Free memory
    locals().update(var_dict)
    del var_dict
    locs = locals()


    # Evaluate each feature
    B_data = np.stack([eval(B[j], {}, locs) for j in range(len(B))]).T


    return B, B_data


def default_parameters(feature_keys):
    """ Defines default parameters for symbolic regression.
    """
    # Definition of raw features and mathematical operations
    operations = {'var': ['const'] + feature_keys,
                  'exp': ['**-3.0', '**-2.0', '**-1.0', '**-0.5', '**0.5', '**1.0', '**2.0', '**3.0'],
                  'fun': ['abs', 'np.log10', 'np.exp', 'np.sin', 'np.cos']}

    # Which raw inputs should be used for the library
    active = {'var': "1" * (len(feature_keys)+1), 'exp': '00000101', 'fun': '00000'}


    return operations, active



# def save_model(identifier, filename):
#     """
#     Save an identifier to file.
#     :param identifier: LogisticRegression identifier.
#     :param filename: Path to file and filename.
#     :return: 1:success.
#     """
#     idf_dict = identifier.__dict__
#
#     # Cannot preserve NoneType and dict
#     del idf_dict['random_state']
#     del idf_dict['l1_ratio']
#     del idf_dict['class_weight']
#
#     assert save_dict_to_mat(filename, idf_dict)
#
#     return 1


# def load_model(filename):
#     """
#     Load identifier configuration to dictionary.
#     Does not preserve random_state, l1_ratio and class_weight!
#     :param filename: path to file and filename.
#     :return: Sklearn.LogisticRegression instance.
#     """
#     # Load model config from data
#     idf_dict = load_mat(filename)
#
#     # Adjustment due to dict conversion
#     idf_dict['coef_'] = idf_dict['coef_'].reshape(1, -1)
#     idf_dict['l1_ratio'] = None  # Could not be preserved
#     idf_dict['random_state'] = None
#     idf_dict['class_weight'] = None
#
#     # Instantiate classifier
#     identifier = logReg()
#
#     for key in idf_dict.keys():
#         identifier.__setattr__(key, idf_dict[key])
#
#     return identifier


#####################################################################
### Test Scikit-learn
#####################################################################
# Define parameters
coef = np.arange(5).reshape(1, -1)  # Requires shape [1, n_features]
intercept = np.zeros(1)
classes = np.array([0, 1])
class_weight = dict(zip([0, 1], [0.807, 1.314]))
C = 100
dual = False
fit_intercept = False
intercept_scaling = 10
penalty = 'elasticnet'
l1_ratio = 0.5
max_iter = 2000
n_jobs = -1
random_state = 42
solver = 'saga'
tol = 1e-10
verbose = 5
warm_start = False


# Construct identifier
idf = logReg()
idf.coef_ = coef
idf.intercept_ = intercept
idf.classes_ = classes
idf.class_weight = class_weight
idf.C = C
idf.penalty = penalty
idf. max_iter = max_iter
idf.n_jobs = n_jobs
idf.tol = tol
idf.verbose = verbose


# Test prediction
feat_test = np.arange(5).reshape(1, 5)-5
print("Decision function gives:\t{0}".format(idf.decision_function(feat_test)))


# Save test
fname = 'SCIKIT'
save_model(idf, fname + "_" + "model", feature_keys=['k_eps_Sij', 'tau_ratio', 'inv03', 'sfsadf', 'dsad_dsafg'], label_index=1)


# Load test
newidf = load_model(logReg, fname + "_" + "model")
print("After save/load\nDecision function gives:\t{0}".format(newidf.decision_function(feat_test)))
del newidf


#####################################################################
### Test SpaRTA
#####################################################################
# Define parameters
fkeys = ['k_eps_Sij', 'tau_ratio', 'asdbasd_Sdsafasd']
operations, active = default_parameters(fkeys)
features = np.arange(5).reshape(1, -1)

B, B_data = build_library_features(operations, active, features, fkeys)


# Test prediction
print("Decision function gives:\t{0}".format(idf.decision_function(feat_test)))



# Safe test
fname2 = 'THISISSPARTA'
save_model(idf, fname2 + "_" + "model", feature_keys=fkeys, label_index=1, candidate_library=B.tolist())



# Load test

newidf = load_model(logReg, fname2 + "_" + "model")
print("After save/load\nDecision function gives:\t{0}".format(newidf.decision_function(feat_test)))

