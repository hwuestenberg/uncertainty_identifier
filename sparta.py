# ###################################################################
# script turboSparseClassification
#
# Description
# Use modified TurboSparseRegression framework for uncertainty marker
# Script adapted from turbo_symbolic_regression/script_simple.py
#
# ###################################################################
# Author: hw
# created: 03. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from uncert_ident.utilities import PHYSICAL_KEYS, INVARIANT_KEYS, FEATURE_KEYS, get_datetime
from uncert_ident.data_handling.data_import import save_model
from uncert_ident.methods.classification import *

from sklearn.svm import l1_min_c

import turbosparsereg.dataprocessing.dataprocessor as dataproc
from turbosparsereg.util import util
from turbosparsereg.methods.sr_elasticnet_julia import TSRElasticNet


#####################################################################
### Functions
#####################################################################
def reconstruct_model(model_constructor, coefficients, intercept, class_weights=None):
    """
    Reconstruct a sklearn model that is ready for predictions.
    :param model_constructor: Class constructor.
    :param coefficients: Array of coefficients.
    :param intercept: The intercept
    :param class_weights: Class_weight dict
    :return: "Fitted" estimator.
    """
    est = model_constructor()
    est.fit(np.array([0, 1, 2, 3]).reshape(-1, 1), np.array([0, 0, 1, 1]))  # Fake fit
    est.coef_ = coefficients.reshape(1, -1)
    est.intercept_ = intercept
    if class_weights:
        est.class_weight = class_weights

    return est


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
    # print(f"n_features:\t{B.shape}\tExpected size of B_data:\t{B.shape[0]*len(var_dict[feature_keys[0]])*8/1e9:3.2f}Gb")
    # assert False, "Finished for debugging size of B and B_data"


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
    active = {'var': "1" * (len(feature_keys)+1), 'exp': '00001111', 'fun': '00000'}

    # Discrete structure of elastic net (at which value pairs l1/alpha should the optimization problem be solved)
    n_alphas = 10
    l1_ratios = [.01, .1, .2, .5, .7, .9, .95, .99, 1.0]

    # Set of ridge parameters for post-selection model inference
    ridge_alphas = [0, 1e-2]

    # Number of subprocesses to be run in parallel
    n_processes = 4

    return operations, active, n_alphas, l1_ratios, ridge_alphas, n_processes


#####################################################################
### Configuration
#####################################################################
# Test scenario and error metric
scenario = 'all'
n_labl = 0


label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[n_labl]
mi_feat_keys = feature_selection(scenario, n_labl, n_feat=5)  # Select number of features (n_feat)


# Sampling
sample_size = int(1e4)


# Solver and gridsearch
operations, active, n_alphas, l1_ratios, ridge_alphas, n_processes = default_parameters(mi_feat_keys)


#####################################################################
### Initialisation
#####################################################################
print("\nStarting SpaRTA for config:\nlabel:\t\t{0}\nscenario:\t{1}".format(label_name, scenario))



# Print timestamp
datetime_str = get_datetime(return_string=True)
print("Started test scenario {0:s} at {1:s}".format(scenario, datetime_str.replace('_', ' ')))


#####################################################################
### Run Symbolic Regression
#####################################################################
# Get all variations of scenario
list_group_names = get_scenario_set_variants(scenario)


# Loop all variations of scenario
for i, group_names in enumerate(list_group_names):
    train_names = group_names[:-1]  # For convenience
    test_name = group_names[-1]     # "


    # Print timestamp, generate filename
    scenario_fname = test_name.replace('-', '_')
    fname = 'spaRTA' + '_' + scenario + '_' + scenario_fname + '_' + label_name + '_' + datetime_str + '_noTau'
    save_directory = "results/" + fname + "/"
    print(f"Running SpaRTA\nTest on:\t{test_name}\nTrain on:\t{train_names}\nResults under:\t{fname}")

    clock = timer()


    #####################################################################
    ### Create library and candidate functions
    #####################################################################
    # Sample data
    sampls = get_group_samples(scenario, mi_feat_keys, n_labl, sample_size, group_names=group_names, undersampling=True)
    feat = np.vstack([s[0] for s in sampls])
    labl = np.hstack([s[1] for s in sampls])


    # Get (evaluated) library of candidates
    B, B_data = build_library_features(operations, active, feat, mi_feat_keys)


    # Test train splits
    feat_train = B_data[:-sample_size, :]   # All but last sample set
    feat_test = B_data[-sample_size:, :]    # Only last sample set
    labl_train = labl[:-sample_size]        # "
    labl_test = labl[-sample_size:]         # "


    #####################################################################
    ### Perform sparse classification using ElasticNet
    #####################################################################
    # Perform discovery of model structures
    tsr = TSRElasticNet(n_alphas=n_alphas,
                        l1_ratios=l1_ratios,
                        ridge_alphas=ridge_alphas,
                        # class_weights=c_weights,
                        standardization=True)
    tsr.fit_map_julia(feat_train, labl_train, processes=n_processes)
    print(f"\nFound {len(tsr.model_structures_)} unique model structures with enet.\n")


    # Perform model inference
    models = tsr.model_inference_parallel_classification(feat_train, feat_test, labl_train, labl_test,
                                                         B, processes=n_processes)


    #####################################################################
    ### Safe models
    #####################################################################
    # Save models
    # for j, mdl in enumerate(models.loc[:, 'model']):
    for row in models.itertuples(index=True):
        row.model.coef_ = row.coef_
        save_model(row.model, save_directory + "models/model_" + f"{row[0]:03d}", feature_keys=mi_feat_keys, label_index=n_labl,
                   test_set=test_name, candidate_library=B.tolist())

    print(f"Finished\t{fname}\tafter \t{timer() - clock}")



#####################################################################
### Evaluate model
#####################################################################
# Run script evaluate_models




# End timestamp
datetime_str_end = get_datetime(return_string=True)
print("Finished test scenario {0:s} at {1:s}".format(scenario, datetime_str_end.replace('_', ' ')))


