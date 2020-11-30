# ###################################################################
# script lr
#
# Description
# Learn identifier using the Logistic Regression algorithm for 
# linear models.
#
# ###################################################################
# Author: hw
# created: 12. Jun. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from timeit import default_timer as timer
from itertools import product
import pandas as pd

from uncert_ident.data_handling.data_import import save_model, load_model
from uncert_ident.methods.classification import *
from uncert_ident.utilities import PHYSICAL_KEYS, INVARIANT_KEYS, FEATURE_KEYS, get_datetime
from uncert_ident.visualisation.plotter import *

from sklearn.linear_model import LogisticRegression as logReg
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import l1_min_c
from sklearn import preprocessing


#####################################################################
### Configuration
#####################################################################
# Test scenario and error metric
scenario = 'all'  # Choose test scenario: all or ph
n_labl = 1


label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[n_labl]


# Exclude features with true tau_ij
if n_labl == 1:
    exclude_feat = ["conv_prod_tke", "tau_ratio"]
else:
    exclude_feat = ["conv_prod_tke", "tau_ratio", "visc_ratio"]
mi_feat_keys = [fkey if fkey not in exclude_feat else None for fkey in FEATURE_KEYS]
for i in range(mi_feat_keys.count(None)):
       mi_feat_keys.remove(None)


# Sampling
sample_size = int(1e4)


# Solver
n_processes = -1
max_iter = 1000
penalty = 'elasticnet'
solver = 'saga'
c = 1
cres = 10


# Gridsearch
l1_ratios = [1.0, 0.99, 0.95, 0.9, 0.7, 0.5, 0.2, 0.1, 0.01]
scores = ['f1', 'precision', 'recall', 'roc_auc', 'accuracy', 'neg_log_loss']
verbosity = 10



#####################################################################
### Initialisation
#####################################################################
print("\nStarting LogReg for config:\nlabel:\t\t{0}\nscenario:\t{1}".format(label_name, scenario))
print("Solver settings:\nmax_iter:\t{0}\npenalty:\t{1}\nsolver:\t\t{2}".format(max_iter, penalty, solver))
print("Cross-Validation setup:\nscores:\t\t{0}\n".format(scores))



# Print timestamp
datetime_str = get_datetime(return_string=True)
print("Started test scenario {0:s} at {1:s}".format(scenario, datetime_str.replace('_', ' ')))



#####################################################################
### Run Gridsearch Logistic Regression
#####################################################################
# Get all variations of scenario
list_group_names = get_scenario_set_variants(scenario)


# Loop all variations of scenario
for i, group_names in enumerate(list_group_names):
    train_names = group_names[:-1]  # For convenience
    test_name = group_names[-1]     # "


    # Print timestamp, generate filename
    scenario_fname = test_name.replace('-', '_')
    fname = 'logReg' + '_' + scenario + '_' + scenario_fname + '_' + label_name + '_' + datetime_str + "_newC"
    save_directory = "results/" + fname + "/"
    print(f"Running Logistic Regression\nTest on:\t{test_name}\nTrain on:\t{train_names}\nResults under:\t{fname}")
    clock = timer()


    # Sample data
    sampls = get_group_samples(scenario, mi_feat_keys, n_labl, sample_size, group_names=group_names, undersampling=True)


    # Train/Test splits
    feat_train = np.vstack([s[0] for s in sampls[:-1]])
    labl_train = np.hstack([s[1] for s in sampls[:-1]])
    grps_train = np.hstack([s[2] for s in sampls[:-1]])

    feat_test = sampls[-1][0]
    labl_test = sampls[-1][1]
    grps_test = sampls[-1][2]


    # Scaling, zero mean + unit variance 
    scaler = preprocessing.StandardScaler().fit(feat_train)
    feat_train = scaler.transform(feat_train)
    feat_test = scaler.transform(feat_test)

    # Set parameter grid, with lower bound for C
    cs = (l1_min_c(feat_train, labl_train, loss='log') * np.logspace(0, 4, cres)).tolist()


    #####################################################################
    ### Setup Cross-Validation and classifier
    #####################################################################
    # Cross-Validation splitter
    if len(np.unique(grps_train)) == 1:
        splitter = StratifiedKFold(n_splits=5)
    else:
        splitter = LeaveOneGroupOut()
    n_splits = splitter.get_n_splits(feat_train, labl_train, grps_train)
    print(f"Number of splits in CV:\t{n_splits}")



    #####################################################################
    ### Model fits
    #####################################################################
    pgrid = product(l1_ratios, cs)

    coefs = []
    f1_trains = []
    complexitys = []
    for idx, params in enumerate(pgrid):
        clock = timer()
        # Instantiate classifier
        l1_ratio = params[0]
        C = params[1]
        idf = logReg(random_state=False,
                     fit_intercept=False,
                     class_weight='none',
                     max_iter=max_iter,
                     penalty=penalty,
                     solver=solver,
                     C=C,
                     l1_ratio=l1_ratio,  # 0.0=l2, 1.0=l1
                     verbose=False,
                     n_jobs=n_processes,
                     )
        idf.fit(feat_train, labl_train)

        # Eval
        complexity = np.flatnonzero(idf.coef_)
        f1_train = f1_score(labl_train, idf.predict(feat_train))
        print("Fitted after {}\nParams:\t{}\t{}".format(timer()-clock, C, l1_ratio))
        print("F1:\t\t\t{:3.2g}".format(f1_train))
        print("Model complexity:\t{}".format(complexity))
        coefs.append(idf.coef_.ravel().copy())
        f1_trains.append(f1_train)
        complexitys.append(complexity)
        save_model(idf, save_directory + "/models/model_{:03d}".format(idx), feature_keys=mi_feat_keys, label_index=n_labl)


    # Write to csv
    df = pd.DataFrame()
    df['f1_train'] = f1_trains
    df['complexity'] = complexitys
    df['coef'] = coefs
    df.to_csv(save_directory + "gridsearch_result.csv")


    # # Plot
    # cs = parameter_grid[1]
    # coefs = np.array(coefs)
    # fig, ax = empty_plot()
    # for j in range(coefs.shape[1]):
    #     lining(np.log10(cs), coefs[:, j], marker='o', append_to_fig_ax=(fig, ax))
    # ymin, ymax = plt.ylim()
    # ax.set_xlabel('log(C)')
    # ax.set_ylabel('Coefficients')
    # save(save_dir + "_complexity_convergence.pdf")


datetime_str_end = get_datetime(return_string=True)
print("Finished test scenario {0:s} at {1:s}".format(scenario, datetime_str_end.replace('_', ' ')))



