# ###################################################################
# test check_test_scenario
#
# Description
# Load data for a given test scenario and inspect/modify the data.
#
# ###################################################################
# Author: hw
# created: 30. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import pandas as pd
from itertools import product

from sklearn.utils.class_weight import compute_class_weight

from uncert_ident.methods.classification import *


# Load data
df_info, df_feat, df_labl = get_databasis_frames(True, True)



# Get all features and labels for a scenario
scenarios = ['sep', 'pg', 'all']
n_labls = [0, 1]


for scenario, n_labl in product(scenarios, n_labls):
    label_name = {
        0: 'non_negative',
        1: 'anisotropic',
        2: 'non_linear'
    }[n_labl]
    print("Scenario:\t{0:s}\nLabel:\t{1:s}\n".format(scenario, label_name))
    train, test = get_test_scenario(scenario)
    alll = train + test
    feat_train, feat_test, labl_train, labl_test = test_train_split_feat_labl(df_feat, df_labl, train, test, FEATURE_KEYS, n_labl, verbose=True)
    feat_all, labl_all = get_feat_labl_for_list(df_feat, df_labl, alll, FEATURE_KEYS, n_labl)


    # Get class weights
    dict_cw = get_class_weights(df_labl, alll, n_labl, verbose=True)
    print("")


