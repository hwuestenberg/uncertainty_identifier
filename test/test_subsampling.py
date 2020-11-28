# ###################################################################
# test subsampling
#
# Description
# Test a sub- or undersampling procedure to reduce the amount of
# data points.
#
# ###################################################################
# Author: hw
# created: 17. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncert_ident.data_handling.flowcase import *
from uncert_ident.methods.classification import *

from sklearn.model_selection import train_test_split

####################################################################
## Load data
####################################################################


def subsample_data_not_classification(list_all, feature_keys, label_index, sample_size=10000):
    # sample_size = int(1e4)
    # label_index = 0
    # feature_keys = FEATURE_KEYS
    label = LABEL_KEYS[label_index]


    # Initiate arrays for all features and labels
    all_feat = np.array([])
    all_labl = np.array([])


    # Loop
    for case in list_all:
        s_size = int(sample_size)
        df_data, df_feat, df_labl = get_databasis_frames(case, 1, 1)


        # Exclude zero feature points
        idx_zeros = df_feat.index[df_feat.loc[:, :'inv46'].eq(0.0).all(axis=1)]
        print(f"{len(idx_zeros)} zeros excluded")
        df_feat.drop(df_feat.index[idx_zeros])
        df_labl.drop(df_feat.index[idx_zeros])


        # N_inactive, N_active for label
        n0, n1 = np.bincount(df_labl.loc[df_labl['case'] == case, label])
        n_minority = np.min([n0, n1])
        if n_minority < s_size:
            print(f"WARNING\tNot enough points in data set {case}\tn_minority {n_minority} < {s_size} samples\t{n_minority*2} points will be used.")
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
        else:
            all_feat = np.concatenate([all_feat, feat], axis=0)
            all_labl = np.concatenate([all_labl, labl], axis=0)

        del df_data, df_feat, df_labl

    return all_feat, all_labl


feat, labl = subsample_data(['PH-Breuer-10595', 'PH-Xiao-15', 'TBL-APG-Bobke-m16', 'NACA4412-Vinuesa-top-4'], FEATURE_KEYS, 0)

X_train, X_test, y_train, y_test = train_test_split(feat, labl, test_size=0.4, random_state=0)  # Conserves balancing
