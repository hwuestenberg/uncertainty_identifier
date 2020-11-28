# ###################################################################
# test bias_variance
#
# Description
# Test the size of the data basis using the bias and variance error.
#
# ###################################################################
# Author: hw
# created: 29. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import pandas as pd

from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score

from uncert_ident.visualisation import plotter as plot
from uncert_ident.methods.classification import get_data, get_groups, get_n_groups
from uncert_ident.utilities import FEATURE_KEYS, LABEL_KEYS


#####################################################################
### Tests
#####################################################################
# Config
n_labl = 0
label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[n_labl]
feat_keys = FEATURE_KEYS  # Choose features: FEATURE_KEYS, INVARIANT_KEYS or PHYSICAL_KEYS
dataset = 'sep'  # Choose test scenario: sep, pg or all


# Get data, group indexes
train_cases, test_cases, df_feat, df_labl = get_data(dataset)
groups = get_groups(df_feat, train_cases + test_cases)
n_groups = get_n_groups(groups, train_cases + test_cases)


# Extract features and labels
X_all = df_feat.loc[df_feat['case'].isin(train_cases + test_cases), feat_keys].to_numpy()
y_all = df_labl.loc[df_labl['case'].isin(train_cases + test_cases), LABEL_KEYS[n_labl]].to_numpy()

# Group Cross-Validation splitter
logo = LeaveOneGroupOut()

# Instantiate classifier
idf = logReg(random_state=False,
             fit_intercept=False,
             class_weight='balanced',
             max_iter=200,
             # penalty='elasticnet',
             solver='lbfgs',
             C=1,
             # l1_ratio=0.0,  # 0.0=l2, 1.0=l1
             verbose=False,
             n_jobs=-1,
             )


# Get learning curve data
train_sizes, \
train_scores, \
valid_scores = learning_curve(idf,
                              X_all,
                              y_all,
                              groups=groups,
                              cv=logo,
                              train_sizes=np.linspace(0.1, 1, 10),
                              scoring='f1',
                              n_jobs=-1,
                              verbose=10
                              )

# Average the scores
mean_train_scores = np.mean(train_scores, axis=1)
mean_valid_scores = np.mean(valid_scores, axis=1)


# Plot learning curve
# fig, ax = plot.empty_plot()
fig1, ax1 = plot.lining(train_sizes, mean_train_scores, line_label='Train accuracy', linestyle='r--', ylim=[0, 1])
# plot.lining(train_sizes, train_scores, append_to_fig_ax=(fig1, ax1))
fig2, ax2 = plot.lining(train_sizes, mean_valid_scores, line_label='Valid accuracy', linestyle='g--', append_to_fig_ax=(fig1, ax1))
# plot.lining(train_sizes, valid_scores, append_to_fig_ax=(fig2, ax2))
