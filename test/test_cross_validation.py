# ###################################################################
# test cross_validation
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
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.metrics import f1_score
from sklearn.base import clone

from uncert_ident.visualisation import plotter as plot
from uncert_ident.methods.classification import get_data, get_groups, get_n_groups, test_train_split_feat_labl
from uncert_ident.utilities import FEATURE_KEYS, LABEL_KEYS, get_datetime


#####################################################################
### Configuration
#####################################################################
n_labl = 1
label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[n_labl]
feat_keys = FEATURE_KEYS  # Choose features: FEATURE_KEYS, INVARIANT_KEYS or PHYSICAL_KEYS
dataset = 'sep'  # Choose test scenario: sep, pg or all



#####################################################################
### Get and select data
#####################################################################
# Define filename and print timestamp
datetime_str = get_datetime(return_string=True)
fname = 'logReg' + '_' + label_name + '_' + dataset + '_' + datetime_str
print("Running test scenario {0:s}, results in {1:s}".format(dataset, fname))


# Get data, group indexes
train_cases, valid_cases, df_feat, df_labl = get_data(dataset)
groups = get_groups(df_feat, train_cases)
n_groups = get_n_groups(groups, train_cases)

# Test/Train splits
feat_train, feat_valid, labl_train, labl_valid = test_train_split_feat_labl(df_feat, df_labl, train_cases, valid_cases, feat_keys, n_labl)


# Instantiate classifier
idf = logReg(random_state=False,
             fit_intercept=False,
             class_weight='balanced',
             max_iter=200,
             # penalty='elasticnet',
             solver='lbfgs',
             C=1,
             # l1_ratio=0.0,  # 0.0=l2, 1.0=l1
             verbose=True,
             n_jobs=-1,
             )


#####################################################################
### Manual Cross-Validation
#####################################################################
test_score = []
train_score = []
for test_case in train_cases:
    # LeaveOneGroupOut splits
    subtrain_cases_bool = np.array(train_cases) != np.array(test_case)
    subtrain_cases = np.array(train_cases)[subtrain_cases_bool].tolist()
    feat_subtrain, feat_test, labl_subtrain, labl_test = test_train_split_feat_labl(df_feat, df_labl, subtrain_cases, [test_case], feat_keys, n_labl)

    # Instantiate new classifier and fit
    tmp_idf = clone(idf)
    tmp_idf.fit(feat_subtrain, labl_subtrain)

    # Scoring
    train_score.append(
        f1_score(labl_subtrain, tmp_idf.predict(feat_subtrain))
    )
    test_score.append(
        f1_score(labl_test, tmp_idf.predict(feat_test))
    )

df_cv_rslt = pd.DataFrame(dict({'test_score': test_score, 'train_score': train_score}))


#####################################################################
### Sklearn Cross-Validation
#####################################################################
logo = LeaveOneGroupOut()
scores = ['f1']
rslt = cross_validate(idf,
                      feat_train,
                      labl_train,
                      groups=groups,
                      cv=logo,
                      return_train_score=True,
                      return_estimator=True,
                      scoring=scores,
                      # fit_params={'sample_weight': s_weights},
                      verbose=False,
                      n_jobs=-1)
df_rslt = pd.DataFrame(rslt)



# Compare both methods
print(df_cv_rslt.mean())
print(df_rslt.mean())


# Conclusion: Mean errors are approximately equal
# Not exactly because,
# Shuffling of data
# Random state in solvers
# Still the manual CV is accurate, so, it can be used for learning curves
