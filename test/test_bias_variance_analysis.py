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

from sklearn.linear_model import LogisticRegression as logReg
from sklearn.metrics import f1_score
from sklearn.base import clone

from uncert_ident.visualisation import plotter as plot
from uncert_ident.methods.classification import get_data, test_train_split_feat_labl
from uncert_ident.utilities import FEATURE_KEYS, get_datetime


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
fname = 'biva' + '_' + 'logReg' + '_' + label_name + '_' + dataset + '_' + datetime_str
print("Running bias-variance analysis for {0:s}, results in {1:s}".format(dataset, fname))


# Get data, group indexes
train_cases, test_cases, df_feat, df_labl = get_data(dataset)
all_cases = train_cases + test_cases

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
num_of_points = []
num_of_datasets = []
for j, valid_case in enumerate(all_cases):
    print("Cross-validation for case {0:01d} out of {1:01d}".format(j+1, len(all_cases)))
    # LeaveOneGroupOut splits
    train_bool = np.array(all_cases) != np.array(valid_case)
    train_cases = np.array(all_cases)[train_bool].tolist()

    # Incrementally use more training data for fitting the identifier, static validation data from LOGO-CV
    for i in range(len(train_cases)):
        print("Fitting with {0:01d} training sets out of {1:01d}".format(i+1, len(train_cases)))
        # Randomly select subset of training data
        random_select = np.random.choice(range(len(train_cases)), i+1, replace=False)
        subtrain_cases = np.array(train_cases)[random_select].tolist()

        # Subtrain/test splits
        feat_subtrain, feat_valid, labl_subtrain, labl_valid = test_train_split_feat_labl(df_feat, df_labl, subtrain_cases, [valid_case], feat_keys, n_labl)

        # Get number of points and datasets
        num_of_datasets.append(i+1)
        num_of_points.append(labl_subtrain.shape[0])

        # Fit new identifier
        tmp_idf = clone(idf)
        # tmp_idf.fit(feat_subtrain, labl_subtrain)
        tmp_idf.fit(np.array([[0, 1, 1, 1, 0], [-1, 0, -1, -1, 0]]).T, np.array([0, 0, 1, 1, 0]))

        # Scoring
        train_score.append(
            f1_score(labl_subtrain, tmp_idf.predict(feat_subtrain))
        )
        test_score.append(
            f1_score(labl_valid, tmp_idf.predict(feat_valid))
        )

df_rslt = pd.DataFrame(
    dict(
        {'test_score': test_score,
         'train_score': train_score,
         'num_of_points': num_of_points,
         'num_of_datasets': num_of_datasets
         }
    )
)



# Look at results both methods
print(df_rslt)


# Write results to csv
print("Finished bias-variance analysis for {0:s}, results in {1:s}".format(dataset, fname))

try:
    df_rslt.to_csv("results/{0}.csv".format(fname))
except Exception:
    df_rslt.to_csv("../results/{0}.csv".format(fname))




#####################################################################
### Plot results
#####################################################################
bias_variance_data = pd.read_csv("results/{0}.csv".format(fname))
n_data = np.unique(bias_variance_data.loc[:, 'num_of_datasets'].to_numpy())
n_datasets = n_data[-1]
tn_scores = bias_variance_data.loc[:, 'train_score'].to_numpy()
tt_scores = bias_variance_data.loc[:, 'test_score'].to_numpy()


# CHECK AVERAGING WITH PROPER VALUES
mean_tn_scores = np.mean(tn_scores.reshape(n_datasets, n_datasets+1), axis=1)
mean_tt_scores = np.mean(tt_scores.reshape(n_datasets, n_datasets+1), axis=1)

plot.lining(np.arange(n_datasets)+1, mean_tn_scores)
plot.lining(np.arange(n_datasets)+1, mean_tt_scores)
