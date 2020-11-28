# ###################################################################
# test eval_model_complexity
#
# Description
# Load a number of models and evaluate them.
#
# ###################################################################
# Author: hw
# created: 07. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import pandas as pd

from glob import glob

# from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score

from uncert_ident.data_handling.data_import import save_model, load_model
from uncert_ident.utilities import FEATURE_KEYS
from uncert_ident.methods.classification import evaluate_model, get_config_from_filename, get_databasis_frames, get_test_scenario, test_train_split_feat_labl, show_model, quantitative_evaluation, qualitative_evaluation, precision_recall_curve, precision_recall_plot

#####################################################################
### Test
#####################################################################
# filename = "gscv_logReg_non_negative_all_20200807_153512"
# filename = "tsc_anisotropic_sep_20200807_175439"
filename = "tsc_anisotropic_sep_20200807_183120"

# Check filename
if filename.find("_model") + 1:
    filename = filename[:-6]  # Remove _model


# Setup
model_constructor, scenario, label_name, label_index = get_config_from_filename(filename)
feature_keys = FEATURE_KEYS


# Load data
# df_info, df_X, df_y = get_databasis_frames(get_features=True, get_labels=True)
# list_train, list_test = get_test_scenario(scenario)
# X_train, X_test, y_train, y_test = test_train_split_feat_labl(df_X, df_y, list_train, list_test, feature_keys, label_index)


# Load all models
model_names = glob("./models" + "/" + filename + "/" + "*.mat")
models = [load_model(model_constructor, name[8:]) for name in model_names]



# Fake data, debug for looping
X_train = np.array([[0, 1, 1, 0, 1, 1, 0, 0],
                       [0, -1, -1, 0, -1, -1, 0, 0]]).T
X_test = np.array([[0, 1, 1, 0, 1, 1, 0, 0],
                      [0, -1, -1, 0, -1, -1, 0, 0]]).T
y_train = np.array([0, 1, 1, 0, 1, 1, 0, 0])
y_test = np.array([0, 1, 1, 0, 1, 1, 0, 0])


# Get test scores
f1 = list()
for model in models:
    f1.append(f1_score(y_test, model.predict(X_test)))

# Get best identifier
best_index = np.argmax(np.array(f1))
best_identifier = models[best_index]


# Model complexity and structure
for model in models:
    show_model(model, print_text=True)
# show_model(identifier, save=filename + '_' + 'model_struct')



# Qualitative evaluation
qualitative_evaluation(identifier, list_test, df_X, df_y, feature_keys, label_index, filename)


# Quantitative evaluation (Print and save classification report)
quantitative_evaluation(identifier, X_test, y_test, filename)


# Save Precision-Recall curve
prc, rcl, thrshlds = precision_recall_curve(y_test, identifier.decision_function(X_test))
precision_recall_plot(prc, rcl, thrshlds, save=filename + "_" + "PRcurve")


