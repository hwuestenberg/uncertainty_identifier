# ###################################################################
# test classifier_performance_metrics
#
# Description
# Test the results of various performance metrics on synthetic and
# real data sets for dummy classifiers and learned classifiers.
#
# ###################################################################
# Author: hw
# created: 20. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import pandas as pd

from uncert_ident.methods.classification import get_databasis_frames
from uncert_ident.utilities import PHYSICAL_KEYS, LABEL_KEYS

from sklearn.linear_model import LogisticRegression as logReg
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.datasets import make_classification

#####################################################################
### Test
#####################################################################
# Generate synthetic data
x, y = make_classification(n_features=10,
                           n_informative=2,
                           n_redundant=2,
                           weights=[0.95],
                           n_classes=2,
                           n_samples=1000000,
                           random_state=False
                           )
bin0, bin1 = np.bincount(y)
print("Synthetic imbalance is: {0} active label".format(bin1/(bin1+bin0)))
train_test_ratio = 0.2
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=False, test_size=train_test_ratio)
print("Train/Test split ratio: {0}".format(train_test_ratio))

print("Shape of train features/labels: {0}\t{1}".format(xtrain.shape, ytrain.shape))
print("Shape of test  features/labels: {0}\t{1}".format(xtest.shape, ytest.shape))


# Load flow data
# hold_out_cases = [
#     # 'CBFS-Bentaleb',
#     'TBL-APG-Bobke-b2',
#     'TBL-APG-Bobke-b1',
#     'TBL-APG-Bobke-m13',
#     'TBL-APG-Bobke-m16',
#     'TBL-APG-Bobke-m18',
#     'CDC-Laval',
#     # 'PH-Breuer-1400',
#     'PH-Breuer-10595',
#     # 'PH-Breuer-2800',
#     # 'PH-Breuer-700',
#     # 'PH-Breuer-5600',
#     'NACA4412-Vinuesa-top-4',
#     'NACA4412-Vinuesa-bottom-10',
#     'NACA4412-Vinuesa-bottom-4',
#     'NACA4412-Vinuesa-bottom-1',
#     'NACA4412-Vinuesa-top-1',
#     'NACA4412-Vinuesa-bottom-2',
#     'NACA4412-Vinuesa-top-10',
#     'NACA4412-Vinuesa-top-2',
#     # 'PH-Xiao-12',
#     'PH-Xiao-15',
#     # 'PH-Xiao-08',
#     # 'PH-Xiao-05',
#     # 'PH-Xiao-10',
#     'NACA0012-Tanarro-top-4'
# ]
# assert isinstance(hold_out_cases, list), 'hold_out_cases needs to be a list type'
#
# test_cases = [
#     'PH-Breuer-10595',
#     'PH-Xiao-15',
#     'CBFS-Bentaleb',
# ]
# assert isinstance(test_cases, list), 'test_cases needs to be a list type'
#
# df_data, df_feat, df_labl = get_databasis_frames(get_features=True, get_labels=True)
#
#
# # Get group-wise data, remove test data
# groups = df_feat.loc[:, 'case'].to_list()  # Get list of names with correct indexes
# groups = [case for case in groups if case not in hold_out_cases]  # Remove hold_out_cases
#
# # Test/Train splits
# xtrain = df_feat.loc[~df_feat['case'].isin(hold_out_cases), PHYSICAL_KEYS].to_numpy()
# ytrain = df_labl.loc[~df_labl['case'].isin(hold_out_cases), LABEL_KEYS[1]].to_numpy()
#
# xtest = df_feat.loc[df_feat['case'].isin(test_cases), PHYSICAL_KEYS].to_numpy()
# ytest = df_labl.loc[df_labl['case'].isin(test_cases), LABEL_KEYS[1]].to_numpy()
#
# train_test_ratio = len(ytest)/(len(ytrain) + len(ytest))
# print("")
# print("Train/Test split ratio: {:3.2f}".format(train_test_ratio))
#
# print("Shape of train features/labels: {0}\t{1}".format(xtrain.shape, ytrain.shape))
# print("Shape of test  features/labels: {0}\t{1}".format(xtest.shape, ytest.shape))
#
#
# bin0, bin1 = np.bincount(ytrain)
# BIN0, BIN1 = np.bincount(ytest)
# print("Imbalancing in train labels: {:3.2f}".format(bin1/(bin1 + bin0)))
# print("Imbalancing in test  labels: {:3.2f}".format(BIN1/(BIN1 + BIN0)))



# Create constant classifier
# dy = DummyClassifier(strategy="constant", random_state=False, constant=0)
# dy.fit(xtrain, ytrain)

# Create random classifier
dy = DummyClassifier(strategy="uniform", random_state=False)
dy.fit(xtrain, ytrain)


# Learn logReg classifier
lr = logReg(random_state=True, fit_intercept=False, class_weight='balanced', verbose=False, n_jobs=-1)
lr.fit(xtrain, ytrain)



# Test classifiers with distinct metrics
print("\n")
print("Dummy performance:")
dy_pred = dy.predict(xtest)
print(classification_report(ytest, dy_pred, target_names=['Inactive', 'Active']))
dy_conf = confusion_matrix(ytest, dy_pred)
dy_TN, dy_FP = dy_conf[0]
dy_FN, dy_TP = dy_conf[1]
dy_sen = dy_TP/(dy_TP + dy_FN)
dy_spe = dy_TN/(dy_TN + dy_FP)
dy_gm = np.sqrt(dy_sen*dy_spe)
print("\t\t\tSensitivity\tSpecificity\tGeom. Mean"
      "\n\tActive\t\t{:2.2f}\t\t{:2.2f}\t\t{:2.2f}".format(dy_sen, dy_spe, dy_gm))



print("\n\n")
print("LogReg performance:")
lr_pred = lr.predict(xtest)
print(classification_report(ytest, lr_pred, target_names=['Inactive', 'Active']))
lr_conf = confusion_matrix(ytest, lr_pred)
lr_TN, lr_FP = lr_conf[0]
lr_FN, lr_TP = lr_conf[1]
lr_sen = lr_TP/(lr_TP + lr_FN)
lr_spe = lr_TN/(lr_TN + lr_FP)
lr_gm = np.sqrt(lr_sen*lr_spe)
print("\t\t\tSensitivity\tSpecificity\tGeom. Mean"
      "\n\tActive\t\t{:2.2f}\t\t{:2.2f}\t\t{:2.2f}".format(lr_sen, lr_spe, lr_gm))

