# ###################################################################
# Script test_knn
#
# Description
# Visualise kNN distance on real features and labels.
#
# ###################################################################
# Author: hw
# created: 26. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.utilities import *
from uncert_ident.visualisation.plotter import *
from uncert_ident.data_handling.flowcase import flowCase


#####################################################################
### Test
#####################################################################
# Load data
# case_name = "PH-Breuer-10595"
# case = flowCase(case_name)
# case.get_features()
# case.get_labels()
#
#
# labl = case.label_dict['anisotropic']
#
# for key in PHYSICAL_KEYS:
#     feat = case.feature_dict[key]
#     scattering(feat, labl, np.ones_like(feat), scale=5)
#
# show()



