# ###################################################################
# script LogReg_identifier
#
# Description
# Learn identifier for the error metrics using the databasis.
#
# ###################################################################
# Author: hw
# created: 12. Jun. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.methods.classification import *
from uncert_ident.utilities import PHYSICAL_KEYS, INVARIANT_KEYS, FEATURE_KEYS, LABEL_KEYS, get_datetime
from uncert_ident.visualisation.plotter import *





#####################################################################
### Configuration
#####################################################################
# Data
n_labl = 1
label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[n_labl]
feat_keys = FEATURE_KEYS  # Choose features: FEATURE_KEYS, INVARIANT_KEYS or PHYSICAL_KEYS
scenario = 'ph'  # Choose test scenario: sep, pg or all
sample_size = 1e4



#####################################################################
### Mutual information
#####################################################################
# mi = mutual_info_classif(feat, labl, n_neighbors=3)
# bar_label = [fkeys for _, fkeys in sorted(zip(mi, feat_keys), key=lambda pair: pair[0], reverse=True)]
# mi = np.sort(mi)[::-1]  # Sort ascending
#
# print(f"5 most-informative features:\n{bar_label[:5]}")
#
# baring(np.arange(mi.shape[0])+1, mi, 0.5, xticklabels=[bl.replace('_', '') for bl in bar_label], color=general_cmap(1))


bar_label = feature_selection(scenario, n_labl, sample_size=1e4, n_feat=10)
print(bar_label)
# save('FAKE_MI.pdf')
# plt.close()
show()
