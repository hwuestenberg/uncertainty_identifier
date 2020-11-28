# ###################################################################
# Script fig_streamlineas
#
# Description
# Visualise flow with streamlines
#
# ###################################################################
# Author: hw
# created: 27. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np

from uncert_ident.utilities import FEATURE_KEYS, feature_to_q_keys
from uncert_ident.visualisation.plotter import baring, empty_plot, latex_textwidth, show, corange
from uncert_ident.methods.classification import get_group_samples, mutual_info_classif



#####################################################################
### Test
#####################################################################
# Test scenario and error metric
scenario = 'all'  # Choose test scenario: all or ph
label_index = 0

label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[label_index]
label_sym = {0: r"y_{\nu_t}", 1: r"y_{II}"}[label_index]
sname_add = {0: "nut", 1: "ii"}[label_index]



# Sample features and labels
if label_index == 1:
       feature_keys = [fkey if fkey not in ["conv_prod_tke", "tau_ratio"] else None for fkey in FEATURE_KEYS]  # Anisotropy
else:
       feature_keys = [fkey if fkey not in ["conv_prod_tke", "tau_ratio", "visc_ratio", "k_eps_Sij"] else None for fkey in FEATURE_KEYS]  # Non-negativity
for i in range(feature_keys.count(None)):
       feature_keys.remove(None)
sampls = get_group_samples(scenario, feature_keys, label_index, sample_size=int(1e5), undersampling=True)
feat = np.vstack([s[0] for s in sampls])
labl = np.hstack([s[1] for s in sampls])



# Evaluate mutual information
mi = mutual_info_classif(feat, labl, n_neighbors=3)
# QKEYS = [r"$q_{" + f"{i+1:d}" + r"}$" for i in range(len(feature_keys))]
QKEYS = [feature_to_q_keys(fkey) for fkey in feature_keys]
selected_q_keys = [qkey for _, qkey in sorted(zip(mi, QKEYS), key=lambda pair: pair[0], reverse=True)]
selected_feature_keys = [fkey for _, fkey in sorted(zip(mi, feature_keys), key=lambda pair: pair[0], reverse=True)]
mi = np.sort(mi)[::-1]  # Sort ascending



# Reduce number of features
n_feat = 5
selected_q_keys = selected_q_keys[:n_feat]
selected_feature_keys = selected_feature_keys[:n_feat]
mi = mi[:n_feat]

print("Feature map:\n")
for qkey, fkey in zip(selected_q_keys, selected_feature_keys):
    print(f" {qkey.replace('$', '')}\t:\t{fkey}")




# Plot
fig, ax = empty_plot(figwidth=latex_textwidth)
baring(pos=np.arange(mi.shape[0]) + 1,
       heights=mi,
       width=0.5,
       # xticklabels=[sfk.replace('_', '') for sfk in selected_feature_keys],
       # xticklabels=[qkey for qkey in selected_q_keys],
       xticklabels=['$q_{6}$', '$q_{8}$', '$q_{25}$', '$q_{19}$', '$q_{2}$'],
       # xticklabels=[r"$\dfrac{k}{\epsilon}\lVert\mathbf{S}\rVert$", r"$k$", r"$\text{Re}_{\text{d}}$", r"$\lvert U_i U_j \dfrac{\partial U_i}{\partial x_j} \rvert$", r"$\text{Q}$"],
       # xticklabels=[r"$\lvert U_i U_j \dfrac{\partial U_i}{\partial x_j} \rvert$", r"$U_i \dfrac{\partial p}{\partial x_i}$", r"$\overline{\mathbf{P} \mathbf{K}}$", r"$\overline{\mathbf{P}^2 \mathbf{S}^2}$", r"k"],
       ylabel=r"$MI(q_{i}; " + f"{label_sym}" + r")$",
       xlabel="",
       color=corange,
       append_to_fig_ax=(fig, ax),
       sname="../figures/mutual_information_all_" + sname_add + f"_{n_feat:01d}",#_equations_half",
       )

show()
