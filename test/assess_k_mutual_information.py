# ###################################################################
# script assess_k_mutual_information
#
# Description
# Assess the influence of varying sample size and varying k on the
# mutual information.
#
# ###################################################################
# Author: hw
# created: 04. Sep. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from uncert_ident.methods.classification import *
from uncert_ident.utilities import PHYSICAL_KEYS, INVARIANT_KEYS, FEATURE_KEYS, LABEL_KEYS, get_datetime
from uncert_ident.visualisation.plotter import *

from sklearn.linear_model import LogisticRegression as logReg
from sklearn.metrics import accuracy_score

#####################################################################
### Configuration
#####################################################################
# Data
n_labl = 1
label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[n_labl]
feat_keys = ['tau_ratio']
scenarios = ['ph']  # Choose test scenario: sep, pg or all
sample_size = 1e5
ks = np.linspace(1, 100, 100, dtype=int).tolist()



#####################################################################
### Learn with varying sample_size
#####################################################################
# Sample and split data
sampls = get_group_samples(scenarios[0], feat_keys, n_labl, sample_size, undersampling=True)
feat = np.vstack([s[0] for s in sampls])
labl = np.hstack([s[1] for s in sampls])


mis = list()
for k in ks:
    print(f"Computing MI for k = {k:d}")
    mis.append(mutual_info_classif(feat, labl, n_neighbors=k))


#####################################################################
### Evaluate and plot results
#####################################################################
# Plot
fig, ax = empty_plot(figwidth=latex_textwidth*2/3)

lining(ks, mis, color=corange, append_to_fig_ax=(fig, ax))
ax.plot(
            color=corange,
            linewidth=0.7,
            linestyle='-',
            lolims=True
            )

set_limits(ax,
           xlim=[1, 12],
           ylim=[min(mis), max(mis)])
ax.set_xlabel('k')
ax.set_ylabel(r'$MI(q_8; y_{II})$')
ax.grid(color=cblack, linestyle='-', linewidth=0.25, which='both', axis='x')


save('./figures/mutual_information_k_dependency.pdf')
# plt.close()
show()
