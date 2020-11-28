# ###################################################################
# script assess_data_mutual_information
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
sample_sizes = np.hstack([np.logspace(1, 5, 13), 1e6])



#####################################################################
### Learn with varying sample_size
#####################################################################
points = list()
for sample_size in sample_sizes:

    mis = list()
    for i in range(15):
        # Sample and split data
        sampls = get_group_samples(scenarios[0], feat_keys, n_labl, sample_size, undersampling=True)
        feat = np.vstack([s[0] for s in sampls])
        labl = np.hstack([s[1] for s in sampls])

        mis.append(mutual_info_classif(feat, labl, n_neighbors=3))

    print(f"Avg. MI:{np.mean(mis):1.3f} +/- {np.std(mis):1.3f}")
    points.append(mis)


#####################################################################
### Evaluate and plot results
#####################################################################
# Compute difference to full data MI and average/std
averages = list()
stds = list()
for mis in points[:-1]:
    errors = [abs(mi - np.mean(points[-1])) for mi in mis]
    averages.append(np.mean(errors))
    stds.append(np.std(errors))


# Plot
fig, ax = empty_plot(figwidth=latex_textwidth*2/3)

ax.errorbar(sample_sizes[:-1], averages,
            yerr=2*np.array(stds),
            fmt='.-',
            capsize=1,
            color=corange,
            linewidth=0.7,
            linestyle='-',
            lolims=True
            )

set_limits(ax,
           xlim=[min(sample_sizes)-2, max(sample_sizes[:-1])+20000],
           ylim=[1e-4, 1e0])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of data points $N$')
ax.set_ylabel(r'$\epsilon_{MI}$')
ax.grid(color=cblack, linestyle='-', linewidth=0.25, which='both', axis='x')


save('./figures/mutual_information_data_dependency.pdf')
# plt.close()
show()
