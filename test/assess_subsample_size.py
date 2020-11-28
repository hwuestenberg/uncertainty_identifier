# ###################################################################
# script assess_subsample_size
#
# Description
# Assess the influence of varying sample size for undersampling on
# the training and test error of LogReg models.
#
# ###################################################################
# Author: hw
# created: 04. Sep. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from itertools import cycle

from uncert_ident.methods.classification import *
from uncert_ident.utilities import PHYSICAL_KEYS, INVARIANT_KEYS, FEATURE_KEYS, LABEL_KEYS, get_datetime
from uncert_ident.visualisation.plotter import *

from sklearn.linear_model import LogisticRegression as logReg
from sklearn.metrics import accuracy_score
from sklearn.svm import l1_min_c


#####################################################################
### Configuration
#####################################################################
# Data
n_labl = 1
label_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[n_labl]
feat_keys = FEATURE_KEYS  # Choose features: FEATURE_KEYS, INVARIANT_KEYS or PHYSICAL_KEYS
scenarios = ['all']  # Choose test scenario: sep, pg or all
sample_sizes = np.logspace(1, 6, 11)

# Solver
max_iter = 1000
penalty = 'elasticnet'
solver = 'saga'
l1_ratio = 1.0  # Strong L1


# test_case = "TBL-APG-Bobke-m18"
# test_case = "CBFS-Bentaleb"
test_case = "PH-Breuer-700"
df_data, df_feat, df_labl = get_databasis_frames(test_case, True, True)
feat_test, labl_test = get_feat_labl_for_list(df_feat, df_labl, [test_case], feat_keys, n_labl)


#####################################################################
### Learn with varying sample_size
#####################################################################
scenario_datas = list()
for scenario in scenarios:

    metrics = list()
    for sample_size in sample_sizes:
        print(f"Sampling with s = {int(sample_size)}")
        f1_trains = list()
        f1_tests = list()
        acc_trains = list()
        acc_tests = list()
        for i in range(10):
            # Sample and split data
            sampls = get_group_samples(scenario, feat_keys, n_labl, sample_size, undersampling=True)
            feat_train = np.vstack([s[0] for s in sampls[1:]])
            labl_train = np.hstack([s[1] for s in sampls[1:]])
            # feat_train = np.vstack([s[0] for s in [sampls[0]] + sampls[2:]])
            # labl_train = np.hstack([s[1] for s in [sampls[0]] + sampls[2:]])


            c = (l1_min_c(feat_train, labl_train, loss='log') * np.logspace(0, 4, 5)).tolist()[1]
            # if sample_size > 100:  # Fix for PH-Breuer
            #     c = 0.02761796
            # Learn identifier
            idf = logReg(random_state=False,
                         fit_intercept=False,
                         class_weight='none',
                         max_iter=max_iter,
                         penalty=penalty,
                         solver=solver,
                         C=c,
                         l1_ratio=l1_ratio,  # 0.0=l2, 1.0=l1
                         verbose=False,
                         n_jobs=-1,
                         )
            idf.fit(feat_train, labl_train)


            # Evaluate trainining error
            labl_pred_train = idf.predict(feat_train)
            f1_trains.append(f1_score(labl_train, labl_pred_train))
            acc_trains.append(accuracy_score(labl_train, labl_pred_train))


            # Evaluate generalisation
            labl_pred = idf.predict(feat_test)
            f1_tests.append(f1_score(labl_test, labl_pred))
            acc_tests.append(accuracy_score(labl_test, labl_pred))
            print(f"Sub F1-score:{f1_tests[-1]:0.3f}\t{i}\tSub Acc:\t{acc_tests[-1]:0.3f}")

        print(f"\tAvg. F1-score:{np.mean(f1_tests):0.3f}")#\tAvg. Acc:\t{np.mean(acc_tests):0.3f}")
        metrics.append([f1_trains, f1_tests])#, acc_trains, acc_tests])

    scenario_datas.append(metrics)



#####################################################################
### Evaluate and plot results
#####################################################################
# Evaluate
f1_train_averages = list()
f1_test_averages = list()
f1_train_stds = list()
f1_test_stds = list()
stds = list()
for scenario_data in scenario_datas:
    for s_sizes in scenario_data:
        f1_train_averages.append(np.mean(s_sizes[0]))
        f1_train_stds.append(np.std(s_sizes[0]))
        f1_test_averages.append(np.mean(s_sizes[1]))
        f1_test_stds.append(np.std(s_sizes[1]))
averages = [f1_test_averages]
stds = [f1_test_stds]


# Plot
fig, ax = empty_plot(figwidth=latex_textwidth*2/3)

colors = cycle(all_colors)
lss = cycle(['-', ':', '--', '-.'])
labels = cycle(['f1test', 'f1train', 'acctrain', 'acctest'])
for average, std, ls, clr, linelabel in zip(averages, stds, lss, colors, labels):
    ax.errorbar(sample_sizes, average,
                yerr=2*np.array(std),
                fmt='.-',
                capsize=5,
                color=clr,
                linewidth=0.8,
                linestyle=ls,
                label=linelabel,
                )

set_limits(ax, xlim=[min(sample_sizes)-5, max(sample_sizes)+int(5e4)], ylim=[0, 1])
ax.set_xscale('log')
ax.set_xlabel('Sample size $s$')
ax.set_ylabel('F-measure')
ax.grid(color=cblack, linestyle='-', linewidth=0.15, which='both', axis='x')


save('./figures/undersample_size_anisotropy.pdf')
# plt.close()
show()
