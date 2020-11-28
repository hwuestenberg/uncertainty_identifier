# ###################################################################
# script evaluation
#
# Description
# Load and evaluate logreg/sparta models.
#
# ###################################################################
# Author: hw
# created: 06. Sep. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from itertools import cycle



from os.path import basename, abspath, dirname
from glob import glob

from uncert_ident.utilities import feature_to_q_keys
from uncert_ident.data_handling.data_import import load_model, find_results
from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.methods.classification import get_sample_cases, get_test_data, predict_case_raw, predict_case_raw_tsc, confusion_matrix
from uncert_ident.visualisation.plotter import scattering, empty_plot, lining, baring, model_matrix_with_score, save, write_model_to_latex, show, update_line_legend, close_all, set_limits, \
    latex_textwidth, cgrey, corange, cred, cyellow, cdarkred, clightyellow, cblack, cwhite, all_colors, Line2D

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, log_loss


def exist_eval_file(name, eval_file):
    dirnames = [basename(dirname(p)) for p in glob(abspath("") + "/results/*/" + eval_file)]
    if name in dirnames:
        return 1
    else:
        return 0


def get_single_result_name(filenames, configuration):
    result_names = []
    for fname in filenames:
        bools = [cfg_item in fname for cfg_item in configuration]
        if all(bools):
            result_names.append(fname)

    if len(result_names) > 1:
        print(f"Multiple results found for config. Newest chosen:\t{result_names[-1]}")

    return result_names[-1]


def get_config_from_result_name(result_name):
    # Define possible scenarios and labels
    algorithms = ['logReg', 'spaRTA']
    scenarios = ['all', 'ph']
    emetrics = ['non_negative', 'anisotropic']
    test_sets = ['PH-Breuer', 'PH-Xiao', 'CBFS', 'TBL-APG', 'NACA']
    ret = []

    # Search each algorithm, scenario, emetric and test_set
    for al in algorithms:
        if result_name.find(al + "_") + 1:
            ret.append(al)
            break
    for sc in scenarios:
        if result_name.find("_" + sc + "_") + 1:
            ret.append(sc)
            break
    for ts in test_sets:
        if result_name.find("_" + ts.replace('-', '_') + "_") + 1:
            ret.append(ts)
            break
    for em in emetrics:
        if result_name.find("_" + em + "_") + 1:
            ret.append(em)
            ret.append({'non_negative': 0, 'anisotropic': 1}[em])
            break

    # ret = [algorithm, scenario, test_set, emetric, emetric_name]
    return ret


def model_and_params(names, algorithm):
    if not isinstance(names, list):
        names = [names]

    models = []
    feat_keyss = []
    n_labls = []
    strings = []

    for name in names:
        i = 0
        while 1:
            try:
                models.append(
                    load_model(LogisticRegression, "./results/" + name + "/models/" + "model_" + f"{i:03d}")
                )
            except FileNotFoundError:
                break

            # Get model params
            idf = models[-1]
            feature_keys = idf.feature_keys
            # hotfix
            if 'conv_prod_tke' in feature_keys:
                feature_keys = [fkey if fkey != 'conv_prod_tke' else 'conv_prod_k' for fkey in idf.feature_keys]

            feat_keyss.append(feature_keys)
            n_labls.append(idf.label_index)
            if algorithm == "spaRTA":
                strings.append(build_model(idf.coef_.flatten(), idf.candidate_library))
            else:
                strings.append(build_model(idf.coef_.flatten(), feature_keys))

            i += 1


    # Combine in dataframe
    df = pd.DataFrame()
    df['model'] = models
    df['feature_keys'] = feat_keyss
    df['label_index'] = n_labls
    df['string'] = strings

    return df


def preload_test_data(algorithm, df, set_test_cases, test_case=None):
    print("Pre-loading test data...")
    if algorithm == "spaRTA":
        candidate_library = df['model'][0].candidate_library
        feat_keys = df['model'][0].feature_keys
        n_labl = df['model'][0].label_index

        if test_case:
            test_cases = [test_case]
        else:
            test_cases = get_sample_cases(set_test_cases)[0]
        test_case_feats = []
        test_case_labls = []

        for test_case in test_cases:
            feat_test, test_case_labl = get_test_data(test_case, feat_keys, n_labl)

            for i, key in enumerate(feat_keys):
                locals().update({key: feat_test[:, i]})
            const = np.ones(feat_test.shape[0])
            locs = locals()
            eval_list = [eval(candidate, {}, locs) for candidate in candidate_library]
            test_case_feat = np.stack(eval_list).T

            test_case_feats.append(test_case_feat)
            test_case_labls.append(test_case_labl)

    else:
        feat_keys = df['model'][0].feature_keys
        n_labl = df['model'][0].label_index

        if test_case:
            test_cases = [test_case]
        else:
            test_cases = get_sample_cases(set_test_cases)[0]
        test_case_feats = []
        test_case_labls = []

        for test_case in test_cases:
            test_case_feat, test_case_labl = get_test_data(test_case, feat_keys, n_labl)
            test_case_feats.append(test_case_feat)
            test_case_labls.append(test_case_labl)

    return test_case_feats, test_case_labls


def quantitative_eval(df, test_case_feats, test_case_labls, set_test_cases, name):
    print("Running quantitative evaluation...")
    # For convenience
    f1_test_avgs = []
    prc_test_avgs = []
    tpr_test_avgs = []
    tnr_test_avgs = []
    fpr_test_avgs = []
    acc_test_avgs = []
    bac_test_avgs = []
    log_test_avgs = []
    complexitys = []

    # Evaluate all models
    clock = timer()
    for row in df.itertuples(index=True):
        print(f"Evaluating model {row[0]:03d}...")

        f1_tests = []
        prc_tests = []
        tpr_tests = []
        tnr_tests = []
        fpr_tests = []
        acc_tests = []
        bac_tests = []
        log_tests = []

        # Loop all test cases in test set
        test_cases = get_sample_cases(set_test_cases)[0]
        for test_case, feat_test, labl_test in zip(test_cases, test_case_feats, test_case_labls):

            # Quantitative evaluation
            labl_pred = row.model.predict(feat_test)
            tp, fp, tn, fn = confusion_matrix(labl_pred, labl_test, return_list=False)

            f1_tests.append(f1_score(labl_test, labl_pred))
            prc_tests.append(precision_score(labl_test, labl_pred))
            tpr_tests.append(recall_score(labl_test, labl_pred))
            tnr_tests.append(tn / (tn + fp))
            fpr_tests.append(fp / (fp + tn))
            acc_tests.append(accuracy_score(labl_test, labl_pred))
            bac_tests.append(balanced_accuracy_score(labl_test, labl_pred))
            log_tests.append(log_loss(labl_test, labl_pred))

        f1_test_avgs.append(np.mean(f1_tests))
        prc_test_avgs.append(np.mean(prc_tests))
        tpr_test_avgs.append(np.mean(tpr_tests))
        tnr_test_avgs.append(np.mean(tnr_tests))
        fpr_test_avgs.append(np.mean(fpr_tests))
        acc_test_avgs.append(np.mean(acc_tests))
        bac_test_avgs.append(np.mean(bac_tests))
        log_test_avgs.append(np.mean(log_tests))
        complexitys.append(np.flatnonzero(row.model.coef_).size)

    print(f"Evaluated models after {timer() - clock:5.3f}")

    # Write to df
    df['f1_test'] = f1_test_avgs
    df['prc_test'] = prc_test_avgs
    df['tpr_test'] = tpr_test_avgs
    df['tnr_test'] = tnr_test_avgs
    df['fpr_test'] = fpr_test_avgs
    df['acc_test'] = acc_test_avgs
    df['bac_test'] = bac_test_avgs
    df['log_test'] = log_test_avgs
    df['complexity'] = complexitys

    # Write to csv
    df.to_csv("./results/" + name + "/quantitative.csv")

    return df


def qualitative_eval(model, test_case, feat_test, labl_test, i_model, name, zoom_data=None):
    fig_directory = "./results/" + name + "/figures/"

    if algo == 'logReg':
        predict_case_raw(model, test_case, model.feature_keys, model.label_index, zoom_data=zoom_data,
                         sname=fig_directory + test_case + f"_model_{i_model:03d}")

    if algo == 'spaRTA':
        predict_case_raw_tsc(model, test_case, feat_test, labl_test, zoom_data=zoom_data,
                             sname=fig_directory + test_case + f"_model_{i_model:03d}")

    print(f"F1_score on {test_case}:\t{f1_score(labl_test, model.predict(feat_test))}")

    return 1


def performance_complexity(name, highlight_idx=None):
    df_quant = pd.read_csv("./results/" + name + "/quantitative.csv")
    complexitys = df_quant['complexity'].to_list()
    f1_tests = df_quant['f1_test'].to_list()

    f1_best = np.sort(f1_tests)[-1]
    idx_best = f1_tests.index(f1_best)
    complexity_best = complexitys[idx_best]
    print(f"Best model: {idx_best}\tF1: {f1_best}\t\tcomplexity: {complexity_best}")

    # All models
    fig, ax = empty_plot(figwidth=latex_textwidth*0.9)
    lining(complexitys, f1_tests,
           linestyle='o',
           marker='o',
           markeredgewidth=0,
           color=cgrey,
           xlim=[0.9, 1200],
           ylim=[0.0, 1.01],
           xlog=True,
           append_to_fig_ax=(fig, ax))

    # Best model
    scattering(complexity_best, f1_best, 1,
               color=cred,
               marker='X',
               scale=100,
               xlabel="Model complexity",
               ylabel="F-measure",
               zorder=2.5,
               append_to_fig_ax=(fig, ax))

    sname_add = "_"

    # Highlights
    if isinstance(highlight_idx, int):
        highlight_idx = [highlight_idx]

    if isinstance(highlight_idx, list):
        for idx, color in zip(highlight_idx, cycle([corange, cyellow, cdarkred, clightyellow, cdarkred])):
            complexity_high = complexitys[idx]
            f1_high = f1_tests[idx]
            print(f"Highlight {idx} at:\tF1: {f1_high}\t\tcomplexity: {complexity_high}")
            scattering(complexity_high, f1_high, 1,
                       color=color,
                       scale=100,
                       marker='^',
                       xlabel="Model complexity",
                       ylabel="F-measure",
                       zorder=2.5,
                       append_to_fig_ax=(fig, ax))
        sname_add = "highlight_"

    legend_elements = [Line2D([0], [0], marker="X", linestyle='', lw=0, markersize=10, color=cred, label="Best"),
                       Line2D([0], [0], marker="^", linestyle='', lw=0, markersize=10, color=corange, label="Algebraic")]
    # Legend
    ax.legend(handles=legend_elements, loc="lower right", numpoints=1)

    # save("./results/" + name + "/figures/performance_vs_complexity" + sname_add + ".pdf")
    save("figures/" + "performance_vs_complexity_" + sname_add + str(algo) + "_" + str(scenario) + "_" + str(
        test_set.replace("-", "_")) + "_" + str(emetric) + "_" + ".pdf")
    # show()

    return idx_best


def receiver_operating_characteristic(configuration):
    # Get all models
    filenames = find_results()
    test_sets = ['PH-Breuer', 'PH-Xiao', 'CBFS', 'TBL-APG', 'NACA']
    algorithms = ["logReg", "spaRTA"]

    f1_bests = []
    tpr_bests = []
    fpr_bests = []
    pbests = []
    f1_highs = []
    tpr_highs = []
    fpr_highs = []
    prc_highs = []
    phighs = []

    for algorithm in algorithms:
        result_names = [get_single_result_name(filenames, [algorithm] + configuration[1:-1] + [test_set_i.replace('-', '_')]) for
                        test_set_i in test_sets]

        if config[2] == "non_negative":
            if algorithm == "logReg":
                highlight = [10, 10, 40, 10, 10]    # LogReg nut
            else:
                highlight = [26, 29, 27, 29, 3]     # SpaRTA nut
        else:
            if algorithm == "logReg":
                highlight = [23, 43, 32, 2, 2]      # LogReg II
            else:
                highlight = [90, 1, 2, 1, 1]        # SpaRTA II


        for name, high in zip(result_names, highlight):
            df_quant = pd.read_csv("./results/" + name + "/quantitative.csv")

            tprs = df_quant['tpr_test'].to_list()
            fprs = df_quant['fpr_test'].to_list()
            prcs = df_quant['prc_test'].to_list()
            f1s = df_quant['f1_test'].to_list()

            # Best model
            idx_best = f1s.index(np.sort(f1s)[-1])
            # tpr_bests.append(np.sort(tprs)[-1])
            # idx_best = tprs.index(tpr_bests[-1])
            f1_bests.append(f1s[idx_best])
            tpr_bests.append(tprs[idx_best])
            fpr_bests.append(fprs[idx_best])
            pbests.append([fpr_bests[-1], tpr_bests[-1]])

            # Highlights
            f1_highs.append(f1s[high])
            tpr_highs.append(tprs[high])
            fpr_highs.append(fprs[high])
            prc_highs.append(prcs[high])
            phighs.append([fpr_highs[-1], tpr_highs[-1]])  # ROC
            # phighs.append([tpr_highs[-1], prc_highs[-1]])  # Precision-Recall curve


    # Total average performance
    total_logreg_best = np.mean(tpr_bests[:5]), np.mean(fpr_bests[:5])
    total_sparta_best = np.mean(tpr_bests[5:]), np.mean(fpr_bests[5:])
    total_logreg_high = np.mean(tpr_highs[:5]), np.mean(fpr_highs[:5])
    total_sparta_high = np.mean(tpr_highs[5:]), np.mean(fpr_highs[5:])

    print(f"Best model performance for Logistic Regression\nTPR, FPR:\t{total_logreg_best}")
    print(f"Best model performance for SpaRTA\nTPR, FPR:\t{total_sparta_best}")
    print(f"Selected model performance for Logistic Regression\nTPR, FPR:\t{total_logreg_high}")
    print(f"Selected model performance for SpaRTA\nTPR, FPR:\t{total_sparta_high}")

    # Setup
    fig, ax = empty_plot(figwidth=latex_textwidth*1.0)
    lining([0, 1], [0, 1],
           linestyle='-k',
           # color=cblack,
           # xlim=[0.9, 1200],
           # ylim=[0.0, 1.01],
           # xlog=True,
           append_to_fig_ax=(fig, ax))

    legend_elements = []
    marker = cycle(['o', '^', 's', 'P', 'D'])
    fillstyles = ["full"] * 5 + ["left"] * 5
    colors = cycle(all_colors[:5])
    # for pbest, phigh, mark, fill, color, test_set_i in zip([total_logreg_high[::-1], total_sparta_high[::-1]], phighs, marker, fillstyles, colors, cycle([r'\textsf{Logistic Regression}', r'\textsf{SpaRTA}'])):
    for pbest, phigh, mark, fill, color, test_set_i in zip(pbests, phighs, marker, fillstyles, colors, cycle([r'\textsf{PH-Re}', r'\textsf{PH-Geo}', r'\textsf{CBFS}', r'\textsf{TBL-APG}', r'\textsf{NACA}'])):
        # scattering(*pbest, 1,
        #            color=cred,
        #            marker=mark,
        #            scale=100,
        #            xlabel="FPR",
        #            ylabel="TPR",
        #            zorder=2.5,
        #            append_to_fig_ax=(fig, ax))

        lining(*phigh,
               color=color,
               xlim=[0.0, 1.0],
               ylim=[0.0, 1.0],
               marker=mark,
               linestyle='-',
               markerfacecoloralt=cblack,
               markersize=10,
               markeredgecolor=cblack,
               fillstyle=fill,
               # xlabel="False-positive rate",
               # ylabel="True-positive rate",
               xlabel=r"$\textsf{FPR}=FP/(FP+TN)$",
               ylabel=r"$\textsf{TPR}=TP/(TP+FN)$",
               line_label="1",
               # zorder=2.5,
               append_to_fig_ax=(fig, ax))
        legend_elements.append(
            Line2D([0], [0], marker=mark, linestyle='-', markerfacecoloralt=cblack, markersize=10, markeredgecolor=cblack, markeredgewidth=1, fillstyle=fill, color=color, label=test_set_i, lw=0)
        )


    # Legend
    ax.legend(handles=legend_elements[:5], loc="lower right", numpoints=1)
    ax.annotate("", xy=[0.55, 0.85], xytext=[0.7, 0.7], arrowprops=dict(fc='k', ec='k', arrowstyle="simple", lw=1.5))
    ax.annotate("", xy=[0.15, 0.45], xytext=[0.3, 0.3], arrowprops=dict(fc='k', ec='k', arrowstyle="simple", lw=1.5))
    ax.set_aspect("equal")

    # show()
    save("figures/" + "roc_" + str(emetric) + ".pdf")

    return 1


def mean_coefficients(configuration):
    # Get all models
    filenames = find_results()
    result_names = [get_single_result_name(filenames, configuration[:-1] + [test_set_i.replace('-', '_')]) for
                    test_set_i in ['PH-Breuer', 'PH-Xiao', 'CBFS', 'TBL-APG', 'NACA']]
    df_eval_all = model_and_params(result_names, configuration[0])


    feature_keys = df_eval_all['model'][0].feature_keys


    # Get and average coefficients
    list_coefs = [row.model.coef_ for row in df_eval_all.itertuples(index=True)]
    coefs = np.vstack(list_coefs)
    mean_coefs = np.abs(coefs).mean(axis=0)
    nzero_coefs = np.count_nonzero(coefs, axis=0)

    # Sort ticklabels and mean coefs
    QKEYS = [feature_to_q_keys(fkey) for fkey in feature_keys]
    selected_q_keys = [qkey for _, qkey in sorted(zip(mean_coefs, QKEYS), key=lambda pair: pair[0], reverse=True)]
    selected_feature_keys = [fkey for _, fkey in sorted(zip(mean_coefs, feature_keys), key=lambda pair: pair[0], reverse=True)]
    sorted_mean_coefs = np.sort(mean_coefs)[::-1]  # Sort ascending


    # Reduce number of features
    n_feat = 5
    selected_q_keys = selected_q_keys[:n_feat]
    selected_feature_keys = selected_feature_keys[:n_feat]
    sorted_mean_coefs = sorted_mean_coefs[:n_feat]

    # Print feature map (might deviate from thesis)
    print("Feature map:\n")
    for qkey, fkey in zip(selected_q_keys, selected_feature_keys):
        print(f" {qkey.replace('$', '')}\t:\t{fkey}")


    fig, ax = empty_plot(figwidth=latex_textwidth/2)
    baring(np.arange(sorted_mean_coefs.size),
           heights=sorted_mean_coefs,
           width=0.5,
           color=corange,
           xticklabels=[qkey for qkey in selected_q_keys],
           # xticklabels=[r"$\overline{\mathbf{S}^3}$", r"$\overline{\mathbf{P}^2 \mathbf{S}^2}$", r"$\dfrac{k}{\epsilon}\lVert\mathbf{S}\rVert$", r"$k$", r"$\overline{\mathbf{S}^2}$"],
           # xticklabels=[r"$\text{Re}_{\text{d}}$", r"$\overline{\mathbf{\Omega}^2}$", r"$k$", r"$\dfrac{k}{\epsilon}\lVert\mathbf{S}\rVert$", r"$\overline{\mathbf{S}^2}$"],
           ylabel="Average magnitude",
           xlabel="",
           append_to_fig_ax=(fig, ax),
           # sname="figures/logreg_model_mean_coef_" + configuration[2] + "_equations",
           )

    return 1


def nzero_coefficients(configuration):
    # Get all models
    filenames = find_results()
    result_names = [get_single_result_name(filenames, configuration[:-1] + [test_set_i.replace('-', '_')]) for
                    test_set_i in ['PH-Breuer', 'PH-Xiao', 'CBFS', 'TBL-APG', 'NACA']]
    df_eval_all = model_and_params(result_names, configuration[0])


    feature_keys = df_eval_all['model'][0].feature_keys


    # Get and average coefficients
    list_coefs = [row.model.coef_ for row in df_eval_all.itertuples(index=True)]
    coefs = np.vstack(list_coefs)
    nzero_coefs = np.count_nonzero(coefs, axis=0)

    # Sort ticklabels and mean coefs
    QKEYS = [feature_to_q_keys(fkey) for fkey in feature_keys]
    selected_q_keys = [qkey for _, qkey in sorted(zip(nzero_coefs, QKEYS), key=lambda pair: pair[0], reverse=True)]
    selected_feature_keys = [fkey for _, fkey in sorted(zip(nzero_coefs, feature_keys), key=lambda pair: pair[0], reverse=True)]
    sorted_nzero_coefs = np.sort(nzero_coefs)[::-1]  # Sort ascending


    # Reduce number of features
    n_feat = 5
    selected_q_keys = selected_q_keys[:n_feat]
    selected_feature_keys = selected_feature_keys[:n_feat]
    sorted_nzero_coefs = sorted_nzero_coefs[:n_feat]

    # Normalise
    sorted_nzero_coefs = sorted_nzero_coefs/max(sorted_nzero_coefs)


    # Print feature map (might deviate from thesis)
    print("Feature map:\n")
    for qkey, fkey in zip(selected_q_keys, selected_feature_keys):
        print(f" {qkey.replace('$', '')}\t:\t{fkey}")


    fig, ax = empty_plot(figwidth=latex_textwidth/2)
    baring(np.arange(sorted_nzero_coefs.size),
           heights=sorted_nzero_coefs,
           width=0.5,
           color=corange,
           xticklabels=[qkey for qkey in selected_q_keys],
           # xticklabels=[r"$\dfrac{k}{\epsilon}\lVert\mathbf{S}\rVert$", r"$k$", r"$U_i \dfrac{\partial p}{\partial x_i}$", r"$\lvert D\Gamma / Ds \rvert$", r"$\text{Re}_{\text{d}}$"],
           # xticklabels=[r"$\text{Re}_{\text{d}}$", r"$\overline{\mathbf{\Omega}^2}$", r"$k$", r"$\dfrac{k}{\epsilon}\lVert\mathbf{S}\rVert$", r"$\lvert U_i U_j \dfrac{\partial U_i}{\partial x_j} \rvert$"],
           ylabel="Relative active feature",
           xlabel="",
           append_to_fig_ax=(fig, ax),
           sname="figures/logreg_model_nzero_coef_" + configuration[2],# + "_equations",
           )

    return 1


def get_model_matrix(name):
    print(f"Creating model matrix for {name}")
    df_quant = pd.read_csv("./results/" + name + "/quantitative.csv")

    df_quant['coef'] = [mdl.coef_.flatten() for mdl in df_quant['model']]
    df_quant['f1_test'] = [f1_test for f1_test in df_quant['f1_test']]

    lib = np.array(df_quant['model'][0].candidate_library)
    inds = df_quant.query("label_index==label_index").index

    model_matrix_with_score(df_quant, lib, inds, sname="./results/" + name + "/model_matrix")

    return 1


def build_model(coefs, candidate_library):
    """ Writes mathematical model as string.

    :param coefs: Coefficient vector (n_features,)
    :param candidate_library: Symbolic library of candidate functions (n_features,)
    :return: model_string
    """
    i_nonzero = np.nonzero(coefs)[0]
    model_string = ''
    coefs = np.round(coefs, 2)
    for i in i_nonzero:
        model_string = model_string + ' + ' + str(abs(coefs[i])) + '*' + candidate_library[i] if coefs[i] > 0 \
            else model_string + " - " + str(abs(coefs[i])) + '*' + candidate_library[i]
    return model_string


def write_models_to_txt(file_name, models, query_inds=None):
    print(f"Writing models to {file_name}")
    if query_inds is None:
        query_inds = models.index.tolist()
    with open(file_name, 'w') as f:
        for i in query_inds:
            f.write('###\n')
            f.write('index:\t\t\t' + str(i) + '\n')
            f.write('F1_test =\t\t' + str(models['f1_test'][i]) + '\n') if "f1_test" in models else f.write("F1_test =\t\t\n")
            f.write('Complexity =\t' + str(np.flatnonzero(models['model'][i].coef_).size) + '\n')
            f.write('model:\t\t\t' + str(models['string'][i]) + '\n')
            f.write('latex:\t\t\t' + write_model_to_latex(feature_to_q_keys(models['string'][i])) + '\n\n')



qualitative = True
batch = False


#####################################################################
### Qualitatively evaluate defined case
#####################################################################
if qualitative:
    # algo = 'logReg'
    algo = 'spaRTA'


    scenario = 'all'
    # scenario = 'ph'


    test_set = 'PH-Breuer'
    # test_set = 'PH-Xiao'
    # test_set = 'CBFS'
    # test_set = 'TBL-APG'
    # test_set = 'NACA'

    test_case = "PH-Breuer-5600"
    # test_case = "PH-Xiao-10"
    # test_case = "CBFS-Bentaleb"
    # test_case = "TBL-APG-Bobke-m13"
    # test_case = "NACA4412-Vinuesa-top-1"
    # assert test_set in test_case, "Test case does not fit selected test set!"


    emetric = 'anisotropic'
    # emetric = 'non_negative'
    # emetric_name = {0: 'non_negative', 1: 'anisotropic', 2: 'non_linear'}[emetric]



    config = [algo, scenario, emetric, test_set.replace('-', '_')]



    # Get result name
    fnames = find_results()
    result_name = get_single_result_name(fnames, config)


    # ROC
    # receiver_operating_characteristic(config)
    # show()
    # assert False, "ROC only"



    # Load all models and params
    df_eval = model_and_params(result_name, algo)
    print(f"Evaluation of scenario {scenario} for {emetric} metric")
    print(f"Model count:\t{len(df_eval)}\nTest set:\t\t{test_set}")


    if algo == "logReg":
        # mean_coefficients(config)
        # nzero_coefficients(config)
        pass


    # Performance vs Complexity
    best_idx = performance_complexity(result_name, 90)
    show()
    assert False, "Performance complexity only"


    # Pre-load test data (avoid multiple loads)
    feat_tests, labl_tests = preload_test_data(algo, df_eval, test_set, test_case)


    # Qualitative evaluation (customise as needed)
    model_idx = 1
    # model_idx = best_idx
    model = df_eval['model'][model_idx]

    zoom_data = None
    # zoom_data = [[[0.2, 0.6]],
    #              [[0.08, 0.15]],
    #              ["crest"],
    #              [10],
    #              ["upper right"],
    #              ["$x/c$"],
    #              ["$y/c$"]]
    # zoom_data = [[[0.2, 0.5], [0.823, 0.971]],
    #              [[0.08, 0.12], [0.008, 0.04]],
    #              ["crest", "aft"],
    #              [10, 10],
    #              [None, None],
    #              ["$x/c$", "$x/c$"],
    #              ["$y/c$", "$y/c$"]]
    # zoom_data = [[[0, 1], [7, 8.9], [1, 4]],
    #              [[0.5, 0.5+0.618], [0, 1.174], [2.5, 3.035]],
    #              ["lee_hill", "wind_hill", "upper_wall"],
    #              [5, 5, 5],
    #              [None, None, None],
    #              ["$x/H$", "$x/H$", "$x/H$"],
    #              ["$y/H$", "$y/H$", "$y/H$"]]

    feat_test = feat_tests[-1]
    labl_test = labl_tests[-1]
    qualitative_eval(model, test_case, feat_test, labl_test, model_idx, result_name, zoom_data=zoom_data)
    show()
    close_all()


    # True error metric
    if not exist_eval_file(result_name, "/figures/true_" + emetric + "_" + test_case + ".jpg"):
        case = flowCase(test_case)
        case.get_labels()
        fig, ax = case.show_label(emetric, show_background=True, labelpos='upper right')#, zoom_box=zoom_data)
        # if zoom_data:
            # set_limits(ax, zoom_data[0][0], zoom_data[1][0])
        save("./results/" + result_name + "/figures/true_" + emetric + "_" + test_case + ".jpg")#_zoom#_zoom_window
        show()
        close_all()


    # Model matrix
    if algo == "spaRTA" and not exist_eval_file(result_name, "model_matrix.pdf"):
        # get_model_matrix(result_name)
        pass




#####################################################################
### Batch run quantitative eval for all models
#####################################################################
if batch:

    # Get result name
    dirnames = find_results()

    for result_name in dirnames:
        # Skip non-result directories
        if "logReg" not in result_name and "spaRTA" not in result_name:
            print(f"Invalid directory:\t{result_name}")
            continue

        algo, scenario, test_set, emetric, emetric_name = get_config_from_result_name(result_name)


        print(f"Evaluating result: \t{result_name}")

        # Load all models and params
        df_eval = model_and_params(result_name, algo)
        print(f"Evaluation of scenario {scenario} for {emetric} metric")
        print(f"Model count:\t{len(df_eval)}\nTest set:\t\t{test_set}")



        # Tabular overview of results
        if not exist_eval_file(result_name, "quantitative.csv"):

            # # Number of candidates and PH-Xiao data is too large for local RAM
            # if algo == "spaRTA" and test_set == "PH-Xiao":
            #     continue


            # Pre-load test data (avoid multiple loads)
            feat_tests, labl_tests = preload_test_data(algo, df_eval, test_set)


            # Produce tabular/overview of performance
            quantitative_eval(df_eval, feat_tests, labl_tests, test_set, result_name)



        # Write to txt
        if not exist_eval_file(result_name, "model_strings.txt"):
            write_models_to_txt("./results/" + result_name + "/model_strings.txt", df_eval)



        # # Get model matrix
        # if not exist_eval_file(result_name, "model_matrix.pdf") and algo == "spaRTA":
        #     get_model_matrix(df_eval, result_name)




print("EOF evaluate_models")
