# ###################################################################
# test physical_confusion
#
# Description
# Test the confusion-scatter plot on a physical domain.
#
# ###################################################################
# Author: hw
# created: 29. Jun. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.data_handling.data_import import find_case_names
from uncert_ident.methods.classification import get_databasis_frames, confusion_matrix, TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE
from uncert_ident.utilities import LABEL_KEYS
from uncert_ident.visualisation import plotter as plot

#####################################################################
### Test
#####################################################################
df_data, df_features, df_labels = get_databasis_frames(get_features=True, get_labels=True)
pred_case = 'PH-Breuer-10595'


# Get test case physical data
pred_data = flowCase(pred_case).flow_dict
stat_x, stat_y = pred_data['x'], pred_data['y']
nx, ny = pred_data['nx'], pred_data['ny']


# Build confusion matrix from predicted and true labels
pred_labl = np.random.choice([0, 1], size=nx*ny)
true_labl = df_labels.loc[df_labels['case'] == pred_case, LABEL_KEYS[1]].to_numpy()
confusion = confusion_matrix(pred_labl, true_labl)

pred_deci = np.random.randint(-100, 100, nx*ny)


# Plot random predictions
plot.physical_confusion(pred_data, confusion, show_background=False)
plot.physical_decision(pred_data, pred_deci, show_background=False)


#
# # Plot setup
# xs, ys, confusions = [], [], []
# labels_color = ["True-Positive", "True-Negative", "False-Positive", "False-Negative"]
# marks = [TRUE_POSITIVE, TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE]
#
# for mark in marks:
#     idx = np.where(confusion == mark)
#     xs.append(stat_x[idx])
#     ys.append(stat_y[idx])
#     confusions.append(confusion[idx])
#
# cmap = plt.cm.RdYlBu
# clrs = cmap(marks)
#
#
# # Scatter plot on physical domain
# fig, ax = plot.empty_plot()
# for x, y, confusion in zip(xs, ys, confusions):
#     ax.scatter(x, y, s=5, c=confusion, cmap=cmap, vmin=0, vmax=1)
# ax.contourf(stat_x.reshape(nx, ny),
#             stat_y.reshape(nx, ny),
#             pred_data['um'].reshape(nx, ny),
#             alpha=0.4)
#
#
# # Legend
# legend_elements = list()
# for clr, label in zip(cmap(marks), labels_color):
#     legend_elements.append(Line2D([0], [0], marker='o', color=clr, markersize=4, markerfacecolor=clr, label=label))
# ax.legend(handles=legend_elements, loc=0)


plot.show()



print('EOF test_physical_confusion')
