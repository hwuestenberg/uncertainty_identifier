# ###################################################################
# Test show_model_bar
#
# Description
# Test for model visualisation using a bar plot for each coefficient.
#
# ###################################################################
# Author: hw
# created: 23. Jul. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.visualisation import plotter as plot
from uncert_ident.utilities import num_of_features, FEATURE_KEYS
from uncert_ident.data_handling.flowcase import flowCase


#####################################################################
### Test
#####################################################################
# Coefficients with colour
coefs = np.random.random(num_of_features)*np.random.randint(-100, 100, num_of_features)
coefs = np.abs(coefs)
colours = plot.general_cmap(coefs/np.max(coefs))

# Bar config
bar_width = 0.9
bar_pos = [i for i in range(num_of_features)]

# Bar plot
fig, ax = plot.empty_plot()
ax.bar(bar_pos, coefs, width=bar_width, color=colours)

# Set xticks position and label
ax.set_xticks(bar_pos)
ax.set_xticklabels(FEATURE_KEYS, rotation=90)

# Adjust position for labels
l, b, w, h = ax.get_position().bounds
ax.set_position([l, 0.2, w, h])

# Set y-scale to log
ax.set_xlim(None)
ax.set_ylim(None)
ax.set_autoscaley_on(True)
ax.set_yscale('log')


plot.baring(bar_pos,
            coefs,
            bar_width,
            colours,
            FEATURE_KEYS,
            xticklabel_bottom_pos=0.2,
            ylog=True)

plot.show()
