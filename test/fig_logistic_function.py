# ###################################################################
# Script fig_logistic_function
#
# Description
# Visualise the logistic function.
#
# ###################################################################
# Author: hw
# created: 31. Aug. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.visualisation.plotter import *
from uncert_ident.methods.classification import sigmoid



#####################################################################
### Plot
#####################################################################
# Define points
x = np.linspace(-5, 5, 50)
y = sigmoid(x)

fig, ax = empty_plot(figwidth=beamer_textwidth/2)
lining(x, y,
       xlabel=r"$x$",
       ylabel=r"$\sigma(x)$",
       linestyle='-',
       color=cred,
       grid=True,
       xlim=[-5, 5],
       ylim=[0, 1],
       # line_label="",
       append_to_fig_ax=(fig, ax))

save("./figures/logistic_function.pdf")
show()
