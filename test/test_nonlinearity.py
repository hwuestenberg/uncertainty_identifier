import pandas as pd
import numpy as np

from uncert_ident.utilities import feature_to_q_keys
from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.visualisation.plotter import *


# case = flowCase("NACA4412-Vinuesa-top-1")
case = flowCase("PH-Xiao-10")
case.get_features()
case.get_labels()

a = 500
# idx = range(a*52, (a+1)*52)
idx = range(a*385, (a+1)*385)


data = case.flow_dict
x = data['x'][idx]
y = data['y'][idx]


labl = case.label_dict['anisotropic'][idx]


feat = case.feature_dict
tke = feat['tke'][idx]
red = feat['Re_d'][idx]
keps = feat["k_eps_Sij"][idx]
orthogonal = feat["orthogonal"][idx]
stream_curv = feat["stream_curv"][idx]
pm_normal_shear_ratio = feat["pm_normal_shear_ratio"][idx]
inv02 = feat['inv02'][idx]
inv03 = feat['inv03'][idx]
inv09 = feat['inv09'][idx]
inv13 = feat['inv13'][idx]

lr_model = 0.45*tke - 2.26*red + 0.31*keps + 0.24*orthogonal - 0.19*stream_curv - 0.03*pm_normal_shear_ratio - 0.71*inv02 + 0.17*inv03 + 0.08*inv09 + 0.15*inv13
sp_model = 5.38*tke**0.5*keps**2 - 1.72*red*3.0 - 0.14*red*3.5 + 0.73*red*4.0 - 0.11*red*4.5
red_poly = - 1.72*red*3.0 - 0.14*red*3.5 + 0.73*red*4.0 - 0.11*red*4.5

fig, ax = empty_plot(figwidth=beamer_textwidth)
lining(5.38*tke**0.5*keps**2, y, append_to_fig_ax=(fig, ax), color='b', line_label="tke-keps term")
lining(red_poly, y, append_to_fig_ax=(fig, ax), linestyle="-.", color='b', line_label="red-polynomial")
lining(red, y, append_to_fig_ax=(fig, ax), color='r', line_label="Red")
lining(sp_model, y, append_to_fig_ax=(fig, ax), line_label="SpaRTA")
lining(lr_model, y, linestyle="--k", append_to_fig_ax=(fig, ax), line_label="LR")
# lining(np.zeros_like(y), y, linestyle="-", color='r', append_to_fig_ax=(fig, ax), line_label="decision")
scattering(np.zeros_like(y), y, labl, cmap=confusion_cmap, scale=50, append_to_fig_ax=(fig, ax))
set_limits(ax, [-5, 5], [2.9, 3.05])
set_labels_title(ax, "Model prediction", "y/H", "")
save("../figures/nonlinearity_anisotropy_ph_xiao_10.pdf")
# lining(tke**0.5*keps**2, y, linestyle="--", color='g', append_to_fig_ax=(fig, ax))
# lining(red, y, linestyle="-.", color='b', append_to_fig_ax=(fig, ax))
# lining(keps, y, linestyle="-.", color='r', append_to_fig_ax=(fig, ax))
show()


# case.show_label("anisotropic")
# show()
