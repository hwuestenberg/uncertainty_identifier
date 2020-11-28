import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.data_handling.flowcase import flowCase


case = flowCase("PH-Breuer-10595")
data = case.flow_dict

x = data['x']
y = data['y']

nx = data['nx']
ny = data['ny']


pos = np.linspace(0, 210, 22)

for p in pos:
    rng = range(int(p*ny), int((p+1)*ny))

    yi = y[rng]
    ui = data['um'][rng]

    # dyi = np.abs(yi[1:] - yi[0:-1])
    # dui = np.mean([ui[1:], ui[:-1]], axis=0)

    # plt.plot(dui, np.cumsum(dyi))
    # plt.plot(ui, yi)

    # volflow = np.sum(dui*dyi)
    volflow = np.trapz(ui, yi)
    mflow = volflow*data['rho']
    print(volflow)
