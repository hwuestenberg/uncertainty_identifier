print('Start of test_sh_script.py')
print('Name is: ', repr(__name__))

import numpy as np
print(np.arange(5))

import uncert_ident as ui
print(ui.PHYSICAL_KEYS)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([0, 1], [1, 0])

import gibtesnicht


print('EOF test_sh_script.py')
