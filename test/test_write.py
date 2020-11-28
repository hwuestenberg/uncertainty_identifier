#!/usr/bin/env python3.6
import numpy as np
import pandas as pd

a = np.arange(1000000)
b = np.zeros_like(a)
c = np.random.random(a.shape)

dictionary = {'arange': a, 'zeros': b, 'random': c}
pd.DataFrame(dictionary).to_csv('job_test')

import uncert_ident
print(uncert_ident.PHYSICAL_KEYS)
