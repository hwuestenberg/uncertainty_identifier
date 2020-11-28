# ###################################################################
# Test test_visualise_databasis
#
# Description
# Test the visualisation of a general databasis of flow cases.
#
# ###################################################################
# Author: hw
# created: 15. Jun. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from uncert_ident.methods.classification import get_databasis_frames


# Get data
df_base = get_databasis_frames()[0]
df_base.info()


data = np.array(df_base['num_of_points'].to_list())
data = data/np.sum(data)*100

# Plot bar
ax = plt.bar(x=df_base['names'].to_list(), height=data)
plt.xticks(df_base['names'].to_list(), rotation='vertical')
plt.title('Data partitioning')
plt.ylabel('Relative number of points')
plt.savefig('data_partitioning.pdf', format='pdf')
plt.show()
