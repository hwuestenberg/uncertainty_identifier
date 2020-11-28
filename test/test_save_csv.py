# ###################################################################
# script test_svae_csv
#
# Description
# Test numpy.savetxt function for simple csv data files.
#
# ###################################################################
# Author: hw
# created: 18. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import numpy as np

import uncert_ident.data_handling.data_import as di


#####################################################################
### Test
#####################################################################

maindict = dict()
subdict = {'x': np.arange(100),
           'y': np.arange(100)*0.01,
           'uu': np.arange(100)*2,
           'vv': np.arange(100)*3,
           'ww': np.arange(100)*4,
           'pp': np.arange(100)*5,
           'nx': 50,
           'ny': 2
           }

save_path = '../' + di.path_to_raw_data + 'Xiao_PH_DNS/case_0p5/dns-data/mean_files_test.dat'
np.savetxt(save_path,
           np.array([subdict['x'], subdict['y'], subdict['uu'], subdict['vv'], subdict['ww'], subdict['pp']]).T,
           fmt=[' %.8e', '%.8e', '%.16e', '%.16e', '%.16e', '%.16e'],
           delimiter='    ',
           newline=' \n',
           header='x\t\ty\t\tuu\t\tvv\t\tww\t\tpp',
           )

new_dict = di.load_csv(save_path[3:], ['x', 'y', 'uu', 'vv', 'ww', 'pp'], skip_header=1, delimiter='', newline_char=' \n')


print('EOF')
