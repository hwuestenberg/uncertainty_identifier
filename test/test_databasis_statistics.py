# ###################################################################
# function databasis_statistics
#
# Description
# Test the evaluation of statistical data from a given data basis.
#
# ###################################################################
# Author: hw
# created: 11. Jun. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import pandas as pd

from uncert_ident.data_handling.flowcase import flowCase
from uncert_ident.data_handling.data_import import find_case_names

#####################################################################
### Tests
#####################################################################
# Get cases
case_names = find_case_names()
hifi_data = list()
for case_name in case_names:
    hifi_data.append(flowCase(case_name, get_features=True, get_labels=True))
print("No of cases: " + str(len(hifi_data)))


# Get metadata
nams, npts, dims, geos, feat, labl = list(), list(), list(), list(), list(), list()
for case in hifi_data:
    nams.append(case.case_name)
    npts.append(case.num_of_points)
    dims.append(case.dimension)
    geos.append(case.geometry)
    case.feature_dict.update({'case': case.case_name})
    case.label_dict.update({'case': case.case_name})
    feat.append(pd.DataFrame(case.feature_dict))
    labl.append(pd.DataFrame(case.label_dict))


# Convert features to array
# case.feature_array = convert_dict_to_ndarray(case.feature_dict)


# Convert to pandas dataframe
databasis_frame = pd.DataFrame({
    "names": nams,
    "dimension": dims,
    "geometry": geos,
    "num_of_points": npts
})

feature_frame = pd.concat(feat)
label_frame = pd.concat(labl)


# Print metastats
for frame in [databasis_frame, feature_frame, label_frame]:
    print(frame.head())
    print(frame.dtypes)
    print(frame.describe())


# Compute useful stats
# Mean of each label by case
# label_means = label_frame.groupby('case').mean()

# sum_all_non_negative = label_frame.groupby('case')['non_negative'].value_counts()
# sum_all_non_negative = label_frame.groupby('case')['non_negative'].value_counts()
# sum_all_non_negative = label_frame.groupby('case')['non_negative'].value_counts()

# sum_non_negative = label_frame['non_negative'].value_counts()
# sum_anisotropic = label_frame['anisotropic'].value_counts()
# sum_non_linear = label_frame['non_linear'].value_counts()


# Ratio C1-C2, Skewness
# rc1 = sum_c1/sum_c
# rc2 = sum_c2/sum_c
# ratio = rc1/rc2
# print('N_C1: %r\nN_C2: %r, C1/C2: %r' % (rc1, rc2, ratio))
