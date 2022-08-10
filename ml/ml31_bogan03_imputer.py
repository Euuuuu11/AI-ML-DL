import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

# print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
# imputer = SimpleImputer()
# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent')
# imputer = SimpleImputer(strategy='constant')
# imputer = SimpleImputer(strategy='constant', fill_value=777)
# imputer = KNNImputer()
imputer = IterativeImputer()

imputer.fit(data)
data2 = imputer.transform(data)
print(data2)



