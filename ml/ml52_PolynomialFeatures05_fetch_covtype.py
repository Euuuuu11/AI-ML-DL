from cgi import test
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor

datasets = fetch_covtype()
x,y = datasets.data, datasets.target
print(x.shape, y.shape)     # (506, 13) (506,)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델
model = make_pipeline(StandardScaler(),
                      XGBClassifier()
                      )
model.fit(x_train,y_train)

print('just score : ',model.score(x_test,y_test))
# 0.7665382927362877

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('just cv : ', scores)
print('just cv n/1 : ', np.mean(scores))

########################################## PolynomialFeatures 후 ##########################################
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, include_bias=False)
xp = pf.fit_transform(x)
# print(xp.shape)     # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, test_size=0.2, random_state=1234,
)
model = make_pipeline(StandardScaler(),
                      XGBClassifier()
                      )
model.fit(x_train,y_train)

print('poly score : ',model.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('poly cv : ', scores)
print('poly cv n/1 : ', np.mean(scores))


# just score :  0.8721461580165745
# just cv :  [0.86888191 0.86906478 0.86806437 0.8702373  0.86992395]
# just cv n/1 :  0.8692344612292047
# poly score :  0.8848394619760247