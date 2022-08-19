from cgi import test
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

datasets = load_diabetes()
x,y = datasets.data, datasets.target
print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델
model = make_pipeline(StandardScaler(),
                      LinearRegression()
                      )
model.fit(x_train,y_train)

print('just score : ',model.score(x_test,y_test))
# 0.7665382927362877

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
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
                      LinearRegression()
                      )
model.fit(x_train,y_train)

print('poly score : ',model.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('poly cv : ', scores)
print('poly cv n/1 : ', np.mean(scores))

# 0.8745129304823863


# just score :  0.46263830098374936
# just cv :  [0.53864643 0.43212505 0.51425541 0.53344142 0.33325728]
# just cv n/1 :  0.47034511859358785
# poly score :  0.4186731903894866
# poly cv :  [0.48845419 0.27385747 0.28899779 0.26214683 0.10885984]
# poly cv n/1 :  0.2844632257068639