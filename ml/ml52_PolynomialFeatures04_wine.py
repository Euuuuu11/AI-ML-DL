from cgi import test
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

datasets = load_wine()
x,y = datasets.data, datasets.target
print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델
model = make_pipeline(StandardScaler(),
                      LogisticRegression()
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
                      LogisticRegression()
                      )
model.fit(x_train,y_train)

print('poly score : ',model.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('poly cv : ', scores)
print('poly cv n/1 : ', np.mean(scores))

# 0.8745129304823863

# just score :  0.9722222222222222
# just cv :  [1.         1.         0.96428571 0.92857143 1.        ]
# just cv n/1 :  0.9785714285714286
# poly score :  0.9722222222222222
# poly cv :  [0.96551724 0.96551724 0.96428571 0.89285714 1.        ]
# poly cv n/1 :  0.9576354679802955