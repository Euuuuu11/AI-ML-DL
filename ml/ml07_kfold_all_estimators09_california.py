from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
# allAlogrithms = all_estimators(type_filter='classifier')
allAlogrithms = all_estimators(type_filter='regressor')

# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>

print('allAlogrithms : ', allAlogrithms)    # 딕셔너리들이 list 형태로 묶여져있다.
print('모델의 개수 : ', len(allAlogrithms))  # 모델의 개수 :  41

# [예외처리] 에러가 떳을 때 무시하고, 넘어가겠다. 
for (name, algorithm) in allAlogrithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        
        scores = cross_val_score(model, x_test, y_test, cv=5)  
        print(name , scores, '\n cross_val_score : ', round(np.mean(scores), 4))
    except :
        # continue    
        print(name, '은 안나온 놈!!') 

# 모델의 개수 :  54
# ARDRegression [0.62798614 0.59392101 0.61858153 0.56960191 0.58544739]
#  cross_val_score :  0.5991
# AdaBoostRegressor [0.51562124 0.51269119 0.52047491 0.34771164 0.50280185] 
#  cross_val_score :  0.4799
# BaggingRegressor [0.76000139 0.76915171 0.74168336 0.73293959 0.75751027] 
#  cross_val_score :  0.7523
# BayesianRidge [0.64269028 0.60819636 0.6231089  0.57317578 0.5886951 ] 
#  cross_val_score :  0.6072
# CCA [0.59584845 0.56436917 0.58405774 0.54339311 0.56320108] 
#  cross_val_score :  0.5702
# DecisionTreeRegressor [0.56144177 0.53031084 0.52405163 0.47818175 0.59234519] 
#  cross_val_score :  0.5373
# DummyRegressor [-1.48191101e-03 -2.84861137e-03 -7.06672196e-06 -2.50922126e-06
#  -2.08016311e-04]
#  cross_val_score :  -0.0009
# ElasticNet [0.4486574  0.4293245  0.45138673 0.4134193  0.44753709] 
#  cross_val_score :  0.4381
# ElasticNetCV [0.52298714 0.59200574 0.52392922 0.54917483 0.60215608] 
#  cross_val_score :  0.5581
# ExtraTreeRegressor [0.52019832 0.45977231 0.4811707  0.44400098 0.48140489] 
#  cross_val_score :  0.4773
# ExtraTreesRegressor [0.80142246 0.79598242 0.77002153 0.76204335 0.79186992] 
#  cross_val_score :  0.7843
# GammaRegressor [-1.49815468e-03 -2.84520170e-03 -7.06983687e-06 -2.51457165e-06
#  -2.09336813e-04]
#  cross_val_score :  -0.0009
# GaussianProcessRegressor [-2.989966   -2.9837044  -2.95417549 -3.21670114 -3.16672381] 
#  cross_val_score :  -3.0623
# GradientBoostingRegressor [0.79683144 0.78511968 0.77893488 0.75027392 0.77776248] 
#  cross_val_score :  0.7778
# HistGradientBoostingRegressor [0.82845624 0.83312681 0.82102517 0.79045137 0.81592583] 
#  cross_val_score :  0.8178
# HuberRegressor [-3.043035    0.5238796   0.48951294  0.47414161  0.42088629] 
#  cross_val_score :  -0.2269
# IsotonicRegression 은 안나온 놈!!
# KNeighborsRegressor [ 0.05336896  0.01991903 -0.02685599  0.06387441 -0.02921406] 
#  cross_val_score :  0.0162
# KernelRidge [0.58857453 0.54905944 0.55793954 0.50166245 0.56427736] 
#  cross_val_score :  0.5523
# Lars [0.6429199  0.60816574 0.62300092 0.57323127 0.58700286] 
#  cross_val_score :  0.6069
# LarsCV [0.50345234 0.6074391  0.50678138 0.57116384 0.58700286] 
#  cross_val_score :  0.5552
# Lasso [0.31460208 0.30870164 0.3195234  0.30896544 0.321871  ] 
#  cross_val_score :  0.3147
# LassoCV [0.53039293 0.59693855 0.53130329 0.55495793 0.60700044] 
#  cross_val_score :  0.5641
# LassoLars [-1.48191101e-03 -2.84861137e-03 -7.06672196e-06 -2.50922126e-06
#  -2.08016311e-04]
#  cross_val_score :  -0.0009
# LassoLarsCV [0.50345234 0.6074391  0.50678138 0.57116384 0.58700286] 
#  cross_val_score :  0.5552
# LassoLarsIC [0.6429199  0.60816574 0.62300092 0.57323127 0.59233505] 
#  cross_val_score :  0.6079
# LinearRegression [0.6429199  0.60816574 0.62300092 0.57323127 0.58700286] 
#  cross_val_score :  0.6069
# LinearSVR [-12.74428286   0.33249627  -0.20121177   0.35996349   0.44215397] 
#  cross_val_score :  -2.3622
# MLPRegressor [-0.23900762  0.00758842  0.53157038  0.46991934  0.50924587] 
#  cross_val_score :  0.2559
# MultiOutputRegressor 은 안나온 놈!!
# MultiTaskElasticNet 은 안나온 놈!!
# MultiTaskElasticNetCV 은 안나온 놈!!
# MultiTaskLasso 은 안나온 놈!!
# MultiTaskLassoCV 은 안나온 놈!!
# NuSVR [-0.0158815   0.00336708 -0.00755387 -0.00956899 -0.01053165] 
#  cross_val_score :  -0.008
# OrthogonalMatchingPursuit [0.50563945 0.46779193 0.51122491 0.44706906 0.49268048]
#  cross_val_score :  0.4849
# OrthogonalMatchingPursuitCV [0.53715276 0.60235038 0.53808783 0.56539188 0.59061879] 
#  cross_val_score :  0.5667
# PLSCanonical [0.44896768 0.42159342 0.41179117 0.33502586 0.31748746]
#  cross_val_score :  0.387
# PLSRegression [0.56402923 0.5370871  0.55472445 0.48700376 0.51347915] 
#  cross_val_score :  0.5313
# PassiveAggressiveRegressor [ -6.55181057  -0.12474928  -0.39940544  -0.09217344 -49.75148376] 
#  cross_val_score :  -11.3839
# PoissonRegressor [-1.58072378e-03 -3.01879436e-03 -7.50431027e-06 -2.66242435e-06
#  -2.21135276e-04]
#  cross_val_score :  -0.001
# RANSACRegressor [ -8.33639768   0.58087039 -23.65022632   0.25214414  -0.1491954 ] 
#  cross_val_score :  -6.2606
# RadiusNeighborsRegressor [nan nan nan nan nan] 
#  cross_val_score :  nan
# RandomForestRegressor [0.79036507 0.7932697  0.75947193 0.75490986 0.78727123] 
#  cross_val_score :  0.7771
# RegressorChain 은 안나온 놈!!
# Ridge [0.64288183 0.60817203 0.62302256 0.5732222  0.58739851] 
#  cross_val_score :  0.6069
# RidgeCV [0.64254484 0.60821494 0.62300306 0.57312284 0.59063757] 
#  cross_val_score :  0.6075
# SGDRegressor [-1.41754972e+30 -2.64129390e+29 -3.36946001e+28 -3.02551840e+28
#  -1.11335844e+29]
#  cross_val_score :  -3.713929474561293e+29
# SVR [-0.06651636 -0.02356704 -0.04755329 -0.05752528 -0.0604402 ] 
#  cross_val_score :  -0.0511
# StackingRegressor 은 안나온 놈!!
# TheilSenRegressor [-1.22326202e+02  6.22667400e-01 -2.14464066e+01 -1.57028193e-02
#  -1.36418238e-01]
#  cross_val_score :  -28.6604
# TransformedTargetRegressor [0.6429199  0.60816574 0.62300092 0.57323127 0.58700286] 
#  cross_val_score :  0.6069
# TweedieRegressor [0.53003375 0.50173727 0.52442387 0.4616551  0.52035203] 
#  cross_val_score :  0.5076
# VotingRegressor 은 안나온 놈!!         