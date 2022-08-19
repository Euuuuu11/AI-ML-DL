# 파일생성 

feil_name = ["01_iris","02_cancer","03__diabets","04_wine","05_fetch_covtype",\
    "06_digits","07_kaggle_titanic","08_boston.py","09_california","10_ddarung",\
        "11_kaggle_bike","12_kaggle_house"]

# feil_name = ["01_iris.py"]

for feil_name in feil_name:
    with open(f"./ml/ml52_PolynomialFeatures{feil_name}.py","w") as file:
        file.write("")