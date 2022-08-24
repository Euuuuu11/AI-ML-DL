# 파일생성 

feil_name = ["09_california","10_ddarung",\
        "11_kaggle_bike","12_kaggle_house"]

# feil_name = ["01_iris.py"]

for feil_name in feil_name:
    with open(f"./tf114/tf14_{feil_name}.py","w") as file:
        file.write("")