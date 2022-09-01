import autokeras as ak
print(ak.__version__) #1,0 20
import keras
import time
#데이터
(x_train , y_train), (x_test , y_test) =\
    keras.datasets.mnist.load_data()
    
print(x_train.shape ,y_train.shape,x_test.shape,y_test.shape) #(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)


model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2   
)

# (class) ImageClassifier(num_classes: int | None = None, multi_label: bool = False, loss: LossType = None, 
#                         metrics: MetricsType | None = None, project_name: str = "image_classifier",
#                         max_trials: int = 100, directory: str | Path | None = None, objective: str = "val_loss", tuner: str 
#                         | Type[AutoTuner] = None, overwrite: bool = False, seed: int | None = None,
#                         max_model_size: int | None = None, **kwargs: Any)


#3.컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs=5)
end = time.time() 
#4.평가, 예측

y_pred = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print('결과:',results)
# print('걸린시간:',round(end-start,4))

# 결과: [0.03779282420873642, 0.9873999953269958]
# 걸린시간: 3376.88