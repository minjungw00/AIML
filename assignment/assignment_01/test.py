import numpy as np
from sklearn.linear_model import LinearRegression
import os

dir = os.getcwd() + "\\assignment\\assignment_01\\"

# Load the data.
data = np.loadtxt(dir + "train_data.csv",
                  delimiter=",",
                  skiprows=1)

# 데이터를 x와 y로 분할
x = data[:, 0]
y = data[:, 1]

# 선형 회귀 모델 생성
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)  # 모델 학습

# 새로운 데이터 생성
new_x = np.random.uniform(71, 101, 30)  # 71~100 사이의 랜덤한 x 값 30개 생성
new_y_pred = model.predict(new_x.reshape(-1, 1))  # 모델을 사용하여 y 값 예측

# 결과 출력
new_data = list(zip(new_x, new_y_pred))
print(new_data)