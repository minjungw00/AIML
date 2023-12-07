import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from classification import Model01

from sklearn.neural_network import MLPClassifier

# Load the dataset.
df_train = pd.read_csv("C:/Users/min/Desktop/Artech/3_2/AIML/assignment/assignment_02/data_train.csv", encoding="utf-8-sig")
df_valid = pd.read_csv("C:/Users/min/Desktop/Artech/3_2/AIML/assignment/assignment_02/data_valid.csv", encoding="utf-8-sig")

def train_and_cal_metrics(model):
    # Train and predict.
    model.train(df_train)

    y_pred = model.predict(df_train)
        
    # Calculate the metrics.
    y_train = df_train["성별"]
    acc = accuracy_score(y_train, y_pred)
    rocauc = roc_auc_score(y_train, y_pred)
    prauc = average_precision_score(y_train, y_pred)

    '''
    print("[Model#01] Acc: %f"%(acc))
    print("[Model#01] ROC-AUC: %f"%(rocauc))
    print("[Model#01] PR-AUC: %f"%(prauc))
    print()
    '''

    return acc + rocauc + prauc, acc, rocauc, prauc



# Create the classification models.
model = Model01()
arr = []

'''
for col in df_train.columns:
   print(f"'{col}',")


for i in range(0, 1000):
   temp = []
   model._model = MLPClassifier(random_state=i)
   temp.append(train_and_cal_metrics(model)[0])
   temp.append(i)
   arr.append(temp)
   
arr.sort()
for a in arr:
    a[0], a[1] = a[1], a[0]
    print(a)
'''

import csv

for a in range(0 + 2354, 10000000):

    with open("C:/Users/min/Desktop/Artech/3_2/AIML/assignment/assignment_02/test_data.csv", "a", newline='') as f:
        writer = csv.writer(f)

        for i in range(0, 10):
            model._model = MLPClassifier(random_state= (10 * a) + i)
            metric, acc, rocauc, prauc = train_and_cal_metrics(model)
            data = [(10 * a) + i, metric, acc, rocauc, prauc]
            writer.writerow(data)
            print(data)