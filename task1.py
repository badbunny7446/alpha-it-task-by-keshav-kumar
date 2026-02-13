import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("diabetes.csv")
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())

X=data.drop(columns="Outcome",axis=1)
Y=data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scalaer = StandardScaler()
x_train = scalaer.fit_transform(x_train) 
X_test= scalaer.transform(x_test)

l1_model = LogisticRegression(penalty='l1', solver='liblinear',C=1.0)
l1_model.fit(x_train, y_train)
y_pred = l1_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of L1: {accuracy}")
print(f"L1 Coefficient: {l1_model.coef_}")
print(f"L1 intercept: {l1_model.intercept_}")

l2_model = LogisticRegression(penalty='l2', solver='lbfgs',C=1.0)
l2_model.fit(x_train, y_train)
y_pred_l2 = l2_model.predict(X_test)
accuracy_l2 = accuracy_score(y_test, y_pred_l2)
print(f"Accuracy of L2: {accuracy_l2}") 
print(f"L2 Coefficient: {l2_model.coef_}")
print(f"L2 intercept: {l2_model.intercept_}")

print("Compersision")
print(f"L1 WEIGHT : {l1_model.coef_}")
print(f"L2 WEIGHT : {l2_model.coef_}")
