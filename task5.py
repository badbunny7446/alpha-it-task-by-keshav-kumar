import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = load_breast_cancer()
df=pd.DataFrame(data.data, columns=data.feature_names)
df['target']=data.target

#eda
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.columns)
# print(df.describe())
# print(df.isnull().sum())

X=df.drop('target', axis=1)
Y=df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

model_Knn=KNeighborsClassifier()
model_Knn.fit(X_train, Y_train)
Y_pre_Knn=model_Knn.predict(X_test)
acc_Knn=accuracy_score(Y_test,Y_pre_Knn)
print(f" knn Accuracy: {acc_Knn*100:.2f}%")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pre_Knn)}")
print(f"Classification Report:{classification_report(Y_test, Y_pre_Knn)}")

model_Lr=LogisticRegression()
model_Lr.fit(X_train, Y_train)
Y_pre_Lr=model_Lr.predict(X_test)
acc_Lr=accuracy_score(Y_test,Y_pre_Lr)
print(f"logistic regression Accuracy: {acc_Lr*100:.2f}%")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pre_Lr)}")
print(f"Classification Report:{classification_report(Y_test, Y_pre_Lr)}")

model_Lr_reg=LogisticRegression(penalty='l2', solver='lbfgs', C=1.0)
model_Lr_reg.fit(X_train, Y_train)
Y_pre_Lr_reg=model_Lr_reg.predict(X_test)
acc_Lr_reg=accuracy_score(Y_test,Y_pre_Lr_reg)
print(f"lr with regAccuracy: {acc_Lr_reg*100:.2f}%")
print(f"reg coefficient:{model_Lr_reg.coef_}")
print(f"reg intercept:{model_Lr_reg.intercept_}")

model_Lr_laso=LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model_Lr_laso.fit(X_train, Y_train)
Y_pre_Lr_laso=model_Lr_laso.predict(X_test)
acc_Lr_laso=accuracy_score(Y_test,Y_pre_Lr_laso)
print(f"lr with lasso Accuracy: {acc_Lr_laso*100:.2f}%")
print(f"laso coefficient:{model_Lr_laso.coef_}")
print(f"laso intercept:{model_Lr_laso.intercept_}")

print(f"l1 weights:{model_Lr_laso.coef_}")
print(f"l2 weights:{model_Lr_reg.coef_}")

model_lin=LinearRegression()
model_lin.fit(X_train, Y_train)
Y_pre_lin=model_lin.predict(X_test)
acc_lin=accuracy_score(Y_test,Y_pre_lin.round())    
print(f"Linear Regression Accuracy: {acc_lin*100:.2f}%")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pre_lin.round())}")
print(f"Classification Report:{classification_report(Y_test, Y_pre_lin.round())}")

model_dt=DecisionTreeClassifier()
model_dt.fit(X_train, Y_train)
Y_pre_dt=model_dt.predict(X_test)
acc_dt=accuracy_score(Y_test,Y_pre_dt)
print(f"dt Accuracy: {acc_dt*100:.2f}%")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pre_dt)}")
print(f"Classification Report:{classification_report(Y_test, Y_pre_dt)}")

model_bag=BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
model_bag.fit(X_train, Y_train)
Y_pre_bag=model_bag.predict(X_test)
acc_bag=accuracy_score(Y_test,Y_pre_bag)
print(f"bagging Accuracy: {acc_bag*100:.2f}%")

model_rf=RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, Y_train)
Y_pre_rf=model_rf.predict(X_test)
acc_rf=accuracy_score(Y_test,Y_pre_rf)
print(f"rf Accuracy: {acc_rf*100:.2f}%")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pre_rf)}")
print(f"Classification Report:{classification_report(Y_test, Y_pre_rf)}")


model_svm_linear = SVC(kernel='linear')
model_svm_linear.fit(X_train, Y_train)
Y_pre_svm_linear = model_svm_linear.predict(X_test)
acc_svm_linear = accuracy_score(Y_test, Y_pre_svm_linear)
print(f"Linear SVM Accuracy: {acc_svm_linear*100:.2f}%")


model_svm_rbf = SVC(kernel='rbf')
model_svm_rbf.fit(X_train, Y_train)
Y_pre_svm_rbf = model_svm_rbf.predict(X_test)
acc_svm_rbf = accuracy_score(Y_test, Y_pre_svm_rbf)
print(f"RBF SVM Accuracy: {acc_svm_rbf*100:.2f}%")

    
model_nb=GaussianNB()
model_nb.fit(X_train, Y_train)
Y_pre_nb=model_nb.predict(X_test)
acc_nb=accuracy_score(Y_test,Y_pre_nb)
print(f"Naive Bayes Accuracy: {acc_nb*100:.2f}%")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pre_nb)}")    
print(f"Classification Report:{classification_report(Y_test, Y_pre_nb)}")

model_lda=LinearDiscriminantAnalysis()
model_lda.fit(X_train, Y_train)
Y_pre_lda=model_lda.predict(X_test) 
acc_lda=accuracy_score(Y_test,Y_pre_lda)
print(f"lda Accuracy: {acc_lda*100:.2f}%")
print(f"Confusion Matrix:{confusion_matrix(Y_test, Y_pre_lda)}")
print(f"Classification Report:{classification_report(Y_test, Y_pre_lda)}")


# acc compare
model_acc={
    'KNN':acc_Knn,
    'Logistic Regression':acc_Lr,
    'Logistic Regression with L2':acc_Lr_reg,
    'Logistic Regression with L1':acc_Lr_laso,
    'Linear Regression':acc_lin,
    'Decision Tree':acc_dt,
    'Bagging':acc_bag,
    'Random Forest':acc_rf,
    'Linear SVM':acc_svm_linear,
    'RBF SVM':acc_svm_rbf,
    'Naive Bayes':acc_nb,
    'LDA':acc_lda

}
for name, acc in model_acc.items():
    print(f"Accuracy of {name}: {acc*100:.2f}%")
print("\n")

check= {
    'KNN': (model_Knn.score(X_train, Y_train), model_Knn.score(X_test, Y_test)),
    'Logistic Regression': (model_Lr.score(X_train, Y_train), model_Lr.score(X_test, Y_test)),      
    'Logistic Regression with L2 ': (model_Lr_reg.score(X_train, Y_train), model_Lr_reg.score(X_test, Y_test)),
    'Logistic Regression with L1 ': (model_Lr_laso.score(X_train, Y_train), model_Lr_laso.score(X_test, Y_test)),
    'Linear Regression': (accuracy_score(Y_train, model_lin.predict(X_train).round()), accuracy_score(Y_test, model_lin.predict(X_test).round())),
    'Decision Tree': (model_dt.score(X_train, Y_train), model_dt.score(X_test, Y_test)),
    'Bagging': (model_bag.score(X_train, Y_train), model_bag.score(X_test, Y_test)),
    'Random Forest': (model_rf.score(X_train, Y_train), model_rf.score(X_test, Y_test)),
    'Linear SVM': (model_svm_linear.score(X_train, Y_train), model_svm_linear.score(X_test, Y_test)),
    'RBF SVM': (model_svm_rbf.score(X_train, Y_train), model_svm_rbf.score(X_test, Y_test)),
    'Naive Bayes': (model_nb.score(X_train, Y_train), model_nb.score(X_test, Y_test)),
    'LDA': (model_lda.score(X_train, Y_train), model_lda.score(X_test, Y_test))
}
for name, scores in check.items():
    print(f"{name} = Train Score: {scores[0]*100:.2f}%, Test Score: {scores[1]*100:.2f}%")

models={
    'KNN': model_Knn,
    'Logistic Regression': model_Lr,
    'Logistic Regression with L2 ': model_Lr_reg,
    'Logistic Regression with L1 ': model_Lr_laso,
    'Decision Tree': model_dt,
    'Bagging': model_bag,
    'Random Forest': model_rf,
    'Linear SVM': model_svm_linear,
    'RBF SVM': model_svm_rbf,
    'Naive Bayes': model_nb,
    'LDA': model_lda
}
for name, model in models.items():
    train_acc = model.score(X_train, Y_train)
    test_acc = model.score(X_test, Y_test)

    diff = train_acc - test_acc

    if diff > 0.05:
        print(f"{name}: High Variance (Overfitting)")
    elif train_acc < 0.90 and test_acc < 0.90:
        print(f"{name}: High Bias (Underfitting)")
    else:
        print(f"{name}: Balanced Model")
