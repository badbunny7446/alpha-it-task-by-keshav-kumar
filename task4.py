import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("Housing.csv")

# print(data.head())
# print(data.shape)
# print(data.info())  
# print(data.isnull().sum())
# print(data.descri be())
# print(data.columns)



median_price = data['price'].median()
data['price'] = data['price'].apply(lambda x: 1 if x >= median_price else 0)
print(data['price'].value_counts())


# encoding 
yes_no_mapping = {'yes': 1, 'no': 0}
data['mainroad'] = data['mainroad'].map(yes_no_mapping)
data['guestroom'] = data['guestroom'].map(yes_no_mapping)   
data['basement'] = data['basement'].map(yes_no_mapping)
data['hotwaterheating'] = data['hotwaterheating'].map(yes_no_mapping)
data['airconditioning'] = data['airconditioning'].map(yes_no_mapping)
data['prefarea'] = data['prefarea'].map(yes_no_mapping)

print(data['furnishingstatus'].value_counts())

data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})



data['mainroad']=data['mainroad'].astype(int)
data['guestroom']=data['guestroom'].astype(int)
data['basement']=data['basement'].astype(int)
data['hotwaterheating']=data['hotwaterheating'].astype(int)
data['airconditioning']=data['airconditioning'].astype(int)
data['prefarea']=data['prefarea'].astype(int)
data['furnishingstatus']=data['furnishingstatus'].astype(int)

print(data.head())
print(data.info())

x=data.drop(columns=['price'],axis=1)
y=data['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# model call
lg=LogisticRegression()
lg.fit(x_train,y_train)

#prediction
y_pred=lg.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
cr=classification_report(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)

print(f"logistic regression accuracy: {accuracy*100:.2f}%")
print(f"logistic regression classification report:{cr}")
print(f"logistic regression confusion matrix:{cm}")

#visualization

plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.show()


# test
new_house = np.array([[ 
    500,   # area
    4,      # bedrooms
    3,      # bathrooms
    2,      # stories
    0,      # mainroad (yes)
    0,      # guestroom (yes)
    0,      # basement (yes)
    0,      # hotwaterheating (no)
    0,      # airconditioning (yes)
    1,      # parking
    1,      # prefarea (yes)
    2       # furnishingstatus (furnished)
]])
new_house_scaled = sc.transform(new_house)
prediction = lg.predict(new_house_scaled)
probability = lg.predict_proba(new_house_scaled)
print("Prediction:", prediction)
print("Probability:", probability)     