import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#reading the daatset
train   = pd.read_csv(r"C:\Users\MY\Desktop\Titanic\train.csv")

train['Sex']= train[['Sex']].replace(['male','female'],[0,1])
train['Embarked']= train['Embarked'].replace(['C','Q','S'],[0,1,2])

train.drop(['Name','SibSp','Parch','Ticket','Cabin'] , axis =1 , inplace=True)

X       = train.iloc[:,2:7].values
y       = train.iloc[:,1].values

test    = pd.read_csv(r"C:\Users\MY\Desktop\Titanic\test.csv")
test    = train.iloc[:,1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

imputer2 = imputer.fit(X[:,4:5 ])
X[:,4:5 ] = imputer2.transform(X[:, 4:5])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_Xtrain = StandardScaler()
X_train = sc_Xtrain.fit_transform(X_train)
sc_Xtest = StandardScaler()
X_test = sc_Xtest.fit_transform(X_test)



#Classifier 
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

y_result = classifier.predict(test)
Passengerid = []
for i in range(892,1310):
    Passengerid.append(i)
    
Passengerid
Result = pd.DataFrame({'PassengerId':Passengerid, 'Survived':y_result})

Result.to_csv(r'C:\Users\MY\Desktop\Titanic\result.csv', index = False)


