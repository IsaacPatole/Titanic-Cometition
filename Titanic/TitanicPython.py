#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# let's read the daatset
train   = pd.read_csv(r"C:\Users\MY\Desktop\Titanic\train.csv")

#let's add parch and sibsp column to know if the passenger is travelling alone or not
train['Family']= train.iloc[:, 6:8].sum(axis=1)
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0
train.describe()
#dropping the columns(you can try using other columns as well)
train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'] , axis =1 , inplace=True)

#fill Nan Values by using 'ffill'.  which fill's NaN values by its previouse value
train['Embarked'] = train[['Embarked']].fillna(method='ffill')
train['Fare'] = train[['Fare']].fillna(method='ffill')

#converting into numeric values
train['Sex']= train[['Sex']].replace(['male','female'],[0,1])
train['Embarked']= train['Embarked'].replace(['C','Q','S'],[0,1,2])



#we dont need passengerId and survived column in our independent variables so we are ignoring them
X       = train.iloc[:,1:7].values

#using dependent variable survived
y       = train.iloc[:,0].values

#let's do the same thing for preprocessing of test set
test    = pd.read_csv(r"C:\Users\MY\Desktop\Titanic\test.csv")
test['Family'] = test.iloc[:, 5:7].sum(axis=1) 
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'] , axis =1 , inplace=True)

test['Embarked'] = test[['Embarked']].fillna(method='ffill')
test['Fare'] = test[['Fare']].fillna(method='ffill')

test['Sex']= test[['Sex']].replace(['male','female'],[0,1])
test['Embarked']= test['Embarked'].replace(['C','Q','S'],[0,1,2])



#We dont need passengerId column in our independent variables so we are ignoring it
test    = test.iloc[:,0:6].values

#lets insert missing values in Age column with median by using Imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
#train set
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
#test set for final result
imputer2 = imputer.fit(test[:,2:3 ])
test[:,2:3 ] = imputer2.transform(test[:, 2:3])





# Splitting the dataset into the Training set and Test set 20% into test set and 80% into training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature Scaling to avoid despersion in the data so that we get values between 0 and 1 
from sklearn.preprocessing import StandardScaler
sc_Xtrain = StandardScaler()
X_train = sc_Xtrain.fit_transform(X_train)
sc_Xtest = StandardScaler()
X_test = sc_Xtest.fit_transform(X_test)
sc_test = StandardScaler()
test = sc_test.fit_transform(test)

#Use any one of the classifier and check the results
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate=0.05,
                           n_estimators=500,
                           max_depth=3,
                           silent = False,
                           min_child_weight=1,
                           subsample=0.65,
                           gamma=0.38,
                           colsample_bytree=0.4,
                           reg_alpha=0.05,
                           seed=1)

classifier.fit(X_train, y_train)

# Fitting Gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(max_depth=3,
                                        learning_rate=0.025,
                                        n_estimators=400, 
                                        subsample=0.2,
                                        min_samples_leaf=8,
                                        verbose=True,
                                      ).fit(X_train,y_train)


print(classifier.score(X_test, y_test))

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

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[100,200,300,400,500,600,700] }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Now let's insert result into result.csv file by using dictionary(Key, Value pair) 
y_result = classifier.predict(test)
Passengerid = []
for i in range(892,1310):
    Passengerid.append(i)
    
Passengerid

Result = pd.DataFrame({'PassengerId':Passengerid, 'Survived':y_result})

Result.to_csv(r'C:\Users\MY\Desktop\Titanic\result.csv', index = False)



