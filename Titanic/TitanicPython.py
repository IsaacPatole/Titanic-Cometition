#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# let's read the daatset
train   = pd.read_csv(r"C:\Users\MY\Desktop\Titanic\train.csv")
test    = pd.read_csv(r"C:\Users\MY\Desktop\Titanic\test.csv")
df_data = [train,test]

#let's add parch and sibsp column to see what impacet does it have on survival.
for df in df_data:
    df['Family']= df['Parch']+df['SibSp']
    df['Family'].loc[df['Family'] == 0] = 0
    df['Family'].loc[df['Family'] == 1] = 1
    df['Family'].loc[df['Family'] == 2] = 2
    df['Family'].loc[(df['Family'] > 2)&(df['Family'] <= 4)] = 3
    df['Family'].loc[df['Family'] > 4] = 4
    df['Family'].describe()
    
print("What Family column can tell us?")
print(train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean())

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
for df in df_data:
    df["Title"] = pd.Series(dataset_title)
    df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df.Title = df.Title.replace('Mlle', 'Miss')
    df.Title = df.Title.replace('Ms', 'Miss')
    df.Title = df.Title.replace('Mme', 'Mrs')
 # Fill in missing data with a placeholder.
    df.Title = df.Title.fillna('Missing')
    
#lets see the average age corresponding to each title
    df.groupby('Title').Age.mean()
#using the same thing again to see the count+mean
    df.groupby('Title').Age.agg(['count','mean'])
#lets fill missing na values in a Age columnby average Age corresponding to each title


    
    
print("Which titles survived?")
print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5, "Missing": 0}
for df in df_data:
    df['Title'] = df['Title'].map(title_mapping)
    df["Title"] = df["Title"].astype(int)
#dropping the columns(you can try using other columns as well)
    df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'] , axis =1 , inplace=True)

#fill Nan Values by using 'ffill'.  which fill's NaN values by its previouse value
for df in df_data:
    df['Embarked'] = df[['Embarked']].fillna(method='ffill')
    df['Fare'] = df[['Fare']].fillna(method='ffill')

#converting into numeric values
    df['Sex']= df[['Sex']].replace(['male','female'],[0,1])
    df['Embarked']= df['Embarked'].replace(['C','Q','S'],[0,1,2])
    
print("What genderer of passenger survived the most?")
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

print("What Embarked can tell us?")
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
    
    
#we dont need passengerId and survived column in our independent variables so we are ignoring them
X       = train.iloc[:,1:8].values
#using dependent variable survived
y       = train.iloc[:,0].values

#We dont need passengerId column in our independent variables so we are ignoring it
test    = test.iloc[:,0:7].values

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

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
print(classifier.score(X_test,y_test))

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 65, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
print(classifier.score(X_test,y_test))

# Fitting Gradient boosting Classification to the Training set
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(max_depth=3,
                                        learning_rate=0.025,
                                        n_estimators=400, 
                                        subsample=0.2,
                                        min_samples_leaf=8,
                                        verbose=True,
                                      ).fit(X_train,y_train)


print(classifier.score(X_test,y_test))

# Fitting XGBoost Classification to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate=0.05,
                           n_estimators=450,
                           max_depth=3,
                           silent = False,
                           min_child_weight=1,
                           subsample=0.65,
                           gamma=0.38,
                           colsample_bytree=0.4,
                           reg_alpha=0.05,
                           seed=1)

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

# Using Grid Search to tune parameters of our classifiers
from sklearn.model_selection import GridSearchCV
parameters = [{}]
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
