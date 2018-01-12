
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn import model_selection

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


# In[15]:


train.info()


# In[16]:


test.info()


# In[5]:


train[['Sex', 'Survived']].groupby('Sex').agg([np.size, np.mean])


# In[6]:


train[['Pclass', 'Survived']].groupby('Pclass').agg([np.size, np.mean])


# In[7]:


train[['Embarked', 'Survived']].groupby('Embarked').agg([np.size, np.mean])


# In[13]:


train.groupby('Survived').Age.describe()


# In[14]:


train.groupby('Survived').Fare.describe()


# In[23]:


sns.pairplot(train[['Survived', 'Age', 'Sex', 'Pclass', 'Embarked', 'Fare', 'SibSp', 'Parch']].dropna(), hue='Survived')


# In[17]:


from sklearn.preprocessing import OneHotEncoder

def data_preprocess(df):
    df = df.drop(columns=['Cabin', 'Ticket', 'Name'])
    #df = df.set_index('PassengerId')
    mean_age = df.Age.mean()
    df['Age'] = df.Age.fillna(mean_age)
    df['Embarked'] = df.Embarked.fillna('S')
    mean_fare = df.Fare.mean()
    df['Fare'] = df.Fare.fillna(mean_fare)
    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype('int')
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype('int')
    #df.loc[df.Fare == 0, 'Fare'] = 1
    #df['Fare'] = df.Fare.map(np.log2)
    enc = OneHotEncoder()
    f = enc.fit_transform(df[['Sex', 'Embarked']]).toarray()
    ddf = pd.DataFrame(f)
    df = pd.merge(df, ddf, left_index=True, right_index=True)
    df = df.drop(columns=['Sex', 'Embarked'])
    df = df.set_index('PassengerId')
    return df


# In[23]:


processed = data_preprocess(train)

dead = processed.loc[processed.Survived == 0]
alive = processed.loc[processed.Survived == 1]

dead_train = dead.sample(frac=0.8)
dead_test = dead.drop(dead_train.index)

alive_train = alive.sample(frac=0.8)
alive_test = alive.drop(alive_train.index)

my_train = pd.concat([dead_train, alive_train])
my_test = pd.concat([dead_test, alive_test])


# In[19]:


corrmat = processed.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.9, square=True, annot=True)


# In[24]:


from sklearn.ensemble import GradientBoostingClassifier
my_train_y = my_train.Survived
my_train_x = my_train.drop(columns=['Survived'])
my_test_y = my_test.Survived
my_test_x = my_test.drop(columns=['Survived'])

clf = GradientBoostingClassifier(max_depth=3, n_estimators=50, verbose=True).fit(my_train_x, my_train_y)
print(clf.score(my_train_x, my_train_y))
print(clf.score(my_test_x, my_test_y))


# In[21]:


test_p = data_preprocess(test)
result = clf.predict(test_p)
f = test_p.reset_index()
f['Survived'] = result
f = f[['PassengerId', 'Survived']]
f.to_csv('result.csv', index=False)


# In[42]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(7, 3), max_iter=10000).fit(my_train_x, my_train_y)
print(clf.score(my_train_x, my_train_y))
print(clf.score(my_test_x, my_test_y))

