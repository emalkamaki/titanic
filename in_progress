"""
Created on Fri Jul  7 14:53:57 2017

It is your job to predict if a passenger survived the sinking of the Titanic or not. 
For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived 
variable.

The training set should be used to build your machine learning models. 
For the training set, we provide the outcome (also known as the “ground truth”) 
for each passenger. Your model will be based on “features” like passengers’ gender and 
class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. 
For the test set, we do not provide the ground truth for each passenger. 
It is your job to predict these outcomes. For each passenger in the test set, use 
the model you trained to predict whether or not they survived the sinking of the Titanic.

@author: Ensio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

read = pd.read_csv('train.csv', header=0) # read in the file 
data = pd.DataFrame(read)

data.columns # 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
              # 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

data.set_index('PassengerId') # Use PassengerId as the index

# possible decisive factors (attributes of interest) Sex, Age, pclass (1st,2nd,3rd class)

# First determine, how likely it is to survive based on your sex?

data.info()   # 891 rows, 12 columns, Age 714 and Cabin 204 data missing. 
data['Age'].describe()

data['Age'] = data.Age.fillna(data.Age.median())
data.info()

survived_sex = data[data.Survived == 1]['Sex'].value_counts()
dead_sex = data[data.Survived == 0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True)
plt.show()
plt.close()

survived_age = data[data.Survived == 1]['Age']
dead_age = data[data.Survived == 0]['Age']
plt.hist([survived_age, dead_age], bins=30, stacked=True, label=['Survived', 'Dead'])
plt.xlabel = 'Age'
plt.ylabel = 'Number of Passengers'
plt.legend()
plt.show()

survived_class = data[data.Survived == 1]['Pclass']
dead_class = data[data.Survived == 0]['Pclass']
plt.hist([survived_class, dead_class], bins=3, rwidth=.8, stacked=True, label=['Dead', 'Survived'])
plt.legend()
plt.show()


