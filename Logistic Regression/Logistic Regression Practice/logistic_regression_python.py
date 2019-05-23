#Logistic Regression Python Practice

"""
Use Titanic Data Set to predict whether a passenger survived or deceased.
Titanic: Machine Learning from Disaster (Kaggle Competition Challenge)
Data sets are already downloaded as .csv files, these are:

titanic_test.csv
titanic_train.csv
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#Read in titanic training data to a dataframe called train
train = pd.read_csv('titanic_train.csv')
print('\n')
print('Titanic Training Set Header:')
print(train.head())
print('\n')

#Print the columns of titanic training dataset
print('Train DataFrame Columns: ')
print(train.columns)
print('\n')

#EXPLORATORY DATA ANALYSIS
#--------------------------------------------------------------------#
#Create a heatmap of dataframe to identify any null values
#yellow line indicates "True" or a null value at location
#From this we can see we are missing a lot of age and cabin data 
print('Showing null value heatmap...')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
print('\n')

#use countplot to see number of people who survived and who did not
#see breakdown by male vs female
print('Showing count of survived vs. deceased by gender...')
sns.countplot(x='Survived',data=train,hue='Sex')
plt.show()
print('\n')

#use countplot to see number of people who survived and who did not
#see breakdown by passenger class
print('Showing count of survived vs. deceased by class...')
sns.countplot(x='Survived',data=train,hue='Pclass')
plt.show()
print('\n')

#use distplot to show age distribution of passengers
#use dropna() to remove null values
print('Showing age distribution...')
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
plt.show()
print('\n')

#explore sibsp column to determine number of siblings or spouses per passenger
#we see that most people didn't come with any siblings or spouses
print('Showing count of Siblings and Spouses...')
sns.countplot(x='SibSp',data=train)
plt.show()
print('\n')

#create a histogram of fare paid per passenger
print('Showing distribution of fares paid...')
sns.distplot(train['Fare'],kde=False,bins=40)
plt.show()
print('\n')



#DATA CLEANING FOR USE IN MACHINE LEARNING ALGORITHMS
#--------------------------------------------------------------------#
#we have many missing age columns, lets see if we can learn more about them
#create a box plot to examine age data by passenger class
#here we see that older passengers are in 1st class, and younger in 3rd
"""
print('Showing box plot for ages based on passenger class...')
sns.boxplot(x='Pclass',y='Age',data=train)
plt.show()
print('\n')
"""

#create a function to assign age if age is null
#takes in columns as parameter
#returns approximate age
def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]

	if pd.isnull(Age):
		if Pclass == 1:
			return 37
		elif Pclass == 2:
			return 29
		else:
			return 24
	else:
		return Age


#use function to apply new ages where ages are null in our data
#axis = 1 so that applied to columns
#age imputation
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
#drop cabin column as there is not enough info to use
train.drop('Cabin',axis=1,inplace=True)
#drop any additional missing values
train.dropna(inplace=True)

#now check heatmap for any null values, and there aren't any!
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title('Heatmap with no null values')
plt.show()
print('\n')

#convert categorical variables into dummy (or indicator) variables
"""
watch for perfect predictors in columns
for male vs. female, this will be multi-colinear
messes up algorithm because columns will be perfect predictors of other columns
to avoid this, include drop_first=True
"""
#sex column is just a 0 or 1 if passenger was male or not
sex = pd.get_dummies(train['Sex'],drop_first=True)
#embark is either Q or S after dropping C for redundancy
embark = pd.get_dummies(train['Embarked'],drop_first=True)
#concatenate database to include dummy variables
train = pd.concat([train,sex,embark],axis=1)

"""
Now we know several columns we no longer need, or we have already
mapped to dummy variables. Let's get rid of those so we can use
our data set for machine learning algorithms. Also notice how PassengerID
is essentially just the index, so we can drop it.

Finally, notice that Pclass is really just a categorical variable. For this example,
we will leave it as either 1,2,3, however we could have also used pd.get_dummies
for this column. We will later see how a machine learning algorithm reacts
to either of these cases.
"""
train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
print('Final Machine Learning Dataset Head:')
print('\n')
print(train.head())
print('\n')


#CREATE TRAIN AND BUILD MACHINE LEARNING CLASSIFICATION MODEL
#USING LOGISTIC REGRESSION
#--------------------------------------------------------------------#
"""
For purposes of this exercise. We will treat train as the entirety of our data
and build our test data off of this. We do have a test.csv file, however
to avoid cleaning this as well, we will go forward with the train data set only.
"""

#Define features as X and target as y
X = train.drop('Survived',axis=1)
y = train['Survived']

#import sklearn libraries needed
from sklearn.model_selection import train_test_split

#perform test train split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#import linear model family Logistic Regression
from sklearn.linear_model import LogisticRegression

#create an instance of the Logistic Model
logmodel = LogisticRegression()

#fit train data to model
logmodel.fit(X_train,y_train)

#get predictions from out model using X test data
predictions = logmodel.predict(X_test)

#import classification report from sklearn metrics
from sklearn.metrics import classification_report

#print classification report
print('Classification Report:')
print(classification_report(y_test,predictions))
print('\n')

#import confusion matric
from sklearn.metrics import confusion_matrix

#print confusion matric
print('Confusion Matrix:')
print(confusion_matrix(y_test,predictions))
print('\n')

"""
How can we improve accuracy?
We could add more features, maybe the ticket, cabin location, etc. 
could indicate an increased likelihood of death or survival.
"""






