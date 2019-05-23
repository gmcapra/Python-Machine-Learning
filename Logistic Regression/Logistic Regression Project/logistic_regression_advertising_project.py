#Logistic Regression Project
"""
Using the advertising.csv file provided. We will create a model
that will predict whether or not a user will click on an ad
based on the features of that user.
"""

#import necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#------------------------------------------------------------------------#
#READ IN DATA, PERFORM SOME EXPLORATORY DATA ANALYSIS AND PLOTTING
#------------------------------------------------------------------------#

#read in advertising.csv file, create dataframe called ad_data
ad_data = pd.read_csv('advertising.csv')
print('Advertising Data Head:')
print(ad_data.head())
print('\n')

#print out ad_data info
print('Advertising Data Info:')
print(ad_data.info())
print('\n')

#print out ad_data description
print('Advertising Data Description:')
print(ad_data.describe())
print('\n')

#Create a histogram of the users' ages
print("Showing Distribution of Users' Age...")
sns.distplot(ad_data['Age'],kde=False,bins=30)
plt.show()
print('\n')

#Create a jointplot showing Area Income vs. Age
print("Showing Area Income vs. Age...")
sns.jointplot(x='Age',y='Area Income',data=ad_data)
plt.show()
print('\n')

#Create a jointplot showing the kde distributions of Daily time spent on site vs. Age
print("Showing KDE Distributions of Daily Time Spent on Site vs. Age...")
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde')
plt.show()
print('\n')

#Create a jointplot of Daily Time Spent on Site vs. Daily Internet Usage
print("Showing Daily Time Spent on Site vs. Daily Internet Usage...")
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)
plt.show()
print('\n')

#Create a pairplot with further segmentation based on if the user clicked on ad
print("Showing pairplot segmented by: User Clicked on Ad = True...")
sns.pairplot(ad_data,hue='Clicked on Ad')
plt.show()
print('\n')

#------------------------------------------------------------------------#
#PERFORM TRAIN TEST SPLIT, TRAIN MODEL
#------------------------------------------------------------------------#

#split data into training set and testing set
"""
Only grab data you are interested in using and will be useful for our model.
In this case, we will grab any numerical categories that describe users.
We will leave out columns holding strings as those would need dummy variables
to be created. Male is either 0 or 1, so it fits.
"""
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']

#import sklearn library to perform split
from sklearn.model_selection import train_test_split

#perform the split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#train and fit a logistic regression model on training set
#import logistic regression model
from sklearn.linear_model import LogisticRegression

#create an instance of the model, fit our training data to the model
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#get predictions from our test data
predictions = logmodel.predict(X_test)

#import classification report from sklearn metrics
from sklearn.metrics import classification_report

#print classification report
print('\n')
print('Classification Report:')
print(classification_report(y_test,predictions))
print('\n')

#import confusion matrix
from sklearn.metrics import confusion_matrix

#print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test,predictions))
print('\n')

"""
The resulting reports show a very high accuracy in predicting whether or not
a user will click on an ad. If we want to refine this further, what actions can be taken?
"""





