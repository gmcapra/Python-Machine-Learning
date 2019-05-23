#PRACTICE WITH DECISION TREES AND RANDOM FORESTS IN PYTHON
#Gianluca Capraro
#Created: May 2019
"""
PRACTICE PROJECT DESCRIPTION
----------------------------------------------------------------------------
Use sample medical data to make predictions regarding whether
or not a patient will be diagnosed with Kyphosis.

This practice will teach how to make use of Decision Trees and RandomForests
to get these predictions.

We will evaluate the performance of a Decision Tree model and compare this
to that of a Random Forest model.
----------------------------------------------------------------------------
"""
#import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read in data from provided kyphosis data file, print the head
df = pd.read_csv('kyphosis.csv')
print('Kyphosis Data Head:')
print('\n')
print(df.head())
print('\n')


#Create a pairplot to visualize relationships in our data
print('Showing Pairplot of Data...')
sns.pairplot(df,hue='Kyphosis',palette='Set1',diag_kind='hist')
plt.show()
print('\n')

#Perform Train Test Split to obtain testing and training data
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#import decision tree classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

#create an instance of the decision tree classifier
dtree = DecisionTreeClassifier()

#fit the model to our training data set
dtree.fit(X_train,y_train)

#make predictions based on our model
predictions = dtree.predict(X_test)

#evaluate the predictions we have obtained using a classification report and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report for Decision Tree Predictions Regarding Kyphosis:')
print(classification_report(y_test,predictions))
print('\n')
print('Confusion Matrix for Decision Tree Predictions Regarding Kyphosis:')
print(confusion_matrix(y_test,predictions))
print('\n')

"""
From our reports, we can see that our current decision tree was 
pretty accurate in its attempts to predict kyphosis in patients. 

Now, let's examine how our results will change after using the 
Random Forest Classifier.
"""

#import Random Forest Classifier from SKLearn Library
from sklearn.ensemble import RandomForestClassifier

#create the random forest classifier model instance (estimator 200 for now)
rfc = RandomForestClassifier(n_estimators=200)

#fit our training data to the random forest model
rfc.fit(X_train,y_train)

#get predictions from new model
rfc_predictions = rfc.predict(X_test)

#obtain new reports for the random forest model
print('Classification Report for Random Forest Predictions Regarding Kyphosis:')
print(classification_report(y_test,rfc_predictions))
print('\n')
print('Confusion Matrix for Random Forest Predictions Regarding Kyphosis:')
print(confusion_matrix(y_test,rfc_predictions))
print('\n')

"""
Examining the accuracy reports from both of our models, we can see that each
time we make predictions using this script, the RandomForest method performs
slightly better than the single Decision Tree model.

Its important to note that as a data set gets larger and larger, the Random Forest
will almost always outperform a single decision tree. However, as the data set in 
this example is somewhat small, we do not see as big of a difference in accuracy.
"""




