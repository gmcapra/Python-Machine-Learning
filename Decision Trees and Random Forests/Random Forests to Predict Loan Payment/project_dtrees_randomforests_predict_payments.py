"""
USING PYTHON, DECISION TREES AND RANDOMFOREST MODELS
TO MAKE PREDICTIONS REGARDING LOAN PAYMENTS

Gianluca Capraro
Created: May 2019

PROJECT DESCRIPTION
-----------------------------------------------------------------------------
Using publicly available data from Lending Club between 2007 and 2010,
this script will classify and try to predict if a borrower paid their loan in full.

Column Descriptions can be found at bottom of script.
The goal is to accurately predict the not.fully.paid column based on the features of the data set.

Data can be found within project files.
Data can also be downloaded directly at: https://www.lendingclub.com/info/download-data.action
-----------------------------------------------------------------------------
"""

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read in data from lending club (this code makes use of downloaded and cleaned data)
loans = pd.read_csv('loan_data.csv')
print('Lending Club Data Head:')
print('\n')
print(loans.head())
print('\n')

#view the info and describe methods for loans dataset
print('Lending Club Data Information:')
print(loans.info())
print('\n')
print('Lending Club Data Description:')
print(loans.describe())
print('\n')


"""
PERFORM EXPLORATORY DATA ANALYSIS
-----------------------------------------------------------------------------
To visualize this data, we will use pandas, seaborn, matplotlib, or your library
of choice, to create different plots and examine relationships that exist.
-----------------------------------------------------------------------------
1-Create a Histogram of two FICO distribution plots on top of each other
	Should be segmented by credit.policy value
	Should show FICO score vs. number of people with that score
	Include a legend to indicate category of credit.policy
	Label x axis 'FICO'

2-Create a similar Histogram, except this time segment by 'not.fully.paid' 

3-Use Seaborn to create a countplot showing the counts of loans by purpose
	Segment this data by the 'not.fully.paid' column

4-Observe trend between FICO score and interest rate using a Seaborn jointplot

5-Create Seaborn lmplots() to see if this trend differs based on
	not.fully.paid - whether or not the loan is paid
	credit.policy - whether or not the lending club credit criteria was met
-----------------------------------------------------------------------------
"""
#1
print('Showing Distribution of FICO Scores Segmented by credit.policy...')
plt.figure(figsize=(10,6))
#select data from the 'fico' column of loans, where the column 'credit.policy' is equal to 1
loans[loans['credit.policy'] == 1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='credit.policy = 1')
#select data from the 'fico' column of loans, where the column 'credit.policy' is equal to 0
loans[loans['credit.policy'] == 0]['fico'].hist(alpha=0.5,color='red',bins=30,label='credit.policy = 0')
plt.legend()
plt.xlabel('FICO')
plt.show()
print('\n')

#2
print('Showing Distribution of FICO Scores Segmented by not.fully.paid...')
plt.figure(figsize=(10,6))
#select data from 'fico' columns, where not.fully.paid is 1
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=1,color='black',bins=30,label='not.fully.paid = 1')
#select data from 'fico' columns, where not.fully.paid is 0
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.4,color='green',bins=30,label='not.fully.paid = 0')
plt.legend()
plt.xlabel('FICO')
plt.show()
print('\n')

#3
print('Showing Countplot of Number of Loans by Purpose Segmented by not.fully.paid...')
plt.figure(figsize=(12,6))
sns.countplot(x='purpose', data=loans, hue='not.fully.paid', palette='Set2')
plt.show()
print('\n')

#4
print('Showing Jointplot of FICO score vs. Interest Rate...')
sns.jointplot(x='fico',y='int.rate',data=loans,kind='scatter')
plt.show()
print('\n')

#5
print('Showing Adjacent LMPlots of FICO score vs. Interest Rate, Segmented by not.fully.paid and credit.policy...')
sns.lmplot(x='fico',y='int.rate',data=loans,col='not.fully.paid',hue='credit.policy',palette='Set1')
plt.show()
print('\n')


"""
SET UP DATA FOR RANDOM FOREST CLASSIFICATION MODEL
-----------------------------------------------------------------------------
The 'purpose' column is categorical, we need to transform it
using dummy variables so sklearn can understand those values.
This can be expanded to multiple categorical features if needed.
-----------------------------------------------------------------------------
1-Create a list containing the string 'purpose'

2-Use get_dummies() to create a fixed larger dataframe that has new
	feature columns with dummy variables.

3-Print the information on this new dataframe to verify dummies
	Looking to see that there are no string variables
-----------------------------------------------------------------------------
"""
#1
categorical_features = ['purpose']

#2
final_data = pd.get_dummies(loans,columns=categorical_features,drop_first=True)

#3
print('Showing Info of DataFrame with Dummy Variables created for Categorical Features:')
print(final_data.info())
print('\n')


"""
PERFORM TRAIN TEST SPLIT ON DATA
-----------------------------------------------------------------------------
Split data into training and testing data that will be fit to our model
and used to create & evaluate predictions regarding target categories.
-----------------------------------------------------------------------------
1-Import necessary method from sklearn
2-Split up data into X and y
3-Pass X and y into the train test split method
-----------------------------------------------------------------------------
"""
#1
from sklearn.model_selection import train_test_split

#2
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

#3
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

"""
TRAIN A DECISION TREE MODEL ON OUR DATA
-----------------------------------------------------------------------------
Create and fit our data to a decision tree model, use this model to make
predictions, and evaluate the performance of the model.
-----------------------------------------------------------------------------
1-Import necessary method from sklearn
2-Create instance of a decision tree classifier
3-Fit this decision tree to our data
4-Use this new model to create predictions from our test set
5-Create confusion matrix and classification reports to evaluate
	You will need to import the necessary methods
-----------------------------------------------------------------------------
"""
#1
from sklearn.tree import DecisionTreeClassifier

#2
dtree = DecisionTreeClassifier()

#3
dtree.fit(X_train,y_train)

#4
predictions = dtree.predict(X_test)

#5
from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report for Decision Tree Predictions Regarding Loan Payment:')
print(classification_report(y_test,predictions))
print('\n')
print('Confusion Matrix for Decision Tree Predictions Regarding Loan Payment:')
print(confusion_matrix(y_test,predictions))
print('\n')

"""
The decision tree model reports show fairly good prediction accuracy.
However, there is room for improvement. Because our data set is fairly large, 
we can train our model using the Random Forest Classifier and see if our 
prediction accuracy can be significantly improved.
"""

"""
TRAIN A RANDOM FOREST CLASSIFICATION MODEL ON OUR DATA
-----------------------------------------------------------------------------
Create and fit our data to a random forest model to improve the accuracy of 
loan payment predictions. Compare the results of this model to the
single decision tree model previously trained.
-----------------------------------------------------------------------------
1-Import necessary method from sklearn
2-Create instance of a random forest classifier
3-Fit this random forest model to our data
4-Use this new model to create predictions from our test set
5-Create confusion matrix and classification reports to evaluate
	Compare this data against the single decision tree classification model
-----------------------------------------------------------------------------
"""
#1
from sklearn.ensemble import RandomForestClassifier

#2
rfc = RandomForestClassifier(n_estimators=600)

#3
rfc.fit(X_train,y_train)

#4
rfc_predictions = rfc.predict(X_test)

#5
print('Classification Report for Random Forest Predictions Regarding Loan Payment:')
print(classification_report(y_test,rfc_predictions))
print('\n')
print('Confusion Matrix for Random Forest Predictions Regarding Loan Payment:')
print(confusion_matrix(y_test,rfc_predictions))
print('\n')

"""
From the new reports based off of the Random Forest Model, it can
be observed that there were significant improvements in the ability
to predict the target variable when using a Random Forest model
as opposed to a single decision tree model.

"""






"""
Description of Each Column
-----------------------------------------------------------------------------
credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
installment: The monthly installments owed by the borrower if the loan is funded.
log.annual.inc: The natural log of the self-reported annual income of the borrower.
dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
fico: The FICO credit score of the borrower.
days.with.cr.line: The number of days the borrower has had a credit line.
revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
-----------------------------------------------------------------------------
"""



