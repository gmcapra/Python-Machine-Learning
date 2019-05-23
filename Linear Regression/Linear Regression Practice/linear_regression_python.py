#Linear Regression with Python
"""
Practice using Linear Regression with Python and Machine Learning techniques.
Learn Regression Evaluation Metrics and how to arrive at their values.
Use SKLearn library to implement, fit, test, and train a linear model
"""

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read in sample USA housing data file
df = pd.read_csv('USA_Housing.csv')

#print the head of the dataframe
print('\n')
print('DataFrame Header:')
print(df.head())
print('\n')

#get additional info on objects stored in dataframe
print('\n')
print('DataFrame Info:')
print(df.info())
print('\n')

#get statistical info on columns in dataframe
print('\n')
print('DataFrame Description:')
print(df.describe())
print('\n')


#Preliminary DATAFRAME INFORMATION GATHERING
#CREATE SOME SAMPLE PLOTS TO UNDERSTAND DATA

#obtain a high-level view of dataframe relationships using seaborn pairplot
print('Displaying USA Housing Data Pairplot...')
print('\n')
sns.pairplot(df)
plt.show()

#display distribution of housing price data
#get an understanding of the data from the column you hope to predict
print('Displaying Housing Price Distribution...')
print('\n')
sns.distplot(df['Price'])
plt.show()

#display heatmap of correlation between each of our dataframe columns
#correlation between each column
print('Displaying Heat Map of Correlation between DataFrame Columns...')
correlationTable = df.corr()
sns.heatmap(correlationTable,cmap='coolwarm')
plt.show()


#BEGIN USING SCIKIT LEARN TO TRAIN LINEAR REGRESSION MODEL
#Import train_test_split from sklearn
from sklearn.model_selection import train_test_split

#get individual column names from dataframe
print('DataFrame column names:')
print(df.columns)
print('\n')
#split data into x array with features to train on
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
#split data into y array for target variable
y = df['Price']

#use tuple unpacking to grab training and test sets
#pass in X and y data, test size (% of data set to allocate to test size)
#pass in random state 101 to ensure standard random splits to check
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40,random_state=101)

#create and train linear regression model
from sklearn.linear_model import LinearRegression
#instantiate an instance of linear model
lm = LinearRegression()
#fit model to training data
lm.fit(X_train,y_train)

#evaluate our model 
#print intercept
print('model intercept:')
print(lm.intercept_)
print('\n')

#print coefficients for each feature in X
print('feature coefficients for X training data columns:')
print(lm.coef_)
print('\n')

#create coefficient dataframe based on coefficients
#print this dataframe
#dataframe shows coefficient increase (dollars) for every unit increase of row
print('Coefficient DataFrame:')
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print(cdf)
print('\n')

#grab predictions from our test set
#pass in features that model has not seen (i.e. testing set)
predictions = lm.predict(X_test)
print('Price Predictions:')
print(predictions)
print('\n')

#create a scatter plot to infer from test data vs predictions
#check how our model did vs the actual output
print('Displaying price prediction scatter plot...')
print('\n')
plt.scatter(y_test,predictions)
plt.show()

#create histogram of distribution of our residuals
#normally distributed residuals means model was a correct choice for our data
#in this case, linear regression model was the correct choice for our dataset
print('Displaying distribution of residuals...')
sns.distplot((y_test-predictions))
plt.show()
print('\n')

#import metrics from sklearn
from sklearn import metrics
#print regression evaluation metrics
print('Mean Absolute Error (MAE): ')
print(metrics.mean_absolute_error(y_test,predictions))
print('\n')
print('Mean Squared Error (MSE): ')
print(metrics.mean_squared_error(y_test,predictions))
print('\n')
print('Root Mean Squared Error (RMSE): ')
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('\n')






