#Linear Regression Project - Ecommerce Analysis
#Gianluca Capraro, 5/20/2019
"""
New York City based Ecommerce company sells clothing online, but they also
have in-store style and clothing sessions. Customers come into the store,
have sessions/meetings with a stylist, then go home to order on mobile or web.

This company is trying to decide whether to focus efforts on their mobile
app or their website.

Use knowledge of Pandas, Numpy, MatplotLib, Seaborn, and Linear Regression
to complete this project
"""

#import necessary libraries
#pandas
import pandas as pd
#numpy
import numpy as np
#matplotlib
import matplotlib.pyplot as plt
#seaborn
import seaborn as sns


#read in data from ecommerce customers csv file
df = pd.read_csv('Ecommerce Customers.csv')
print('\n')
print('Ecommerce Customers head: ')
print(df.head())
print('\n')

#show column names specifically
print('DataFrame Columns:')
print(df.columns)
print('\n')

#show dataframe info and description
print('DataFrame Info:')
print(df.info())
print('\n')
print('DataFrame Description:')
print(df.describe())
print('\n')


#PERFORM PRELIMINARY DATA ANALYSIS

#use a jointplot to compare time spent on the website vs the yearly amount spent
print('Showing Time on Website vs. Yearly Amount Spent...')
print('\n')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)
plt.show()
print('\n')

#Does the correlation between web time and amount spent make sense?
"""
These results show that there is not a correlation between the time 
spent on the website and the amount of money spent yearly.
"""

#use a jointplot to compare time spent in app vs yearly amount spent
print('Showing Time on App vs. Yearly Amount Spent...')
print('\n')
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)
plt.show()

#Does the correlation between app time and amount spent make sense?
"""
These results show that there is a possible correlation between app 
time and yearly amount spent.
"""

#use a jointplot to create a hex bin comparing time on app and length of membership
print('Showing Time on App vs. Length of Membership...')
print('\n')
sns.jointplot(x='Time on App',y='Length of Membership',data=df,kind='hex')
plt.show()

#use a seaborn pairplot to explore relationships across entire data set
print('Showing Pairplot of Entire Data set...')
print('\n')
sns.pairplot(df)
plt.show()

#Based off the pairplot, what looks to be the most correlated feature with Yearly Amount Spent?
"""
Observing the pairplot, it appears that length of membership is most
closelt correlated with yearly amount spent.
"""

#Create a linear model plot of Yearly Amount Spent vs. Length of Membership
print('Showing linear model plot of Yearly Amount Spent vs. Length of Membership...')
print('\n')
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)
plt.show()


#TRAINING AND TESTING DATA

#set a variable X equal to the numerical features of the customers
#set a variable y equal to the Yearly Amount Spent column
X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']

#Import train_test_split from sklearn
from sklearn.model_selection import train_test_split

#split the data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=101)

#Import Linear Regression Model from SKLearn
from sklearn.linear_model import LinearRegression

#create an instance of the LinearRegression model and fit lm on the training data
lm = LinearRegression()
lm.fit(X_train,y_train)

#print out the Coefficients of the model
print('Model Coefficients:')
print(lm.coef_)
print('\n')

#use linear model predict() off the X test set of data
predictions = lm.predict(X_test)

#using these predictions, create a scatter plot comparing the y_test values to our predictions
print('Showing Scatter Plot of Predictions vs. Test Values...')
plt.scatter(predictions,y_test)
plt.show()

#Evaluate our model by calculating the regression evaluation metrics
#import metrics from sklearn
from sklearn import metrics

print('Mean Absolute Error (MAE):')
print(metrics.mean_absolute_error(y_test,predictions))
print('\n')

print('Mean Squared Error (MSE):')
print(metrics.mean_squared_error(y_test,predictions))
print('\n')

print('Root Mean Squared Error (RMSE):')
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('\n')

#Finally, plot the distribution of residuals as a histogram to check the fit of our model
print('Showing distribution of residuals...')
print('\n')
sns.distplot(y_test - predictions, bins=100)
plt.show()
print('\n')

#Create a Dataframe containing the Coefficient for each numerical customer category
coefficientDF = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
print('Coefficient DataFrame:')
print(coefficientDF)
print('\n')

#What can be interpreted from these coefficients?
"""
For one, we can see that the length of a customer's membership is the most
significant indicator of amount spent yearly. (If a person has a membership for an additional year,
we can expect to see an average additional spend of $61)

Additionally, we can see that spending more time in the app is likely to increase yearly spend by more
than it would increase if that time were spent in the website.

We can also see that a longer average session time also increases the yearly spend.

Finally, we have been asked to make a recommendation on the return of allocating resources to further
developing the app vs. the website. In light of these coefficients, we can see that yearly spend increases more
if time is spent in the app, therefore we could suggest spending more time to develop the app.
However, this can go another way as it could also be beneficial to devote time to the website and increase this
coefficient as it clearly has room to improve.
To make a final decision regarding this dilemma, we would need cost information to further identify pros and cons
of each. Simply going off of the current coefficients would not be guarenteed to provide the best solution.

"""



