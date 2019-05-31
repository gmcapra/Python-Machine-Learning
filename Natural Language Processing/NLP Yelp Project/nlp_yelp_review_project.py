"""
--------------------------------------------------------------------------------------------
Using Natural Language Processing with Python to Filter SMS Spam Messages
--------------------------------------------------------------------------------------------
Gianluca Capraro, Python for Data Science and Machine Learning Project
Created: May 2019
--------------------------------------------------------------------------------------------
In this project, we will use Python with the NLTK library to attempt to classify
Yelp Reviews into 1 or 5 star categories based off of the context of text in the
review.

This project utilizes the Yelp Review Data Set provided by Kaggle:
https://www.kaggle.com/c/yelp-recsys-2013

The data set can be downloaded or the .csv file within this folder can be used.
--------------------------------------------------------------------------------------------
Each observation within the dataset is a review of a business by some user.
The 'stars' column refers to the rating (1-5) the reviewer gave to the business.
The 'cool' column refers to the number of 'cool' votes the review received from other users
The 'useful' and 'funny' columns are similar to the 'cool' column.
--------------------------------------------------------------------------------------------
"""

#import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#read in the yelp.csv file
yelp_data = pd.read_csv('yelp.csv')

#create a new column, called 'text length' that stores the length of the text column
yelp_data['text_length'] = yelp_data['text'].apply(len)

#print the head, info, and descriptions of the dataset
print('\nYelp Review Data Head:')
print(yelp_data.head())
print('\n')
print('Yelp Review Information:')
print(yelp_data.info())
print('\n')
print('Yelp Review Description:')
print(yelp_data.describe())
print('\n')


"""
--------------------------------------------------------------------------------------------
Exploratory Data Analysis
--------------------------------------------------------------------------------------------
1 - Create a FacetGrid using Seaborn to get distributions of the text length based on the star rating

2 - Create a boxplot of text length for each star category

3 - Create a countplot for occurrences of star ratings

4 - Use groupby to get the mean values of the numerical columns
	Create a dataframe for this new table and print the head

5 - Use the .corr() method on the grouped dataframe to get correlation
	between numerical fields
	Print the data head of this matrix

6 - Create a heatmap to better visualize the numerical column correlations
--------------------------------------------------------------------------------------------
"""
#1
print('Showing FacetGrid of Text Length Distribution by Star Rating...')
stars_text_length = sns.FacetGrid(yelp_data, col = 'stars')
stars_text_length.map(plt.hist, 'text_length')
plt.show()
print('\n')

#2
print('Showing Boxplot of Text Length Distribution by Star Rating...')
sns.boxplot(x='stars',y='text_length',data=yelp_data,palette='rainbow')
plt.title('Boxplot of Review Length by Star Rating')
plt.show()
print('\n')

#3
print('Showing Occurrences of each Star Rating...')
sns.countplot(x='stars',data=yelp_data,palette='rainbow')
plt.title('Countplot of Star Ratings')
plt.show()
print('\n')

#4
yelp_mean_stars_data = yelp_data.groupby('stars').mean()
print('Yelp - Mean Value for Numerical Columns based on Star Rating Data Head:')
print(yelp_mean_stars_data.head())
print('\n')

#5
stars_corr_matrix = yelp_mean_stars_data.corr()
print('Yelp - Correlation between Numerical Columns:')
print(stars_corr_matrix.head())
print('\n')

#6
print('Showing Heatmap of Star Rating and Numerical Column Correlation...')
sns.heatmap(stars_corr_matrix, cmap = 'coolwarm', annot = True)
plt.title('Heatmap of Correlation Between Numerical Columns and Star Rating')
plt.show()
print('\n')


"""
--------------------------------------------------------------------------------------------
Natural Language Processing Classification of Reviews
--------------------------------------------------------------------------------------------
In this project, we will attempt to either classify the review as a 1 or a 5.
The process to build our classifier is broken down step-by-step:

1 - Create new dataframe called yelp_rating that contains the columns of our
	original dataframe, but with only 1 or 5 star reviews

2 - Create two objects, X and y, that will be the features and target variable of yelp_rating
	X should be the 'text' column
	y should be the 'stars' column

3 - Import the CountVectorizer method and create an instance of it

4 - Overwrite X by using the fit_transform method on cv and passing X

5 - Perform a train test split on our data to obtain testing and training sets

6 - Train the model using the Naive Bayes Algorithm

7 - Fit the algorithm to our data

8 - Obtain rating classification predictions from our model using the test set
--------------------------------------------------------------------------------------------
"""
#1
yelp_rating = yelp_data[(yelp_data.stars == 1) | (yelp_data.stars == 5)]

#2
X = yelp_rating['text']
y = yelp_rating['stars']

#3 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

#4
X = cv.fit_transform(X)

#5 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#6 
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

#7
nb.fit(X_train, y_train)

#8
rating_predictions = nb.predict(X_test)


"""
--------------------------------------------------------------------------------------------
Evaluating the Model Predictions
--------------------------------------------------------------------------------------------
- Obtain classification report and confusion matrix for predictions
- Compare true test data classifications to those predicted by the Naive Bayes model
--------------------------------------------------------------------------------------------
"""
from sklearn.metrics import confusion_matrix, classification_report
print('Classification Report for Naive Bayes Model Accuracy in Predicting Yelp Rating:')
print(classification_report(y_test, rating_predictions))
print('\n')
print('Confusion Matrix for Naive Bayes Model Accuracy in Predicting Yelp Rating:')
print(confusion_matrix(y_test, rating_predictions))
print('\n')

"""
Based on the reports obtained, we can see that the model was fairly accurate 
in predicting a 1 or 5 star yelp rating based on the data we had to train
our model. 

To continue, this classification model can be refined to utilize a pipeline
for text preprocessing. Does this necessarily improve the accuracy? 

What about a different Machine Learning Model than the NB to classify our data?
"""
