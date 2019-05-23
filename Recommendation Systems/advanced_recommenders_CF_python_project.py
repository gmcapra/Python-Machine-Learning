"""
Building Collaborative Filtering Recommendation Systems in Python
--------------------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
--------------------------------------------------------------------------------
PROJECT DESCRIPTION
--------------------------------------------------------------------------------
In this script we will make use of the MovieLens public dataset. The data set
contains 100,000 movie ratings from 943 users, and a selection of 1682 movies.

Using a Collaborative Filtering Model, we will produce recommendations based on
the knowledge of users' attitudes to items. The algorithm has the ability to do 
feature learning on its own, meaning it can learn which features to use.

Collaborative Filtering (CF) can be divided into two categories:
- Memory-Based Collaborative Filtering
- Model-Based Collaborative Filtering

In this script, we will implement Model-Based CF through singular value
decomposition (SVD), and we will implement Memory-Based CF through cosine
similarity.
--------------------------------------------------------------------------------
"""

# import libraries
import pandas as pd
import numpy as np

"""
DATA PREPARATION
---------------------------------------------------------------------
"""
# create column names for u.data file contents
column_names = ['user_id','item_id','rating','timestamp']

# create initial movie_data dataframe and read in u.data file
df = pd.read_csv('u.data', sep='\t',names=column_names)

# get movie titles from csv file
movie_titles = pd.read_csv('Movie_Id_Titles.csv')

# merge movie_titles with movie_data to finalize our dataframe
df = pd.merge(df,movie_titles,on='item_id')
 
# print out the head of the final movie dataframe
print('\nMovie Recommendation DataFrame Head:')
print(df.head())
print('\n')


#print out the number of unique users and movies in the set
unique_users = df.user_id.nunique()
unique_movies = df.item_id.nunique()
print('Number of Unique Users: '+ str(unique_users))
print('Number of Unique Movies: '+str(unique_movies))
print('\n')


"""
Train Test Split
---------------------------------------------------------------------
By nature, recommendation systems are difficult to evaluate, however
we will demonstrate how to evaluate them in this project.

To do this, we need to split the data into a testing and training set.
---------------------------------------------------------------------
"""
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)


"""
Memory-Based Collaborative Filtering
---------------------------------------------------------------------
Memory-based algorithms can be considered easier to implement and
produce reasonable predition quality. The drawback would be that it
does not scale to real-world scenarios and does not address the cold
start problem when a new user or item enters the system.

Can be divided into two main sections:
- user-item filtering
- item-item filtering

user-item: takes a particular user, finds users that are similar
based on ratings, and recommend items that similar users liked.

item-item: takes a particular item, find users that liked that item,
and find other items that those users or similar users also like

In both cases, we need to create a user-item matrix built from the
entire dataset.
"""
#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((unique_users, unique_movies))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((unique_users, unique_movies))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#use the pairwise distance function from sklearn to calculate cosine similarity
#output will range from 0 to 1 because all ratings are positive
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#apply formula for user-based CF to predict based on user or based on item
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

#make actual predictions from training set using predict method
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


"""
Evaluation of Memory-Based Collaborative Filtering Model
----------------------------------------------------------------------------------
Use Root Mean Squared Error (RMSE) to evaluate the accuracy of predicted ratings.
----------------------------------------------------------------------------------
Because we only want to consider predicted ratings that are in the test data set,
we will filter out all other elements in the prediction matrix with:

prediction[ground_truth.nonzero()]
----------------------------------------------------------------------------------
"""

from sklearn.metrics import mean_squared_error
from math import sqrt

#function to return the mean squared error based on test set
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

#print out the results of the root mean square error evaluation
print('Memory-Based CF:')
print('User-based Collaborative Filtering RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based Collaborative Filtering RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
print('\n')





"""
Model-Based Collaborative Filtering
---------------------------------------------------------------------------------
Model-Based CF methods are scalable and can deal with a higher sparsity
level than memory-based models. However, model-based models can also suffer when
new users or items with no ratings enter the system.

Model-Based CF is based on matrix factorization (MF).
The goal of MF is to learn the preferences of users and the attributes of items
from known ratings (learning features that describe characteristics of ratings)
to then predict the unkown ratings through the dot product of the features of 
users and items.
---------------------------------------------------------------------------------
"""
"""
Use Singular Value Decomposition (SVD) to perform Matrix Factorization
---------------------------------------------------------------------------------
1- In the below SVD formulation the parameters are described 
	given an (m) x (n) matrix X:

	- u is an (m) x (r) orthogonal matrix
	- s is an (r) x (r) diagonal matrix with non-negative real numbers on the diagonal
	- vt is an (r) x (n) orthogonal matrix

Matrix X can be factorized to u, s, and v.

The u matrix represents the feature vectors that correspond to the users in the
hidden feature space.
The v matrix represents the feature vectors corresponding to the items (movies) in the 
hidden feature space.

2- Elements on the diagonal of s are known as singular values of X

3- Make a prediction by taking the dot product of u, s, and vt

4- Calculate the sparsity level of the MovieLens dataset

5- Print out the RMSE for the Model Based CF model

"""
import scipy.sparse as sp
from scipy.sparse.linalg import svds

#1
u, s, vt = svds(train_data_matrix, k = 20)

#2
s_diag_matrix=np.diag(s)

#3
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

#4
print('Model-Based CF:')
sparsity = round(1.0-len(df)/float(unique_users*unique_movies),3)
print('The sparsity level of the MovieLens Dataset is ' +  str(sparsity*100) + '%')

#5
print('User-based Collaborative Filtering RMSE: ' + str(rmse(X_pred, test_data_matrix)))
print('\n')


"""
CONCLUSION, SUMMARY
----------------------------------------------------------------------------------------
This script has demonstrated how to implement Collaborative Filtering Methods,
both using a memory-based CF and model-based CF approach.

Memory-based models are based on similarity between items or users, and we used
cosine-similarity to build them.

Model-based CF is based on matrix factorization where singular value decomposition was
used to actorize the matrix.
----------------------------------------------------------------------------------------
"""


