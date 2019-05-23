"""
Building a Simple Content-Based Movie Recommendation System in Python
--------------------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
--------------------------------------------------------------------------------
PROJECT DESCRIPTION
--------------------------------------------------------------------------------
In this script, a content-based recommender system will be built for a
data set of movies.

We will then use the recommendation system to see what movies a user is most likely
to enjoy based on a given movie.
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
movie_data = pd.read_csv('u.data', sep='\t',names=column_names)

# get movie titles from csv file
movie_titles = pd.read_csv('Movie_Id_Titles.csv')

# merge movie_titles with movie_data to finalize our dataframe
movie_data = pd.merge(movie_data,movie_titles,on='item_id')
 
# print out the head of the final movie dataframe
print('\nMovie Recommendation DataFrame Head:')
print(movie_data.head())
print('\n')

"""
EXPLORATORY DATA ANALYSIS
---------------------------------------------------------------------
Visualize and create different combinations of the movie data.
---------------------------------------------------------------------
"""

# import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# create ratings dataframe with average rating grouped by the movie title
ratings = pd.DataFrame(movie_data.groupby('title')['rating'].mean())

# add the total number of ratings per movie to this dataframe as a column
ratings['number of ratings'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())

# print out the head of this new data frame
print('Ratings DataFrame:')
print(ratings.head())
print('\n')

# create a histogram showing the distribution of the number of ratings for each movie
print('Showing Distribution of Number of Ratings for each Movie...')
plt.figure(figsize=(10,4))
ratings['number of ratings'].hist(bins=70)
plt.title('Distribution of the Number of Ratings for each Movie')
plt.xlabel('Number of Ratings')
plt.show()
print('\n')

# create a histogram showing the distribution of movie ratings
print('Showing Distribution of Movie Ratings...')
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Movie Rating 0-5')
plt.show()
print('\n')

# create a jointplot that compares the rating of a movie to the number of ratings
print('Showing Movie Ratings vs. Number of Ratings Jointplot...')
sns.jointplot(x='rating',y='number of ratings',data=ratings, alpha=0.55)
sns.set_context('paper')
plt.show()
print('\n')

"""
CREATING A SIMPLE RECOMMENDATION SYSTEM
-------------------------------------------------------------------------------
"""

# create a matrix that has user ids on axis and movie title on another axis
# this will result in each cell consisting of the user's rating for that movie
# will result in many NaN values, because not everyone has seen every movie
moviematrix = movie_data.pivot_table(index='user_id',columns='title',values='rating')

# to build our recommender system, lets see the movies with the most ratings 
# print out the top 10 most rated movies
print('Top 10 Most Rated Movies:')
print(ratings.sort_values('number of ratings',ascending=False).head(10))
print('\n')

"""
USING THE RECOMMENDER SYSTEM TO MAKE MOVIE RECOMMENDATIONS
-------------------------------------------------------------------------------
"""

starwars_ratings = moviematrix['Star Wars (1977)']
liarliar_ratings = moviematrix['Liar Liar (1997)']

# now, we can use the corrwith() method to get correlations between two pandas series
# the method will return a series with 
similar_to_starwars = moviematrix.corrwith(starwars_ratings)
similar_to_liarliar = moviematrix.corrwith(liarliar_ratings)

# instead of a series, lets use a DataFrame
corr_starwars = pd.DataFrame(similar_to_starwars, columns = ['Correlation'])
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns = ['Correlation'])

# this will contain a lot of NaN values, let's remove them
corr_starwars.dropna(inplace=True)
corr_liarliar.dropna(inplace=True)

"""
Now, printing the head of these dataframes, we can see that sorting by correlation,
we can arrive at the most similar movies. However, currently these values don't make 
sense. This is because there are many movies that have only been watched once by users 
who have also watched starwars and liar liar
--------------------------------------------------------------------------------------
Copy and Paste the code below to show top 10 movies (incorrectly) correlated 
with star wars and liar liar:
print('\nTop 10 Movies Correlating with Star Wars (Incorrect):')
print(corr_starwars.sort_values('Correlation',ascending=False).head(10))
print('\n')
print('\nTop 10 Movies Correlating with Liar Liar (Incorrect):')
print(corr_liarliar.sort_values('Correlation',ascending=False).head(10))
print('\n')
--------------------------------------------------------------------------------------
This can be fixed by filtering out movies with less than 100 reviews
the value 100 was chosen from observation of previously plotted histograms.
"""

# use a join to merge ratings column 'number of ratings' to the correlation dataframe
corr_starwars = corr_starwars.join(ratings['number of ratings'])
corr_liarliar = corr_liarliar.join(ratings['number of ratings'])

# now, sort the values and filter by our minimum 100 ratings criteria
print('Top 10 Most Correlated Movies for Star Wars:')
print(corr_starwars[corr_starwars['number of ratings']>100].sort_values('Correlation',ascending=False).head(10))
print('\n')
print('Top 10 Most Correlated Movies for Liar Liar:')
print(corr_liarliar[corr_liarliar['number of ratings']>100].sort_values('Correlation',ascending=False).head(10))
print('\n')

"""
Observing the output of the final table, we can see that our recommender system is able
to show the correlation between movies for users. Using this simple script, one could
repeat the process starting from the section:

USING THE RECOMMENDER SYSTEM TO MAKE RECOMMENDATIONS

to gather recommendations for other movies within the data set. Additionally, the 100
review criteria could be further adjusted for different results.
"""



