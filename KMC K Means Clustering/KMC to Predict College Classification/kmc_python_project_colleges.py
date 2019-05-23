"""
Using K Means Clustering to Predict Whether a University is Public or Private
-------------------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
------------------------------------------------------------------
PROJECT DESCRIPTION
------------------------------------------------------------------
Attempt to use K Means Clustering to cluster universities into 
two groups: Public or Private.

This project will make use of a public data frame containing
777 Universities with 18 different variables related to each.

Although this data frame already contains the label indicating
whether the University is public or private, this project will
cluster the data points in an unsupervised setting where the true
classification information is not used to make predictions.

These K Means Cluster Predictions will be compared to the known
classification to demonstrate the application of the KMC as an
unsupervised classification algorithm.
------------------------------------------------------------------
"""

#import necessary libraries
import pandas as pd
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt


#read in college data .csv file and set the first column as the index
college_data = pd.read_csv('College_Data.csv', index_col = 0)

#print out the head()
print("\nCollege Data Head:")
print(college_data.head())

#print out the info()
print("\nCollege Data Info:")
print(college_data.info())

#print out the describe() method, showing statistics across columns
print("\nCollege Data Description of Columns:")
print(college_data.describe())

"""
EXPLORATORY DATA ANALYSIS
-----------------------------------------------------------------------
Create various data visualizations for practice and to develop a better
understanding of the relationships within the college data set.

Within this section, feel free to use the Public vs. Private label for
visualization. However, this label will not be used when we use the
K Means Clustering algorithm later on.
-----------------------------------------------------------------------
1- Create a scatterplot of the grad rate vs. the room and board

	a) Prior to removal, there will be a school with a grad rate > 100%
	Set this outlier to be 100% so that it does not skew data

	b)Segment the data by Public vs. Private
	Optionally, add in lines for linear relationships
	

2- Create a scatterplot of full time undergrads vs. out of state tuition
	Points should also be segmented by Public vs. Private

3- Create a stacked histogram showing Out of State Tuition based on the
	whether University is Public or Private
		Use Seaborns FacetGrid to do this

4- Create a similar histogram for the Grad Rate column

"""
#1a - remove graduation rate > 100% outlier, set = 100
college_data['Grad.Rate']['Cazenovia College'] = 100

#1b
print('Showing Scatterplot of Room and Board vs. Graduation Rate...')
sns.lmplot('Room.Board','Grad.Rate',data=college_data,hue='Private')
plt.title('Room and Board vs. Graduation Rate')
plt.xlabel('Room and Board Cost (Dollars)')
plt.ylabel('Graduation Rate (%)')
plt.show()
print('\n')


#2
print('Showing Scatterplot of Out of State Tuition vs. # of Full Time Undergrads...')
sns.lmplot('Outstate','F.Undergrad',data=college_data,hue='Private')
plt.title('Out of State Tuition vs. # of Full Time Undergraduates')
plt.xlabel('Out of State Tuition (Dollars)')
plt.ylabel('# of Full Time Undergraduates')
plt.show()
print('\n')


#3
print('Showing Histograms of Out of State Tuition based on University Classification...')
sns.set_style('darkgrid')
snsGrid = sns.FacetGrid(college_data,hue='Private',palette='coolwarm',height=6,aspect=1.5,legend_out=True)
snsGrid = (snsGrid.map(plt.hist,'Outstate',bins=20,alpha=0.7).add_legend())
plt.title('Out of State Tuition based on University Classification')
plt.xlabel('Out of State Tuition (Dollars)')
plt.ylabel('# of Universities')
plt.show()
print('\n')

#4
print('Showing Histograms of Graduation Rate based on University Classification...')
sns.set_style('darkgrid')
snsGrid = sns.FacetGrid(college_data,hue='Private',palette='coolwarm',height=6,aspect=1.5,legend_out=True)
snsGrid = (snsGrid.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7).add_legend())
plt.title('Graduation Rate based on University Classification')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('# of Universities')
plt.show()
print('\n')

"""
K MEANS CLUSTER FITTING
-------------------------------------------------------------
Use the K Means Algorithm to fit the model to our data
determine the predicted labels, and evaluate the results
against the actual college classifications.
-------------------------------------------------------------
1- Import KMeans from SciKit Learn
	Create an instance of a K Means Model with 2 clusters

2- Fit the model to the data (excluding the 'Private Label')
	We don't want the private label in our model because that
	is the information we want to predict and group our data
	with using K Means

3- Print the cluster center vectors
"""
#1
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
print('K Means Model Created with 2 Clusters')

#2
kmeans.fit(college_data.drop('Private',axis=1))
print('Classification Column Dropped\nK Means Model Fitted to College Data\n')


#3
print('Cluster Center Vectors:')
print(kmeans.cluster_centers_)
print('\n')

"""
EVALUATION
-----------------------------------------------------------------------------------
Now that we have used the K Means model to determine the cluster groups for our data,
we can compare these groups to the actual, known labels that we have, and evaluate
the performance of the model.

Note: In the real-world this luxury won't always be available. Usually, an
unsupervised algorithm such as K Means would be used in a case where the labels
are not known, and therefore normally wouldn't be able to compare against and 
evaluate the model's effectiveness.
-----------------------------------------------------------------------------------
1- Create a new column called 'Cluster'
	Convert to numbers, should contain a 1 if Private, a 0 if Public
	Use a lambda function to transform the existing column

3- Create a confusion matrix and classification report
	Compare K Means to True clusters
	Observe the K Means Clustering accuracy in determining the correct
	Public or Private University classification
-----------------------------------------------------------------------------------
"""
#1
def converter(cluster):
	if cluster == 'Yes':
		return 1
	else:
		return 0

college_data['Cluster'] = college_data['Private'].apply(converter)


#2 (true values, k means predicted)
from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report for K Means Classification of Public vs. Private Universities:')
print(classification_report(college_data['Cluster'], kmeans.labels_))
print('\n')
print('Confusion Matrix for K Means Classification of Public vs. Private Universities:')
print(confusion_matrix(college_data['Cluster'], kmeans.labels_))
print('\n')


"""
CONCLUSIONS
---------------------------------------------------------------------------------
The reports obtained show a moderate effectiveness in using 18 feature variables
from 777 colleges to accurately predict whether a college is Public or Private 
using the K Means Clustering Algorithm.

Although accuracy was nowhere near perfect, this example demonstrates how K Means
can be used to approximately cluster groups of unlabeled data based on feature
variables.
---------------------------------------------------------------------------------
"""

