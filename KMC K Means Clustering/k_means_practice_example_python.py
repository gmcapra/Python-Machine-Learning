"""
Practicing Using K Means Clustering With Python
------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
------------------------------------------------------------------
-Use Scikit learn and python to create artificial data
-Use Scikit learn and python to manipulate and visualize this data 
	and implement a K Means Clustering Algorithm 
------------------------------------------------------------------
"""

#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#generate 'blob-like' groups of data that can be manipulated and analyzed
from sklearn.datasets import make_blobs

#create data set using make_blobs method
data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8, random_state = 101)

"""
The data created is a tuple containing two indexes.
The first index contains a numpy array with the number of samples (200) and features (2)
The second index in the tuple contains information regarding the cluster (center)
each sample belongs to

Within this data, there are 4 'blobs' or clusters, as created by the make_blobs method

Print out the data to understand the contents of the tuple
"""
print('\nData Created:')
print(data)
print('\n')


"""
Create a scatter plot to try to visualize the data created.
In this plot the x and y parameters should represent the features for the 200 samples
data[0] is the array of samples and features data
data[0][:,0] represents x, the first feature for all 200 samples
data[0][:,1] represents y, the second features for all 200 samples
"""
print('Showing Scatter Plot...')
plt.title('Feature 1 vs. Feature 2 for 200 Artificially Created Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(data[0][:,0], data[0][:,1])
plt.show()
print('\n')


"""
The data plots correctly, however we cannot distinguish the 4 distinct clusters
that were created.

To visualize these groups, we need to make use of the second index of our
data tuple, the cluster information for each sample. This information can 
be passed into our scatter function to segment the data by cluster group.

After running the below code, we can see everything grouped correctly by cluster.
"""
print('Showing Scatter Plot Segmented by Cluster...')
plt.title('Feature 1 vs. Feature 2 Segmented by Cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap='coolwarm')
plt.show()
print('\n')


"""
In the above example, our data already contains the correct cluster information.
If we did not have the appropriate cluster information, how would we determine
the clusters our data points should be grouped in?

To do this, we can make use of the K Means Clustering Algorithm
----------------------------
K Means Clustering Algorithm
----------------------------
Allows clustering unlabeled data in an unsupervised machine learning algorithm.

Choose a number of clusters K
Randomly assign each data point to a cluster
Until the clusters stop changing, repeat:
	For each cluster, compute the centroid by taking the mean vector of cluster points
	Assign each data point to the cluster to which the centroid is closest

The overall goal is to divide the data into distinct groups such that observations
within each group are similar.

For optimization of K Value, one would employ the elbow method and identify
the best K value for their data set.

"""

#import the algorithm
from sklearn.cluster import KMeans

#create kmeans object
kmeans = KMeans(n_clusters=4)

#fit the algorithm to our data features
kmeans.fit(data[0])

#now we can actually predict the locations of our 4 clusters based on our data
print('Cluster Location Coordinates:\n')
print(kmeans.cluster_centers_)
print('\n')


#visualize the predicted cluster centers by plotting them
#create a list of the coordinates
cluster_center_locations = kmeans.cluster_centers_

#create two empty lists to store x and y values respectively
cluster_x = []
cluster_y = []

#loop through the coordinates to store x and y values in their lists
for i in range(0, len(cluster_center_locations)):
	cluster_x.append(cluster_center_locations[i][0])
	cluster_y.append(cluster_center_locations[i][1])

#plot the coordinates on a scatter plot
print('Showing Cluster Center Locations Plot...')
plt.scatter(cluster_x, cluster_y, c = [0,1,2,3], cmap = 'coolwarm')
plt.title('K Means Predicted Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
print('\n')


#to view the individual cluster predictions for each sample
print('K Means Clustering Predictions for Each Sample:')
print(kmeans.labels_)
print('\n')


#use matplotlib to compare our kmeans predicted clusters against the true values
print('Showing Comparison Between K Means Clustering Predictions and Known Data Clusters...')
fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,6))

#create plot for the K means predicted data
ax1.set_title('K Means Predicted Clusters')
ax1.scatter(data[0][:,0], data[0][:,1],c=kmeans.labels_,cmap='rainbow')

#create plot for the original true cluster data
ax2.set_title('Known Clusters')
ax2.scatter(data[0][:,0], data[0][:,1],c=data[1],cmap='rainbow')

plt.show()
print('\n')


"""
Although the colors may show up slightly differently in each plot, it can be seen
that the K Means Algorithm was able to cluster the data in the same groups as
we knew to be correct.

For further exploration, modify this script's choice of K Value, and observe
how the clusters would have been predicted for K = 2, 3, 8, etc.
"""





