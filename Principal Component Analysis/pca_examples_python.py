"""
Practice with Principal Component Analysis (PCA) in Python
-------------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
-------------------------------------------------------------------------
Principal Component Analysis (PCA) is an unsupervised learning algorithm.

PCA is essentially a transformation of our data followed by attempts to
determine the features that explain the most variance in our data.

This type of factor analysis would normally be performed prior to 
implementing a machine learning classification algorithm. PCA facilitates
identification of the most impactful features in the data set which can
then be utilized to more effectively build a machine learning model.

This practice script will demonstrate the use of PCA using Python.
In this example, we will work with the Breast Cancer Dataset provided by
SKLearn. Using PCA, we will identify which tumor features lead to the
most variance in data.
-------------------------------------------------------------------------
"""

#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


#import sklearn sample database - for this example, the load_breast_cancer data
from sklearn.datasets import load_breast_cancer

#create database
cancer_data = load_breast_cancer()

#print out the information contained in cancer sample data we have loaded
print('\n')
print('Sample Data Keys for Exploration:')
print(cancer_data.keys())
print('\n')

#print out the information contained in the DESCR key
print('DESCR:')
print(cancer_data['DESCR'])
print('\n')

#store the relevant data from cancer_data into our working dataframe
#the working data frame refers to a dataframe of only features
df = pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])

#print out the head and verify it contains only feature data
print('Working DataFrame Head:')
print(df.head())
print('\n')

#scale data so that each feature has a single unit variance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

#import the PCA method from sklearn
from sklearn.decomposition import PCA

#create an instance of PCA with 2 components to keep
pca = PCA(n_components=2)

#fit the scaled data to the PCA model
pca.fit(scaled_data)

#transform the data to its first 2 principal components
x_pca_data = pca.transform(scaled_data)

#call shape on the scaled data to see original dataframe dimensions
print('Original DataFrame Dimensions:')
print(scaled_data.shape)
print('\n')

#now, observe the new shape of the data after transformation with PCA
print('Principal Component Data Dimensions:')
print(x_pca_data.shape)
print('\n')

#the 30 component dimensions have been reduced to the desired 2 components
#these two components can be shown through the use of a scatter plot
print('Showing First Principal Component vs. Second Principal Component...')
print('Shows Separation Between Malignant and Benign Tumor Based on 2 Principal Components...')

plt.figure(figsize=(8,6))
plt.title('Separation Between Malignant and Benign Tumor Classification Based on 2 Principal Components')
plt.scatter(x_pca_data[:,0],x_pca_data[:,1],c=cancer_data['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
print('\n')

"""
INTERPRETING THE COMPONENTS
--------------------------------------------------------------------------------
The plot defined above clearly demonstrates that the two components alone can
easily separate the two classes. However, understanding what those two
components represent will take additional work.

The principal components identified correspond to combinations of the original
features. The components themselves are stored as attributes of the fitted
PCA object.
--------------------------------------------------------------------------------
"""

#view the components attribute of pca object
print('PCA Components:')
print(pca.components_)
print('\n')

#create a dataframe of the pca components corresponding to each feature
df_components = pd.DataFrame(pca.components_,columns=cancer_data['feature_names'])

#show the contents of this dataframe, it will display the relationship 
#between each feature and the principal components (0 and 1)
print('Relationship between each feature and the 2 principal components:')
print(df_components)
print('\n')

#visualize the component relationships using a heatmap
print('Showing PCA Components Heatmap...')
plt.figure(figsize=(12,6))
plt.title('Feature Correlations to Principal Components')
sns.heatmap(df_components,cmap='plasma')
plt.show()
print('\n')

"""
CONCLUSION
-----------------------------------------------------------------
The resulting heatmap allows us to visualize how each feature
correlates to the principal component. This information can be
used to inform decisions regarding model parameters, optimize
their values, and build more capable models.
-----------------------------------------------------------------
"""


