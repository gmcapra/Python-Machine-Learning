#Use K Nearest Neighbor Algorithm to Choose an Optimal K Value
#Gianluca Capraro
#Created: May, 2019
"""
KNN PROJECT DESCRIPTION
----------------------------------------------------------------------------------
This project will demonstrate ability to:
Use Seaborn to visualize given DataFrame and analyze column relationships.
Standardize database variables for use in a train test split.
Fit training data to a KNN Model and use model to predict test values.
Create a confusion matrix and classification report based on this model.
Use the elbow method to optimize model k value selection by plotting against error rate.
Retrain KNN Model with optimized k value and reprint new classification report and
confusion matrix.
----------------------------------------------------------------------------------
"""

#import necessary python data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read in classified data and print the head
df = pd.read_csv('KNN_Project_Data.csv')
print('\n')
print('Project Data Head: ')
print(df.head())
print('\n')

"""
EXPLORATORY DATA ANALYSIS
----------------------------------------------------------------------------------
Because we are sampling from artificial data, we will do a large seaborn
pairplot to visualize as many relationships as possible and determine
what correlations may exist.

Additionally, because we are specifically interested in the 'Target Class' column
we will specify a hue to further segment the pairplot by this column value (0 or 1).

Finally, we want the diagonal to show histogram plots, so we will pass in
this parameter as well.
----------------------------------------------------------------------------------
"""
print('Showing Seaborn Pairplot of Project Data...')
print('Large graph, may take a few seconds...')
print('\n')
sns.pairplot(df, hue='TARGET CLASS',diag_kind='hist')
plt.show()
print('\n')


"""
STANDARDIZE THE VARIABLES
--------------------------------------------------------------------------------
1-Use SciKit learn library to import the Standard Scaler.
2-Create an instance of this scaler to standardize our data and prepare
	it for use in KNN classification algorithm.
3-Fit the scaler to our dataframe features, making sure to drop the target column
	and any other columns that are not needed.
4-Use the transform() method to transform the features of our data to a
	scaled version.
5-Convert the scaled features to a data frame and check the head to verify. Also
	verify the columns in the new data frame.
--------------------------------------------------------------------------------
"""
#1
from sklearn.preprocessing import StandardScaler
#2
scaler = StandardScaler()
#3
scaler.fit(df.drop('TARGET CLASS',axis=1))
#4
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
#5
df_features = pd.DataFrame(data=scaled_features,columns=df.columns[:-1])
print('Transformed Features DataFrame:')
print(df_features.head())
print('\n')
print(df_features.columns)
print('\n')

"""
PERFORM TRAIN TEST SPLIT ON OUR DATA
----------------------------------------------------------------------------------
1-Import train test split method
2-Split Data into Testing and Training Sets
----------------------------------------------------------------------------------
"""
#1
from sklearn.model_selection import train_test_split
#2
X = df_features
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

"""
SET UP AND FIT KNN MODEL TO DATA WITH K = 1
----------------------------------------------------------------------------------
1-Import necessary KNeighborsClassifier from SKLEARN
2-Create an instance of KNN classifier with K value = 1
3-Fit training data to KNN model
4-Predict values off of X test set, print these values
5-Import classification report and confusion matrix from SKLEARN metrics library
6-Create and print the classification report and confusion matrix for k = 1
	Compare our predictions to the y_test data that we know matches with X_test
	Observe the overall accuracy of our current model with k = 1
----------------------------------------------------------------------------------
"""
#1
from sklearn.neighbors import KNeighborsClassifier
#2
knn = KNeighborsClassifier(n_neighbors=1)
#3
knn.fit(X_train,y_train)
#4
predictions = knn.predict(X_test)
print('Predictions for data being in TARGET CLASS:')
print(predictions)
print('\n')
#5
from sklearn.metrics import classification_report,confusion_matrix
#6
print('Confusion Matrix with k = 1:')
print(confusion_matrix(y_test,predictions))
print('\n')
print('Classification Report with k = 1:')
print(classification_report(y_test,predictions))
print('\n')

"""
USING ELBOW METHOD TO OPTIMIZE K VALUE SELECTION IN KNN MODEL
----------------------------------------------------------------------------------
1-Create a for loop that trains various KNN Models with different k values
	Keep track of the error_rate using a list that stores these values for each model
2-Plot k vs. error_rate using matplotlib, determine best k value
	Observing the data, we should find that an optimal choice for K is around 31
----------------------------------------------------------------------------------
"""
#1
error_rate = []

for i in range(1,40):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train)
	pred_i = knn.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))

#2
print('Showing Error Rate vs. K Value...')
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()
print('\n')


"""
RETRAIN DATA USING OPTIMAL K VALUE TO REDUCE ERROR
----------------------------------------------------------------------------------
1-Instantiate new KNN model with K = 31
2-Fit Training Data to KNN model
3-Obtain predictions from our model based on the X_test set
4-Use classification report and confusion matrix to analyze new model results
	Our new model shows all around improvements in accuracy according to our reports
	For further exploration, retrain with other k values to improve accuracy
----------------------------------------------------------------------------------
"""
#1
knn = KNeighborsClassifier(n_neighbors = 31)
#2
knn.fit(X_train,y_train)
#3
optimized_predictions = knn.predict(X_test)
#4
print('Confusion Matrix with k = 31:')
print(confusion_matrix(y_test,optimized_predictions))
print('\n')
print('Classification Report with k = 31:')
print(classification_report(y_test,optimized_predictions))
print('\n')




