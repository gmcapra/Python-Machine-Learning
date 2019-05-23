#Practice Using K Nearest Neighbor Classification Algorithm
"""
Given some 'classified' data where we do not
know what the columns represent, use KNN to classify and determine
whether or not data belongs to target class. Further, show how to 
use the Elbow method to optimize selection of K value.
"""

#import necessary python data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read in classified data and print the head
df = pd.read_csv('Classified Data.csv',index_col=0)
print('\n')
print('Showing Classified Data Head: ')
print(df.head())
print('\n')

#data must be standardized to the same scale
#import sklearn to use standard scaler
from sklearn.preprocessing import StandardScaler

#create an instance of a scaler
scaler = StandardScaler()

#fit the scaler to our data features (drop target class so we only include feature cols)
scaler.fit(df.drop('TARGET CLASS',axis=1))

#perform standardization on all features by using transform function
#returns an array of values (the scaled version of original data in df)
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

#create dataframe of features using scaled features transformed data
df_features = pd.DataFrame(data=scaled_features,columns=df.columns[:-1])

#print the head of this new dataframe to be used in our machine learning algorithm
print('Transformed Classified Features DataFrame:')
print('\n')
print(df_features.head())
print('\n')

#move on to train test split now that data is ready
from sklearn.model_selection import train_test_split

#use features dataframe as x, target class column as y
X = df_features
y = df['TARGET CLASS']

#perform train test split on data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#import KNN algorithm
from sklearn.neighbors import KNeighborsClassifier

#instantiate knn variable (k = ?)
knn = KNeighborsClassifier(n_neighbors=1)

#fit the data to KNN classification
knn.fit(X_train,y_train)

#predict values off of test set and print to terminal
pred = knn.predict(X_test)
print('Predictions for data being in TARGET CLASS:')
print(pred)
print('\n')

#import metrics from sklearn
from sklearn.metrics import classification_report, confusion_matrix

#print both classification report and confusion matrix, where k = 1
print('Confusion Matrix with k = 1:')
print(confusion_matrix(y_test,pred))
print('\n')
print('Classification Report with k = 1:')
print(classification_report(y_test,pred))
print('\n')

#use elbow method to determine optimal k value
"""
- define error rate
- iterate several models using different k values and determine maximum efficiency
- in this case check for k values from 1 to 40
- create model, fit model to training set, predict from test_set, append mean of where predictions were not equal to test values
"""
error_rate = []
for i in range(1,40):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train,y_train)
	pred_i = knn.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))


#print out the error_rate array and then determine the minimum error rate
print('Error Rates for Model Iteration:')
print(error_rate)
print('\n')

#plot the error rate as a function of chosen k value to determine best k value for classification
print('Showing Error Rate vs. K Value...')
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()
print('\n')

#observing the graph will show a minimum error rate around 17 and then continuing around k = 34
#let's retry the model with k = 17 and get new classification reports for our model
knn17 = KNeighborsClassifier(n_neighbors=17)
knn17.fit(X_train,y_train)
pred17 = knn17.predict(X_test)

#print both classification report and confusion matrix for k = 17 model
print('Confusion Matrix with k = 17:')
print(confusion_matrix(y_test,pred17))
print('\n')
print('Classification Report with k = 17:')
print(classification_report(y_test,pred17))
print('\n')

"""
By viewing the new classification report, we can see increases across the board
in precision, recall, f1-score, and support. Looking at the confusion matrix also
supports this increased accuracy, showing more True Positives and True Negatives 
as well as less False Positives and False Negatives.
"""

