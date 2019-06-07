"""
--------------------------------------------------------------------------------------------------
TensorFlow Estimators Project - Predicting Iris Species Using a DNN Classifier
--------------------------------------------------------------------------------------------------
In this project, a TensorFlow Estimator Object will be created and trained
to classify the species of Iris flower based on the numerical feature columns
of the iris dataset.
--------------------------------------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
"""
#import libraries 
import pandas as pd

#read in the iris data set
iris = pd.read_csv('iris.csv')

#print the head of the dataset
print('\nIris Dataset Head:')
print(iris.head())
print('\n')

#for use with tensorflow estimator object, there are a few things that we need to change
#column names cant have spaces or special characters
#target value for classification has to be integer (currently a float)
iris.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']
iris['target'] = iris['target'].apply(int)

#separate the features from the target variable
X = iris.drop('target',axis=1)
y = iris['target']

#perform Train Test Split on data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#import tensorflow to begin creating estimator object
import tensorflow as tf

#create numeric feature columns for the estimator
feature_cols = []
for col in X.columns:
	feature_cols.append(tf.feature_column.numeric_column(col))

#create input function for training using pandas input function
#x is training input function
#y is training labels
#num epochs is how many times you want to go through training data
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True)

#create DNN classifier (our estimator)
#hidden units defines # of hidden layers and how many neurons in each layer
#specify number of classes, we have 3, as there are 3 species of flowers
#define the feature columns, created above
classifier = tf.estimator.DNNClassifier(hidden_units = [10,20,10], n_classes = 3, feature_columns = feature_cols)

#train the estimator using our input function, specify how many steps
classifier.train(input_fn = input_func, steps = 50)
print('\n')

"""
--------------------------------------------------------------------------
Evaluate the Model Performance
--------------------------------------------------------------------------
To do this, create another input function similar to the one above,
except now pass in the test data.
--------------------------------------------------------------------------
"""
#create the prediction input function
#only need to pass x because we will not know the labels, we are predicting them
prediction_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test), shuffle=False)

#get the predictions from the classifier
#this contains class_ids, classes, logits, and probabilities as a dictionary for each prediction made
predictions = list(classifier.predict(input_fn = prediction_func))

#let's get the final individual predictions
#will return a list of 0,1,or 2 as the values referring to each predicted class
final_predictions = []
for prediction in predictions:
	final_predictions.append(prediction['class_ids'][0])


#now, we can import the classification and confusion matrix from sklearn
from sklearn.metrics import classification_report,confusion_matrix
print('\nClassification Report for Iris Species Prediction using a DNN Classifier:')
print(classification_report(y_test,final_predictions))
print('\n')
print('Confusion Matrix for Iris Species Prediction using a DNN Classifier:')
print(confusion_matrix(y_test,final_predictions))
print('\n')

"""
The reports obtained indicate a fairly strong ability to predict the 
iris species based on the feature columns using our DNN model. This example
specifically demonstrates the use of the TensorFlow Estimator to create a DNN 
predictive model.
"""
