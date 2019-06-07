"""
--------------------------------------------------------------------------------------------
Using TensorFlow Estimators to Create a DNN Model and Predict Bank Note Authenticity
--------------------------------------------------------------------------------------------
Gianluca Capraro, Python for Data Science and Machine Learning Project
Created: May 2019
--------------------------------------------------------------------------------------------
This is the final project for the Python Data Science and Machine Learning class offered by
Udemy. In this project, a model will be created and trained on the bank note data set to 
make predictions regarding whether a bank note is fake or authentic.
--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------
"""

#import pandas
import pandas as pd

#read in the data from bank note csv file
df = pd.read_csv('bank_note_data.csv')

#print data head to understand features and target variables
print('\nBank Note Data Head:')
print(df.head())
print('\n')


"""
--------------------------------------------------------------------------------------------
Exploratory Data Analysis
--------------------------------------------------------------------------------------------
"""
#import seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

#create a countplot of the two classes of bank note (authentic-1 or fake-0)
print('Showing Countplot of Bank Notes by Authenticity...')
sns.countplot(x='Class', data = df)
plt.title('Countplot of Bank Notes by Authenticity (0 = Fake Bill)')
plt.show()
print('\n')

#create a pairplot of the two classes of bank note (authentic-1 or fake-0)
print('Showing Pairplot of Bank Notes by Authenticity...')
sns.pairplot(df, hue = 'Class')
plt.show()
print('\n')


"""
Based on the pairplot, we can see that the bank note data is quite separable.
Based on this, we can expect our model to do pretty well once we have trained it on
our dataset.
"""

"""
--------------------------------------------------------------------------------------------
Data Preparation
--------------------------------------------------------------------------------------------
"""
#perform standard scaling on the data set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#fit the scaler to our dataset's features
#drop the target variable
scaler.fit(df.drop('Class',axis=1))

#use the transform method to get the features into a scaled version
scaled_features = scaler.transform(df.drop('Class',axis=1))

#convert scaled features to dataframe
df_features = pd.DataFrame(scaled_features, columns = df.columns[:-1])

"""
--------------------------------------------------------------------------------------------
Train Test Split Data
--------------------------------------------------------------------------------------------
"""
X = df_features
y = df['Class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


"""
--------------------------------------------------------------------------------------------
Utilize TensorFlow Estimators to create and train our model
--------------------------------------------------------------------------------------------
"""
#import tensorflow library
import tensorflow as tf

#get tensorflow useable feature columns 
feature_cols = []
for col in df_features.columns:
	feature_cols.append(tf.feature_column.numeric_column(col))

#create dnn classifier
classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10],n_classes=2,feature_columns=feature_cols)

#create the input function
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 20, shuffle = True)

#train classifier to input function
classifier.train(input_fn = input_func, steps = 400)


"""
--------------------------------------------------------------------------------------------
Evaluate the DNN Classifier
--------------------------------------------------------------------------------------------
"""
#create prediction function
prediction_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test), shuffle = False)

#use predict method from classifier model to get predictions from X_test
bank_note_predictions = list(classifier.predict(input_fn = prediction_func))

#get the final predictions from the dictionary of bank note predictions at index 0
final_predictions = []
for prediction in bank_note_predictions:
	final_predictions.append(prediction['class_ids'][0])

#get classification report and confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
print('\nClassification Report for Bank Note Authenticity Predictions Using DNN Classifier:')
print(classification_report(y_test,final_predictions))
print('\n')
print('Confusion Matrix for for Bank Note Authenticity Predictions Using DNN Classifier:')
print(confusion_matrix(y_test,final_predictions))
print('\n')


"""
The reports obtained show very high accuracy of bank note authenticity predictions. To
compare, let's see how a different classifier model would perform.

Use a RandomForestClassifier instead and compare this model's accuracy to that of our DNN Model.
"""

#import random forest classifier
from sklearn.ensemble import RandomForestClassifier

#create an instance of the rf classifier
rfc = RandomForestClassifier(n_estimators = 200)

#fit the rfc to our training data
rfc.fit(X_train,y_train)

#grab predictions from the rfc model using our X_test set
rfc_predictions = rfc.predict(X_test)

#get classification report and confusion matrix from random forest classifier predictions
print('\nClassification Report for Bank Note Authenticity Predictions Using RandomForestClassifier Model:')
print(classification_report(y_test,rfc_predictions))
print('\n')
print('Confusion Matrix for for Bank Note Authenticity Predictions Using RandomForestClassifier Model:')
print(confusion_matrix(y_test,rfc_predictions))
print('\n')

"""
As we can see, even the RFC model performs excellently with this data as it is so separable.
"""


