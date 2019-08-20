"""
Intro to Support Vector Machines with Python
Gianluca Capraro
Created: May 2019

PRACTICE EXAMPLE DESCRIPTION
---------------------------------------------------------------------
Use a Support Vector Machines Classification Model with sample data
to predict whether a tumor is malignant or benign.
---------------------------------------------------------------------
"""

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#use sample data from sklearn, it is returned as a dictionary
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#print out the information contained in cancer sample data we have loaded
print('\n')
print('Sample Data Keys for Exploration:')
print(cancer.keys())
print('\n')

#print out the information contained in the DESCR key
print('DESCR:')
print(cancer['DESCR'])
print('\n')

#explore the attributes or numerical features that will be predictors
#contained in the cancer sample data dictionary as 'feature_names'
print('Numerical Features:')
print(cancer['feature_names'])
print('\n')

#set up dataframe to contain feature names
df_features = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

#print out info regarding features data frame
print('Feature Variable DataFrame Info:')
print(df_features.info())
print('\n')

#create dataframe for target variables
df_targets = pd.DataFrame(cancer['target'],columns=['Cancer'])

#print out info regarding targets dataframe
print('Targets Dataframe Info:')
print(df_targets.info())
print('\n')

#perform train test split on data
from sklearn.model_selection import train_test_split
X = df_features
y = df_targets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#get support vector classifier model, use default parameter values to make predictions
from sklearn.svm import SVC
svcmodel = SVC()
svcmodel.fit(X_train,y_train)
predictions = svcmodel.predict(X_test)

"""
EXPLANATION OF SVC PARAMETERS (the above example is intentionally blank)
--------------------------------------------------------------------
SVC(C,cache_size,class_weight,coef0,decision_function_shape,degree,gamma,
	kernel,max_iter,probability,random_state,shrinking,tol,verbose)

C-controls cost of misclassification on training data
	large c value gives low bias and high variance
	smaller c value cost is not as high, higher bias, lower variance

gamma-'auto' is usually the default
	small gamma means large variance
	large gamma means small variance

both of these parameters can be adjusted using a grid search, discussed later
--------------------------------------------------------------------
"""

#print out classification report and confusion matrix to evaluate predictions made
from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report for Support Vector Machines Predictions of Tumor Classification:')
print(classification_report(y_test,predictions))
print('\n')
print('Confusion Matrix for Support Vector Machines Predictions of Tumor Classification:')
print(confusion_matrix(y_test,predictions))
print('\n')

"""
The reports show that everything is being grouped into a single class.
This means our model needs to have its parameters adjusted. It might be
necessary to normalize the data as well.

To search for parameters we can use a GridSearch.
"""

#perform grid search to optimize parameter choices in our model
from sklearn.model_selection import GridSearchCV

#create param grid to be passed into grid search
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}

#higher verbose, more text output, default 3
grid = GridSearchCV(SVC(),param_grid,verbose=3)

#fit the grid on our training data
#will run the same loop with cross validation to determine best parameter combo
#once it has best combo, run fit again on all data passed to our fit without 
#cross validation to build a single new model using the best parameter setting
grid.fit(X_train,y_train)
print('\n')

#return combination of parameters with best cross validation scores
print('Best Parameters for SVC Function as Determined by Grid Search:')
print(grid.best_params_)
print('\n')

#can also return best estimator
grid.best_estimator_

#create predictions based on grid parameters instead of using defaults
grid_predictions = grid.predict(X_test)

#obtain new reports for new model with grid search parameters
#evaluate accuracy compared to the previous model with default parameters
print('Classification Report for Support Vector Machines\nUsing Grid Search to Identify Optimal SVC Parameters\nPrediction Accuracy of Tumor Classification:')
print(classification_report(y_test,grid_predictions))
print('\n')
print('Confusion Matrix for Support Vector Machines\nUsing Grid Search to Identify Optimal SVC Parameters\nPrediction Accuracy of Tumor Classification:')
print(confusion_matrix(y_test,grid_predictions))
print('\n')

"""
Conclusion, after using a grid search to optimize our Support Vector Model 
parameters, the accuracy improved dramatically.
"""

