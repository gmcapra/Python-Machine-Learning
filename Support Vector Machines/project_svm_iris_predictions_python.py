"""
SUPPORT VECTOR MACHINES IN PYTHON
Gianluca Capraro
Created: May 2019

PROJECT DESCRIPTION
------------------------------------------------------------------------------
In this project, I will be using data from the iris data set.
The Iris flower data set is a multivariate data set introduced by Sir Ronald
Fisher in 1936 to show examples of discriminant analysis.

The data set consists of 50 samples from each of three species of Iris (150 total)
Four features are measured from each sample, and we will use these features, along
with a Support Vector Classification model, to make predictions regarding the
species of Iris.

Prior to creating the Support Vector Model, I will perform some exploratory data
analysis on the data set to observe any relationships of interest.
------------------------------------------------------------------------------
"""

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the iris data set using seaborn 
iris = sns.load_dataset('iris')

#print the contents of our loaded data
print('\nIris Dataset:\n')
print(iris)
print('\n')

"""
------------------------------------------------------------------------
Create a pairplot of the data.
What flower species appears to be the most separable?
------------------------------------------------------------------------
Observing the resulting pairplot, it can be seen that setosa is the most 
clearly separable species in our data when compared to physical similarities
between the virginica and versicolor species. (Code directly below)
------------------------------------------------------------------------
"""
print('Showing Pairplot of Iris Data, Segmented by Iris Species...')
sns.pairplot(iris,hue='species',palette='coolwarm')
plt.show()
print('\n')


"""
------------------------------------------------------------------------
Create a kde plot of sepal_length vs. sepal width for the setosa species
------------------------------------------------------------------------
1-Create sub dataframe that only contains data from the setosa species
2-Plot the KDE plot using Seaborn kdeplot function
------------------------------------------------------------------------
"""
#1
df_setosa = iris[iris['species'] == 'setosa']

#2
print('Showing KDE Plot of sepal_width vs. sepal_length for the Setosa species...')
sns.kdeplot(df_setosa['sepal_width'], df_setosa['sepal_length'], cmap = 'plasma', shade = True, shade_lowest = False)
plt.title('KDE Plot of sepal_width vs. sepal_length for Setosa species')
plt.show()
print('\n')


"""
Perform Train Test Split on Data
X refers to numerical feature columns of data
y refers to our target variable, or what we want to predict
"""
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


"""
TRAIN THE SUPPORT VECTOR MACHINE CLASSIFIER
----------------------------------------------------------------
We will train our data on the SVC model and make predictions off of
this model with default parameters, and with the optimal parameters
as determined through a grid search.
----------------------------------------------------------------
"""
#get support vector classifier model, use default parameter values to make predictions
from sklearn.svm import SVC
svcmodel = SVC()
svcmodel.fit(X_train,y_train)
predictions = svcmodel.predict(X_test)

"""
EXPLANATION OF SVC PARAMETERS (the above example is intentionally blank)
---------------------------------------------------------------------------
SVC(C,cache_size,class_weight,coef0,decision_function_shape,degree,gamma,
	kernel,max_iter,probability,random_state,shrinking,tol,verbose)

C-controls cost of misclassification on training data
	large c value gives low bias and high variance
	smaller c value cost is not as high, higher bias, lower variance

gamma-'auto' is usually the default
	small gamma means large variance
	large gamma means small variance

both of these parameters will be adjusted using a grid search
--------------------------------------------------------------------------
"""

#print out classification report and confusion matrix to evaluate predictions made
from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report for Support Vector Machines Predictions of Species:')
print(classification_report(y_test,predictions))
print('\n')
print('Confusion Matrix for Support Vector Machines Predictions of Species:')
print(confusion_matrix(y_test,predictions))
print('\n')

"""
------------------------------------------------------------------------
How did the model perform with default SVC parameters?
------------------------------------------------------------------------
Observing the resulting classification report and confusion matrix, the
model performed very well, even capable of achieving 100% accuracy!

It will be hard to improve our predictions for this data problem, however
we will optimize our parameter choice using a grid search for practice.
------------------------------------------------------------------------
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
print('Classification Report for Support Vector Machines Prediction of Iris Species\nSVC Parameters Optimized Using GridSearch:')
print(classification_report(y_test,grid_predictions))
print('\n')
print('Confusion Matrix for Support Vector Machines Prediction of Iris Species\nSVC Parameters Optimized Using GridSearch:')
print(confusion_matrix(y_test,grid_predictions))
print('\n')

"""
In this case, the grid search for optimal SVC parameters doesn't improve our
model's ability to predict the species of iris flower.
"""





