# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:17:21 2021

@author: jb1033
"""


# Import model we want to use
# In sklearn, all machine learning models are implemented as Python classes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Make an instance of the model, all parameters not specified will be set to their defaults
# This way we don't have to save the outcomes from model fit and prediction separately
#   but logisticRegr already has these variables, so it will be enough to update them
logisticRegr = LogisticRegression()

from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.data.shape)
print(cancer.target.shape)

(X_cancer, y_cancer) = load_breast_cancer(return_X_y=True)
# We split the data into training and test sets
# We do this to make sure that after we train our classification algorithm,
# it is able to generalise well to new data.
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
                                                    test_size=0.75, random_state=0)


# Call the Logisitic Regression function
logisticRegr.fit(X_train, y_train)

# Predict label for the first test data
logisticRegr.predict(X_test[0].reshape(1, -1))
# Predict for all the test data
predictions = logisticRegr.predict(X_test)

# Use score method to get accuracy of the model
score = logisticRegr.score(X_test, y_test)
print(score)
# Our accuracy was 95.8%


# Compute and print confusion matrix
confusion = confusion_matrix(y_test, predictions)
print(confusion)

from sklearn.metrics import accuracy_score, precision_score
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, predictions)))
print('Precision: {:.2f}'.format(precision_score(y_test, predictions)))

