# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:42:27 2018

@author: Isaac
"""

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#using K-neirest neighbours
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

#using a basic decision tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))