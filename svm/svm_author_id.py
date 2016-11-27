#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#More value of C, more complicated decision boundary will be there

#First we optimise value of C and kernel on a smaller dataset and 
# then again increase our dataset to improve the efficiency.
clf = SVC(C = 10000, kernel = 'rbf')


# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 
clf.fit(features_train, labels_train)

count = 0
print(count)
for num in range(0, len(features_test)):
	if clf.predict(features_test[num]) == 1:
		count = count + 1

print(count)
print(clf.predict(features_test[50]))
print(accuracy_score((clf.predict(features_test)), labels_test))

#########################################################
### your code goes here ###



#########################################################


