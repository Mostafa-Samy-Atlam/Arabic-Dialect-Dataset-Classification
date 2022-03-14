# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:39:45 2022

@author: Eng. Mostafa
"""
# Importing Libraries
import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import requests
import string
import nltk
import sklearn
import time
from sklearn.model_selection import train_test_split
#################################################################################
"""
# Import Dialect Dataset to use Id column to retrieves sentences.
"""
df=pd.read_csv("dialect_dataset.csv")
y = df.iloc[:, 1].values
#################################################################################
"""
# Convert the id to a lists of stirng with each list consists of 1000 element and
  the last one consists of 196 element.
"""
ids_dict=dict()
for i in range(int(df.shape[0]/1000)+1):
    ids_list=[]
    for j in range(1000):
        if(i==int(df.shape[0]/1000) and j==df.shape[0]%1000):
            break
        ids_list.append(str(df.loc[j,"id"]))
    ids_dict[i]=ids_list
#################################################################################
"""
# Post request to retrieve arabic sentences to be classified.
"""
r = []
for key in ids_dict:
    response = requests.post("https://recruitment.aimtechnologies.co/ai-tasks",json=ids_dict[key])
    #print(r.json())
    a = response.json()
    s = list(a.values())
    for i in range(1000):
        #r.append(response.json())
        r.append(s[i])
        if(key == (len(ids_dict)-1) and i == (len(s) - 1)):
            break

"""
# Concatenation between the arabic sentences and their classes to form the dataset
  to be classified.
"""                
y = pd.DataFrame(y)
r1 = pd.DataFrame(r)

data = pd.concat([r1, y], axis=1)
data.columns = ["text", "class"]
##################################################################################
"""
# Take a fraction of the dataset.
"""
data = data.sample(frac = 0.2)

"""
# Splitting the features and the targer variables.
"""
sentences = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
#################################################################################
"""
# Vectorizing the text (Features).
"""
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sentences.flatten())
sentences = vectorizer.transform(sentences.flatten())
#X_test  = vectorizer.transform(sentences_test.flatten())
#################################################################################
"""
# Apply feature selection to reduce features in order to simplify the model to 
  fit to the available resources. (An optional step)
"""
# feature extraction
test = SelectKBest(score_func=f_classif, k=500)
fit = test.fit(sentences, y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
sentences = fit.transform(sentences)
# summarize selected features
#print(features[0:5,:])
#################################################################################
"""
# Splitting the data to training set and testing set.
"""
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.40, random_state=1000)

"""
# Applying labelencoder on the target variable, to resolve categorical data
  problem.
"""
from sklearn.preprocessing import LabelEncoder
tags = data['class']
encoder = LabelEncoder()
encoder.fit(tags)
tagst=encoder.fit_transform(tags)
num_classes = int((len(set(tagst))))
print((len(set(tagst))))
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

X_train = X_train.toarray()
X_test = X_test.toarray()
#################################################################################
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
#################################################################################

# Fitting Naive Bayes to the Training set
"""
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
score = classifier.score(X_test, y_test)
print("Accuracy:", score)
score = classifier.score(X_train, y_train)
print("Accuracy:", score)
"""



# Fitting Kernel SVM to the Training set`
"""
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
classifier.fit(X_train, y_train)
# Predicting the Test set results
score = classifier.score(X_test, y_test)
print("Accuracy:", score)
# Predicting the Test set results
score = classifier.score(X_train, y_train)
print("Accuracy:", score)
"""



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
score = classifier.score(X_test, y_test)
print("Accuracy:", score)
################################################################################

# Create Pickle File:
import pickle
pickle.dump(classifier, open('model.pkl', 'wb'))
#################################################################################