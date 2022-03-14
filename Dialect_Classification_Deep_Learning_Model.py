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
import keras
import sklearn
import time
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import numpy as np
from time import time
from nltk.corpus import stopwords
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
import keras.preprocessing.text
from tensorflow.keras import layers

"""
# Import Dialect Dataset to use Id column to retrieves sentences.
"""
df=pd.read_csv("dialect_dataset.csv")
y = df.iloc[:, 1].values
#################################################################################
"""
# Convert the id to lists of stirng with each list consists of 1000 element and
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
#################################################################################
"""
# Seperatimg the text column from classes column.
"""
texts= data['text']
tags = data['class']

"""
# Removing Non-arabic characters.
"""
import re
for i in range(458197):   
    texts[i] = re.sub('[a-zA-Z0-9_]|#|http\S+', '', texts[i])
#################################################################################
"""
# Taking only a small fraction of the data, as neither resources of my machine, 
  nor the resources of google colab are enough to process the whole data.
  
# Taking a fraction from the data results in very lower accuracy, but there is
  no other choice.
"""
data = data.sample(frac = 0.15)

# Setting the size of training sample as 60% of the fraction subset.
train_size = int(len(data) * .6)
# Setting training data.
train_posts = data['text'][:train_size]
train_tags = data['class'][:train_size]
# Setting testing data
test_posts = data['text'][train_size:]
test_tags =  data['class'][train_size:]
###############################################################################
"""
# tokenizer class allows to vectorize a text corpus, by turning each text into 
  either a sequence of integers.
"""
tokenizer = Tokenizer(num_words=None,lower=False)
tokenizer.fit_on_texts(texts)
x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')
###############################################################################
"""
# Applying labelencoder on the target variable, to resolve categorical data
  problem.
"""
encoder = LabelEncoder()
encoder.fit(tags)
tagst=encoder.fit_transform(tags)

# Get the number of classes, that is 18.
num_classes = int((len(set(tagst))))
print((len(set(tagst))))
y_train = encoder.fit_transform(train_tags)
y_test = encoder.fit_transform(test_tags)
###############################################################################
"""
# In some versions of keras, using to categorical below would cause some error, so
  we can call the library as following to solve this problem:
      from tensorflow import keras
      from keras.utils import np_utils
"""
from tensorflow import keras
from keras.utils import np_utils

y_train= keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# I have tried to use PCA to reduce the number of features, beacause of availavle
# resources.

# from sklearn.decomposition import PCA
# pca = PCA(n_components = 100)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

num_labels = int(len(y_train.shape))
vocab_size = len(tokenizer.word_index) + 1

max_words=vocab_size
###############################################################################
"""
# Bulding the deep learning model:
"""
embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(1024, embedding_dim, input_length=max_words))
model.add(layers.Conv1D(512, 3, activation='relu'))
model.add(layers.Conv1D(128, 3, activation='relu'))
#model.add(Dropout(0.4)
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(32, activation='relu'))
#model.add(Dropout(0.5)
model.add(layers.Dense(num_classes, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(x_test, y_test),
                    batch_size=32)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

"""
# Another model using LSTM:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(max_words, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    verbose=True,
                    validation_data=(x_test, y_test),
                    batch_size=32)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
"""

"""
# Both models show bad accuracy results, although increasing the complexity of
  the model, applying feature engineering techniques, applying feature selection 
  and dimensionality reduction methods, and finally with different structures of
  deep learning model.

# Note: This is the most complex architecture and largest fraction of data, that 
        I can apply on my resources  
"""
#################################################################################