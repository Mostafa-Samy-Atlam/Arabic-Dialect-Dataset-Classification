# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:28:23 2022

@author: Eng. Mostafa
"""
import pandas as pd
from numpy import set_printoptions
import requests
import string
import sklearn
import time

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