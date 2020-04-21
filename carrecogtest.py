# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:15:35 2020

@author: Shreyansh Satvik
"""
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from numpy import asarray
from os import listdir
import os
import sys
import csv
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle
DATADIR="G:/CAR dATASET/CARVSNONCARTEST"
CATEGORIES=["NONCAR","CARS"] 
for category in CATEGORIES:
    path=os.path.join(DATADIR,category) #paths to car or non car directory
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img) )
        plt.imshow(img_array)
        plt.show()
        break
    break
#fix shape
IMG_SIZE=128
new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array)
plt.show()    
##################
test_data=[]
def create_test_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)                 #paths to car or non car directory
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img) )
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                test_data.append([new_array,class_num])
            except Exception as e:
                pass
            

create_test_data()

#shuffling training data

random.shuffle(training_data)
X=[]
y=[]
for features ,label in training_data:
    X.append(features)
    y.append(label)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

"""

np.save("xtrain",X)
np.save("ytrain",y)
"""
pickle_out= open("X_traindata.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("Y_traindata.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
