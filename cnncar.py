# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:18:08 2020

@author: Shreyansh Satvik
"""
import pickle
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import metrics



import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot 

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


###########LOADING DATASET##################################

X_train=pickle.load(open("X_traindata.pickle","rb"))
Y_train=pickle.load(open("Y_traindata.pickle","rb"))

##########PLOT THE IMAGE#############################

index=1258
plt.imshow(X_train[index])

#########PREPROCESSING THE IMAGE#########################

x_train=X_train/255
model=Sequential()
"""
X = tf.placeholder(tf.float32,name="X",shape=[6061,128,128,3])
Y = tf.placeholder(tf.float32,name="Y",shape=[6061])
sess = tf.Session()
X= sess.run(X,feed_dict={X:x_train})
Y= sess.run(Y,feed_dict={Y:Y_train}) 
    
# close the session 
sess.close()    
###############APPLYING CNN DEEP NET####################

def carmodel(input_shape):
    X_input=input(input_shape)
    X=ZeroPadding2D((4,4),input_shape=(128,128,3))
    X=Conv2D(40,(9,9),strides=(1,1),name='conv0')(X)
    X=BatchNormalization(axis=3,name='bn0')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2),name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model=Model(inputs=X_input,outputs=X,name='CarModel')
    return model
CARMODEL=carmodel(X.shape[1:])

CARMODEL.compile(optimizer="adam",loss="mean_squared_error",metrices=["accuracy"])
CARMODEL.fit(x_train,y_train,epochs=5500,batch_size=32)
"""
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(BatchNormalization(axis=3,name='bn0'))
model.add(Conv2D(16, (3, 3),strides=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=3,name='bn1'))

model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(BatchNormalization(axis=3,name='bn2'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=3,name='bn3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer="adam",metrics=["accuracy"])
model.fit(x_train,Y_train,epochs=10,batch_size=32) 

####################LOADDING TEST DATA#######################

X_test=pickle.load(open("X_testdata.pickle","rb"))
Y_test=pickle.load(open("Y_testdata.pickle","rb"))
x_test=X_test/255

score = model.evaluate(x_test, Y_test, batch_size=32,verbose=1,sample_weight=None)
print("Test Accuracy:" + str(score[1]))


