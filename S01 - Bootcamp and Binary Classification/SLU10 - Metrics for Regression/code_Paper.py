# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:35:40 2019

@author: mh.olyaei
"""
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, merge, Activation, ZeroPadding2D, concatenate
from keras.layers import AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json
#import shutil
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib
import os                                                                                                                                    
import tensorflow as tf
from PIL import Image
#from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
#from plot_history import plot_history

#from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from keras import regularizers

#%%
# mini batch size  
bath_Size =32

numberClass = 3

NumberEpoch =200

img_Channels = 3

#%%

# load data train

path1 = 'Dataset/train'
path2 = 'Dataset/trainResized'

listing = os.listdir(path1)
num_samples = np.size(listing)
print (num_samples,'num_samples')

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((32 , 32))
    IMMGG = img.copy()
    gray = img.convert('L')
    gray.save(path2 + '\\' + file, "png")
imlist = os.listdir(path2)

im1 = np.array(Image.open(path2 + '\\' + imlist[0]))
m,n = im1.shape[0:2]
imagenbr = len(imlist)


imMatrix = np.array([np.array(Image.open(path2 + '\\' + im2)).flatten() for im2 in imlist], 'f')

lable_train = np.ones((num_samples,) , dtype=int)
lable_train[0:112]=0 #CLL
lable_train[113:251]=1 #FL
lable_train[252:373]=2 #MCL

data, lable = shuffle(imMatrix , lable_train , random_state = 2)
train_data = [data , lable]


#%%
# The data, split between train and test sets:


datanum=len(data)
test_num=round(datanum*0.25)
train_num= (datanum-test_num)


X_test = data[0:test_num]
Y_test = lable[0:test_num]

X_train = data[test_num:]
Y_train = lable[test_num:]


print('X_train shape:', X_train.shape)
print(train_num, 'train samples')
print('X_test shape:', X_test.shape)
print(test_num, 'test samples')



X_train = X_train.reshape(X_train.shape[0], 32 , 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32 , 32, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, numberClass)
Y_test  = np_utils.to_categorical(Y_test, numberClass)



#%%  load validation data

# val data 
path1 = 'Dataset/validation'
path2 = 'Dataset/validation_resize'

listing = os.listdir(path1)
num_samples = np.size(listing)
print (num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((32 , 32))
    IMMGG = img.copy()
    gray = img.convert('L')
    gray.save(path2 + '\\' + file, "png")
imlistVal = os.listdir(path2)

im1 = np.array(Image.open(path2 + '\\' + imlistVal[0]))
m,n = im1.shape[0:2]
imagenbr = len(imlistVal)

imMatrixVal = np.array([np.array(Image.open(path2 + '/' + im2)).flatten() for im2 in imlistVal], 'f')

lable_val = np.ones((num_samples,) , dtype=int)
lable_val[0:112]=0 #CLL
lable_val[113:251]=1 #FL
lable_val[252:373]=2 #MCL

val_data, val_lable = shuffle(imMatrixVal[0:900] , lable_val , random_state = 2)
validation_data = [val_data , val_lable]

(X_val, y_val) = (validation_data[0], validation_data[1])

XX_test = X_val.copy()

X_val = X_val.reshape(X_val.shape[0], 32 , 32, 1)

X_val = X_val.astype('float32')


X_val /= 255

Y_val  = np_utils.to_categorical(y_val, numberClass)


#%%

#from keras.preprocessing.image import ImageDataGenerator
##datagen = ImageDataGenerator(zca_whitening=True)
#
#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=10,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True)
#
#datagen.fit(X_train)
#
#datagen.fit(X_val)

#%%
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.5,
        zoom_range=0.5,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)



train_datagen.fit(X_train)

test_datagen.fit(X_test)


#%%noise
#
#from skimage.util import random_noise
#
#X_train = random_noise(X_train, mode='s&p',amount=0.01)
#
#
#def add_salt_pepper_noise(X_imgs):
#    # Need to produce a copy as to not modify the original image
#    X_imgs_copy = X_imgs.copy()
#    row, col, _ = X_imgs_copy[0].shape
#    salt_vs_pepper = 0.2
#    amount = 0.004
#    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
#    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
#    
#    for X_img in X_imgs_copy:
#        # Add Salt noise
#        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
#        X_img[coords[0], coords[1], :] = 1
#
#        # Add Pepper noise
#        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
#        X_img[coords[0], coords[1], :] = 0
#    return X_imgs_copy
#  
#X_train = add_salt_pepper_noise(X_train)

#%%
def FireModule(s_1x1, e_1x1, e_3x3, name):
    """
        Fire module for the SqueezeNet model. 
        Implements the expand layer, which has a mix of 1x1 and 3x3 filters, 
        by using two conv layers concatenated in the channel dimension. 
        Returns a callable function
    """
    def layer(x):
        squeeze = keras.layers.Convolution2D(s_1x1, 1, 1, activation='relu', init='he_normal', name=name+'_squeeze',  kernel_regularizer=regularizers.l2(0.01))(x)
        squeeze = keras.layers.BatchNormalization(name=name+'_squeeze_bn')(squeeze)
        # Set border_mode to same to pad output of expand_3x3 with zeros.
        # Needed to merge layers expand_1x1 and expand_3x3.
        expand_1x1 = keras.layers.Convolution2D(e_1x1, 1, 1, border_mode='same', activation='relu', init='he_normal', name=name+'_expand_1x1',  kernel_regularizer=regularizers.l2(0.01))(squeeze)
        # expand_1x1 = BatchNormalization(name=name+'_expand_1x1_bn')(expand_1x1)

        # expand_3x3 = ZeroPadding2D(padding=(1, 1), name=name+'_expand_3x3_padded')(squeeze)
        expand_3x3 = keras.layers.Convolution2D(e_3x3, 3, 3, border_mode='same', activation='relu', init='he_normal', name=name+'_expand_3x3',  kernel_regularizer=regularizers.l2(0.01))(squeeze)
        # expand_3x3 = BatchNormalization(name=name+'_expand_3x3_bn')(expand_3x3)

        expand_merge = keras.layers.concatenate([expand_1x1, expand_3x3], axis=3, name=name+'_expand_merge')
        return expand_merge
    return layer

#%% layers

input_image = keras.layers.Input(shape=(32,32,1))

#noise_layer=keras.layers.GaussianNoise(0.1)(input_image)


padd_conv1 = keras.layers.ZeroPadding2D(padding=(2, 2))(input_image)

conv1 =keras.layers.Conv2D(32, (3, 3), activation='relu',strides=(1, 1), kernel_regularizer=regularizers.l2(0.01))(padd_conv1)

maxpool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

###########
fire2 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire2')(maxpool1)
fire3 = FireModule(s_1x1=16, e_1x1=32, e_3x3=32, name='fire3')(fire2)

########
gu=keras.layers.GaussianNoise(0.01)(fire3)

padd_conv4 = keras.layers.ZeroPadding2D(padding=(1, 1))(gu)

conv4 = keras.layers.Conv2D(3, (1, 1), activation='relu', subsample=(2, 2), init='he_normal', name='conv2', kernel_regularizer=regularizers.l2(0.01))(padd_conv4)

Averagepool4 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3')(conv4)

out_dropout = Flatten()(Averagepool4)
drop=Dropout(0.1)(out_dropout)

#gu=keras.layers.GaussianNoise(0.01)(drop)

out1=keras.layers.Dense(3, kernel_regularizer=regularizers.l2(0.01))(drop)
drop2=Dropout(0.1)(out1)
softmax = keras.layers.Activation('softmax', name='softmax')(drop2)
model = Model(input=input_image, output=[softmax])



#%%
#model.set_weights(weights)
# Tiiiiime !!


import datetime
start = datetime.datetime.now()

# Compile model

#  Stochastic gradient descent optimizer
#sgd = optimizers.SGD(lr=0.001, decay=1e-9, momentum=0.9)
adamax =keras.optimizers.Adamax(lr=0.001)

#adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#A metric is a function that is used to judge the performance of your mode

from keras import regularizers
#model.add(kernel_regularizer=regularizers.l2(0.01))

model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy']) 
#history = model.fit(X_train, Y_train , validation_split=0.33, bath_Size, NumberEpoch)

####   DECAY TEEEEST
from keras.callbacks import LearningRateScheduler
import math
#NumberEpoch = 1000
#def step_decay(epoch):
#   initial_lrate = 0.001
#   drop = 0.1
#   epochs_drop = 200
#   lrate = initial_lrate * math.pow(drop,  
#           math.floor((1+epoch)/epochs_drop))
#   return lrate

def step_decay(epoch):
    print('---',epoch)
    init_lr = 0.001
    drop = 0.1
    epochs_drop = 200
    lrate = init_lr*math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))


loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
#callbacks_list = [loss_history, lrate]

from keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True

##########


#EarlyStoppingByAccuracy=EarlyStoppingByAccuracy()



callbacks_list = [TerminateOnBaseline(monitor='acc', baseline=1.00),loss_history, lrate]


##### END

#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.

history = model.fit(X_train, Y_train, bath_Size, NumberEpoch,callbacks=callbacks_list, verbose=1, validation_data=(X_val,Y_val))

#y_prediction = model.fit(X_train, y_train).predict(X_test)



#
#plot_history(history)


#history = model.fit(X_train, Y_train, bath_Size, NumberEpoch, verbose=1, validation_split=0)

#record the training / validation loss / accuracy at each epoch
loss_acc=history.history
#print(history.history)
loss = history.history['loss']
#print (loss)
acc = history.history['accuracy']
#print(acc)



model.summary()

# model.save('my_Trained_model.h5')
print('Ok, Model Saved!!')


###  END Time
end = datetime.datetime.now()
traintime = end - start
print('Total training time: ', str(traintime))



#%%  plot
##   plot
#history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()




#%%
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, Y_test, batch_size=32)
print('test loss, test acc:', results)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on val data')
results = model.evaluate(X_val, Y_val, batch_size=32)
print('val loss, val acc:', results)


# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on Train data')
results = model.evaluate(X_train, Y_train, batch_size=32)
print('train loss, train acc:', results)



#%%
######################  save    ################
# model.save_weights('weights_model4.h5')
# model.save('model_model4.h5')


#%%
##########################################    PREDICT #######################################
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 

#%%  load prediction data

# val data 
path1 = 'Dataset/img4predection'
path2 = 'Dataset/img4predection-resize'

#
#path1 = 'E:/0hatami duc/pre2019'
#path2 = 'Dataset/img4predection-resize'
#



listing = os.listdir(path1)
num_samples = np.size(listing)
print (num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((32 , 32))
    IMMGG = img.copy()
    gray = img.convert('L')
    gray.save(path2 + '\\' + file, "png")
imlistpre = os.listdir(path2)

#im1 = np.array(Image.open(path2 + '\\' + imlistVal[0]))
#m,n = im1.shape[0:2]
#imagenbr = len(imlistpre)

imMatrix_pre = np.array([np.array(Image.open(path2 + '/' + im2)).flatten() for im2 in imlistpre], 'f')

lable_pre = np.ones((num_samples,) , dtype=int)
lable_pre[0:4]=0 #Erosion
lable_pre[5:9]=1 #Polyp
lable_pre[10:14]=2 #Ulcer

pre_data, pre_lable = shuffle(imMatrix_pre[0:15] , lable_pre , random_state = 2)
predection_data = [pre_data , pre_lable]

(X_pre, y_pre) = (predection_data[0], predection_data[1])

XX_pre = X_pre.copy()

X_pre = X_pre.reshape(X_pre.shape[0], 32 , 32, 1)

X_pre = X_pre.astype('float32')


X_pre /= 255

Y_pre  = np_utils.to_categorical(y_pre, numberClass)

#
#
#X_pre = X_val
#y_pre = Y_val




#%%

pre = model.predict(X_test)

############

import numpy as np
labels_predection = np.argmax(pre, axis=1) 
labels_original = np.argmax(Y_test, axis=1) 
y=labels_original
###########  confusion_matrix for prediction

cm_pre=confusion_matrix(labels_predection ,labels_original)
print(cm_pre)



#%%


from sklearn.metrics import classification_report

labels_predection = np.argmax(pre, axis=1) 
labels_original = np.argmax(Y_test, axis=1) 
target_names = ['CLL', 'FL', 'MCL']

mesure= classification_report(labels_predection, labels_original, target_names=target_names)

print(mesure)


cm = confusion_matrix(labels_predection ,labels_original)


#%%




#%%
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 
########################################## ########################################## 

#%%

from sklearn.metrics import plot_confusion_matrix
class_names = ['CLL', 'FL', 'MCL']

lp=np.array(labels_predection)

lo = np.array(labels_original)
#import plot_confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_predection ,labels_original)
np.set_printoptions(precision=2)
plt.matshow(cnf_matrix)
plt.title('Confusion matrix, without normalization')
plt.colorbar()
# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()





#%%
from sklearn.metrics import plot_confusion_matrix
lp=np.array(labels_predection)

lo = np.array(labels_original)

# Compute confusion matrix
cnf_matrix = confusion_matrix(lp ,lo)
np.set_printoptions(precision=2)
class_names = ['0', '1', '2']
# Plot non-normalized confusion matrix
plt.matshow(cnf_matrix)
plt.title('Confusion matrix, without normalization')
plt.colorbar()
# plt.figure()
# plot_confusion_matrix( lp ,lo, classes=class_names,title='Confusion matrix, without normalization')
# plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

cm = confusion_matrix(labels_predection ,labels_original)
# or
#cm = np.array([[1401,    0],[1112, 0]])

plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

#%%



# from sklearn.metrics import confusion_matrix

# pred = model.predict(X_val)

# conf = confusion_matrix(labels_original, pred)

# import seaborn as sns
# sns.heatmap(cm, annot=True)

#%%
cm = confusion_matrix(labels_predection ,labels_original)

plt.imshow(cm, cmap='binary')
from pandas import DataFrame
df_cm = DataFrame(cm) #, index=columns, columns=columns)
import seaborn as sn
ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)


#%%




columns = ['CLL', 'FL', 'MCL']
#get pandas dataframe
df_cm = DataFrame(cm)# , index=columns, columns=columns)
    #colormap: see this and choose your more dear
cmap = 'PuRd'
# pretty_plot_confusion_matrix(df_cm, cmap=cmap)




#%%

import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_cm)








#%%  OOKKKK


columns = ['CLL', 'FL', 'MCL']
annot = True;
cmap = 'Oranges';
fmt = '.2f'
lw = 0.5
cbar = False
show_null_values = 2
pred_val_axis = 'y'
#size::
fz = 12;
figsize = [9,9];
if(len(Y_val) > 10):
    fz=9; figsize=[14,14];
# plot_confusion_matrix_from_data(labels_original, labels_predection, columns,
#   annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


#%%

#%%
##########################################   RETRAIN #######################################
##########################################   RETRAIN #######################################
##########################################   RETRAIN #######################################
##########################################   RETRAIN #######################################

##########################################   RETRAIN #######################################
##########################################   RETRAIN #######################################
##########################################   RETRAIN #######################################

##########################################   RETRAIN #######################################
#%%
####
#


# del model


#weights=model.load_weights('weights_asli.h5')

# model=load_model('model_model1000.h5')

NumberEpoch=1100
ie=1000



#%%
#model.set_weights(weights)
# Tiiiiime !!


import datetime
start = datetime.datetime.now()





# Compile model

#  Stochastic gradient descent optimizer
sgd = optimizers.SGD(lr=0.001, decay=1e-9, momentum=0.5)

#A metric is a function that is used to judge the performance of your mode

from keras import regularizers
#model.add(kernel_regularizer=regularizers.l2(0.01))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
#history = model.fit(X_train, Y_train , validation_split=0.33, bath_Size, NumberEpoch)

####   DECAY TEEEEST
from keras.callbacks import LearningRateScheduler
import math

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.01
   epochs_drop = 200
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

#def step_decay(epoch):
#    print('---',epoch)
#    init_lr = 0.001
#    drop = 0.1
#    epochs_drop = 50
#    lrate = init_lr*math.pow(drop,math.floor((1+epoch)/epochs_drop))
#    return lrate

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))


loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
#callbacks_list = [loss_history, lrate]

from keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True

##########


#EarlyStoppingByAccuracy=EarlyStoppingByAccuracy()



callbacks_list = [TerminateOnBaseline(monitor='val_acc', baseline=0.99),loss_history, lrate]

history = model.fit(X_train, Y_train, bath_Size, NumberEpoch,callbacks=callbacks_list, verbose=1, validation_data=(X_val,Y_val),initial_epoch=ie)

##### END

#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.

#history = model.fit(X_train, Y_train, bath_Size, NumberEpoch,callbacks=callbacks_list, verbose=1, validation_data=(X_val,Y_val))

#y_prediction = model.fit(X_train, y_train).predict(X_test)



#
#plot_history(history)


#history = model.fit(X_train, Y_train, bath_Size, NumberEpoch, verbose=1, validation_split=0)

#record the training / validation loss / accuracy at each epoch
loss_acc=history.history
#print(history.history)
loss = history.history['loss']
#print (loss)
acc = history.history['accuracy']
#print(acc)
#
#

#model.summary()

# model.save('my_Trained_model.h5')
print('Ok, Model Saved!!')


###  END Time
end = datetime.datetime.now()
traintime = end - start
print('Total training time: ', str(traintime))



#%%
##   plot
#history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()



#%%
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, Y_test, batch_size=32)
print('test loss, test acc:', results)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on val data')
results = model.evaluate(X_val, Y_val, batch_size=32)
print('val loss, val acc:', results)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on train data')
results = model.evaluate(X_train, Y_train, batch_size=32)
print('val loss, val acc:', results)





#%%
######################  save    ################
# model.save_weights('weights_model1000.h5')
# model.save('model_model1000.h5')






from keras.utils import plot_model

plot_model(model, to_file='0GPD_Model.png', show_shapes=True)


























