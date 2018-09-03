# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D

correction = 0.2
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = 'data/IMG/'+os.path.split(batch_sample[0])[1]
                name_left = 'data/IMG/'+os.path.split(batch_sample[1])[1]
                name_right = 'data/IMG/'+os.path.split(batch_sample[2])[1]
                #Load images from center, left and right cameras.
                image_center = cv2.imread(name_center)
                image_left = cv2.imread(name_left)
                image_right = cv2.imread(name_right)
                #Load steering angles corresponding to the three cameras  
                angle_center = float(batch_sample[3])
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                images.extend([image_center,image_left,image_right])
                angles.extend([angle_center,angle_left,angle_right])
                
                #Flip three images above and add to the dataset
                image_center_flip = np.fliplr(image_center)
                image_left_flip = np.fliplr(image_left)
                image_right_flip = np.fliplr(image_right)
                #Load steering angles corresponding to the three cameras  
                angle_center_flip =  -angle_center
                angle_left_flip = -angle_left
                angle_right_flip = -angle_right
                images.extend([image_center_flip,image_left_flip,image_right_flip])
                angles.extend([angle_center_flip,angle_left_flip,angle_right_flip])


            # trim image to only see section with road
            X_train = np.asarray(images)
            y_train = np.asarray(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Build network 
model = Sequential()
# Set up cropping2D layer
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(65,320,3)))
#Set 5 convolutional layers
model.add(Conv2D(24,5,5,subsample=(2,2),activation = "relu")) 
model.add(Conv2D(36,5,5,subsample=(2,2),activation = "relu")) 
model.add(Conv2D(48,5,5,subsample=(2,2),activation = "relu")) 
model.add(Conv2D(64,3,3,activation = "relu")) 
model.add(Conv2D(64,3,3,activation = "relu")) 

#Flatten layer
model.add(Flatten())
#Dropout layer
#Set 3 fully connected layer
model.add(Dense(100))
#Dropout layer
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dropout(0.25))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
                    samples_per_epoch = 6*len(train_samples),
                    validation_data=validation_generator, 
                    nb_val_samples = 6*len(validation_samples),
                    nb_epoch=10, verbose = 1)

model.save('model.h5')



import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()