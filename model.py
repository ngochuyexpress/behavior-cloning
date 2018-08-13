import csv
import cv2
import numpy as np
import tensorflow as tf

lines = []
with open("../data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn
from random import randint
batch_size = 80
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Randomly pick camera
                camera = randint(0,2)
                angle = float(batch_sample[3])
                # Force using side camera when angle is too small
                if abs(angle) < 0.01 and randint(30) != 0:
                    camera = randint(0,1) + 1
                name = '../data/IMG/'+batch_sample[camera].split('/')[-1]
                image = cv2.imread(name)
                # Change angle based on the camera
                if camera == 1:
                    angle += 0.3
                elif camera == 2:
                    angle -= 0.3
                # Randomly flipping the image so that it won't biased to left
                # turn
                if randint(0,1) == 1:
                    image = np.fliplr(image)
                    angle = -angle
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,24),(0,0))))
model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,3),strides=(2,3)))
model.add(Convolution2D(36, 4, 4, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss="mse", optimizer="adam")
model.fit_generator(train_generator,
        samples_per_epoch=len(train_samples), \
        validation_data=validation_generator,  \
        nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')
print("Done!")
