import os
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.misc import imread
from scipy.misc import imresize
from keras.optimizers import SGD, Adam, Adamax


root_dir = os.path.abspath('.')
data_dir = ''

train = pd.read_csv(os.path.join(data_dir, 'train/train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test/test.csv'))


temp = []
for img_name in train.ID:
    img_path = os.path.join(data_dir, 'train/Train', img_name)
    img = imread(img_path)
    img = imresize(img, (64,64))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

train_x = np.stack(temp)

temp = []
for img_name in test.ID:
    img_path = os.path.join(data_dir, 'test/Test', img_name)
    img = imread(img_path)
    img = imresize(img, (64,64))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

test_x = np.stack(temp)

train_x = train_x / 255.
test_x = test_x / 255.
train.Class.value_counts(normalize=True)

lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)


epochs = 15
batch_size = 128

model = Sequential()
model.add(Convolution2D(32, (3, 3),  padding='valid',activation='relu', input_shape=(64,64,3)))

model.add(Convolution2D(16, (3, 3), padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(8, (3, 3), padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))




model.summary()
model.load_weights('weights.h5')
model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-9, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1)

model.save_weights('weights.h5')
pred = model.predict_classes(test_x)
pred = lb.inverse_transform(pred)

test['Class'] = pred
test.to_csv('sub.csv', index=False)

