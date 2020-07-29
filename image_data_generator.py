import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
import skimage.color
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras import layers
from keras import models
import glob
import code
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from PIL import Image
import csv
import random
import pickle

## file:train_test_val_data.npz
def load_data(filename):
    data = np.load(filename, mmap_mode='r')
    x_train = data['a']
    x_test = data['b']
    
    y_train = data['c']
    y_test = data['d']
    
    x_val = data['e']
    y_val = data['f']
    return x_train, x_test, y_train, y_test, x_val, y_val

x_train, x_test, y_train, y_test, x_val, y_val = load_data('train_test_val_data.npz')

IMAGE_SHAPE = (150, 150, 3)
vgg_conv = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE)
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
for layer in vgg_conv.layers:
    print(f'Is {layer.__class__.__name__} traininable? {layer.trainable}')
    
    
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)


from keras.preprocessing.image import ImageDataGenerator
train_gen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,    
)
train_gen.fit(x_train)

augmented_model = keras.models.Sequential()
augmented_model.add(vgg_conv)
augmented_model.add(keras.layers.Flatten())
augmented_model.add(keras.layers.Dense(1024, activation='relu'))
augmented_model.add(keras.layers.Dropout(0.5))
augmented_model.add(keras.layers.Dense(46, activation='softmax'))
augmented_model.summary()

BATCH_SIZE =32
EPOCHS =10
from sklearn.metrics import classification_report
augmented_model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
history = augmented_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

y_pred = augmented_model.predict_classes(x_test)
print(classification_report(y_test, y_pred))
