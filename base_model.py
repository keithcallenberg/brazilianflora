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

#train_paths - all images
#train_labels- all lables

def get_image_features(image_paths, shape):
    height, width, channels = shape
    array = np.zeros((len(image_paths), height, width, channels))
    for i, path in enumerate(image_paths):
        im = Image.open(path)
        as_array = np.asarray(im)
        if len(as_array.shape) == 2: # convert to 3 channels (required input to VGG)
            as_array = skimage.color.gray2rgb(as_array)
        resized = skimage.transform.resize(as_array, shape)
        array[i,:,:] = resized
        if (i + 1) % 300 == 0:
            print(f'Finished loading {i+1} samples')
    return array

categorical_mapping = { label: i for i, label in enumerate(set(train_labels)) }

def labels_to_np_array(labels, mapping):
    return np.array([mapping[label] for label in labels])

full_training_X = get_image_features(train_paths, IMAGE_SHAPE)
full_training_y = labels_to_np_array(train_labels, categorical_mapping)


train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(full_training_X, full_training_y, 
                                                    test_size=1 - train_ratio, random_state=SEED, shuffle =True)

# validation is now 10% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(full_training_X, full_training_y,random_state=SEED, shuffle =True,
                                                test_size=test_ratio/(test_ratio + validation_ratio)) 




model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150,150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(46, activation='sigmoid'))

EPOCHS, BATCH_SIZE = 15, 32


model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4), 
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
y_pred = model.predict_classes(X_test)
print(classification_report(y_test, y_pred))


