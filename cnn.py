# Nzinga Eduardo Code Labs Brazilian Flora
# Build CNN model and training

import os
import pickle
import argparse

from preprocessing import label_img
from sift import create_features_BOW, sift_feature_extractor, kmean_BOW

import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers


plt.style.use('fivethirtyeight')

arg = argparse.ArgumentParser()
arg.add_argument("-td", "--inputdata", required=True, help="path to training data")
args = vars(arg.parse_args())

path = args["inputdata"]

training_data, label, label_encoded = label_img(path)

label = np.array(label)
num_classes = len(label_encoded)
print("num classes: ", num_classes)

print("training data len: ", len(training_data))

img_desctiptors = sift_feature_extractor(training_data)
print("descriptors: ", img_desctiptors)

descriptors_list = []


for descriptor in img_desctiptors:
    if descriptor is not None:
        for item in descriptor:
            descriptors_list.append(item)

num_cluster = 60
bow = kmean_BOW(descriptors_list, num_cluster)  # create bag-of-words

X_features = create_features_BOW(img_desctiptors, bow, num_cluster)  # get features vector
X_features = np.array(X_features)

print("X_features.shape: ", X_features.shape)
print("label.shape: ", label.shape)

X_train, X_test, y_train, y_test = train_test_split(X_features, label, test_size=0.2, random_state=1)
#model_cnn = sklearn.svm.SVC(C=30, random_state=0, probability=True)

#model_cnn.fit(X_train, y_train)
#pickle.dump(model_cnn, open('sift_cnn.sav', 'wb'))


# normalize features
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[1], 3)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[1], 3)

y_train = np.array(to_categorical(y_train))
y_test = np.array(to_categorical(y_test))

print('x_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)



# build the model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(X_train.shape[1], X_train.shape[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# training the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train,
                 batch_size=150,
                 epochs=10,
                 validation_split=0.2
                 )

# save model
model.save('sift_cnn.h5')
pickle.dump(model, open('sift_cnn.sav', 'wb'))

# evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# visualization
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('cnn_accuracy.png')

# Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('cnn_loss.png')

"""
