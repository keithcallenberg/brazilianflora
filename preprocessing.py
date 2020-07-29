# Nzinga Eduardo Code Lab, Brazilian Flora
# Loads images and converts to numpy array

import os
import glob
import cv2
import numpy as np


def load_image(img_path):
    return cv2.imread(img_path)


def label_img(path_to_data):
    data_train = []
    data_label = []

    labels_list = os.listdir(path_to_data)

    # Encode label to integer id
    label_encoded = dict(zip(labels_list, np.arange(len(labels_list))))

    for label in labels_list:
        for img in glob.glob(path_to_data + "/" + label + "/*"):
            image_arr = load_image(img)
            label_image = label_encoded[label]
            data_train.append(image_arr)
            data_label.append(label_image)
    return data_train, data_label, label_encoded
