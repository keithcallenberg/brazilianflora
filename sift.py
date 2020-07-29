# Nzinga Eduardo Code Labs Brazilian Flora
# Extract features using SIFT and build Bag-of-Words using Kmeans

import numpy as np
import os
import cv2
import pickle
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# extract image features using SIFT


def sift_feature_extractor(list_images):
    img_descriptors = []
    orb = cv2.ORB_create(nfeatures=500)

    for image in list_images:
        _, descriptor = orb.detectAndCompute(image, None)
        img_descriptors.append(descriptor)

    return img_descriptors


# Create Bag-of-Words dictionary using Kmeans
def kmean_BOW(descriptors, num_cluster):
    bow_dict = []

    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(descriptors)

    bow_dict = kmeans.cluster_centers_

    if not os.path.isfile('bow_dictionary.pkl'):
        pickle.dump(bow_dict, open('bow_dictionary.pkl', 'wb'))

    return bow_dict

# Create feature vector from Bag-of-Words
def create_features_BOW(img_descriptors, bow, num_cluster):

    X_features = []

    for i in range(len(img_descriptors)):
        features = np.array([0] * num_cluster)

        if img_descriptors[i] is not None:
            distance = cdist(img_descriptors[i], bow)
            min_dist = np.argmin(distance, axis=1)

            for j in min_dist:
                features[j] += 1

        X_features.append(features)

    return X_features
