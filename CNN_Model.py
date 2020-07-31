import numpy as np
import json
import pickle
from sift import create_features_BOW_img, sift_feature_extractor_img, kmean_BOW_img


class CNN(object):
    def __init__(self):
        # Initialize all necessary models
        self.model = pickle.load(open("sift_cnn.sav", "rb"))
        self.labels = json.load(open("label_encoded.json"))
        self.bow_dict = pickle.load(open("bow_dictionary.pkl", "rb"))

    def predict(self, image: np.ndarray, num_cluster=60):
        # getting the image
        img = np.array(image, dtype=np.uint8)

        # extracting features
        descriptor = sift_feature_extractor_img(img)
        bow = kmean_BOW_img(descriptor, num_cluster)
        features = create_features_BOW_img(descriptor, bow, num_cluster).reshape((1, -1))

        # return a prediction
        predict_probs = self.model.predict_proba(features)
        predict_list = np.argsort(predict_probs)[:, :-5 - 1:-1][0]
        predictions = [self.labels[str(idx)] for idx in predict_list]
        return predictions
