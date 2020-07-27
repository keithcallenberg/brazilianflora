from typing import Dict
from typing import Iterable
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64
import json
import cv2
import pickle

class MyModel(object):
	def __init__(self):
		#Initialize all necessary models
		self.feature_extractor = VGG16(weights="imagenet", include_top = False) 
		self.model = pickle.load(open("Logistic.pickle","rb"))		
	def predict(self, X:np.ndarray, names:Iterable[str] = None, meta: Dict = None):
		#getting the image		
		img = np.array(X, dtype = np.uint8)
		#preprocess image before extracting features
		#resize the image to 224x224
		img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img =imagenet_utils.preprocess_input(img)
		#extracting features
		features = self.feature_extractor.predict(img, batch_size = 1)
		features = features.reshape((features.shape[0], 512*7*7))
		#return a prediction
		return self.model.predict(features)		
				
