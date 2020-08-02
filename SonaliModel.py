from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
class SonaliModel(object):
	def __init__(self):
		self.model = keras.models.load_model("ImageDataGenerator_model.h5")
		self.labels = pd.read_csv("sonaliLabels.csv")	
	def predict(self, X:np.ndarray):
		IMAGE_SIZE = (150, 150)
		#convert image to numpy array
		image = np.array(X, dtype = np.uint8)
		#resize image to IMAGE_SIZE
		image = cv2.resize(image, IMAGE_SIZE, interpolation= cv2.INTER_AREA)
		#reshape the image dimension based on the CNN model's input
		image = image.astype("float")
		image = np.reshape(image, [1, 150, 150,3])
		results = self.labels[self.labels["ID"] == self.model.predict_classes(image)[0]]["SPECIES"].values	 
		return results
