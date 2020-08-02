import LeModel
import SonaliModel
#import NzingaModel

from typing import Dict
from typing import Iterable

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array

import random
import numpy as np
import pandas as pd
import base64
import json
import cv2
import pickle

class DeployModel(object):
	def __init__(self):
		#Initialize all necessary models Le, Nzinga, Sonali
		self.leModel = LeModel.LeModel()
		self.sonaliModel = SonaliModel.SonaliModel()
		#self.nzingaModel = NzingaModel.NzingaModel()
		self.ensembleModel = pickle.load(open("EnsembleRandomForest.pickle","rb"))
		self.labels = pd.read_csv("labels.csv")
		self.results = []
	def predict(self, X:np.ndarray, names:Iterable[str] = None, meta: Dict = None):
		self.results.clear()		
		leResult = self.leModel.predict(X)		
		
		sonaliResult = self.sonaliModel.predict(X)
		self.results.append(sonaliResult[0])
		
		if(sonaliResult[0] == "Acrocarpus-fraxinifolius"):
			sonaliResult = self.labels[self.labels["SPECIES"]=="Acrocarpus fraxinifolius"]["ID"].values[0]
		else:
			sonaliResult = self.labels[self.labels["SPECIES"]==sonaliResult[0]]["ID"].values[0]

		EnsembleResult = self.ensembleModel.predict([[self.labels[self.labels["SPECIES"]==leResult[0]]["ID"].values[0], random.randrange(0, 46), sonaliResult]])
		
		self.results.append(leResult[0])		
		self.results.append(self.labels[self.labels["ID"]==EnsembleResult[0]]["SPECIES"].values[0])
		
		  		
		return self.results
				
