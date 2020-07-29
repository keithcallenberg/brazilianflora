import numpy as np
import cv2
import requests
import json
import base64
from json import JSONEncoder		
payload = {}
#reading the image, getting the image
#for i in range(10):
fa = cv2.imread('image.jpg')
#flat the image to send as json file
payload = {"data":{"tensor":{"shape":fa.shape, "values":fa.reshape(-1).tolist()}}}
#make a post request
response = requests.post('http://0.0.0.0:5000/predict', json=payload)
#result returns
data = response.json()["data"]
print(data["ndarray"][0])
'''
shape = data["tensor"]["shape"]
output = data["tensor"]["values"]
output = np.array(output, dtype=np.uint8).reshape(shape[0], shape[1], shape[2])

cv2.imshow("TEST", output)
cv2.waitKey()
'''
