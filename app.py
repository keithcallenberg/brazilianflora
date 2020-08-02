import flask
import requests
from PIL import Image
import numpy as np
from flask import Flask, render_template,flash, url_for, redirect
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = {"jpg", "png", "jpeg"}
app = Flask(__name__, template_folder="template")
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=["GET", "POST"])
def home():
	if flask.request.method == "POST":
		'''
		if 'file' not in flask.request.files:
			print("No file part")
			return redirect(flask.request.url)
		'''
		file = flask.request.files["image"]
		if file.filename =='':
			print("No selected file")
			return render_template("index.html", result="NOT GOT IT")
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			fa = np.asarray(Image.open(file))
			payload = {}
			payload = {"data":{"tensor":{"shape":fa.shape, "values":fa.reshape(-1).tolist()}}}
			response = requests.post('http://172.17.0.1:5000/predict', json=payload)
			data = response.json()["data"]
			results = data['ndarray'] 	
			First_judge = "First Predict: " + results[0] + "\n"
			Second_judge = "Second Predict: " + results[1] + "\n"
			Final_judge = "Final Verdict: " + results[2]
			return render_template("results.html", First_judge= First_judge, Second_judge=Second_judge, Final_judge = Final_judge)
	return render_template("index.html")

@app.route('/results', methods=['GET','POST'])
def back2Main():
	if request.method == 'POST':
		return redirect(url_for('home'))
	return render_template('results.html')
app.run()
