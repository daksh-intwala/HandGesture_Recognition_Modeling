from flask import Flask, render_template, request
from IPython.display import HTML
from base64 import b64encode
import cv2
import js2py
from util.capture_user_video import CaptureVideo
from util.extract_keyponts import ExtractKeypoints
from util.prediction import Predict
import os

app = Flask(__name__,template_folder='templates')

@app.route('/')
def start():
    print("Service is running! Navigate to /home")
    return "Service is running! Navigate to /home"

@app.route('/home', methods = ['POST','GET'])
def index():
    return render_template('try.html')


@app.route('/predict-your-gesture')
def predict():
    prd = ExtractKeypoints('/Users/dakshintwala/HGR_Application/src/77777.mp4')
    path = prd.make_csv()
    return path
    

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3000)