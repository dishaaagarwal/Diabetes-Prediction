# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 12:52:10 2020

@author: This PC
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load the Random Forest CLassifier model
filename = 'random_classification_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result1.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)

#classifier=pickle.load(open('diabetes-prediction-model.pkl','rb'))


        