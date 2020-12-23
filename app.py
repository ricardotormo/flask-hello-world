from flask import Flask, render_template, request, url_for
import os   
app = Flask(__name__)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
import pickle

open_hi_hat = [-3.12911920e+02, -2.92649135e+01, -1.85740846e+02,  4.53350203e+00, 4.97865354e+00,  4.29631003e+01, -2.01392647e+01,  3.65025148e+01, -1.74142162e+01,  1.65687643e+01, -1.35218854e+01,  9.14118486e+00, -2.94986171e+01,  3.06986676e-01,  3.52033210e+00,  1.52007691e+01, 1.20489711e+01,  1.72260742e+00,  7.26698843e+00, -3.08770386e+00]

def loadModel():
    site_root = os.path.realpath(os.path.dirname(__file__))
    model = os.path.join(site_root, "data", "vocal_drum_classification.pkl")
    with open(model, 'rb') as f:
        clf = pickle.load(f)
    return clf

@app.route('/')
def hello_world():
    global open_hi_hat
    clf = loadModel()
    res = clf.predict([open_hi_hat])
    return res[0]
