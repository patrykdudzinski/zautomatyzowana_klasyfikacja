#Magisterka
#Author: Patryk Dudziński
#ver. 0.0 03.2020

from flask import Flask, request, render_template, jsonify, redirect
from pprint import pprint
import numpy as np
import scipy 
import os
import urllib.request
import json
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import metrics as m

#config
app = Flask(__name__)  
UPLOAD_FOLDER = 'C:\\Users\\tries\\OneDrive\\Documents\\mgr\\application\\app\\static\\assets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#router indexu
@app.route('/')
def index():
    return render_template('index.html')

#@desc: zapisuje plik w upload folder 
#@param: (object)file - plik wejściowy
#@return: parameters_list - lista parametrów z pliku
@app.route('/', methods=['POST'])
def _send_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    data = arff.loadarff(app.config['UPLOAD_FOLDER']+'/'+filename)
    df = pd.DataFrame(data[0])
    parameters_list = data[1].names()
    filename = filename.split(".")
    return render_template('index.html', parameters_list=parameters_list, filename = filename[0])

#@desc: prognozuje atrybut na podstawie jego nazwy oraz zbioru
#@param: (string)filename - nazwa pliku
#@param: (string)attr - nazwa atrybutu (z listy po stronie widoku)
#@return: na razie nic XD
@app.route('/_forecast_data/', methods=['POST'])
def forecast_data():
    x_test = []
    x_train = []    
    filename = request.form['filename']
    attr = request.form['attr']
    data = arff.loadarff(app.config['UPLOAD_FOLDER']+'/'+filename)
    df = pd.DataFrame(data[0])   
    parameters_list = data[1].names()
    df[parameters_list[-1]] = pd.factorize(df[parameters_list[-1]])[0].astype(int)
    attr_index = parameters_list.index(attr)
    x = np.array(df)
    x_train ,x_test = train_test_split(x,test_size=0.7) 
    class_params_train = x_train[:,-1]
    x_train = np.delete(x_train, -1, 1)
    class_params_test = x_test[:, -1]
    x_test = np.delete(x_test, -1, 1) 
    class_names = data[1][data[1].names()[-1]][1]     
#    class_names, class_indices = np.unique(class_names, return_inverse=True)
    clf = LogisticRegression(random_state=0).fit(x_train, class_params_train)
    clf.predict(x_test[:-1, :])
    proba = clf.predict_proba(x_test[:-1, :])
    result_json = []
    for data_attr in proba:
        temp_dict = {
             'proba': data_attr,
        }        
        result_json.append(temp_dict) 
    score = clf.score(x_test, class_params_test)
#    score_result = {
#         'score_acc': score,
#    }        
#    result_json.append(score_result) 
    result = pd.Series(result_json).to_json(orient='values')
    return jsonify({'data': render_template('chart.html', class_names = class_names, probes = result, score_result = score)})
    

@app.route('/_get_file_data/', methods=['POST'])
def _get_file_data():
    filename = request.form['filename']
    data = arff.loadarff(app.config['UPLOAD_FOLDER']+'/'+filename)
    df = pd.DataFrame(data[0])
    parameters_list = data[1].names()
    return jsonify({'data': render_template('parameters_list.html', parameters_list=parameters_list, filename = filename)})


if __name__ == "__main__":
    app.run(debug=True)

    
    