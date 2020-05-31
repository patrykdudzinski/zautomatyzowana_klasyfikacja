#Praca magisterska
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn import metrics as m
from sklearn.preprocessing import LabelEncoder


#config
app = Flask(__name__)  
UPLOAD_FOLDER = 'C:\\Users\\tries\\OneDrive\\Documents\\mgr\\application\\app\\static\\assets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def train_model():
    
    return score


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
    try: 
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        data = arff.loadarff(app.config['UPLOAD_FOLDER']+'/'+filename)
        df = pd.DataFrame(data[0])
        parameters_list = data[1].names()
        class_names = data[1][data[1].names()[-1]][1] 
        df.columns = map(str.lower, df.columns)
        pprint('KOLUMNY WGRYWANEGO PLIKU')
        pprint(df.columns)
        if 'class' not in df.columns:
            code = 1
            class_names_length = 0
        else:
            code = 0
            class_names_length = len(set(df['class']))
            if(class_names_length > 10):
                code = 2
        filename = filename.split(".")
    except:
        code = 3
        class_names_length = 0
    return render_template('index.html', code = code, class_names_length=class_names_length, filename = filename[0])

#@desc: prognozuje atrybut na podstawie jego nazwy oraz zbioru
#@param: (string)filename - nazwa pliku
#@param: (string)attr - nazwa atrybutu (z listy po stronie widoku)
#@return: przekierowanie do template'u chart.js
#@return: class_names - tablica z nazwami klas (na potrzeby wyświetlenia na wykresie)
#@return: probes - wyniki dla poszczególnych próbek
#@return: score_result - dokładność modelu
@app.route('/_forecast_data/', methods=['POST'])
def forecast_data():
    x_test = []
    x_train = []  
    #Przygotowanie zestawu uczącego - odczytanie wartości z jsona oraz zadanego pliku#
    hyperparams_json = json.loads(request.form['hyperparams'])
    hyperparams = {}
    for param in hyperparams_json:
        temp_param = param.split(':')
        hyperparams[temp_param[0]] = temp_param[1]

    filename = request.form['filename']
    test_size_value = int(hyperparams['form_proportion'])/100
    data = arff.loadarff(app.config['UPLOAD_FOLDER']+'/'+filename)
    df = pd.DataFrame(data[0])   
    parameters_list = data[1].names()    
    df[parameters_list[-1]] = pd.factorize(df[parameters_list[-1]])[0].astype(int)
    newdf = df.select_dtypes(include=['int32','int64','float64'])
    newdf = newdf.dropna(axis='rows')
    pprint('KOLUMNY')
    pprint(df.columns)
    pprint(df.dtypes)   
    x = np.array(newdf)

    #pobierz indeks kolumny klasy
    newdf.columns = map(str.lower, newdf.columns)
    class_column_idx = newdf.columns.get_loc("class")

    #przygotuj zbiór testowy i uczący#
    x_train, x_test = train_test_split(x,test_size=test_size_value) 
    class_params_train = x_train[:,class_column_idx]
    params_index = set(class_params_train)
    x_train = np.delete(x_train, class_column_idx, 1)
    class_params_test = x_test[:, class_column_idx]
    x_test = np.delete(x_test, class_column_idx, 1) 
    
    class_names = data[1][data[1].names()[-1]][1] 
    if(hyperparams['form_method'] == '0'): #Regresja log
        clf = LogisticRegression(random_state=0, 
                                 penalty = hyperparams['form_lr_penalty'],
                                 tol=float(hyperparams['form_score']), 
                                 max_iter=int(hyperparams['form_iterations']), 
                                 solver=hyperparams['form_lr_algortithm'],
                                 dual=bool(int(hyperparams['form_lr_dual'])), 
                                 l1_ratio=float(hyperparams['form_lr_l1_ratio']), 
                                 C=float(hyperparams['form_lr_reg_strength']),
                                 fit_intercept=bool(hyperparams['form_lr_fit_intercept']),
                                 warm_start=hyperparams['form_lr_warm_start'])
    if(hyperparams['form_method'] == '1'): #KNN
        clf = KNeighborsClassifier(n_neighbors=3)
    if(hyperparams['form_method'] == '2'): #Bayes
        priors = hyperparams['form_nb_probas'].split(',')
        i = 0
        for item in priors:
            priors[i] = float(item)
            i = i+1

        clf = GaussianNB(priors = priors,
                             var_smoothing = float(hyperparams['form_nb_smooth']))
    if(hyperparams['form_method'] == '3'): #SVC
        clf = SVC(probability=hyperparams['form_svc_probability'], 
                  max_iter=int(hyperparams['form_iterations']), 
                  shrinking=hyperparams['form_svc_shrinking'], 
                  tol=float(hyperparams['form_score']), 
                  C=float(hyperparams['form_svc_reg_strength']),
                  kernel=hyperparams['form_svc_kernel'], 
                  degree=float(hyperparams['form_svc_deg']),
                  gamma = hyperparams['form_svc_gamma'],
                  decision_function_shape = hyperparams['form_svc_decision']
                 )
                  
    if(hyperparams['form_method'] == '4'): #Drzewo decyzyjne - KONIECZNIE TRZEBA ZROBIĆ MECHANIZM DO ZARZĄDZANIA HIPERPARAMETREM max_features
        if(hyperparams['form_bt_max_features'] == 'user'):
            max_features = float(hyperparams['form_bt_max_features_float'])
        else:
            max_features = hyperparams['form_bt_max_features']
        
        clf = tree.DecisionTreeClassifier(criterion=hyperparams['form_bt_criterion'], 
                                          splitter=hyperparams['form_bt_splitter'], 
                                          max_depth = int(hyperparams['form_bt_max_depth']),
                                          min_samples_split = int(hyperparams['form_bt_samples_split']),
                                          min_samples_leaf =int(hyperparams['form_bt_leaf_node']),
                                          max_features = max_features,
                                          min_weight_fraction_leaf = float(hyperparams['form_bt_weight_fraction']),
                                          max_leaf_nodes =int(hyperparams['form_bt_max_leaf_nodes']),
                                          min_impurity_decrease =float(hyperparams['form_bt_impurity']),
                                          ccp_alpha =float(hyperparams['form_bt_ccp_alpha'])
                                         )
    if(hyperparams['form_method'] == '5'): #sieć neuronowa MLP - ZALEŻNOŚCI DO OBKODOWANIA
        pprint(hyperparams)
        max_fun_value = 1500 if (hyperparams['form_mlp_max_fun'] == '') else hyperparams['form_mlp_max_fun']
        layers = hyperparams['form_mlp_hidden_layer'].split(',')
        i = 0
        for item in layers:
            layers[i] = int(item)
            i = i+1

        clf = MLPClassifier(solver=hyperparams['form_mlp_solver'], 
                            alpha=float(hyperparams['form_mlp_alpha']), 
                            activation=hyperparams['form_mlp_activation'], 
                            hidden_layer_sizes= layers, 
                            batch_size= int(hyperparams['form_mlp_batch_size']), 
                            learning_rate= hyperparams['form_mlp_learning_rate'], 
                            learning_rate_init= float(hyperparams['form_mlp_learning_rate_init']), 
                            power_t= float(hyperparams['form_mlp_learning_rate_powert']), 
                            momentum= float(hyperparams['form_mlp_momentum']), 
                            nesterovs_momentum= bool(hyperparams['form_mlp_nesterov']), 
                            n_iter_no_change= int(hyperparams['form_mlp_epochs']), 
                            epsilon= float(hyperparams['form_mlp_eps']), 
                            beta_1= float(hyperparams['form_mlp_exp1']), 
                            beta_2= float(hyperparams['form_mlp_exp2']), 
                            early_stopping= bool(hyperparams['form_mlp_early_stop']), 
                            shuffle= bool(hyperparams['form_mlp_shuffle']), 
                            warm_start= hyperparams['form_mlp_warm_start'], 
                            max_fun= int(max_fun_value), 
                            random_state=1)
    try:
        clf = clf.fit(x_train, class_params_train)
        clf.predict(x_test[:-1, :])
        proba = clf.predict_proba(x_test[:-1, :])
        result_json = []
        for data_attr in proba:
            temp_dict = {
                 'proba': data_attr,
            }        
            result_json.append(temp_dict) 
        score = round(clf.score(x_test, class_params_test), 2)
        result = pd.Series(result_json).to_json(orient='values')
        code = 0
    except:
        code = 2
        params_index = 0
        probes = 0
        result = 0
        score = 0
    return jsonify({'data': render_template('chart.html', code = code, class_names = params_index, probes = result, score_result = score)})
    

@app.route('/_get_file_data/', methods=['POST'])
def _get_file_data():
    filename = request.form['filename']
    data = arff.loadarff(app.config['UPLOAD_FOLDER']+'/'+filename)
    df = pd.DataFrame(data[0])
    parameters_list = data[1].names()
    return jsonify({'data': render_template('parameters_list.html', parameters_list=parameters_list, filename = filename)})


if __name__ == "__main__":
    app.run(debug=True)
 