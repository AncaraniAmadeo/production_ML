import pandas as pd
import xgboost 
from aux_func import data_preprocess, predict_optimized
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer


app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return 'Welcome All'


@app.route('/predict_file', methods=['POST'])
def predict_file_csv():
    
    data = pd.read_csv(request.files.get('file'), sep=',')
    print(data.columns.tolist())
    print(data)
    if 'Unnamed: 0' in data.columns:
        data.set_index(['Unnamed: 0'], inplace=True)
        data.index.names = ['']
    
    print(data)
    prediction = predict_optimized(data_preprocess(data))
    return 'The predicted value is ' + str(list(prediction))


@app.route('/predict',methods=['GET','POST'])
def predict():
    d = None
    if request.method == 'POST':
        
        d = request.form.to_dict()
    else:
        
        d = request.args.to_dict()

    data = pd.DataFrame([d.values()], columns=d.keys())
    data_processed = data_preprocess(data)
    prediction = predict_optimized(data_processed)
    
    return 'prediction is: ' + str(prediction)


if __name__ == '__main__':
    app.run()