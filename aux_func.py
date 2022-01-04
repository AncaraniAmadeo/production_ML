# primero todas las librerías por bloques
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import pickle
from random import seed
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, \
                            silhouette_score, recall_score, precision_score, make_scorer, \
                            roc_auc_score, f1_score, precision_recall_curve

from sklearn.metrics import accuracy_score, roc_auc_score, \
                            classification_report, confusion_matrix


from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

import re

def data_preprocess(data):
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    import pickle
    from category_encoders import TargetEncoder
    from sklearn.impute import SimpleImputer

    data = data.drop_duplicates()
    if 'C_SEV' in data.columns:
        y = np.where(data['C_SEV']==2,0,1)
         
        data['C_SEV'] = y
    
    data.C_VEHS.fillna('UU', inplace=True)
    
    numerical_vars = ['C_YEAR', 'C_MNTH', 'C_WDAY', 'C_HOUR', 'P_ID', 'V_ID', 'V_YEAR', 'P_AGE'] 
    unknown = ['N', 'NN', 'NNN', 'NNNN', 'U', 'UU', 'UUU', 'UUUU']
    
    for i in numerical_vars:
        for j in unknown:
            data[i + '_' + j] = np.where(data[i] == j, 1, 0)
    
    for i in numerical_vars:
        data[i] = pd.to_numeric(data[i], errors='coerce', downcast='integer')
    
    data = data.loc[:, (data != 0).any(axis=0)]
    
    categorical_vals = [c for c in data if c not in numerical_vars][1:10]
    
    data['V_TYPE'] = np.where(data['V_TYPE'].isin(['05', '06', '07', '08', '18', '19', '20', '21']), 'truck', data['V_TYPE'])
    data['V_TYPE'] = np.where(data['V_TYPE'].isin(['09', '10', '11']), 'bus', data['V_TYPE'])
    data['V_TYPE'] = np.where(data['V_TYPE'].isin(['14', '16', '17', '22']), 'bike', data['V_TYPE'])
    data['V_TYPE'] = np.where(data['V_TYPE'].isin(['01', '23']), 'car', data['V_TYPE'])
    
    data = data[~data['V_TYPE'].isin(['QQ', 'UU'])]
    
    data['P_PSN'] = np.where(data['P_PSN'].isin(['11', '12', '13']), 'front', data['P_PSN'])
    data['P_PSN'] = np.where(data['P_PSN'].isin(['21', '22', '23']), 'middle', data['P_PSN'])
    data['P_PSN'] = np.where(data['P_PSN'].isin(['31', '32', '33']), 'back', data['P_PSN'])
    data['P_PSN'] = np.where(data['P_PSN'].isin(['96']), 'occupant', data['P_PSN'])
    data['P_PSN'] = np.where(data['P_PSN'].isin(['98', '97']), 'unsecure', data['P_PSN'])
    data['P_PSN'] = np.where(data['P_PSN'].isin(['99']), 'pedestrian', data['P_PSN'])
    
    data['P_PSN'] = np.where(data['P_PSN'].isin(['UU', 'NN', 'QQ']), 'other', data['P_PSN'])
    
    data['P_SAFE'] = np.where(data['P_SAFE'].isin(['02', '09', '11', '12']), 'safe', data['P_SAFE'])
    data['P_SAFE'] = np.where(data['P_SAFE'].isin(['01', '10', '13']), 'unsafe', data['P_SAFE'])
    
    data['P_SAFE'] = np.where(data['P_SAFE'].isin(['QQ', 'NN', 'UU']), 'other', data['P_SAFE'])
    
    data['P_USER'] = np.where(data['P_USER'].isin(['1', '2']), 'vehicle', data['P_USER'])
    data['P_USER'] = np.where(data['P_USER'].isin(['3']), 'pedestrian', data['P_USER'])
    data['P_USER'] = np.where(data['P_USER'].isin(['4']), 'bicyclist', data['P_USER'])
    data['P_USER'] = np.where(data['P_USER'].isin(['5']), 'motorcyclist', data['P_USER'])
    data['P_USER'] = np.where(data['P_USER'].isin(['U']), 'other', data['P_USER'])
    
    data = data[~data['P_SEX'].isin(['N'])]
    
    data['V_ANT'] = data['C_YEAR'] - data['V_YEAR']
    data.drop(columns=['V_YEAR'], inplace=True)
    
    data = data[(data['V_ANT'] > 0)|(data['V_ANT'].isna())]
    
    data['V_ANT'] = np.where((data['V_ANT'] > 25), 25, data['V_ANT'])
    
    data['C_RCFG'] = np.where((~data['C_RCFG'].isin(['02','01','UU','03','QQ','05'])), 'OT', data['C_RCFG'])
    
    data = data[~data['C_WTHR'].isin(['Q'])]
    data['C_WTHR'] = np.where((data['C_WTHR'].isin(['5','6','7'])), 'O', data['C_WTHR'])
    
    data['C_RSUR'] = np.where((data['C_RSUR'].isin(['9','8','7', '6'])), 'O', data['C_RSUR'])
    
    for i in ['C_MNTH_UU', 'C_WDAY_U', 'C_HOUR_UU', 'P_ID_NN', 'P_ID_UU', 'V_ID_UU', 'P_AGE_NN']:
        if i in data.columns.tolist():
            data.drop(i, axis=1, inplace=True)
            
    data.dropna(subset=['V_ID', 'P_ID'], inplace=True)
    
    columns_with_na = ['C_MNTH', 'C_WDAY', 'C_HOUR', 'P_AGE', 'V_ANT']
    
    data_to_fillna = data[columns_with_na]
    
    model_list = ['SimpleImputer_trained', 'TargetEncoder_model']
    unpickeled_models = []

    for model in list(model_list):
        ''' use after the model is pickled '''
        imputer = open(model + '.pickle',"rb")
        unpickeled_models.append(pickle.load(imputer))
        
        imputer.close()
    
    if data_to_fillna.isnull().any().any() == True:
        data_to_fillna = unpickeled_models[0].transform(data_to_fillna)
    
    data[columns_with_na] = data_to_fillna
    
    data['C_HOUR_SIN'] = np.sin(2 * np.pi * data['C_HOUR']/24)
    data['C_HOUR_COS'] = np.cos(2 * np.pi * data['C_HOUR']/24)
    
    data['C_WDAY_SIN'] = np.sin(2 * np.pi * data['C_WDAY']/7)
    data['C_WDAY_COS'] = np.cos(2 * np.pi * data['C_WDAY']/7)
    
    data['C_MNTH_SIN'] = np.sin(2 * np.pi * data['C_MNTH']/12)
    data['C_MNTH_COS'] = np.cos(2 * np.pi * data['C_MNTH']/12)
    
    data.drop(columns=['C_HOUR', 'C_WDAY', 'C_MNTH'], inplace=True) 
    
    not_treated_columns = [c for c in data.columns if c not in ['C_HOUR_SIN', 'C_HOUR_COS', 'C_MNTH_SIN',
                                         'C_MNTH_COS', 'C_WDAY_SIN', 'C_WDAY_COS', 'C_YEAR', 'C_AGE', 'V_ANT',
                                          'C_SEV', 'V_YEAR_NNNN', 'V_YEAR_UUUU', 'P_AGE_UU', 'V_ID', 'P_ID', 'P_AGE']]
    
    
    data.drop(columns= ['V_ID', 'P_ID'], inplace=True)
    for i in ['C_YEAR', 'C_VEHS', 'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'V_TYPE',
       'P_SEX', 'P_AGE', 'P_PSN', 'P_SAFE', 'P_USER', 'C_SEV', 'V_YEAR_NNNN',
       'V_YEAR_UUUU', 'P_AGE_UU', 'V_ANT', 'C_HOUR_SIN', 'C_HOUR_COS',
       'C_WDAY_SIN', 'C_WDAY_COS', 'C_MNTH_SIN', 'C_MNTH_COS']:
        if i not in data.columns:
            data[i] = 0
            
    data = unpickeled_models[1].transform(data)
    
    
    return data



def predict_optimized(data):
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    import pickle

    if 'C_SEV' in data.columns:
        data = data.drop(columns=['C_SEV'])
    
    best_model_grid = []
    classifier_f = open('grid_search_gb_smote' +'.pickle',"rb")
    best_model_grid.append(pickle.load(classifier_f))
    classifier_f.close()

    y_pred_best = (best_model_grid[0].predict_proba(data)[:,1] >= 0.00056324113).astype(int)

    return y_pred_best


