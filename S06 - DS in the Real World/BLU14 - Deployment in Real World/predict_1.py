import pandas as pd
import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
with open(os.path.join('data', 'columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('data', 'pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('data', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)

def predict_1(request):
    new_obs_dict = request['data']
    obs = pd.DataFrame([new_obs_dict], columns=columns)
    obs = obs.astype(dtypes)
    pred_proba = pipeline.predict_proba(obs)
    if pred_proba[0][0]>pred_proba[0][1]:
        prediction = False
    else:
        prediction = True
        response_1 = dict()
        response_1["observation_id"] = request["observation_id"]
        response_1['age'] =  request['data']['age']
        response_1['sex'] =  request['data']['sex']
        response_1['cp'] =  request['data']['cp']
        response_1['trestbps'] =  request['data']['trestbps']
        response_1['fbs'] =  request['data']['fbs']
        response_1['restecg'] =  request['data']['restecg']
        response_1['oldpeak'] =  request['data']['oldpeak']
        response_1['ca'] =  request['data']['ca']
        response_1['thal'] =  request['data']['thal']
        response_1['prediction'] = prediction
        response_1['probability'] = pred_proba[0][1]
    return response_1