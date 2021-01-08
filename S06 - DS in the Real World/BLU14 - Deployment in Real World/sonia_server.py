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
from predict_1 import predict_1


########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open(os.path.join('data', 'columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('data', 'pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('data', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions


def check_request(request):
    """
        Validates that our request is well formatted

        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    if "observation_id" not in request.keys():
        error = "Field 'observation_id' missing from request: {}".format(request)
        return False, error
    elif 'data' not in request.keys():
        error = "Field `data` missing from request: {}".format(request)
        return False, error
    else:
        return True, ""


def check_valid_column(observation):
    """
        Validates that our observation only has valid columns

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """

    valid_columns = set(columns)
    keys = set(observation.keys())

    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    if len(keys - valid_columns) > 0:
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error

    if len(keys - valid_columns) == 0:
        return True, ""


def check_values(observation):
    valid_category_map = {
        "age": list(range(10, 100)),
        "sex": [0, 1],
        "cp": [0, 1, 2, 3],
        "trestbps": list(range(50, 200)),
        "fbs": [0, 1],
        "restecg": [0, 1, 2],
        # "oldpeak": Interval(0,8),
        "ca": [0, 1, 2, 3],
        "thal": [0, 1, 2]
    }
    for key, item in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in item:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in item]))
                return False, error
        elif key not in observation:
            error = "{} is not in the observation".format(key)
            return False, error
    return True, ""


def oldpeak(observation):
    """
        Validates that observation contains valid hour value

        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """
    old_peak_value = observation['oldpeak']

    if old_peak_value < 0 or old_peak_value > 8:
        error = "Field {} `oldpeak` is not between 0 and 8".format(old_peak_value)
        return False, error
    return True, ""


# End input validation functions
########################################

########################################
# Begin webserver stuffdef check_

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()

    response = dict()
    request_checked_1, request_checked_2 = check_request(obs_dict )
    if request_checked_1:
        observation = obs_dict ['data']
        result_valid_columns_1, result_valid_columns_2 = check_valid_column(observation)
        if result_valid_columns_1:
            cv_1, cv_2 = check_values(observation)
            cv_3, cv_4 = oldpeak(observation)
            if cv_1 == True and cv_3 == True:
                pre= predict_1(obs_dict)
                # pr = dict()
                response['observation_id'] = obs_dict['observation_id']
                response['prediction'] = pre['prediction']
                response['probability'] = pre['probability']
            elif cv_1 == False:
                # pr = dict()
                response['observation_id'] = obs_dict['observation_id']
                response['error'] = cv_2
            elif cv_3 == False:
                # pr = dict()
                response['observation_id'] = obs_dict['observation_id']
                response['error'] = cv_4

        elif ~result_valid_columns_1:
            # pr = dict()
            response['observation_id'] = obs_dict['observation_id']
            response['error'] = result_valid_columns_2
            # response = pr
    elif ~request_checked_1:
        # pr = dict()
        response['observation_id'] = None
        response['error'] = request_checked_2
        # response = pr
    return jsonify(response)

    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'prediction': bool(prediction), 'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)

    
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


    
if __name__ == "__main__":
    app.run()
