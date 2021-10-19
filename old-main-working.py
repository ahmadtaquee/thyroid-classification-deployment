from flask import Flask, request, render_template, redirect, url_for
import pickle, gzip
import joblib
import numpy as np
import xgboost
from pymongo import MongoClient
from flask_pymongo import PyMongo
import pandas as pd
import sampling_data
from scipy.stats import yeojohnson
import warnings

warnings.filterwarnings('ignore')

pickled_model = pickle.load(open('random_forest_model_new.pkl', 'rb'))

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'Thyroid'
app.config[
    "MONGO_URI"] = 'mongodb+srv://thyroid:project@ahmad.5gjdl.mongodb.net/patient_database?retryWrites=true&w=majority'

mongo = PyMongo(app)


@app.route('/')
def home():
    online_uses = mongo.db.users.find({"online": True})
    return render_template('index.html', online_uses=online_uses)


@app.route('/predict', methods=['POST'])
def predict():
    db = mongo.db.patient_data

    age = float(request.form.get('age', False))
    TSH = float(request.form.get('TSH', False))
    T3 = float(request.form.get('T3', False))
    T4U = float(request.form.get('T4U', False))
    FTI = float(request.form.get('FTI', False))
    sex = float(request.form.get('sex', False))
    onthyroxine = float(request.form.get('onthyroxine', False))
    queryonthyroxine = float(request.form.get('queryonthyroxine', False))
    onantithyroidmedication = float(request.form.get('onantithyroidmedication', False))
    sick = float(request.form.get('sick', False))
    pregnant = float(request.form.get('pregnant', False))
    thyroidsurgery = float(request.form.get('thyroidsurgery', False))
    I131treatment = float(request.form.get('I131treatment', False))
    queryhypothyroid = float(request.form.get('queryhypothyroid', False))
    queryhyperthyroid = float(request.form.get('queryhyperthyroid', False))
    lithium = float(request.form.get('lithium', False))
    goitre = float(request.form.get('goitre', False))
    tumor = float(request.form.get('tumor', False))
    hypopituitary = float(request.form.get('hypopituitary', False))
    psych = float(request.form.get('psych', False))

    values = ({"age": [age],
               "TSH": [TSH], "T3": [T3], "T4U": [T4U], "FTI": [FTI],
               "sex": [sex],
               "onthyroxine": [onthyroxine], "queryonthyroxine": [queryonthyroxine],
               "onantithyroidmedication": [onantithyroidmedication],
               "sick": [sick], "pregnant": [pregnant], "thyroidsurgery": [thyroidsurgery],
               "I131treatment": [I131treatment],
               "queryhypothyroid": [queryhypothyroid], "queryhyperthyroid": [queryhyperthyroid],
               "lithium": [lithium], "goitre": [goitre], "tumor": [tumor],
               "hypopituitary": [hypopituitary],
               "psych": [psych]})

    df_transform = pd.DataFrame.from_dict(values)
    # print(df_transform)

    # print("after transformation\n")

    df_transform.age = df_transform.age ** (1 / 2)
    # print("After transformation: ", df_transform.age)

    df_transform.TSH = np.log1p(df_transform.TSH)
    # print(df_transform.tsh_level)

    df_transform.T3 = df_transform.T3 ** (1 / 2)
    # print(df_transform.t3_level)

    df_transform.T4U = np.log1p(df_transform.T4U)
    # print(df_transform.t4u_level)

    df_transform.FTI = df_transform.FTI ** (1 / 2)
    # print(df_transform.fti_level)

    data_dict = df_transform.to_dict()
    # print(df_transform.to_dict())


    my_data = db.insert_one(data_dict)
    print(my_data)

    arr = np.array([[age,
                     TSH, T3, T4U, FTI,
                     sex,
                     onthyroxine, queryonthyroxine,
                     onantithyroidmedication,
                     sick, pregnant, thyroidsurgery,
                     I131treatment,
                     queryhypothyroid, queryhyperthyroid,
                     lithium, goitre, tumor,
                     hypopituitary,
                     psych]])
    print(arr)

    pred = pickled_model.predict(arr)[0]
    print(pred)

    if pred == 0:
        res_Val = "Hyperthyroid"
        print(res_Val)
    elif pred == 1:
        res_Val = "Hypothyroid"
        print(res_Val)
    else:
        res_Val = 'Negative'
        print(res_Val)

    return render_template('index.html', prediction_text='Patient has {}'.format(res_Val))


if __name__ == '__main__':
    app.run(port=5000)
