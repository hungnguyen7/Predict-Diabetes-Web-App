import numpy as np
from flask import Flask, request, render_template, make_response
from flask_restful import Api, Resource, reqparse
import pickle
import pandas as pd
app=Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # retriving values from form
    init_features=[int(x) for x in request.form.values()]
    final_features=[np.array(init_features)]
    # return str(final_features).strip('[]')
    prediction=model.predict(final_features)
    return render_template('index.html', prediction_text= prediction)

import io
import csv
@app.route('/uploadcsv')
def uploadCSV():
    return render_template('uploadCSV.html')

def tranformData(arrayToTransform):
    "Chuyen doi du lieu tu Yes, No sang 0, 1"
    for i in range(len(arrayToTransform)):
        if(arrayToTransform[i]=='Yes' or arrayToTransform[i]=='Male'):
            arrayToTransform[i]=1
        elif(arrayToTransform[i]=='No' or arrayToTransform[i]=='Female'):
            arrayToTransform[i]=0


@app.route('/predictCSV', methods=['POST'])
def predictCSV():
    # Lay file cua nguoi dung upload
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode('UTF-8-sig'), newline=None)
    csv_input = csv.reader(stream)

    # header cho file output.csv
    header=next(csv_input)
    header.append('Class')
    output=[header]

    # Du doan tung dong du lieu
    for row in csv_input:
        tranformData(row)
        # print(row)
        if(type(row[1])==int): # khong xu li dong header
            prediction=model.predict([np.array(row)])
            row.append(list(prediction))
            output.append(row)
    # print(output)

    # print(stream)

    # Xuat file result.csv
    pd.DataFrame(output).to_csv('./result.csv', index=False, index_label=False)
    f=open('./result.csv', 'r')
    response=make_response(f.read())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

API=Api(app)
@app.route('/api/predict', methods=['POST'])
def apiPredict():
    parser=reqparse.RequestParser(bundle_errors=True)
    parser.add_argument('age', required=True, help='Age cannot be blank')
    parser.add_argument('gender', required=True, help='Gender cannot be blank')
    parser.add_argument('polyuria', required=True, help='Polyuria cannot be blank')
    parser.add_argument('polydipsia', required=True, help='Polydipsia cannot be blank')
    parser.add_argument('suddenWeightLoss', required=True, help='Sudden Weight Loss cannot be blank')
    parser.add_argument('weakness', required=True, help='Weakness cannot be blank')
    parser.add_argument('polyphagia', required=True, help='Polyphagia cannot be blank')
    parser.add_argument('genitalThrush', required=True, help='Genital Thrush cannot be blank')
    parser.add_argument('visualBlurring', required=True, help='Visual Blurring cannot be blank')
    parser.add_argument('itching', required=True, help='Itching cannot be blank')
    parser.add_argument('irritability', required=True, help='Irritability cannot be blank')
    parser.add_argument('delayedHealing', required=True, help='Delayed Healing cannot be blank')
    parser.add_argument('partialParesis', required=True, help='Partial Paresis cannot be blank')
    parser.add_argument('muscleStiffness', required=True, help='Muscle Stiffness cannot be blank')
    parser.add_argument('alopecia', required=True, help='Alopecia cannot be blank')
    parser.add_argument('obesity', required=True, help='Obesity cannot be blank')

    args=parser.parse_args()
    X=np.fromiter(args.values(), dtype=int)
    output={'Prediction': model.predict([X])[0]}
    return output, 200
if __name__=="__main__":
    app.run()