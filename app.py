import numpy as np
from flask import Flask, request, render_template, make_response
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

    # Xuat file output.csv
    pd.DataFrame(output).to_csv('./output.csv', index=False, index_label=False)
    f=open('./output.csv', 'r')
    response=make_response(f.read())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response
if __name__=="__main__":
    app.run(debug=True)