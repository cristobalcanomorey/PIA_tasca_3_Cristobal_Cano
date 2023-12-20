import pickle

from flask import Flask, request
from flor_predict_service import predict_single
from vista.response import json_response
from sklearn import datasets

iris = datasets.load_iris()
clases = iris.target_names

app = Flask('flor-predict')

def get_prediction(flor, modelo):
    prediction = predict_single(flor, modelo)
    return 'Iris ' + clases[int(prediction[0])]

#KNN
with open('models/flor-knn.pck', 'rb') as f:
    knn, sc = pickle.load(f)

@app.route('/predict/knn', methods=['POST'])
def predict_w_knn():
    flor = [request.get_json()['flor']]
    flor_std = sc.transform(flor)
    prediction = get_prediction(flor_std,knn)
    return json_response(prediction,flor[0], 'KNN')


#Tree model
with open('models/flor-tree_model.pck', 'rb') as f:
    decision_trees = pickle.load(f)

@app.route('/predict/decision_trees', methods=['POST'])
def predict_w_decision_tree():
    flor = [request.get_json()['flor']]
    prediction = get_prediction(flor,decision_trees)
    return json_response(prediction,flor[0], 'Decision trees')


#Linear regression
with open('models/flor-lr.pck', 'rb') as f:
    lr, sc = pickle.load(f)

@app.route('/predict/linear_regression', methods=['POST'])
def predict_w_lr():
    flor = [request.get_json()['flor']]
    flor_std = sc.transform(flor)
    prediction = get_prediction(flor_std,lr)
    return json_response(prediction,flor[0], 'Linear regression')


#SVM
with open('models/flor-svm.pck', 'rb') as f:
    svm, sc = pickle.load(f)

@app.route('/predict/svm', methods=['POST'])
def predict_w_svm():
    flor = [request.get_json()['flor']]
    flor_std = sc.transform(flor)
    prediction = get_prediction(flor_std,svm)
    return json_response(prediction,flor[0], 'SVM')


if __name__ == '__main__':
    app.run(debug=True, port=8000)  