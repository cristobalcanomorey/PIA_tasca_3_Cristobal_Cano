import pickle

from flask import Flask, request
from flor_predict_service import predict_single
from vista.response import json_response
# from sklearn.preprocessing import StandardScaler

app = Flask('flor-predict')

#KNN
with open('models/flor-knn.pck', 'rb') as f:
    knn, sc = pickle.load(f)

@app.route('/predict/knn', methods=['POST'])
def predict_w_knn():
    flor = [request.get_json()['flor']]
    flor_std = sc.transform(flor)
    prediction = predict_single(flor_std, knn)
    return json_response(float(prediction[0]),flor[0])


#Tree model
with open('models/flor-tree_model.pck', 'rb') as f:
    decision_trees = pickle.load(f)

@app.route('/predict/decision_trees', methods=['POST'])
def predict_w_decision_tree():
    flor = [request.get_json()['flor']]
    prediction = predict_single(flor, decision_trees)
    return json_response(float(prediction[0]),flor[0])


#Linear regression
with open('models/flor-lr.pck', 'rb') as f:
    lr, sc = pickle.load(f)

@app.route('/predict/linear_regression', methods=['POST'])
def predict_w_lr():
    flor = [request.get_json()['flor']]
    flor_std = sc.transform(flor)
    prediction = predict_single(flor_std, knn)
    return json_response(float(prediction[0]),flor[0])


#SVM
with open('models/flor-svm.pck', 'rb') as f:
    svm, sc = pickle.load(f)

@app.route('/predict/svm', methods=['POST'])
def predict_w_svm():
    flor = [request.get_json()['flor']]
    flor_std = sc.transform(flor)
    prediction = predict_single(flor_std, knn)
    return json_response(float(prediction[0]),flor[0])


if __name__ == '__main__':
    app.run(debug=True, port=8000)  