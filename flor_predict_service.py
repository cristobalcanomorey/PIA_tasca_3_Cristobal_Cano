def predict_single(flor, model):
    # x = dv.transform([flor])
    y_pred = model.predict(flor)
    return y_pred 