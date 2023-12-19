from flask import jsonify
def json_response(prediction,flor):
    result = {
        'flor': flor,
        'prediction': prediction
    }

    return jsonify(result)