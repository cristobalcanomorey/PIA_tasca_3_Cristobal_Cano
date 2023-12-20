from flask import jsonify

def json_response(prediction,flor, using):
    result = {
        'flower': flor,
        'prediction': prediction,
        'using': using
    }

    return jsonify(result)