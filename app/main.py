import os
from flask import Flask, request, jsonify

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

VN_LABEL_ARR = ['bang-vuong', 'cam-duong', 'dau-co-bien', 'muong-bien', 'sa-sam']
LABEL_ARR = [
    'Launaea sarmentosa (Willd.) Alston',
    'Limnocitrus littoralis (Miq.) Swingle',
    'Canavalia cathartica Thouars',
    'Ipomoea pes-caprae (L.) Sweet',
    'Prenanthes sarmentosa Willd',
    ]

VN_LABEL_ARR_FLOWER = ['bang-vuong', 'cam-duong', 'dau-co-bien', 'sa-sam']
LABEL_ARR_FLOWER = [
    'Launaea sarmentosa (Willd.) Alston',
    'Limnocitrus littoralis (Miq.) Swingle',
    'Canavalia cathartica Thouars',
    'Prenanthes sarmentosa Willd',
    ]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(type_predict):
    if request.method =='POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error':'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error':'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor, type_predict)
            print(prediction[0])
            return jsonify({
            'status': 'success',
            'accuracy': str(prediction[0]),
            'predict': str(prediction[1]),
            'vn-label': VN_LABEL_ARR[prediction[1]],
            'label': LABEL_ARR[prediction[1]],
        })     
        except:
            return jsonify({
            'error': 'Prediction error!'
        })

@app.route('/predict/leaf', methods=['POST'])
def predict_leaf():
    return predict('leaf')

@app.route('/predict/flower', methods=['POST'])
def predict_flower():
    return predict('flower')

@app.route('/predict/fruit', methods=['POST'])
def predict_fruit():
    return predict('fruit')

@app.route('/predict/overall', methods=['POST'])
def predict_overall():
    return predict('overall')

if __name__ == '__main__':
    app.run()