import os
from flask import Flask, request, jsonify

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

VN_LABEL_ARR = ['cam-duong', 'muong-bien', 'dau-co-bien', 'sa-sam', 'bang-vuong']
LABEL_ARR = [
    'Limnocitrus littoralis (Miq.) Swingle',
    'Ipomoea pes-caprae (L.) Sweet',
    'Canavalia cathartica Thouars',
    'Prenanthes sarmentosa Willd',
    'Launaea sarmentosa (Willd.) Alston',
    ]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error':'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error':'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
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

if __name__ == '__main__':
    app.run()