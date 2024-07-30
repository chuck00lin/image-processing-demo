from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from segmentation import process_image
from connected_components import process_image_q2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
SAMPLE_FOLDER = 'dataset/question-1'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return jsonify({'filename': file.filename})

@app.route('/sample')
def get_sample():
    source = os.path.join(SAMPLE_FOLDER, 'sample_source.png')
    groundtruth = os.path.join(SAMPLE_FOLDER, 'sample_groundtruth.png')
    return jsonify({
        'source': '/sample_image/sample_source.png',
        'groundtruth': '/sample_image/sample_groundtruth.png'
    })

@app.route('/sample_image/<path:filename>')
def sample_image(filename):
    return send_from_directory(SAMPLE_FOLDER, filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
    result1, result2 = process_image(image_path)
    return jsonify({
        'result1': f'/static/{result1}',
        'result2': f'/static/{result2}'
    })

@app.route('/process_q2', methods=['POST'])
def process_q2():
    data = request.json
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
    result, analysis = process_image_q2(image_path)
    return jsonify({
        'result': f'/static/{result}',
        'analysis': analysis
    })

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
    # app.run(debug=True)
