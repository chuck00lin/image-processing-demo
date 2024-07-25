from flask import Flask, request, send_file
from PIL import Image
import io

app = Flask(__name__)

def process_image(image):
    # Your image processing algorithm goes here
    # This is a placeholder that simply inverts the image
    return Image.eval(image, lambda x: 255 - x)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    img = Image.open(image)
    processed_img = process_image(img)

    img_io = io.BytesIO()
    processed_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # app.run(debug=True)
