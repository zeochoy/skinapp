import os
import glob
from flask import Flask
from flask import jsonify
from flask import request, render_template

from skinapp import app
from model.utils import *
from model.skinmodel import *

valid_mimetypes = ['image/jpeg', 'image/png']

@app.route('/')
def index():
    samples = glob.glob("%s/*" % app.config['SAMPLE_FOLDER'])
    return render_template('index.html', samples=samples)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'no file'}), 400
        # Image info
        img_file = request.files.get('file')
        img_name = img_file.filename
        mimetype = img_file.content_type
        # Return an error if not a valid mimetype
        if mimetype not in valid_mimetypes:
            return jsonify({'error': 'bad-type'})

        # Write image to static directory
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        img = open_image(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        # Run Prediction on the model
        res = get_predictions(img)

        # Delete image when done with analysis
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        return jsonify(res)
