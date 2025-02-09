from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDE

alzhimers=pickle.load(open('alz.pkl','rb'))

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
@app.route('/first')
def first():
	return render_template('first.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            return render_template('result.html', prediction=predicted_class, image_url=url_for('static', filename='uploads/' + filename))

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)