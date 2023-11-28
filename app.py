from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
model = load_model('model/model.h5')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/blog")
def blog():
    return render_template("blog1.html")

@app.route("/tester")
def tester():
    return render_template("tester.html")

@app.route("/produk")
def produk():
    return render_template("produk.html")

@app.route("/hubungi")
def hubungi():
    return render_template("hubungi.html")    


@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        # Cek apakah file telah dipilih
        if 'file' not in request.files:
            return render_template('deteksi.html', message='File belum dipilih')

        file = request.files['file']

        # Cek apakah file kosong
        if file.filename == '':
            return render_template('deteksi.html', message='File belum dipilih')

        # Simpan file ke direktori tertentu (opsional)
        file_path = 'histori/' + file.filename
        file.save(file_path)

        # Proses file menggunakan model
        img = Image.open(file_path)
        img = np.array(img.resize((94, 55)))
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)

        # Hapus file yang diunggah setelah diproses (opsional)
        # import os
        # os.remove(file_path)

        # Tampilkan hasil prediksi di halaman web
        # actual_class = 'normal'  # sesuaikan dengan kelas yang sesuai dengan gambar
        predicted_class = 'Normal' if pred[0] > 0.5 else 'Katarak'

        # Konversi gambar ke format yang dapat ditampilkan di HTML
        img_str = image_to_base64(img)

        return render_template('deteksi.html', message=f'Prediksi: {predicted_class}',
                                 predicted_class=predicted_class, img_str=img_str)

    return render_template('deteksi.html')


def image_to_base64(image):
    # Konversi gambar menjadi format base64 agar bisa ditampilkan di HTML
    img = Image.fromarray(image[0])
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = "data:image/png;base64," + \
        base64.b64encode(buffer.getvalue()).decode()
    return img_str


if __name__ == '__main__':
    app.run(debug=True)
