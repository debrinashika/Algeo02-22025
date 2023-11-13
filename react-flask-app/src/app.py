from flask import Flask, render_template, request, send_from_directory, session
from werkzeug.utils import secure_filename
from tqdm import tqdm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import cv2
import time
import os
from multiprocessing import Pool
import multiprocessing
import numpy as np
import imageprocess as pros
from os.path import basename
import shutil

def create_pdf(top_images, destination_folder,time):
    pdf_path = os.path.join(destination_folder, 'result.pdf')

    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.drawString(400, 720 - 10, f'Total Elapsed Time: {time:.5f} seconds')
    for i, (image_path, similarity_score) in enumerate(top_images):
        c.drawImage(image_path, 20, 720 - (i%4 + 1) * 150, width=150, height=150)
        c.drawString(200, 720 - (i%4 + 1) * 150 + 75, f'Similarity: {similarity_score:.5f}')

        if(i%4==3):
            c.showPage()

    c.save()

    return pdf_path

def htgduaelmnt (mat,hasil):
    for i in range(len(mat)):
        for j in range(len(mat)-1):
            x = mat[i][j][0]
            y = mat[i][j+1][0]
            hasil[x-1][y-1] += 1

def contrast(mat):
    i, j = np.indices(mat.shape)
    return np.sum(mat * (i - j)**2)

def homogeneity(mat):
    i, j = np.indices(mat.shape)
    return np.sum(mat / (1 + (i - j)**2)) 

def entropy(mat):
    non_zero_indices = np.where(mat != 0)
    nonzero_elements = mat[non_zero_indices]
    entropy = -np.sum(nonzero_elements * np.log(nonzero_elements))
    return entropy

def similarity (mat1,mat2):
    bagi = (np.dot(mat1, mat2) / (np.linalg.norm(mat1)*np.linalg.norm(mat2)))*100
    return bagi

def compare(gambar1,gambar2):
    # start_time = time.time()  # Waktu mulai proses
    # occurence matrix
    occurmat = np.zeros((256, 256), dtype=int)
    occurmat2 = np.zeros((256, 256), dtype=int)

    htgduaelmnt (gambar1,occurmat)
    htgduaelmnt (gambar2,occurmat2)

    occurmat = (occurmat + occurmat.T) / occurmat.sum()
    occurmat2 = (occurmat2 + occurmat2.T) / occurmat2.sum()

    vek1 = [contrast(occurmat),homogeneity(occurmat),entropy(occurmat)]
    vek2 = [contrast(occurmat2),homogeneity(occurmat2),entropy(occurmat2)]

    sim = similarity(vek1,vek2)
    # end_time = time.time()  # Waktu selesai proses
    # elapsed_time = end_time - start_time  # Hitung lama waktu proses

    # print(f"Lama waktu proses pemrosesan: {elapsed_time:.2f} detik")
    return sim

def process_image_color(image_path, gambar1, results):
    gambar2 = cv2.imread(image_path)

    similarity_score = pros.compare_color(gambar1, gambar2)
    results.append((image_path, similarity_score))

def main_process_color(gambar1_path, destination_folder):
    gambar1_matrix = cv2.imread(gambar1_path)
    gambar1_matrix = pros.ubahwarna(gambar1_matrix)
    files = os.listdir(destination_folder)
    total_files = len(files)

    with tqdm(total=total_files, desc="Processing Images") as progress:
        start_time = time.time()
        num_processes = 4
        pool = Pool(processes=num_processes)
        manager = multiprocessing.Manager()
        results = manager.list()

        for file in files:
            if file != '':
                image_path = os.path.join(destination_folder, file)
                pool.apply_async(process_image_texture, args=(image_path, gambar1_matrix, results))

        pool.close()
        pool.join()

        print(results)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        top_images = sorted_results[:5]  # Ambil 10 sementara deh
        print(top_images)
        end_time = time.time()
        elapsed_time = end_time - start_time

    return top_images, elapsed_time

def calculate_similarity(features1, features2):
    # Hitung similarity antara dua vektor fitur
    # Misalnya, menggunakan cosine similarity
    dot_product = sum(a * b for a, b in zip(features1, features2))
    norm1 = sum(a ** 2 for a in features1) ** 0.5
    norm2 = sum(b ** 2 for b in features2) ** 0.5

    similarity = dot_product / (norm1 * norm2 + 1e-10)
    return similarity

def process_image_texture(image_path, gambar1, results):
    gambar2 = cv2.imread(image_path)

    similarity_score = compare(gambar1, gambar2)
    results.append((image_path, similarity_score))

def main_process_texture(gambar1_path, destination_folder):
    gambar1_matrix = cv2.imread(gambar1_path)
    gambar1_matrix = pros.ubahbw(gambar1_matrix)
    files = os.listdir(destination_folder)
    total_files = len(files)

    with tqdm(total=total_files, desc="Processing Images") as progress:
        start_time = time.time()
        num_processes = 4
        pool = Pool(processes=num_processes)
        manager = multiprocessing.Manager()
        results = manager.list()

        for file in files:
            if file != '':
                image_path = os.path.join(destination_folder, file)
                pool.apply_async(process_image_texture, args=(image_path, gambar1_matrix, results))

        pool.close()
        pool.join()

        print(results)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        top_images = sorted_results[:5]  # Ambil 10 sementara deh
        print(top_images)
        end_time = time.time()
        elapsed_time = end_time - start_time

    return top_images, elapsed_time

app = Flask(__name__, static_folder='static')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'datasave')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    gambar1 = request.files['gambar1']
    folder = request.files.getlist('folder[]')
    destination_folder = app.config['UPLOAD_FOLDER']

    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)

    os.makedirs(destination_folder)

    if folder:
        for uploaded_file in folder:
            if uploaded_file.filename != '':
                file_path = os.path.join(destination_folder, secure_filename(uploaded_file.filename))
                uploaded_file.save(file_path)
                print(f'Saving file: {file_path}')
        if request.form.get('mode') == 'texture':
            pros.ubahbwfolder(destination_folder)
    else:
        if request.form.get('mode') == 'texture':
            return render_template('error.html', message='No folder selected for texture mode.')

    gambar1_path = os.path.join(destination_folder, secure_filename(gambar1.filename))
    gambar1.save(gambar1_path)

    mode = request.form.get('mode', 'color')  # Mode default: color

    if mode == 'color':
        top_images, elapsed_time = main_process_color(gambar1_path, destination_folder)
    elif mode == 'texture':
        top_images, elapsed_time = main_process_texture(gambar1_path, destination_folder)
    else:
        return render_template('error.html', message='Invalid mode selected.')

    session['top_images'] = top_images
    session['elapsed_time'] = elapsed_time

    return render_template('index.html', top_images=top_images, elapsed_time=elapsed_time, gambar = gambar1.filename, basename=basename)

@app.route('/download-pdf')
def download_pdf():
    top_images = session.get('top_images', [])
    elapsed_time = session.get('elapsed_time', 0)

    pdf_path = create_pdf(top_images, "test/datasave", elapsed_time)
    return send_from_directory("test/datasave", 'result.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)