from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from tqdm import tqdm
import cv2
import time
import os
from multiprocessing import Pool
import multiprocessing
import numpy as np
import imageprocess as pros
from os.path import basename

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

def process_image(image_path, gambar1, results):
    gambar2 = cv2.imread(image_path)
    pros.ubahbw(gambar1)

    similarity_score = compare(gambar1, gambar2)
    results.append((image_path, similarity_score))

def main_process(gambar1_path, destination_folder):
    gambar1_matrix = cv2.imread(gambar1_path)
    pros.ubahbw(gambar1_matrix)

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
                pool.apply_async(process_image, args=(image_path, gambar1_matrix, results))

        pool.close()
        pool.join()

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        top_images = sorted_results[:5]  # Ambil 10 sementara deh
        print(top_images)
        end_time = time.time()
        elapsed_time = end_time - start_time

    return top_images, elapsed_time

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "test/datasave"

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
    os.makedirs(destination_folder, exist_ok=True)

    if folder:
        for uploaded_file in folder:
            if uploaded_file.filename != '':
                file_path = os.path.join(destination_folder, secure_filename(uploaded_file.filename))
                uploaded_file.save(file_path)
                print(f'Saving file: {file_path}')

    gambar1_path = os.path.join(destination_folder, secure_filename(gambar1.filename))
    gambar1.save(gambar1_path)
    print(f'gambar1: {gambar1.filename}')
    print(f'folder: {folder}')

    top_images, elapsed_time = main_process(gambar1_path, destination_folder)

    return render_template('result.html', top_images=top_images, elapsed_time=elapsed_time, basename=basename)

if __name__ == '__main__':
    app.run(debug=True)