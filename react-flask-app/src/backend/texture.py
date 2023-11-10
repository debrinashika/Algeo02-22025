from flask import Flask, render_template, request
import cv2
import time
import os
from multiprocessing import Pool
import multiprocessing
import numpy as np
import imageprocess as pros


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

app = Flask(__name__)
@app.route('/upload', methods=['POST'])
def upload():
    gambar1 = request.files['gambar1']
    folder_path = request.form['folder_path']

    gambar1_matrix = cv2.imread(gambar1)
    pros.ubahbw(gambar1_matrix)

    start_time = time.time()
    num_processes = 4
    pool = Pool(processes=num_processes)
    manager = multiprocessing.Manager()
    results = manager.list()

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        gambar2_matrix = cv2.imread(image_path)
        pool.apply_async(compare, args=(gambar1_matrix, gambar2_matrix, results))

    pool.close()
    pool.join()

    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    top_images = sorted_results[:10]  # Ambil 10 sementara deh
    end_time = time.time()
    elapsed_time = end_time - start_time

    return render_template('result.html', top_images=top_images, elapsed_time=elapsed_time)

if __name__ == '__main__':
    app.run(debug=True)