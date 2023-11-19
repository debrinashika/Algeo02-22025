import numpy as np
import cv2
import os
import time
from multiprocessing import Pool, Lock
import multiprocessing
from tqdm import tqdm
import cbir_color as col
import csv

def compress(image_matrix):
    image = np.array(image_matrix, dtype=np.uint8)
    resized_image = cv2.resize(image, (256, 256))
    # Ubah kembali ke matrix pixel
    compressed_matrix = np.array(resized_image)
    return compressed_matrix

def cosine_similarity(vektor1, vektor2):
    # Hitung cosine similarity antara dua vektor fitur
    dot_product = sum(a * b for a, b in zip(vektor1, vektor2))
    norm1 = sum(a ** 2 for a in vektor1) ** 0.5
    norm2 = sum(b ** 2 for b in vektor2) ** 0.5
    similarity = dot_product / (norm1 * norm2 + 1e-10)
    return similarity

def ubahbw(matriksgambar):
    blue_channel = matriksgambar[:, :, 0]
    green_channel = matriksgambar[:, :, 1]
    red_channel = matriksgambar[:, :, 2]
    grayscale_values = 0.114 * blue_channel + 0.587 * green_channel + 0.299 * red_channel
    matriksgambar = np.stack([grayscale_values, grayscale_values, grayscale_values], axis=-1)
    matriksgambar = compress(matriksgambar)
    return matriksgambar

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
    nonzero_elements = np.clip(nonzero_elements, 1e-10, None)
    entropy = -np.sum(nonzero_elements * np.log(nonzero_elements))
    return entropy


def compare(gambar1,gambar2):
    # start_time = time.time()  # Waktu mulai proses
    # occurence matrix
    occurmat = np.zeros((256, 256), dtype=int)
    occurmat2 = np.zeros((256, 256), dtype=int)

    htgduaelmnt(gambar1,occurmat)
    htgduaelmnt(gambar2,occurmat2)

    occurmat = (occurmat + occurmat.T) / occurmat.sum()
    occurmat2 = (occurmat2 + occurmat2.T) / occurmat2.sum()

    vek1 = [contrast(occurmat),homogeneity(occurmat),entropy(occurmat)]
    vek2 = [contrast(occurmat2),homogeneity(occurmat2),entropy(occurmat2)]

    sim = cosine_similarity(vek1,vek2)*100
    # end_time = time.time()  # Waktu selesai proses
    # elapsed_time = end_time - start_time  # Hitung lama waktu proses

    # print(f"Lama waktu proses pemrosesan: {elapsed_time:.2f} detik")
    return sim

def process_image_texture(image_path, gambar1, results):
    gambar2 = cv2.imread(image_path)
    gambar2 = ubahbw(gambar2)

    similarity_score = round(compare(gambar1, gambar2),8)
    results.append((image_path, similarity_score))

def main_process_texture(gambar1_path, destination_folder, csv_file_path):
    gambar1_matrix = cv2.imread(gambar1_path)
    gambar1_matrix = ubahbw(gambar1_matrix)

    csv_lock = Lock()
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            header = ['Input_Path', 'Dataset_Path', 'Similarity_Score']
            csv.writer(csvfile).writerow(header)
    cached_results = col.read_results_from_csv(gambar1_path, destination_folder, csv_file_path)
    if cached_results:
        # Gunakan data dari CSV jika sudah ada
        print("Using cache...")
        top_images = cached_results
    else:
        files = os.listdir(destination_folder)
        total_files = len(files)

        with tqdm(total=total_files, desc="Processing Images"):
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

            results = [(image_path, similarity_score) for image_path, similarity_score in results if similarity_score > 60]
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            top_images = sorted_results + cached_results

            col.save_results_to_csv(top_images, gambar1_path, csv_file_path, csv_lock)

            end_time = time.time()
            elapsed_time = end_time - start_time

    return top_images, elapsed_time