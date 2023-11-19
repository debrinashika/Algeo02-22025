# File: cbir_color.py
# CBIR dengan parameter warna

import numpy as np
import cv2
import os
import time
from multiprocessing import Pool, Lock
import multiprocessing
from tqdm import tqdm
import csv

def compress(image_matrix):
    # Mengompres matrix pixel gambar menjadi 256 x 256
    # Konversi matrix pixel menjadi gambar
    image = np.array(image_matrix, dtype=np.uint8)
    # Ubah ukuran gambar menjadi 256 x 256
    resized_image = cv2.resize(image, (256, 256))
    # Ubah kembali ke matrix pixel
    compressed_matrix = np.array(resized_image)
    return compressed_matrix

def extract_rgb(image_matrix):
    # Mengambil nilai RGB dari matrix gambar
    red_matrix = image_matrix[:, :, 0]
    green_matrix = image_matrix[:, :, 1]
    blue_matrix = image_matrix[:, :, 2]
    return red_matrix, green_matrix, blue_matrix

def rgb_to_hsv(matrix_R, matrix_G, matrix_B):
    # Melakukan konversi warna dari RGB ke HSV
    # Normalisasi nilai RGB ke range [0, 1]
    r, g, b = matrix_R / 255.0, matrix_G / 255.0, matrix_B / 255.0
    # Mencari Cmax, Cmin, dan delta
    Cmax = np.maximum.reduce([r, g, b])
    Cmin = np.minimum.reduce([r, g, b])
    delta = Cmax - Cmin
    # Hitung nilai H
    h = np.where(delta != 0,
        np.where(Cmax == r, (g - b) / (delta + np.finfo(float).eps) % 6,
        np.where(Cmax == g, (b - r) / (delta + np.finfo(float).eps) + 2,
        np.where(Cmax == b, (r - g) / (delta + np.finfo(float).eps) + 4, 0))),
        0)
    h = (h * 60) % 360 # Konversi H ke range [0, 360]
    # Hitung nilai S
    s = np.where(Cmax != 0, delta / (Cmax + np.finfo(float).eps), 0)
    # Hitung nilai V
    v = Cmax
    return h, s, v

def cosine_similarity(vektor1, vektor2):
    # Hitung cosine similarity antara dua vektor fitur
    dot_product = sum(a * b for a, b in zip(vektor1, vektor2))
    norm1 = sum(a ** 2 for a in vektor1) ** 0.5
    norm2 = sum(b ** 2 for b in vektor2) ** 0.5
    similarity = dot_product / (norm1 * norm2 + 1e-10)
    return similarity

def block_similarity(H1, S1, V1, H2, S2, V2):
    # Menghitung similarity antara 2 blok
    # Meratakan matriks H, S, dan V menjadi vektor satu dimensi
    vector1_H = H1.flatten().astype(float)
    vector1_S = S1.flatten().astype(float)
    vector1_V = V1.flatten().astype(float)
    vector2_H = H2.flatten().astype(float)
    vector2_S = S2.flatten().astype(float)
    vector2_V = V2.flatten().astype(float)
    # Menghitung cosine similarity untuk setiap komponen (H, S, V)
    similarity_H = cosine_similarity(vector1_H, vector2_H)
    similarity_S = cosine_similarity(vector1_S, vector2_S)
    similarity_V = cosine_similarity(vector1_V, vector2_V)
    # Penanganan nilai tidak valid
    if np.isnan(similarity_H) or np.isnan(similarity_S) or np.isnan(similarity_V):
        return 0.0
    # Menghitung rata-rata similarity vektor H, S, V 2 blok
    block_sim = (similarity_H + similarity_S + similarity_V) / 3
    return block_sim

def process_image_color(image_path, gambar1_matrix, results):
    # Membandingkan dua gambar dan mendapatkan similarity scorenya
    # Mendapatkan matrix dari gambar2
    gambar2_matrix = cv2.imread(image_path)
    # Mengompres matrix gambar1 dan gambar2
    compressed_gambar1 = compress(gambar1_matrix)
    compressed_gambar2 = compress(gambar2_matrix)
    # Mengambil matriks R, G, B dari matriks gambar1 dan gambar2
    R1, G1, B1 = extract_rgb(compressed_gambar1)
    R2, G2, B2 = extract_rgb(compressed_gambar2)
    # Konversi matriks R, G, B menjadi matriks H, S, V
    H1, S1, V1 = rgb_to_hsv(R1, G1, B1)
    H2, S2, V2 = rgb_to_hsv(R2, G2, B2)
    # Inisialisasi similarity scores
    total_similarity = 0.0
    # Iterasi untuk 16 blok 4 x 4
    for y in range(0, 256, 64):
        for x in range(0, 256, 64):
            # Mendapatkan blok dari matriks H, S, V untuk gambar1
            block1_H = H1[y:y + 64, x:x + 64]
            block1_S = S1[y:y + 64, x:x + 64]
            block1_V = V1[y:y + 64, x:x + 64]
            # Mendapatkan blok dari matriks H, S, V untuk gambar2
            block2_H = H2[y:y + 64, x:x + 64]
            block2_S = S2[y:y + 64, x:x + 64]
            block2_V = V2[y:y + 64, x:x + 64]
            # Menghitung similarity score untuk blok saat ini
            block_sim = block_similarity(block1_H, block1_S, block1_V, block2_H, block2_S, block2_V)
            # Menambahkan similarity score blok ke total similarity
            if not np.isnan(block_sim):
                total_similarity += block_sim
    # Menghitung rata-rata similarity untuk mendapatkan similarity score akhir
    similarity_score = (total_similarity / 16) * 100
    similarity_score = round(similarity_score, 8)
    # Menambahkan hasil ke dalam list results
    results.append((image_path, similarity_score))

def save_results_to_csv(results, input_image_path, csv_file_path, csv_lock):
    # Menyimpan data ke file csv
    write_header = not os.path.exists(csv_file_path)  # Check if the file exists
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['input_image_path', 'image_path', 'similarity_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Acquire lock
        csv_lock.acquire()
        # Tulis header jika file csv masih kosong
        if write_header:
            writer.writeheader()
        # Menyimpan results ke file csv
        for image_path, similarity_score in results:
            writer.writerow({'input_image_path': input_image_path, 'image_path': image_path, 'similarity_score': similarity_score})
        # Release lock
        csv_lock.release()

def read_results_from_csv(input_image_path, destination_folder, csv_file_path):
    # Membaca data dari file csv
    results = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['input_image_path'] == input_image_path:
                image_path = row['image_path']
                similarity_score = float(row['similarity_score'])
                if os.path.exists(os.path.join(destination_folder, image_path)):
                    results.append((image_path, similarity_score))
    return results

def main_process_color(input_path, destination_folder, csv_file_path):
    # Fungsi utama untuk pemrosesan gambar berdasarkan CBIR fitur warna
    # Mendapatkan matrix dari gambar input
    input_matrix = cv2.imread(input_path)
    # Mengkompres matrix gambar input
    comp_input_matrix = compress(input_matrix)
    # Membuat lock untuk sinkronisasi file CSV
    csv_lock = Lock()
    # Mengecek apakah file csv sudah ada
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            header = ['Input_Path', 'Dataset_Path', 'Similarity_Score']
            csv.writer(csvfile).writerow(header)
    # Mengecek apakah data sudah ada di file CSV
    cached_results = read_results_from_csv(input_path, destination_folder, csv_file_path)
    if cached_results:
        # Gunakan data dari CSV jika sudah ada
        print("Using cache...")
        top_images = cached_results
    else:
        # Mengambil daftar file di folder tujuan (dataset)
        files = os.listdir(destination_folder)
        total_files = len(files)
        # Inisialisasi progress bar
        with tqdm(total=total_files, desc="Processing Images") as progress:
            # Memulai waktu proses
            start_time = time.time()
            # Jumlah proses paralel yang dijalankan
            num_processes = 4
            # Membuat pool untuk multiprocessing
            pool = Pool(processes=num_processes)
            # Membuat manajer untuk berbagi data antar proses
            manager = multiprocessing.Manager()
            # List hasil yang bisa dibagi antar proses
            results = manager.list()
            # Iterasi setiap file di folder
            for file in files:
                if file != '':
                    # Mendapatkan path lengkap file gambar
                    image_path = os.path.join(destination_folder, file)
                    # Menerapkan fungsi process_image_color secara asynchronous
                    pool.apply_async(process_image_color, args=(image_path, comp_input_matrix, results))
            # Menutup pool setelah semua tugas selesai
            pool.close()
            pool.join()
            # Menyaring hasil dan menyimpan hanya yang memiliki similarity_score > 60
            results = [(image_path, similarity_score) for image_path, similarity_score in results if similarity_score > 60]
            # Mengurutkan hasil berdasarkan similarity_score
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            top_images = sorted_results + cached_results
            # Menyimpan data ke file csv
            save_results_to_csv(top_images, input_path, csv_file_path, csv_lock)
            # Menghitung waktu total pemrosesan
            end_time = time.time()
            elapsed_time = end_time - start_time

    return top_images, elapsed_time