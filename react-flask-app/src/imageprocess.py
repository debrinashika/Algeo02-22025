# imageprocess.py
import cv2
import os
from PIL import Image
from io import BytesIO
import numpy as np
import shutil

def ubahbw(matriksgambar):
    for i in range(matriksgambar.shape[0]):
        for j in range(matriksgambar.shape[1]):
            matriksgambar[i][j] = matriksgambar[i][j][0] * 0.114 + matriksgambar[i][j][1] * 0.587 + matriksgambar[i][j][2] * 0.29
    matriksgambar = cv2.resize(matriksgambar, (256, 256), interpolation=cv2.INTER_AREA)
    matriksgambar = compressimage(matriksgambar)
    return matriksgambar

def ubahbwfolder(folderinput):
    path = folderinput
    output = os.path.join("test/dataolah")

    if os.path.exists(output):
        shutil.rmtree(output)

    os.makedirs(output)

    for filename in os.listdir(path):
        input_path = os.path.join(path, filename)

        if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
            continue

        try:
            image = cv2.imread(input_path)
            if image is None:
                raise Exception(f"Failed to read image: {input_path}")

            bw_image = ubahbw(image)
            output_path = os.path.join(output, filename)
            print(f"Processing {filename} and saving to {output_path}")
            cv2.imwrite(output_path, bw_image)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def compressimage(gambar1):
    gambar1 = Image.fromarray(gambar1)
    compressed_image = BytesIO()
    gambar1.save(compressed_image, 'JPEG', quality=90)
    compresimage = np.array(Image.open(compressed_image))
    return compresimage

def rgb_to_hsv(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val

    # Hitung nilai Hue
    if delta == 0:
        hue = 0
    elif max_val == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif max_val == g:
        hue = 60 * (((b - r) / delta) + 2)
    elif max_val == b:
        hue = 60 * (((r - g) / delta) + 4)

    # Hitung nilai Saturation
    saturation = 0 if max_val == 0 else delta / max_val

    # Hitung nilai Value
    value = max_val

    return hue, saturation, value

def extract_color_features(image):
    # Convert RGB to HSV
    hsv_image = np.array([rgb_to_hsv(pixel) for pixel in image.reshape(-1, 3)]).reshape(image.shape)
    # Split the image into 3x3 blocks
    blocks = np.array_split(hsv_image, 3, axis=0)
    blocks = [np.array_split(block, 3, axis=1) for block in blocks]

    # Calculate the average HSV values for each block
    avg_hsv_blocks = []
    for row in blocks:
        for block in row:
            avg_hsv = np.mean(block, axis=(0, 1))
            avg_hsv_blocks.append(avg_hsv)

    return np.concatenate(avg_hsv_blocks)

def compare_color(gambar1, gambar2):
    features1 = extract_color_features(gambar1)
    features2 = extract_color_features(gambar2)

    # Calculate cosine similarity between color features
    similarity = similarity(features1, features2)

    return similarity

def ubahwarna(matriksgambar):
    matrikswarna = cv2.resize(matriksgambar, (256, 256), interpolation=cv2.INTER_AREA)
    matrikswarna = compressimage(matrikswarna)
    return matrikswarna