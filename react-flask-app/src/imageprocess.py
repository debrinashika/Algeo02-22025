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
