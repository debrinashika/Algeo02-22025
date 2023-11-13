import numpy as np
import cv2

class ColorDescriptor:
    def __init__(self, bins):
        # Inisialisasi objek dengan jumlah bins untuk histogram
        self.bins = bins

    def describe(self, image):
        # Deskripsi gambar dengan ekstraksi fitur warna
        image_hsv = self.rgb_to_hsv(image)
        features = []

        # Bagi gambar menjadi blok 3x3
        blocks = self.divide_image(image_hsv, 3)

        # Loop over blok-blok tersebut
        for block in blocks:
            # Hitung nilai rata-rata HSV dari blok
            block_avg_hsv = self.calculate_block_average(block)
            features.extend(block_avg_hsv)

        return features
        
    def rgb_to_hsv(self, image):
        # Implementasi konversi warna RGB ke HSV secara manual

        # Normalize RGB values to the range [0, 1]
        image_normalized = image / 255.0

        # Extract individual color channels
        R, G, B = image_normalized[:,:,0], image_normalized[:,:,1], image_normalized[:,:,2]

        # Find Cmax, Cmin, and ∆
        Cmax = np.maximum(R, np.maximum(G, B))
        Cmin = np.minimum(R, np.minimum(G, B))
        delta = Cmax - Cmin

        # Initialize arrays for H, S, and V
        H = np.zeros_like(Cmax)
        S = np.zeros_like(Cmax)
        V = Cmax

        # Calculate Hue (H)
        H[Cmax == Cmin] = 0  # ∆ = 0
        non_zero_delta = delta != 0
        # Case: Cmax = R
        H[R == Cmax] = 60.0 * (((G - B) / delta) % 6)
        H[R == Cmax] = np.extract(non_zero_delta, H[R == Cmax])
        # Case: Cmax = G
        H[G == Cmax] = 60.0 * (((B - R) / delta) + 2)
        H[G == Cmax] = H[G == Cmax][non_zero_delta]
        # Case: Cmax = B
        H[B == Cmax] = 60.0 * (((R - G) / delta) + 4)
        H[B == Cmax] = H[B == Cmax][non_zero_delta]
        # Calculate Saturation (S)
        S[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]

        return H, S, V

    def divide_image(self, image, num_blocks):
        # Bagi gambar menjadi blok-blok
        
        # Get the dimensions of the image
        height, width, _ = image.shape

        # Calculate the size of each block
        block_height = height // num_blocks
        block_width = width // num_blocks

        # Initialize a list to store the divided blocks
        blocks = []

        # Iterate over rows and columns to extract each block
        for i in range(num_blocks):
            for j in range(num_blocks):
                # Calculate the starting and ending indices for the block
                start_row = i * block_height
                end_row = (i + 1) * block_height
                start_col = j * block_width
                end_col = (j + 1) * block_width

                # Extract the block from the image
                block = image[start_row:end_row, start_col:end_col, :]

                # Append the block to the list
                blocks.append(block)

        return blocks

    def calculate_block_average(self, block):
        # Hitung nilai rata-rata HSV dari blok

        # Convert RGB to HSV for the entire block
        H, S, V = self.rgb_to_hsv(block)

        # Calculate the average HSV values for the block
        avg_H = np.mean(H)
        avg_S = np.mean(S)
        avg_V = np.mean(V)

        return avg_H, avg_S, avg_V