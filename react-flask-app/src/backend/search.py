import numpy as np
import csv
from color_descriptor import ColorDescriptor

class Searcher:
    def __init__(self, indexPath):
        # Inisialisasi objek dengan path indeks
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=5):
        # Inisialisasi dictionary hasil pencarian
        results = {}

        with open(self.indexPath) as f:
            # Baca file indeks
            reader = csv.reader(f)
            for row in reader:
                # Parse data dari file indeks
                features = [float(x) for x in row[1:]]
                # Hitung cosine similarity antara fitur query dan fitur dataset
                d = self.cosine_similarity(features, queryFeatures)
                results[row[0]] = d

            f.close()

        # Urutkan hasil pencarian
        results = sorted([(v, k) for (k, v) in results.items()])
        return results[:limit]

    def cosine_similarity(self, histA, histB, eps=1e-10):
        # Cosine Similarity antara dua vektor
        dot_product = np.dot(histA, histB)
        norm_A = np.linalg.norm(histA)
        norm_B = np.linalg.norm(histB)

        similarity = dot_product / (norm_A * norm_B + eps)
        return similarity