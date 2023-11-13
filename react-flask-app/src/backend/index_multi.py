import argparse
import glob
from cbir_utils import MapReduce, feature_extraction

if __name__ == '__main__':
    # Ambil argumen dari command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path to the directory that contains the images to be indexed")
    ap.add_argument("-i", "--index", required=True,
                    help="Path to where the computed index will be stored")
    args = vars(ap.parse_args())

    # Ambil daftar path gambar dari folder dataset
    inputs = glob.glob(args["dataset"] + "/*.jpg")
    mapper = MapReduce(feature_extraction)

    # Gunakan pagination untuk mengatur jumlah gambar per halaman
    page_size = 10
    num_pages = len(inputs) // page_size + 1

    for page in range(num_pages):
        start_idx = page * page_size
        end_idx = (page + 1) * page_size
        mapper_result = mapper(inputs[start_idx:end_idx])

        output = open(args["index"], "a")  # Gunakan mode "a" untuk menambahkan hasil ke file yang sudah ada
        for i in range(len(mapper_result)):
            output.write(mapper_result[i])

        output.close()