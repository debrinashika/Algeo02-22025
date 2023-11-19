from flask import Flask, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from multiprocessing import Pool
from os.path import basename
import shutil
import cbir_color as col
import cbir_texture as tex
from urllib.parse import unquote


def create_pdf(top_images, destination_folder,time):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.pdf')

    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.drawString(400, 720 - 10, f'Total Elapsed Time: {time:.5f} seconds')
    for i, (image_path, similarity_score) in enumerate(top_images):
        c.drawImage(image_path, 20, 720 - (i%4 + 1) * 150, width=150, height=150)
        c.drawString(200, 720 - (i%4 + 1) * 150 + 75, f'Similarity: {similarity_score:.5f}')

        if(i%4==3):
            c.showPage()

    c.save()

    return pdf_path

def file_exists(file_path):
    return os.path.exists(file_path)

app = Flask(__name__, static_url_path='/static')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'datasave')
app.config['UPLOAD_IMAGES'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'input_images')

@app.route('/')
def home():
    return render_template('home.html', file_exists=file_exists)

@app.route('/start')
def start():
    return render_template('index.html')

@app.route('/how-to-use')
def how_to_use():
    return render_template('home.html')

@app.route('/about-us')
def about_us():
    return render_template('home.html')

@app.route('/uploaded-images/<path:filename>')
def uploaded_image(filename):
    filename = unquote(filename)
    return send_from_directory(app.config['UPLOAD_IMAGES'], secure_filename(filename))

@app.route('/images/<path:filename>')
def download_file(filename):
    print("Uploaded Image Path:", os.path.join(app.config['UPLOAD_FOLDER']))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    gambar1 = request.files['gambar1']
    folder = request.files.getlist('folder[]')
    destination_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'datasave')
    input_images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'input_images')
    csv_file_path = os.path.join(destination_folder, 'result.csv')  # Path ke file CSV

    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    if os.path.exists(input_images_folder):
        shutil.rmtree(input_images_folder)

    os.makedirs(destination_folder)
    os.makedirs(input_images_folder)

    mode = request.form.get('mode', 'color')  # Mode default: color
    
    if mode == 'on':
        mode = 'texture'
    print("Received mode:", mode)

    if folder:
        for uploaded_file in folder:
            if uploaded_file.filename != '':
                file_path = os.path.join(destination_folder, secure_filename(uploaded_file.filename))
                uploaded_file.save(file_path)
                print(f'Saving file: {file_path}')
    else:
        return render_template('error.html', message='No folder selected.')

    gambar1_path = os.path.join(input_images_folder, secure_filename(gambar1.filename))
    gambar1.save(gambar1_path)

    if mode == 'color':
        top_images, elapsed_time = col.main_process_color(gambar1_path, destination_folder, csv_file_path)
    elif mode == 'texture':
        top_images, elapsed_time = tex.main_process_texture(gambar1_path, destination_folder, csv_file_path)
    else:
        return render_template('error.html', message='Invalid mode selected.')

    session['top_images'] = top_images
    session['elapsed_time'] = elapsed_time
    session['gambar'] = gambar1.filename
    session['gambar_path'] = gambar1_path
    session['page'] = 1  # Initialize the page number

    top2_images = top_images + ["No Image"]
    banyak_gambar = len(top_images)

    return render_template('index.html', top2_images=top2_images, top_images=top_images[:12], elapsed_time=elapsed_time, banyakgambar=banyak_gambar,
                           gambar=gambar1.filename, gambar_path=gambar1_path, basename=basename, page=1, total_pages=(banyak_gambar + 11) // 12)

@app.route('/pagination', methods=['GET', 'POST'])
def pagination():
    top_images = session.get('top_images', [])
    elapsed_time = session.get('elapsed_time', 0)
    gambar_path = session.get('gambar_path', '')
    gambar=session.get('gambar', '')
    page = int(request.args.get('page', 1))
    session['page'] = page

    start_idx = (page - 1) * 12
    end_idx = start_idx + 12

    banyak_gambar = len(top_images)
    total_pages = (banyak_gambar + 11) // 12

    return render_template('index.html', top_images=top_images[start_idx:end_idx], elapsed_time=elapsed_time,
                           banyakgambar=banyak_gambar, gambar=gambar, gambar_path=gambar_path,
                           basename=basename, page=page, total_pages=total_pages)


@app.route('/download-pdf')
def download_pdf():
    top_images = session.get('top_images', [])
    elapsed_time = session.get('elapsed_time', 0)

    pdf_path = create_pdf(top_images, "test/datasave",elapsed_time)
    
    # Check if the file exists
    if os.path.exists(pdf_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'result.pdf', as_attachment=True)
    else:
        return render_template('error.html', message='PDF file not found.')

if __name__ == '__main__':
    app.run(debug=True)