# Algeo02-21025
<h2 align="center">
  Image Processing App<br/>
</h2>
<hr>

> To watch the demo of the program [_here_](). 


## General Information
Content-Based Image Retrieval (CBIR) system revolutionizes image searches by allowing users to find similar pictures based on visual content rather than keywords. Utilizing advanced algorithms, the program extracts features such as color and texture, enabling accurate and efficient retrieval. With a user-friendly interface, this CBIR system is versatile, scalable, and applicable across diverse domains, offering a seamless experience for users seeking visually related images.

<a name="member-list"></a>

## Member List

| Nama                  | NIM      |
| --------------------- | -------- |
| Debrina Veisha Rashika| 13522025 |
| Nabila Shikoofa Muida | 13522069 |
| Novelya Putri R       | 13522096 |

<a name="features"></a>

## Features
- Upload a photo and dataset from your computer
- The result of comparison between the uploaded photo and the image from the dataset
- Download the result as a PDF file

<a name="technologies-used"></a>

## Technologies Used
- CustomTkinter - version 4.6.3
- OpenCV2 - version 4.5.4
- PIL - version 8.4.0
- fpdf - version 1.7.2
- numpy - version 1.21.3
- sys - version 3.9.4
- tkinter - version 8.6

> Note: The version of the libraries above is the version that we used in this project. You can use the latest version of the libraries.

<a name="setup"></a>

## Setup
You can setup your project by cloning this repository and install the libraries above.

For specific version of the libraries, please check the `requirements.txt` file. You can install the libraries by using the command below.

```bash
pip install -r requirements.txt
```

<a name="usage"></a>

## Usage
You can run the program by using the command below.

```bash
python app.py
```

<a name="screenshots"></a>

## Screenshots
<p align=center>
  <img src="">
  <p>Figure 1.</p>
  <nl>
</p>

<a name="structure"></a>

## Structure
```bash
│   README.md
│   requirement.txt
│
├───.vscode
│       settings.json
│
├───doc
│       Tubes2-Algeo-2022.pdf
│
├───test
│   │
│   └───dataset
│           ss_1.png
│           ss_2.png
│           ss_3.png
├───src
│   │   app.py
|   |   cbir_color.py
|   |   cbir_texture.py
│   │
│   ├───template
│   │   │   index.html
│   │   │   home.html
│   │
│   ├───static
│   │   │   styles.css
│   │   │   styles2.css

```

