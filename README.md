# Image Processing App

> To watch the demo of the program [_here_](https://youtu.be/ZLtBSGCD3Ks). 

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

<a name="setup"></a>

## Setup

For the libraries, please check the `requirements.txt` file. You can install the libraries by using the command below.

```bash
pip install -r requirements.txt
```

<a name="usage"></a>

## Usage
You can run the program by using the command below.

```bash
cd src
python app.py
```

<a name="screenshots"></a>

## Screenshots
<p align=center>
  <h2>Home Page</h2>
  <img src="/img/home.png/">
  <nl>
  <h2>Color Mode</h2>
  <img src="/img/1.png/">
  <img src="/img/2.png/">
  <nl>
  <h2>texture Mode</h2>
  <img src="/img/3.png/">
  <img src="/img/4.png/">
  <nl>
</p>

<a name="structure"></a>

## Structure
```bash
└─── README.md
└─── requirement.txt
│
├───doc
│
├───test
│   │
│   └───database
│   └───dataa
├───img
│   │
│   └───1.png
│   └───2.png
│   └───3.png
│   └───4.png
│   └───home.png
│  
├───src
     └───app.py
     └─── cbir_color.py
     └───cbir_texture.py
     │
     ├───template
     │   └─── index.html
     │   └─── home.html
     │
     ├───static
         └───styles.css
         └───styles2.css

```

