// ImageUpload.js
import React, { useState } from 'react';
import axios from 'axios';

const ImageUpload = ({ setTopImages }) => {
  const [gambar1, setGambar1] = useState(null);
  const [folderPath, setFolderPath] = useState(null);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('gambar1', gambar1);
    formData.append('folder_path', folderPath);
  
    try {
      await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log(`Upload Progress: ${percentCompleted}%`);
          // Update state or UI with the progress information
        },
      });
    } catch (error) {
      console.error('Error uploading images:', error);
    }
  };

  const handleFolderSelect = (e) => {
    const folderPath = e.target.files[0]?.path;
    setFolderPath(folderPath);
  };

  const handleImageSelect = (e) => {
    const imagePath = e.target.files[0]?.path;
    setGambar1(imagePath);
  };

  return (
    <div>
      <h1>Image Upload</h1>
      <div>
        <label>
          Upload Gambar 1:
          <input type="file" accept="image/*" onChange={handleImageSelect} />
        </label>
      </div>
      <div>
        <label>
          Pilih Folder:
          <input type="file" webkitdirectory="true" directory="true" onChange={handleFolderSelect} />
        </label>
      </div>
      <div>
        <button onClick={handleUpload}>Upload</button>
      </div>
    </div>
  );
};

export default ImageUpload;
