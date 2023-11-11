// ResultDisplay.js
import React from 'react';

const ResultDisplay = ({ topImages }) => {
  return (
    <div>
      <h1>Top Images</h1>
      {topImages && (
        <div>
          <p>Image Path: {topImages[0]}</p>
          <p>Similarity Score: {topImages[1]}</p>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;
