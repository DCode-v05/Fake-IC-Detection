import React, { useState, useRef } from 'react';
import '../styles/ImageUpload.css';

/**
 * ImageUpload Component
 * Handles drag-and-drop and click-to-upload functionality for IC images
 */
function ImageUpload({ onImageUpload, currentImage, onClearImage }) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  /**
   * Handle drag events
   */
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  /**
   * Handle file drop
   */
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  /**
   * Handle file selection via input
   */
  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  /**
   * Process the selected file
   */
  const handleFile = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (JPG, PNG, etc.)');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(file);

    // Pass file to parent component
    onImageUpload(file);
  };

  /**
   * Trigger file input click
   */
  const handleClick = () => {
    fileInputRef.current.click();
  };

  /**
   * Clear the uploaded image
   */
  const handleClear = () => {
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onClearImage();
  };

  return (
    <div className="image-upload-container">
      <h3 className="upload-title">Upload IC Component Image</h3>
      
      {!preview ? (
        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleClick}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          
          <div className="upload-icon">
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
              <path d="M32 8L32 40M32 8L20 20M32 8L44 20" stroke="#00D4FF" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 32L12 48C12 50.2091 13.7909 52 16 52L48 52C50.2091 52 52 50.2091 52 48L52 32" stroke="#00D4FF" strokeWidth="3" strokeLinecap="round"/>
            </svg>
          </div>
          
          <p className="upload-text-primary">Drag and drop IC image here</p>
          <p className="upload-text-secondary">or click to browse</p>
          
          <div className="upload-requirements">
            <p>Supported formats: JPG, PNG, BMP</p>
            <p>Maximum file size: 10MB</p>
            <p>Recommended: Clear, well-lit images of IC markings</p>
          </div>
        </div>
      ) : (
        <div className="preview-container">
          <div className="preview-header">
            <h4>Image Preview</h4>
            <button className="clear-button" onClick={handleClear}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M15 5L5 15M5 5L15 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              Clear
            </button>
          </div>
          
          <div className="preview-image-wrapper">
            <img src={preview} alt="IC Component Preview" className="preview-image" />
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
