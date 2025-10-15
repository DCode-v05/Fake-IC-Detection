import React, { useState } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import InspectionDashboard from './components/InspectionDashboard';
import UserValidationDashboard from './components/UserValidationDashboard';
import './styles/App.css';

/**
 * Main Application Component
 * Manages the overall state and layout of the AOI IC Marking Detection System
 */
function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [inspectionStatus, setInspectionStatus] = useState('idle'); // idle, processing, completed
  const [inspectionResult, setInspectionResult] = useState(null);
  const [activeTab, setActiveTab] = useState('detection'); // detection, validation

  /**
   * Handle image upload from ImageUpload component
   * @param {File} file - The uploaded image file
   */
  const handleImageUpload = (file) => {
    setUploadedImage(file);
    setInspectionStatus('idle');
    setInspectionResult(null);
  };

  /**
   * Clear the uploaded image and reset inspection state
   */
  const handleClearImage = () => {
    setUploadedImage(null);
    setInspectionStatus('idle');
    setInspectionResult(null);
  };

  /**
   * Handle tab change - also resets inspection state
   */
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setInspectionStatus('idle');
    setInspectionResult(null);
  };

  return (
    <div className="app">
      <Header activeTab={activeTab} onTabChange={handleTabChange} />
      
      <main className="main-content">
        <div className="container">
          {/* System Info Section */}
          <section className="system-info">
            <h2>
              {activeTab === 'detection' 
                ? 'Automated Optical Inspection System' 
                : 'IC Marking Validation System'}
            </h2>
            <p className="description">
              {activeTab === 'detection'
                ? 'Upload IC component images for automated marking verification and fake detection. The system compares IC markings against OEM specifications to ensure component authenticity.'
                : 'Provide expected IC details first, then upload the IC image. The system will verify if the detected markings match your expectations.'}
            </p>
          </section>

          {/* Detection Mode Layout: Two Columns (Upload | Analysis) */}
          {activeTab === 'detection' && (
            <div className="two-col-layout">
              {/* Left: Upload Panel */}
              <section className="upload-section panel-card">
                <ImageUpload 
                  onImageUpload={handleImageUpload}
                  currentImage={uploadedImage}
                  onClearImage={handleClearImage}
                />
              </section>

              {/* Right: Analysis Panel (sticky/scroll area) */}
              <section className="inspection-section panel-card">
                <InspectionDashboard 
                  image={uploadedImage}
                  status={inspectionStatus}
                  result={inspectionResult}
                  onStatusChange={setInspectionStatus}
                  onResultChange={setInspectionResult}
                />
              </section>
            </div>
          )}

          {/* Validation Mode Layout: User Inputs First, Then Upload */}
          {activeTab === 'validation' && (
            <>
              {/* User Validation Dashboard - Shows inputs and upload together */}
              <section className="inspection-section">
                <UserValidationDashboard 
                  image={uploadedImage}
                  status={inspectionStatus}
                  result={inspectionResult}
                  onStatusChange={setInspectionStatus}
                  onResultChange={setInspectionResult}
                  onImageUpload={handleImageUpload}
                  onClearImage={handleClearImage}
                />
              </section>
            </>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>&copy; 2025 AOI IC Marking Detection System | Quality Assurance Division</p>
      </footer>
    </div>
  );
}

export default App;

