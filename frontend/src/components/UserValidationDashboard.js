import React, { useState } from 'react';
import '../styles/InspectionDashboard.css';
import { detectManufacturer } from '../services/api';
import ImageUpload from './ImageUpload';

/**
 * UserValidationDashboard Component
 * Focused on validating IC against user-provided expected values
 */
function UserValidationDashboard({ image, status, result, onStatusChange, onResultChange, onImageUpload, onClearImage }) {
  const [error, setError] = useState(null);
  
  // User input fields - always required in this mode
  const [expectedManufacturer, setExpectedManufacturer] = useState('');
  const [expectedPartNumber, setExpectedPartNumber] = useState('');
  
  const handleValidateIC = async () => {
    if (!image) {
      setError('Please upload an IC image first');
      return;
    }

    if (!expectedManufacturer || !expectedPartNumber) {
      setError('Please provide both expected manufacturer and part number');
      return;
    }

    onStatusChange('processing');
    setError(null);
    
    try {
      // Always pass user inputs in this mode
      const userInputs = {
        expectedManufacturer: expectedManufacturer.trim(),
        expectedPartNumber: expectedPartNumber.trim()
      };
      
      const response = await detectManufacturer(image, userInputs);
      
      if (response.success) {
        onStatusChange('completed');
        onResultChange({
          status: response.result.status,
          message: response.result.message,
          manufacturer: response.result.manufacturer,
          confidence: response.result.confidence,
          ocr: response.ocr_extraction,
          marking: response.marking_analysis,
          verification: response.verification,
          logo: response.logo_detection,
          userValidation: response.user_validation
        });
      } else {
        throw new Error(response.error || 'Validation failed');
      }
    } catch (err) {
      console.error('Validation error:', err);
      onStatusChange('idle');
      setError(err.message);
      onResultChange({
        status: 'error',
        message: `Error: ${err.message}`
      });
    }
  };

  return (
    <div className="inspection-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h3>✓ Batchwise IC Validation</h3>
        <span className={`status-badge status-${status}`}>
          {status === 'idle' && '⚪ Ready'}
          {status === 'processing' && '🔄 Validating...'}
          {status === 'completed' && '✅ Complete'}
        </span>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Two-column Inputs (Left) and Image Upload (Right) */}
        <div className="two-col-layout">
          {/* Left: User Inputs */}
          <section className="panel-card">
            <div className="user-validation-section validation-mode">
              <div className="validation-header-static">
                <span className="validation-title">📝 Enter Expected IC Details</span>
                <span className="required-badge">Required</span>
              </div>
              <div className="validation-inputs">
                <div className="input-group">
                  <label htmlFor="expected-manufacturer">
                    Expected Manufacturer: <span className="required-star">*</span>
                  </label>
                  <select
                    id="expected-manufacturer"
                    value={expectedManufacturer}
                    onChange={(e) => setExpectedManufacturer(e.target.value)}
                    className="input-field"
                    required
                    disabled={status === 'processing'}
                  >
                    <option value="">-- Select Manufacturer --</option>
                    <option value="Texas Instruments">Texas Instruments</option>
                    <option value="STMicroelectronics">STMicroelectronics</option>
                    <option value="NXP Semiconductors">NXP Semiconductors</option>
                    <option value="ON Semiconductor">ON Semiconductor</option>
                    <option value="Microchip">Microchip</option>
                    <option value="Infineon">Infineon</option>
                  </select>
                </div>
                <div className="input-group">
                  <label htmlFor="expected-part">
                    Expected Part Number: <span className="required-star">*</span>
                  </label>
                  <input
                    type="text"
                    id="expected-part"
                    value={expectedPartNumber}
                    onChange={(e) => setExpectedPartNumber(e.target.value)}
                    placeholder="e.g., LM324, STM32F103, 6007329"
                    className="input-field"
                    required
                    disabled={status === 'processing'}
                  />
                </div>
              </div>
            </div>
          </section>

          {/* Right: Image Upload */}
          <section className="panel-card">
            <ImageUpload 
              onImageUpload={onImageUpload}
              currentImage={image}
              onClearImage={onClearImage}
            />
          </section>
        </div>

        {/* Validate Button below two columns */}
        <div className="start-section" style={{paddingTop: 0}}>
          <button 
            className="btn-inspect btn-validate"
            onClick={handleValidateIC}
            disabled={status === 'processing' || !image || !expectedManufacturer || !expectedPartNumber}
          >
            {status === 'processing' ? (
              <>
                <span className="spinner"></span>
                Validating IC...
              </>
            ) : (
              <>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                Validate IC Marking
              </>
            )}
          </button>
          {error && (
            <div className="error-alert">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Results Display */}
        {result && (
          <div className="results-container">
            {/* Primary Result Card - Match with Detect's style */}
            {result.userValidation && (
              <>
                <div className={`result-card ${result.userValidation.matches_expectations ? 'result-genuine' : 'result-fake'}`}>
                  <div className="result-icon" aria-hidden>
                    {result.userValidation.matches_expectations ? '✓' : '✗'}
                  </div>
                  <h4 className="result-title">
                    {result.userValidation.matches_expectations ? 'IC matches user expectations' : 'IC does not match user expectations'}
                  </h4>
                  {result.manufacturer && (
                    <p className="result-manufacturer">
                      Detected: <strong>{result.manufacturer}</strong>
                      {typeof result.confidence === 'number' && (
                        <> ({(result.confidence * 100).toFixed(1)}% confidence)</>
                      )}
                    </p>
                  )}
                </div>

                {/* Compact details in the shared info style */}
                <div className="details-section validation-compact">
                  <h5 className="section-title">✓ User Validation</h5>
                  <div className="info-card">
                    <div className="info-row">
                      <span className="info-label">Manufacturer</span>
                      <span className="info-value">
                        Expected: <strong>{expectedManufacturer || 'N/A'}</strong>
                        {" \u2022 "}
                        Detected: <strong>{result.manufacturer || 'Not detected'}</strong>
                        {" \u00A0"}
                        <span className={`match-badge ${result.userValidation.manufacturer_match ? 'match' : 'mismatch'}`}>
                          {result.userValidation.manufacturer_match ? '✓ Match' : '✗ Mismatch'}
                        </span>
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Part Number</span>
                      <span className="info-value">
                        Expected: <strong>{expectedPartNumber || 'N/A'}</strong>
                        {" \u2022 "}
                        Detected: <strong>{(result.marking && (result.marking.part_number || (result.marking.part_number?.part_number))) || 'Not detected'}</strong>
                        {" \u00A0"}
                        <span className={`match-badge ${result.userValidation.part_number_match ? 'match' : 'mismatch'}`}>
                          {result.userValidation.part_number_match ? '✓ Match' : '✗ Mismatch'}
                        </span>
                      </span>
                    </div>
                    {result.userValidation.message && (
                      <div className="info-row">
                        <span className="info-label">Notes</span>
                        <span className="info-value">{result.userValidation.message}</span>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}

            {/* Fallback: if userValidation missing, still show the detect-style main result card */}
            {!result.userValidation && (
              <div className={`result-card result-${result.status}`}>
                <div className="result-icon" aria-hidden>
                  {result.status === 'genuine' && '✓'}
                  {result.status === 'fake' && '✗'}
                  {result.status === 'error' && '!'}
                </div>
                <h4 className="result-title">{result.message}</h4>
              </div>
            )}

            {/* New Validation Button - match Detect's style */}
            <button
              className="btn-new-inspection"
              onClick={() => {
                onStatusChange('idle');
                onResultChange(null);
                setExpectedManufacturer('');
                setExpectedPartNumber('');
              }}
            >
              🔄 New Validation
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default UserValidationDashboard;
