import React, { useState } from 'react';
import '../styles/InspectionDashboard.css';
import { detectManufacturer } from '../services/api';

/**
 * InspectionDashboard Component - Professional UI
 * OCR-based IC logo detection and verification
 */
function InspectionDashboard({ image, status, result, onStatusChange, onResultChange }) {
  const [error, setError] = useState(null);
  
  const handleStartInspection = async () => {
    if (!image) {
      setError('Please upload an IC image first');
      return;
    }

    onStatusChange('processing');
    setError(null);
    
    try {
      const response = await detectManufacturer(image);
      
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
          logo: response.logo_detection
        });
      } else {
        throw new Error(response.error || 'Detection failed');
      }
    } catch (err) {
      console.error('Inspection error:', err);
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
        <h3>🔍 IC Authenticity Inspector</h3>
        <span className={`status-badge status-${status}`}>
          {status === 'idle' && '⚪ Ready'}
          {status === 'processing' && '🔄 Analyzing...'}
          {status === 'completed' && '✅ Complete'}
        </span>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        
        {/* Start Button */}
        {!result && (
          <div className="start-section">
            <button 
              className="btn-inspect"
              onClick={handleStartInspection}
              disabled={status === 'processing' || !image}
            >
              {status === 'processing' ? (
                <>
                  <span className="spinner"></span>
                  Analyzing IC...
                </>
              ) : (
                <>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                  </svg>
                  Start Inspection
                </>
              )}
            </button>
            {error && (
              <div className="error-alert">
                ⚠️ {error}
              </div>
            )}
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="results-container">
            
            {/* Main Result Card */}
            <div className={`result-card result-${result.status}`}>
              <div className="result-icon" aria-hidden>
                {result.status === 'genuine' && '✓'}
                {result.status === 'fake' && '✗'}
                {result.status === 'error' && '!'}
              </div>
              <h4 className="result-title">{result.message}</h4>
            </div>

            {/* Results Grid Layout */}
            <div className="results-grid">
              
              {/* Left Column: Detection & OCR */}
              <div className="grid-column">
                
                {/* Detection Details */}
                {result.manufacturer && (
                  <div className="details-section">
                    <h5 className="section-title">📋 Detection Results</h5>
                    <div className="info-card">
                      <div className="info-row">
                        <span className="info-label">Manufacturer</span>
                        <span className="info-value manufacturer-name">{result.manufacturer}</span>
                      </div>
                      <div className="info-row">
                        <span className="info-label">Confidence</span>
                        <span className="info-value">
                          <div className="confidence-bar">
                            <div className="confidence-fill" style={{width: `${result.confidence * 100}%`}}></div>
                          </div>
                          <span className="confidence-text">{(result.confidence * 100).toFixed(1)}%</span>
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* OCR Extracted Text */}
                {result.ocr && result.ocr.success && result.ocr.extracted_text && result.ocr.extracted_text.length > 0 && (
                  <div className="details-section">
                    <h5 className="section-title">📝 Extracted Text</h5>
                    <div className="info-card">
                      <div className="text-chips">
                        {result.ocr.extracted_text.map((item, idx) => (
                          <span key={idx} className="chip">
                            {typeof item === 'string' ? item : item.text}
                            {item.confidence && (
                              <span className="chip-confidence">
                                {(item.confidence * 100).toFixed(0)}%
                              </span>
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: Marking & Verification */}
              <div className="grid-column">
                
                {/* IC Marking Analysis */}
                {result.marking && (
                  <div className="details-section">
                    <h5 className="section-title">🔍 IC Marking Analysis</h5>
                    <div className="info-card">
                      {result.marking.part_number && (
                        <div className="info-row">
                          <span className="info-label">Part Number</span>
                          <span className="info-value code-text">{result.marking.part_number}</span>
                        </div>
                      )}
                      {result.marking.suffix && (
                        <div className="info-row">
                          <span className="info-label">Suffix</span>
                          <span className="info-value code-text">{result.marking.suffix}</span>
                        </div>
                      )}
                      {result.marking.date_code && (
                        <div className="info-row">
                          <span className="info-label">Date Code</span>
                          <span className="info-value code-text">{result.marking.date_code}</span>
                        </div>
                      )}
                      {result.marking.country_code && (
                        <div className="info-row">
                          <span className="info-label">Country Code</span>
                          <span className="info-value code-text">{result.marking.country_code}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Verification Results */}
                {result.verification && (
                  <div className="details-section">
                    <h5 className="section-title">✓ Verification Results</h5>
                    <div className="info-card">
                      <div className="info-row">
                        <span className="info-label">Authenticity</span>
                        <span className={`authenticity-badge ${result.verification.authenticity}`}>
                          {result.verification.authenticity?.toUpperCase()}
                        </span>
                      </div>
                      <div className="info-row">
                        <span className="info-label">Confidence Score</span>
                        <span className="info-value">
                          {(result.verification.confidence_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="info-row">
                        <span className="info-label">Status</span>
                        <span className="info-value">{result.verification.verification_status}</span>
                      </div>
                      
                      {/* Web Verification Info */}
                      {result.verification.details?.verified_via && (
                        <>
                          <div className="info-row">
                            <span className="info-label">Verified Via</span>
                            <span className="info-value">
                              {result.verification.details.verified_via === 'Nexar Web Database' ? (
                                <span className="web-badge">🌐 {result.verification.details.verified_via}</span>
                              ) : (
                                <span className="local-badge">💾 {result.verification.details.verified_via}</span>
                              )}
                            </span>
                          </div>
                          {result.verification.details.verified_via === 'Nexar Web Database' && (
                            <>
                              {result.verification.details.in_stock !== undefined && (
                                <div className="info-row">
                                  <span className="info-label">Availability</span>
                                  <span className="info-value">
                                    {result.verification.details.in_stock ? (
                                      <span className="stock-badge in-stock">✓ In Stock ({result.verification.details.web_availability} units)</span>
                                    ) : (
                                      <span className="stock-badge out-of-stock">✗ Out of Stock</span>
                                    )}
                                  </span>
                                </div>
                              )}
                              {result.verification.details.sellers > 0 && (
                                <div className="info-row">
                                  <span className="info-label">Sellers</span>
                                  <span className="info-value">{result.verification.details.sellers} authorized sellers</span>
                                </div>
                              )}
                            </>
                          )}
                        </>
                      )}
                      
                      {/* Only show errors and warnings for GENUINE ICs */}
                      {result.status === 'genuine' && (
                        <>
                          {/* Errors */}
                          {result.verification.errors && result.verification.errors.length > 0 && (
                            <div className="verification-messages errors">
                              <p className="message-title">⚠️ Issues</p>
                              <ul>
                                {result.verification.errors.map((err, idx) => (
                                  <li key={idx}>{err}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {/* Warnings */}
                          {result.verification.warnings && result.verification.warnings.length > 0 && (
                            <div className="verification-messages warnings">
                              <p className="message-title">⚡ Warnings</p>
                              <ul>
                                {result.verification.warnings.map((warn, idx) => (
                                  <li key={idx}>{warn}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* New Inspection Button */}
            <button 
              className="btn-new-inspection"
              onClick={() => {
                onResultChange(null);
                onStatusChange('idle');
                setError(null);
              }}
            >
              🔄 New Inspection
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default InspectionDashboard;
