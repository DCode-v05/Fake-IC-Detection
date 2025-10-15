import React, { useState } from 'react';
import '../styles/InspectionDashboard.css';
import { detectManufacturerStream } from '../services/api';

/**
 * InspectionDashboard Component - Progressive UI
 * Shows real-time verification steps as they complete
 */
function InspectionDashboard({ image, status, result, onStatusChange, onResultChange }) {
  const [error, setError] = useState(null);
  const [progressSteps, setProgressSteps] = useState({
    preprocessing: { status: 'pending', message: 'Preprocessing image...', data: null },
    logo_detection: { status: 'pending', message: 'Detecting logo...', data: null },
    ocr_extraction: { status: 'pending', message: 'Extracting IC marking...', data: null },
    verification: { status: 'pending', message: 'Verifying authenticity...', data: null }
  });
  
  const handleStartInspection = async () => {
    if (!image) {
      setError('Please upload an IC image first');
      return;
    }

    onStatusChange('processing');
    setError(null);
    
    // Reset progress steps
    setProgressSteps({
      preprocessing: { status: 'pending', message: 'Preprocessing image...', data: null },
      logo_detection: { status: 'pending', message: 'Detecting logo...', data: null },
      ocr_extraction: { status: 'pending', message: 'Extracting IC marking...', data: null },
      verification: { status: 'pending', message: 'Verifying authenticity...', data: null }
    });
    
    try {
      await detectManufacturerStream(image, (update) => {
        console.log('Progress update:', update);
        
        if (update.error) {
          setError(update.error);
          onStatusChange('idle');
          return;
        }
        
        // Update the specific step
        if (update.step && update.step !== 'complete') {
          // Only update if the step exists in our progressSteps
          setProgressSteps(prev => {
            if (prev.hasOwnProperty(update.step)) {
              return {
                ...prev,
                [update.step]: {
                  status: update.status,
                  message: update.message,
                  data: update.data || null
                }
              };
            }
            return prev;
          });
        }
        
        // Handle completion
        if (update.step === 'complete') {
          onStatusChange('completed');
          // Build final result from progress data
          const finalResult = {
            status: update.final_status || 'fake',
            message: update.message,
            manufacturer: progressSteps.logo_detection?.data?.manufacturer,
            confidence: progressSteps.logo_detection?.data?.confidence,
          };
          onResultChange(finalResult);
        }
      });
      
    } catch (err) {
      console.error('Inspection error:', err);
      onStatusChange('idle');
      setError(err.message);
    }
  };

  const getStepIcon = (stepStatus) => {
    if (stepStatus === 'completed') return '✅';
    if (stepStatus === 'failed') return '❌';
    if (stepStatus === 'processing') return '🔄';
    return '⏳';
  };

  const getStepClass = (stepStatus) => {
    if (stepStatus === 'completed') return 'step-completed';
    if (stepStatus === 'failed') return 'step-failed';
    if (stepStatus === 'processing') return 'step-processing';
    return 'step-pending';
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
        {!result && status === 'idle' && (
          <div className="start-section">
            <button 
              className="btn-inspect"
              onClick={handleStartInspection}
              disabled={!image}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
              </svg>
              Start Inspection
            </button>
            {error && (
              <div className="error-alert">
                ⚠️ {error}
              </div>
            )}
          </div>
        )}

        {/* Progressive Steps Display - Show during processing AND after completion */}
        {(status === 'processing' || status === 'completed') && (
          <div className="progress-steps">
            <h4 className="progress-title">
              {status === 'processing' ? 'Analysis Progress' : 'Analysis Complete'}
            </h4>
            
            {/* Render steps in fixed order */}
            {['preprocessing', 'logo_detection', 'ocr_extraction', 'verification'].map((stepKey) => {
              const step = progressSteps[stepKey];
              if (!step) return null;
              
              return (
                <div key={stepKey} className={`progress-step ${getStepClass(step.status)}`}>
                  <div className="step-icon">{getStepIcon(step.status)}</div>
                  <div className="step-content">
                    <div className="step-label">
                      {stepKey === 'preprocessing' && 'Image Preprocessed'}
                      {stepKey === 'logo_detection' && 'Logo Detection using YOLO'}
                      {stepKey === 'ocr_extraction' && 'IC Marking Extraction using PaddleOCR'}
                      {stepKey === 'verification' && 'Database Verification'}
                    </div>
                    <div className="step-message">{step.message}</div>
                    {step.data && (
                      <div className="step-data">
                        {step.data.manufacturer && <span className="data-chip">{step.data.manufacturer}</span>}
                        {step.data.marking && <span className="data-chip">{step.data.marking}</span>}
                        {step.data.status && <span className={`data-chip status-${step.data.status}`}>{step.data.status}</span>}
                        {step.data.source && (
                          <span className={`data-chip source-${step.data.source}`}>
                            {step.data.source === 'nexar' ? '🌐 Nexar Web' : '💾 Local DB'}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  {step.status === 'processing' && (
                    <div className="step-spinner"></div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Final Result Card - Show after completion */}
        {result && status === 'completed' && (
          <div className="final-result-section">
            {/* Big Genuine/Fake Result */}
            <div className={`big-result-banner result-${result.status}`}>
              <div className="big-result-icon">
                {result.status === 'genuine' ? '✓' : '✗'}
              </div>
              <div className="big-result-text">
                {result.status === 'genuine' ? 'GENUINE' : 'FAKE'}
              </div>
            </div>

            {/* New Inspection Button */}
            <button 
              className="btn-new-inspection"
              onClick={() => {
                onResultChange(null);
                onStatusChange('idle');
                setError(null);
                setProgressSteps({
                  preprocessing: { status: 'pending', message: 'Preprocessing image...', data: null },
                  logo_detection: { status: 'pending', message: 'Detecting logo...', data: null },
                  ocr_extraction: { status: 'pending', message: 'Extracting IC marking...', data: null },
                  verification: { status: 'pending', message: 'Verifying authenticity...', data: null }
                });
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
