import React, { useState } from 'react';
import '../styles/InspectionDashboard.css';
import { detectManufacturerStream } from '../services/api';
import ImageUpload from './ImageUpload';

function UserValidationDashboard({ image, status, result, onStatusChange, onResultChange, onImageUpload, onClearImage }) {
  const [error, setError] = useState(null);
  
  const [expectedManufacturer, setExpectedManufacturer] = useState('');
  const [expectedPartNumber, setExpectedPartNumber] = useState('');
  
  const [progressSteps, setProgressSteps] = useState({
    preprocessing: { status: 'pending', message: 'Preprocessing image...', data: null },
    logo_detection: { status: 'pending', message: 'Detecting logo...', data: null },
    ocr_extraction: { status: 'pending', message: 'Extracting IC marking...', data: null },
    user_validation: { status: 'pending', message: 'Validating against user inputs...', data: null }
  });
  
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
    
    setProgressSteps({
      preprocessing: { status: 'pending', message: 'Preprocessing image...', data: null },
      logo_detection: { status: 'pending', message: 'Detecting logo...', data: null },
      ocr_extraction: { status: 'pending', message: 'Extracting IC marking...', data: null },
      user_validation: { status: 'pending', message: 'Validating against user inputs...', data: null }
    });
    
    try {
      const userInputs = {
        expectedManufacturer: expectedManufacturer.trim(),
        expectedPartNumber: expectedPartNumber.trim()
      };
      
      await detectManufacturerStream(image, (update) => {
        console.log('Progress update:', update);
        
        if (update.error) {
          setError(update.error);
          onStatusChange('idle');
          return;
        }
        
        if (update.step && update.step !== 'complete') {
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
        
        if (update.step === 'complete') {
          if (update.user_validation) {
            const validationStatus = update.user_validation.matches_expectations ? 'completed' : 'failed';
            const validationMessage = update.user_validation.matches_expectations 
              ? '✓ IC matches user expectations' 
              : '✗ IC does not match user expectations';
            
            setProgressSteps(prev => ({
              ...prev,
              user_validation: {
                status: validationStatus,
                message: validationMessage,
                data: {
                  manufacturer_match: update.user_validation.manufacturer_match,
                  part_number_match: update.user_validation.part_number_match,
                  matches: update.user_validation.matches_expectations
                }
              }
            }));
          }
          
          onStatusChange('completed');
          const finalResult = {
            status: update.final_status || 'fake',
            message: update.message,
            manufacturer: progressSteps.logo_detection?.data?.manufacturer,
            confidence: progressSteps.logo_detection?.data?.confidence,
            marking: progressSteps.ocr_extraction?.data ? {
              part_number: progressSteps.ocr_extraction.data.marking
            } : null,
            userValidation: update.user_validation || null
          };
          onResultChange(finalResult);
        }
      }, userInputs);
      
    } catch (err) {
      console.error('Validation error:', err);
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
      <div className="dashboard-header">
        <h3>✓ Batchwise IC Validation</h3>
        <span className={`status-badge status-${status}`}>
          {status === 'idle' && '⚪ Ready'}
          {status === 'processing' && '🔄 Validating...'}
          {status === 'completed' && '✅ Complete'}
        </span>
      </div>

      <div className="dashboard-content">
        <div className="two-col-layout">
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

          <section className="panel-card">
            <ImageUpload 
              onImageUpload={onImageUpload}
              currentImage={image}
              onClearImage={onClearImage}
            />
          </section>
        </div>

        {!result && status === 'idle' && (
          <div className="start-section" style={{paddingTop: 0}}>
            <button 
              className="btn-inspect btn-validate"
              onClick={handleValidateIC}
              disabled={status === 'processing' || !image || !expectedManufacturer || !expectedPartNumber}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
              Validate IC Marking
            </button>
            {error && (
              <div className="error-alert">
                ⚠️ {error}
              </div>
            )}
          </div>
        )}

        {(status === 'processing' || status === 'completed') && (
          <div className="progress-steps">
            <h4 className="progress-title">
              {status === 'processing' ? 'Validation Progress' : 'Validation Complete'}
            </h4>
            
            {['preprocessing', 'logo_detection', 'ocr_extraction', 'user_validation'].map((stepKey) => {
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
                      {stepKey === 'user_validation' && 'User Validation'}
                    </div>
                    <div className="step-message">{step.message}</div>
                    {step.data && (
                      <div className="step-data">
                        {step.data.manufacturer && <span className="data-chip">{step.data.manufacturer}</span>}
                        {step.data.marking && <span className="data-chip">{step.data.marking}</span>}
                        {step.data.status && <span className={`data-chip status-${step.data.status}`}>{step.data.status}</span>}
                        {step.data.manufacturer_match !== undefined && (
                          <span className={`match-badge ${step.data.manufacturer_match ? 'match' : 'mismatch'}`}>
                            {step.data.manufacturer_match ? '✓ Manufacturer Match' : '✗ Manufacturer Mismatch'}
                          </span>
                        )}
                        {step.data.part_number_match !== undefined && (
                          <span className={`match-badge ${step.data.part_number_match ? 'match' : 'mismatch'}`}>
                            {step.data.part_number_match ? '✓ Part Number Match' : '✗ Part Number Mismatch'}
                          </span>
                        )}
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

        {result && status === 'completed' && (
          <div className="results-container">
            {result.userValidation && (
              <>
                <div className={`big-result-banner result-${result.userValidation.matches_expectations ? 'genuine' : 'fake'}`}>
                  <div className="big-result-icon">
                    {result.userValidation.matches_expectations ? '✓' : '✗'}
                  </div>
                  <div className="big-result-text">
                    {result.userValidation.matches_expectations ? 'GENUINE' : 'FAKE'}
                  </div>
                </div>
              </>
            )}

            {!result.userValidation && (
              <div className={`big-result-banner result-${result.status}`}>
                <div className="big-result-icon">
                  {result.status === 'genuine' && '✓'}
                  {result.status === 'fake' && '✗'}
                  {result.status === 'error' && '!'}
                </div>
                <div className="big-result-text">
                  {result.status === 'genuine' ? 'GENUINE' : result.status === 'fake' ? 'COUNTERFEIT' : 'ERROR'}
                </div>
              </div>
            )}

            <button
              className="btn-new-inspection"
              onClick={() => {
                onStatusChange('idle');
                onResultChange(null);
                setExpectedManufacturer('');
                setExpectedPartNumber('');
                setProgressSteps({
                  preprocessing: { status: 'pending', message: 'Preprocessing image...', data: null },
                  logo_detection: { status: 'pending', message: 'Detecting logo...', data: null },
                  ocr_extraction: { status: 'pending', message: 'Extracting IC marking...', data: null },
                  user_validation: { status: 'pending', message: 'Validating against user inputs...', data: null }
                });
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
