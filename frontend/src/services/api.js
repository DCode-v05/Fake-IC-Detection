/**
 * API Service for IC Marking Detection Backend
 * Handles all communication with Django backend
 */

const API_BASE_URL = 'http://localhost:8000/api';

/**
 * Stream IC detection with progressive updates (SSE)
 * 
 * @param {File} imageFile - The IC image file to analyze
 * @param {Function} onProgress - Callback for each progress update
 * @param {Object} userInputs - Optional user expectations
 * @returns {Promise<void>}
 */
export const detectManufacturerStream = async (imageFile, onProgress, userInputs = {}) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    if (userInputs.expectedManufacturer) {
      formData.append('expected_manufacturer', userInputs.expectedManufacturer);
    }
    if (userInputs.expectedPartNumber) {
      formData.append('expected_part_number', userInputs.expectedPartNumber);
    }

    const response = await fetch(`${API_BASE_URL}/detect-stream/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Stream detection failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.substring(6));
          onProgress(data);
        }
      }
    }

  } catch (error) {
    console.error('API Error - detectManufacturerStream:', error);
    throw error;
  }
};

/**
 * Upload IC image and get detection results
 * Includes: Logo detection, OCR extraction, Marking validation
 * 
 * @param {File} imageFile - The IC image file to analyze
 * @param {Object} userInputs - Optional user expectations { expectedManufacturer, expectedPartNumber }
 * @returns {Promise<Object>} Detection results with logo, OCR, and verification data
 */
export const detectManufacturer = async (imageFile, userInputs = {}) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    // Add user inputs if provided
    if (userInputs.expectedManufacturer) {
      formData.append('expected_manufacturer', userInputs.expectedManufacturer);
    }
    if (userInputs.expectedPartNumber) {
      formData.append('expected_part_number', userInputs.expectedPartNumber);
    }

    const response = await fetch(`${API_BASE_URL}/detect/`, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - browser will set it automatically with boundary
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Detection failed');
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('API Error - detectManufacturer:', error);
    throw error;
  }
};

/**
 * Check backend health status
 * 
 * @returns {Promise<Object>} Health status with service availability
 */
export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health/`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error('Health check failed');
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('API Error - checkHealth:', error);
    throw error;
  }
};

/**
 * Get inspection history
 * 
 * @param {number} limit - Maximum number of inspections to retrieve
 * @returns {Promise<Object>} List of past inspections
 */
export const getInspectionHistory = async (limit = 50) => {
  try {
    const response = await fetch(`${API_BASE_URL}/history/?limit=${limit}`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error('Failed to fetch history');
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('API Error - getInspectionHistory:', error);
    throw error;
  }
};

/**
 * Get system statistics
 * 
 * @returns {Promise<Object>} System statistics and analytics
 */
export const getStatistics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/statistics/`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error('Failed to fetch statistics');
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('API Error - getStatistics:', error);
    throw error;
  }
};

/**
 * Format confidence percentage
 * 
 * @param {number} confidence - Confidence value (0-1)
 * @returns {string} Formatted percentage string
 */
export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(1)}%`;
};

/**
 * Get status color based on authenticity
 * 
 * @param {string} status - Authenticity status (only 'genuine' or 'fake')
 * @returns {string} Color code for UI
 */
export const getStatusColor = (status) => {
  switch (status?.toLowerCase()) {
    case 'genuine':
      return '#10b981'; // Green - Genuine IC
    case 'fake':
      return '#ef4444'; // Red - Fake IC
    default:
      return '#6b7280'; // Gray - Unknown
  }
};

/**
 * Get status icon based on authenticity
 * 
 * @param {string} status - Authenticity status (only 'genuine' or 'fake')
 * @returns {string} Icon/emoji for UI
 */
export const getStatusIcon = (status) => {
  switch (status?.toLowerCase()) {
    case 'genuine':
      return '✅';
    case 'fake':
      return '❌';
    default:
      return '⚪';
  }
};

export default {
  detectManufacturer,
  checkHealth,
  getInspectionHistory,
  getStatistics,
  formatConfidence,
  getStatusColor,
  getStatusIcon,
};
