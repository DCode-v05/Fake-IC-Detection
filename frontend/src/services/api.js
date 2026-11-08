const API_BASE_URL = 'http://localhost:8000/api';

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

export const detectManufacturer = async (imageFile, userInputs = {}) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    if (userInputs.expectedManufacturer) {
      formData.append('expected_manufacturer', userInputs.expectedManufacturer);
    }
    if (userInputs.expectedPartNumber) {
      formData.append('expected_part_number', userInputs.expectedPartNumber);
    }

    const response = await fetch(`${API_BASE_URL}/detect/`, {
      method: 'POST',
      body: formData,
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

export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(1)}%`;
};

export const getStatusColor = (status) => {
  switch (status?.toLowerCase()) {
    case 'genuine':
      return '#10b981';
    case 'fake':
      return '#ef4444';
    default:
      return '#6b7280';
  }
};

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
