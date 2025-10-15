# IC Marking Detection - Django Backend

Modular Django backend for IC manufacturer logo detection and verification.

## Overview

This backend handles:
- Image upload from React frontend
- Logo detection and cropping from IC images
- Manufacturer prediction using trained EfficientNet model
- Results storage in JSON database
- Inspection history and statistics

## Architecture

### Modular Structure

```
backend/
├── ic_detection/              # Django project
│   ├── settings.py           # Configuration (CORS, media, etc.)
│   ├── urls.py               # Main URL routing
│   └── wsgi.py               # WSGI application
│
├── api/                       # Main API app
│   ├── views.py              # API endpoints and request handlers
│   ├── urls.py               # API URL routing
│   ├── apps.py               # App configuration
│   │
│   ├── services/             # Business logic (modular)
│   │   ├── logo_detector.py  # Logo detection & cropping
│   │   └── model_predictor.py # Model inference
│   │
│   └── utils/                # Helper utilities
│       ├── image_utils.py    # Image processing helpers
│       └── database.py       # JSON database operations
│
├── media/                     # Uploaded images storage
│   └── uploads/
│
├── database/                  # JSON database
│   └── inspections.json
│
├── manage.py                  # Django management script
└── requirements.txt           # Python dependencies
```

## API Endpoints

### 1. Detect Manufacturer
**POST** `/api/detect/`

Upload IC image and get manufacturer prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` (file)

**Response:**
```json
{
  "success": true,
  "inspection_id": "INS_20251007123456_1",
  "result": {
    "manufacturer": "Texas Instruments",
    "confidence": 0.9876,
    "confidence_percentage": "98.76%",
    "status": "genuine",
    "message": "Detected as Texas Instruments logo with 98.76% confidence"
  },
  "top_predictions": [
    {
      "manufacturer": "Texas Instruments",
      "confidence": 0.9876,
      "confidence_percentage": "98.76%"
    },
    {
      "manufacturer": "STMicroelectronics",
      "confidence": 0.0089,
      "confidence_percentage": "0.89%"
    },
    {
      "manufacturer": "Infineon",
      "confidence": 0.0023,
      "confidence_percentage": "0.23%"
    }
  ],
  "all_probabilities": {
    "Infineon": 0.0023,
    "Microchip": 0.0008,
    "NXP Semiconductors": 0.0003,
    "ON Semiconductor": 0.0001,
    "STMicroelectronics": 0.0089,
    "Texas Instruments": 0.9876
  },
  "cropped_logo": "data:image/png;base64,..."
}
```

### 2. Get Inspection History
**GET** `/api/history/?limit=50`

Retrieve past inspection records.

**Response:**
```json
{
  "success": true,
  "count": 10,
  "inspections": [...]
}
```

### 3. Get Statistics
**GET** `/api/statistics/`

Get overall system statistics.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_inspections": 150,
    "manufacturer_distribution": {
      "Texas Instruments": 45,
      "STMicroelectronics": 32,
      ...
    },
    "average_confidence": 0.9234,
    "last_updated": "2025-10-07T12:34:56"
  }
}
```

### 4. Health Check
**GET** `/api/health/`

Check if all services are running.

**Response:**
```json
{
  "success": true,
  "status": "healthy",
  "services": {
    "logo_detector": true,
    "predictor": true,
    "database": true
  }
}
```

## Module Details

### 1. Logo Detector (`api/services/logo_detector.py`)
**Purpose:** Detect and crop logo regions from IC images

**Features:**
- Contour-based detection
- Edge-based detection
- Smart detection (tries multiple strategies)
- Logo enhancement for better prediction
- Automatic fallback to center crop

**Methods:**
- `detect_and_crop_logo()` - Main detection method
- `detect_logo_smart()` - Smart multi-strategy detection
- `enhance_logo_for_detection()` - Image enhancement

### 2. Model Predictor (`api/services/model_predictor.py`)
**Purpose:** Load trained model and predict manufacturer

**Features:**
- Loads trained EfficientNet model
- Inference with confidence scores
- Top-K predictions
- Batch prediction support
- Confidence threshold checking

**Methods:**
- `predict()` - Main prediction
- `get_top_k_predictions()` - Get top K results
- `is_confident_prediction()` - Confidence check

### 3. Image Utils (`api/utils/image_utils.py`)
**Purpose:** Image processing utilities

**Functions:**
- `save_uploaded_image()` - Save uploads to disk
- `validate_image()` - Validate uploaded files
- `image_to_base64()` - Convert to base64
- `resize_image()` - Resize with aspect ratio

### 4. Database (`api/utils/database.py`)
**Purpose:** JSON-based storage for inspection results

**Features:**
- Simple JSON file storage
- Inspection history
- Statistics generation
- Manufacturer filtering

**Methods:**
- `add_inspection()` - Save new inspection
- `get_all_inspections()` - Retrieve history
- `get_statistics()` - Generate stats

## Installation

1. Navigate to backend directory:
```cmd
cd "d:\Deni\Mr. Tech\AI\Projects\Fake IC Marking Detection\backend"
```

2. Install dependencies:
```cmd
pip install -r requirements.txt
```

3. Run migrations (if using database in future):
```cmd
python manage.py migrate
```

4. Start development server:
```cmd
python manage.py runserver
```

Backend will run at: `http://localhost:8000`

## Testing API Endpoints

### Using cURL:
```cmd
curl -X POST http://localhost:8000/api/detect/ -F "image=@path/to/ic_image.jpg"
```

### Using Python requests:
```python
import requests

url = "http://localhost:8000/api/detect/"
files = {'image': open('ic_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Configuration

### CORS Settings
Edit `ic_detection/settings.py`:
```python
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # React frontend
]
```

### File Upload Settings
```python
FILE_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10 MB
```

### Model Path
Default: `../trained_models/logo_classifier_best.pth`

To change, modify `model_predictor.py`:
```python
predictor = ManufacturerPredictor(model_path="/path/to/model.pth")
```

## Workflow

1. **Image Upload** → Frontend sends image via POST request
2. **Validation** → Backend validates file type and size
3. **Logo Detection** → CV algorithms detect and crop logo
4. **Enhancement** → Logo is enhanced for better recognition
5. **Prediction** → Model predicts manufacturer
6. **Storage** → Results saved to JSON database
7. **Response** → JSON response sent to frontend

## Error Handling

All endpoints return consistent error format:
```json
{
  "success": false,
  "error": "Error description"
}
```

HTTP Status Codes:
- 200: Success
- 400: Bad request (validation error)
- 500: Internal server error

## Database Schema

JSON structure in `database/inspections.json`:
```json
{
  "inspections": [
    {
      "id": "INS_20251007123456_1",
      "timestamp": "2025-10-07T12:34:56",
      "original_filename": "ic_chip.jpg",
      "manufacturer": "Texas Instruments",
      "confidence": 0.9876,
      "all_probabilities": {...},
      "top_predictions": [...]
    }
  ],
  "metadata": {
    "total_inspections": 150,
    "last_updated": "2025-10-07T12:34:56"
  }
}
```

## Modular Benefits

✅ **Easy Debugging:** Each module is isolated and testable  
✅ **Maintainable:** Clear separation of concerns  
✅ **Extensible:** Easy to add new features  
✅ **Testable:** Each service can be unit tested independently

## Next Steps

1. Connect React frontend to backend API
2. Add authentication if needed
3. Add validation dataset for accuracy verification
4. Implement caching for faster responses
5. Add comprehensive error logging
6. Deploy to production server

## Technology Stack

- **Framework:** Django 4.2+
- **Deep Learning:** PyTorch, timm
- **Computer Vision:** OpenCV
- **Image Processing:** Pillow
- **Database:** JSON (file-based)
- **API:** Django REST views

## Production Deployment

For production:
1. Set `DEBUG = False` in settings
2. Configure `ALLOWED_HOSTS`
3. Use proper database (PostgreSQL/MySQL)
4. Set up static file serving
5. Configure HTTPS
6. Add authentication/authorization
7. Set up proper logging

---

**Status:** ✅ Backend Complete and Ready for Integration  
**API Endpoints:** 4 endpoints implemented  
**Modular Services:** 4 independent modules
