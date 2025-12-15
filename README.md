# IC Marking Detection System

## Project Description
This project aims to detect and verify the authenticity of Integrated Circuits (ICs) using advanced machine learning techniques. By analyzing manufacturer logos and extracting chip markings (OCR), the system validates parts against a known genuine database. The solution is deployed as a full-stack web application with a Django backend and React frontend, providing real-time authenticity assessments (Genuine/Fake/Uncertain) to help prevent the use of counterfeit components.

---

## Project Details

### Problem Statement
The electronics supply chain is plagued by counterfeit Integrated Circuits (ICs), which can lead to critical system failures. Identifying these fakes manually is error-prone and slow. This project automates the verification process by combining object detection for logos and OCR for text markings, cross-referencing them with a verified database to ensure component authenticity.

### Data Preprocessing
- **Logo Dataset:** Synthetic data generation is used to create training samples by pasting authentic logos onto metallic backgrounds with various augmentations.
- **Image Augmentation:** Rotation, scaling, and opacity adjustments are applied to simulate real-world conditions.
- **OCR Preprocessing:** Images undergo CLAHE (Contrast Limited Adaptive Histogram Equalization), denoising, and sharpening to improve text extraction accuracy.

### Model Training & Evaluation
- **Models Used:**
  - **YOLOv8 (Ultralytics):** Custom-trained for manufacturer logo detection.
  - **PaddleOCR:** For Optical Character Recognition (text extraction).
- **Training Pipeline:** Automated pipeline generates synthetic data and fine-tunes the YOLOv8 model.
- **Evaluation Metrics:** Mean Average Precision (mAP), Precision, and Recall are used to validate the logo detection model.
- **Best Model:** YOLOv8n (Nano) optimized for potential real-time inference on edge devices.

### Hyperparameter Tuning
The YOLOv8 model training process is configurable to optimize performance. Key parameters include:
```
{
  'epochs': 60,
  'imgsz': 640,
  'batch_size': 16,  # 8 for CPU
  'model': 'yolov8n.pt',
  'augmentation': {
     'min_scale': 0.03,
     'max_scale': 0.18
  }
}
```

### Visualizations
- **Detection Results:** Bounding boxes drawn around detected logos with confidence scores.
- **Authenticity Status:** Color-coded badges (Green for Genuine, Red for Fake, Yellow for Uncertain).
- **Extracted Text:** Display of raw text read from the chip surface alongside parsed values.

### Web Application
The application features a modern React-based UI that provides:
- **Image Upload:** Drag-and-drop interface for IC images.
- **Real-time Analysis:** Immediate visual feedback on logo detection and OCR text.
- **Detailed Reports:** Breakdown of part numbers, date codes, and country of origin validation.
- **Responsive Design:** Accessible on desktop and tablet devices.

---

## Tech Stack
- **Languages:** Python 3.x, JavaScript (ES6+)
- **Frontend:** React, Axios, CSS Modules
- **Backend:** Django Rest Framework
- **AI/ML:** PyTorch, Ultralytics YOLOv8, PaddleOCR, OpenCV
- **Database:** SQLite (Default), JSON-based verified parts database
- **Tools:** npm, pip

---

## Getting Started

### 1. Clone the repository
```
git clone <repository-url>
cd Fake-IC-Detection
```

### 2. Install dependencies

**Backend:**
```
cd backend
pip install -r requirements.txt
```

**Frontend:**
```
cd frontend
npm install
```

### 3. Run the Application

**Start Backend:**
```
cd backend
python manage.py runserver
```

**Start Frontend:**
```
cd frontend
npm start
```
The application will be available at `http://localhost:3000`.

---

## Usage
- **Upload Image:** Navigate to the home page and upload a clear image of an IC chip.
- **Inspect:** Click "Start Inspection" to trigger the AI analysis.
- **Review Results:** Check the "Detection Results" panel for the manufacturer logo and the "Verification" panel for the authenticity status.
- **Train Model:** Use the `model_training` scripts to retrain the YOLOv8 model with new datasets if needed.

---

## Project Structure
```
Fake-IC-Detection/
│
├── backend/                  # Django Backend
│   ├── api/                  # API logic and ML services
│   ├── database/             # JSON database for IC parts
│   ├── manage.py             # Django entry point
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React Frontend
│   ├── public/               # Static assets
│   ├── src/                  # React components and logic
│   └── package.json          # Node dependencies
├── model_training/           # Training pipeline
│   ├── yolo_logo_train_pipeline.py
│   └── config/               # Training configurations
├── trained_models/           # Saved model artifacts
│   └── best.pt               # Production YOLOv8 model
├── Dataset/                  # Raw training images (Logos)
├── dataset_synth/            # Generated synthetic training data
├── runs/                     # Training runs and logs
└── README.md                 # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
