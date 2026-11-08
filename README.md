# IC Marking Detection System

**A full-stack web app that checks whether an integrated circuit is genuine. It reads the chip's surface markings, identifies the manufacturer, and validates the part against a known-good database, with a web-database fallback.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white) ![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![PaddleOCR](https://img.shields.io/badge/PaddleOCR-0062B0?style=flat) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=flat) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

## Overview

Counterfeit and re-marked ICs are a recurring problem in the electronics supply chain. A fake op-amp or microcontroller can pass a visual glance and still fail in the field, and checking by eye is slow and inconsistent. This project automates a first pass of that inspection. It reads and sanity-checks the marking, and it does not replace electrical or decapsulation testing.

You upload a photo of a chip. The backend runs OCR over the chip surface, works out which manufacturer the markings belong to, parses the fields out of the marking (part number, suffix, date code, country code), and checks them against a database of genuine parts. If the part isn't in the local database, it falls back to the Nexar electronic-parts API before deciding. The result comes back as one verdict, **genuine** or **fake**, with a confidence score and a breakdown of which checks passed and failed.

It's a solo project: a Django REST backend, a React frontend, a multi-factor verification engine, and a separate YOLOv8 training pipeline (with synthetic-data generation) that produces a custom logo-detection model. Built September–October 2025.

## Key Features

- **Image upload and inspection**: drag and drop a chip photo into the React UI and get an authenticity verdict back.
- **OCR-based manufacturer identification**: PaddleOCR reads the text on the chip, and per-manufacturer keyword and regex patterns map it to one of six manufacturers (Texas Instruments, STMicroelectronics, NXP, ON Semiconductor, Microchip, Infineon).
- **Marking parser**: pulls structured fields out of the raw OCR text: part number, package suffix, `YYWW` date code and country code.
- **Multi-factor verification**: each field is validated on its own. That means a part-number lookup, a valid-suffix check, a date-code check (not in the future, not older than 20 years, week between 1 and 53) and a country-of-origin check against the expected manufacturing locations for that part.
- **Confidence scoring with a fail-safe default**: the verifier averages the per-field confidences and the manufacturer-detection confidence, and needs a score of at least 0.70 to call a chip genuine. Anything below that, or any error, defaults to **fake**.
- **Nexar fallback**: if a part number isn't in the local JSON database, the system asks the Nexar API whether the part exists and how many authorized sellers carry it before issuing a verdict.
- **Two detection endpoints**: a standard JSON endpoint and a streaming (Server-Sent Events) endpoint that pushes progress for each step (logo detection, OCR extraction, verification) so the UI can show it live.
- **Expected-value validation**: you can optionally enter the manufacturer and part number you think the chip is, and the system reports whether the detected values match.
- **Inspection history and statistics**: every inspection is stored in a JSON history file, with endpoints to fetch past inspections and aggregate stats.
- **Custom YOLOv8n logo detector**: a separately trained object-detection model with its own synthetic-data pipeline (details below).

## How It Works

The repo has three parts: the served inspection pipeline (Django + PaddleOCR), the verification engine, and the YOLOv8 training pipeline.

### Inspection pipeline (the live `/api/detect/` path)

1. **Upload and validation**: the image is received, validated (type and size) and saved to disk.
2. **Combined OCR pass**: one PaddleOCR call detects the manufacturer text and extracts the full chip marking. It was collapsed from two OCR passes into one for speed (`detect_and_extract_combined`).
3. **Manufacturer identification**: the OCR text is matched against per-manufacturer keyword patterns (for example `\bTI\b`, `\bTEXAS\s*INSTRUMENTS?\b` for Texas Instruments). There's also a part-number-prefix fallback, such as `LM`/`SN`/`TPS` for TI and `STM`/`L###` for ST, for when the brand name isn't legible.
4. **Marking parsing**: `marking_parser.py` splits the raw text into part number, suffix, date code (`YYWW`) and country code.
5. **Verification**: the parsed fields go to the verifier (below), which returns the verdict.
6. **Response**: the API returns the verdict, confidence score, per-field validation details, the manufacturer, the raw OCR text and, optionally, the comparison with your expected values.

`/api/detect-stream/` does the same but streams progress as server-sent events. Inspections are appended to `backend/database/inspections.json`.

Note on logo detection: the served API identifies the manufacturer from **OCR text**, not from the YOLO model. A YOLOv8 logo detector is in the repo (`backend/api/services/yolo_logo_detector.py`, weights in `trained_models/best.pt`), but the detect views import only the OCR detector.

### Verification engine (`ic_verifier.py`)

This is where the genuine/fake decision is made. The verifier is a singleton that loads the parts database once and runs a series of independent checks:

- **Part number**: case-insensitive exact match against the manufacturer's parts in `backend/database/ic_parts_database.json`. If it isn't found locally, it asks Nexar (needs the `NEXAR_ACCESS_TOKEN` environment variable; without it that check is skipped). If Nexar confirms the part, that counts as verified, with seller and availability info attached. If neither has it, the part fails.
- **Suffix**: checks the package suffix against the valid suffixes for the matched part.
- **Date code**: validates the `YYWW` code. Years or weeks in the future are suspicious, chips older than 20 years are flagged as possibly re-marked, and week numbers outside 1–53 are rejected.
- **Country code**: checks the marking's country code against the expected manufacturing locations recorded for that part.

Each passing check adds its confidence to a list, the manufacturer-detection confidence is added too, and the mean becomes the overall score. The decision rule is conservative: **any error means fake**, a score of **0.70 or more means genuine**, and anything else means **fake**. The result also carries warnings, errors and a per-field breakdown, so the verdict can be explained.

### YOLOv8 logo-detection training (`model_training/`)

Separate from the served pipeline, the repo has a training pipeline for a custom logo detector:

- **Synthetic data generation**: real manufacturer logos (a small set under `Dataset/`, one folder per brand) are pasted onto generated metallic-looking backgrounds with rotation, scaling and opacity changes to imitate chips photographed under different conditions, and exported in YOLO label format (the generated set is in `dataset_synth/`).
- **Training**: fine-tunes `yolov8n.pt` (nano) with `epochs=60`, `imgsz=640` and `batch_size=16` (8 on CPU), picked as the nano variant so inference can be fast on small hardware.
- **Output**: the trained weights are in `trained_models/best.pt`, and the Ultralytics run artifacts (curves, confusion matrices, `results.csv`) are under `runs/detect/yolov8_logo_run/`.

`model_training/README.md` also describes an older EfficientNet contour-crop classifier (v1, ~70.9% accuracy) that the YOLO approach replaced. That path doesn't run: `model_training/train.py` imports `models/efficientnet_model.py`, which isn't in the repo.

## Results / Highlights

From the final epoch (60) of `runs/detect/yolov8_logo_run/results.csv`:

- **mAP@50 ≈ 0.98**
- **mAP@50–95 ≈ 0.975**
- **Precision ≈ 0.96**, **Recall ≈ 0.96**
- Trained `best.pt` model, ~6 MB (YOLOv8n)
- Six manufacturers supported end to end
- Single-OCR-pass inspection (halved from the original two-pass approach)

Caveat: the training and validation images are synthetic, built from a small set of real logos, so these metrics say little about real chip photos. The earlier README gave mAP@50–95 as 0.77, which was actually the `dfl_loss` column.

## Tech Stack

- **Languages:** Python 3.x, JavaScript (ES6+)
- **Frameworks:** Django (REST API, `django-cors-headers`), React 18 (Create React App)
- **Data / ML:** PaddleOCR + PaddlePaddle (OCR), Ultralytics YOLOv8 (logo detection), PyTorch / torchvision, OpenCV, Pillow, NumPy
- **External services:** Nexar parts API (web-database verification fallback)
- **Storage:** JSON parts database + JSON inspection history (no external DB required to run)
- **Frontend libraries:** Axios for the API layer; plain CSS for styling

## Getting Started

### Prerequisites

- Python 3.10+ and `pip`
- Node.js 16+ and `npm`
- The trained model at `trained_models/best.pt` (included) if you want to run the YOLO detector
- Optional: a `NEXAR_ACCESS_TOKEN` environment variable to enable the Nexar web-database fallback (the app runs without it; that check is just skipped)

### Installation

```bash
git clone https://github.com/DCode-v05/Fake-IC-Detection.git
cd Fake-IC-Detection

# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### Running

```bash
# Terminal 1: backend (http://localhost:8000)
cd backend
python manage.py runserver

# Terminal 2: frontend (http://localhost:3000)
cd frontend
npm start
```

The first inspection is slow while PaddleOCR downloads and caches its models. The frontend has the API URL hard-coded as `http://localhost:8000/api` in `frontend/src/services/api.js`.

## Usage

1. Open `http://localhost:3000`.
2. Upload a clear photo of an IC chip (drag-and-drop or file picker).
3. Optionally enter the manufacturer and part number you expect, so the system can confirm whether the chip matches your expectation.
4. Run the inspection. The dashboard shows the detected manufacturer, the raw and parsed markings, each validation step, and the final **genuine / fake** verdict with its confidence score.

### API endpoints

All under `/api/`:

| Endpoint | Method | What it does |
| --- | --- | --- |
| `detect/` | POST | Upload an image, get the full verdict as JSON |
| `detect-stream/` | POST | Same, but streams step-by-step progress (SSE) |
| `history/` | GET | Returns past inspections |
| `statistics/` | GET | Aggregate inspection stats |
| `health/` | GET | Health check |

Example:

```bash
curl -X POST http://localhost:8000/api/detect/ \
  -F "image=@chip.jpg" \
  -F "expected_manufacturer=Texas Instruments" \
  -F "expected_part_number=LM324"
```

## Project Structure

```
Fake-IC-Detection/
├── backend/                          # Django backend
│   ├── api/
│   │   ├── services/
│   │   │   ├── ocr_logo_detector.py  # PaddleOCR text -> manufacturer (served path)
│   │   │   ├── ocr_extractor.py      # OCR text extraction
│   │   │   ├── yolo_logo_detector.py # YOLOv8 logo detector (loads best.pt, not used by the views)
│   │   │   ├── ic_verifier.py        # multi-factor genuine/fake verification
│   │   │   └── nexar_service.py      # Nexar web-database fallback
│   │   ├── utils/
│   │   │   ├── marking_parser.py     # part no / suffix / date code / country
│   │   │   ├── database.py
│   │   │   └── image_utils.py
│   │   ├── views.py                  # detect, detect-stream, history, stats, health
│   │   └── urls.py
│   ├── ic_detection/                 # Django project (settings, urls, wsgi)
│   ├── database/
│   │   ├── ic_parts_database.json    # genuine-parts reference data (6 manufacturers)
│   │   └── inspections.json          # inspection history store
│   ├── download_models.py
│   ├── manage.py
│   └── requirements.txt
├── frontend/                         # React app
│   └── src/
│       ├── components/               # ImageUpload, InspectionDashboard, Header,
│       │                             #   UserValidationDashboard, ExpectedValuesInspector
│       ├── services/api.js           # axios API layer
│       └── styles/
├── model_training/
│   ├── yolo_logo_train_pipeline.py   # synthetic data -> YOLO labels -> train
│   ├── train.py                      # older EfficientNet path (broken: missing models/efficientnet_model.py)
│   ├── config/training_config.py
│   ├── README.md                     # describes the older EfficientNet pipeline
│   └── yolov8n.pt, yolo11n.pt        # base weights
├── trained_models/best.pt            # trained YOLOv8n logo model
├── Dataset/                          # real manufacturer logos (by brand)
├── dataset_synth/                    # generated synthetic YOLO dataset
├── IC/                               # two sample chip photos
├── runs/detect/                      # training artifacts + results.csv (yolov8_logo_run, yolov8_logo_run2)
└── README.md
```

**Notes:** `backend/ic_detection/settings.py` has `DEBUG = True`, `ALLOWED_HOSTS = ['*']` and a development secret key, so change these before deploying. The parts database covers a limited set of parts per manufacturer, and anything not in it and not found on Nexar is marked fake.

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
