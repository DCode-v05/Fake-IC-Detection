# IC Marking Detection System

**A full-stack web app that checks whether an integrated circuit is genuine — it reads the chip's surface markings, identifies the manufacturer, and validates the part against a known-good database (with a live web-database fallback).**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Django](https://img.shields.io/badge/Django-092E20?style=flat&logo=django&logoColor=white) ![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![PaddleOCR](https://img.shields.io/badge/PaddleOCR-0062B0?style=flat) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=flat) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

## Overview

Counterfeit and re-marked ICs are a recurring problem in the electronics supply chain — a fake op-amp or microcontroller can pass a visual glance but fail in the field. Catching them by eye is slow and inconsistent. This project automates the first pass of that inspection.

You upload a photo of a chip. The backend runs OCR over the chip surface, figures out which manufacturer the markings belong to, parses the structured fields out of the marking (part number, suffix, date code, country code), and checks them against a database of genuine parts. If the part isn't in the local database, it falls back to the Nexar electronic-parts web API before deciding. The result comes back as a single verdict — **genuine** or **fake** — with a confidence score and a breakdown of which checks passed and which failed.

It's a solo project: Django REST backend, React frontend, a multi-factor verification engine, and a separate YOLOv8 training pipeline (synthetic-data generation included) that produces a custom logo-detection model. Built September–October 2025.

## Key Features

- **Image upload and inspection** — drag-and-drop a chip photo through the React UI and get an authenticity verdict back.
- **OCR-based manufacturer identification** — PaddleOCR reads text off the chip, and a set of per-manufacturer keyword/regex patterns maps that text to one of six manufacturers (Texas Instruments, STMicroelectronics, NXP, ON Semiconductor, Microchip, Infineon).
- **Marking parser** — pulls structured fields out of the raw OCR text: part number, package suffix, `YYWW` date code, and country code.
- **Multi-factor verification** — each field is validated independently: part-number lookup, valid-suffix check, date-code sanity (not in the future, not absurdly old, valid week number), and country-of-origin check against the expected manufacturing locations for that part.
- **Confidence scoring with a fail-safe default** — the verifier averages the per-field confidences and the logo-detection confidence, then requires a score of at least 0.70 to call a chip genuine. Anything below the threshold, or any hard error, defaults to **fake**.
- **Nexar web-database fallback** — if a part number isn't in the local JSON database, the system queries the Nexar API to confirm whether the part actually exists and how many authorized sellers carry it, before issuing a verdict.
- **Two detection endpoints** — a standard JSON endpoint and a streaming (Server-Sent Events) endpoint that pushes progressive updates for each step (logo detection, OCR extraction, verification) so the UI can show progress live.
- **Expected-value validation** — the user can optionally enter the manufacturer and part number they *think* the chip is, and the system reports whether the detected values match those expectations.
- **Inspection history and statistics** — every inspection is stored to a JSON history file, with endpoints to retrieve past inspections and aggregate stats.
- **Custom YOLOv8n logo detector** — a separately trained object-detection model with its own synthetic-data generation pipeline and verified metrics (details below).

## How It Works

The repo has three parts that are worth describing separately: the served inspection pipeline (Django + PaddleOCR), the verification engine, and the YOLOv8 training pipeline.

### Inspection pipeline (the live `/detect/` path)

1. **Upload and validation** — the image is received, validated (type/size), and saved to disk.
2. **Combined OCR pass** — a single PaddleOCR call does double duty: it detects the manufacturer logo *text* and extracts the full chip marking. This was deliberately collapsed from two OCR passes into one for speed (`detect_and_extract_combined`).
3. **Manufacturer identification** — the OCR text is matched against per-manufacturer keyword patterns (for example `\bTI\b`, `\bTEXAS\s*INSTRUMENTS?\b` for Texas Instruments). There's also a part-number-prefix fallback — e.g. an `LM`/`SN`/`TPS` prefix points to TI, an `STM`/`L###` prefix points to ST — for when the brand name itself isn't legible.
4. **Marking parsing** — the parser structures the raw text into part number, suffix, date code (`YYWW`), and country code.
5. **Verification** — the parsed fields go to the verifier (below), which returns the verdict.
6. **Response** — the API returns the verdict, confidence score, per-field validation details, the manufacturer, the raw OCR text, and (optionally) the user-expectation comparison.

Note on logo detection: the live API path identifies the manufacturer from **OCR text**, not from the YOLO model. The OCR detector's own docstring describes it as the text-matching replacement for visual logo detection. The trained YOLOv8 detector is a complete, separate component (see the training section) but isn't wired into the served endpoint.

### Verification engine (`ic_verifier.py`)

This is where the genuine/fake decision is made. The verifier is a singleton that loads the parts database once and runs a sequence of independent checks:

- **Part number** — exact match (case-insensitive) against the manufacturer's parts in the local database. If not found locally, it queries Nexar; if Nexar confirms the part, that counts as verified (with seller/availability info attached). If neither database has it, the part fails.
- **Suffix** — checks the package suffix against the list of valid suffixes for the matched part.
- **Date code** — validates the `YYWW` code: rejects years/weeks in the future as suspicious, flags chips older than 20 years as possibly re-marked, and rejects week numbers outside 1–53.
- **Country code** — checks the marking's country code against the expected manufacturing locations recorded for that part.

Each passing check contributes its confidence to a running list; the logo confidence is added in, and the mean becomes the overall confidence score. The decision rule is intentionally conservative: **any error → fake**, score **≥ 0.70 → genuine**, anything else → **fake** ("better safe than sorry"). The result object also carries warnings, errors, and a full per-field breakdown so the verdict is explainable, not a black box.

### YOLOv8 logo-detection training (`model_training/`)

Separate from the served pipeline, the repo includes an end-to-end training pipeline for a custom logo detector:

- **Synthetic data generation** — real manufacturer logos (a small set under `Dataset/`, organized by the six brands) are pasted onto metallic-looking backgrounds with augmentation (rotation, scaling, opacity changes) to simulate chips photographed under different conditions, and exported in YOLO label format.
- **Training** — fine-tunes a `YOLOv8n` (nano) model — `epochs=60`, `imgsz=640`, `batch_size=16` (8 on CPU) — chosen as the nano variant for the option of fast/edge inference.
- **Output** — the trained weights land in `trained_models/best.pt`, and the full Ultralytics run artifacts (curves, confusion matrices, `results.csv`) are kept under `runs/detect/yolov8_logo_run/`.

The README also documents the history here: v1 used an EfficientNet contour-crop classifier (~70.9% accuracy); v2 moved to YOLO for the logo task.

## Results / Highlights

From the actual training run (`runs/detect/yolov8_logo_run/results.csv`, 60 epochs):

- **mAP@50 ≈ 0.98**
- **mAP@50–95 ≈ 0.77**
- **Precision ≈ 0.96**, **Recall ≈ 0.96**
- Trained `best.pt` model, ~6 MB (YOLOv8n)
- Six manufacturers supported end-to-end
- Single-OCR-pass inspection (halved from the original two-pass approach)

These numbers come from the committed run logs, not from the README's targets — they line up, so the headline metrics are real. Caveat: the detector was trained largely on synthetic data over a small real-logo set, so the metrics reflect that distribution.

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
- A trained model at `trained_models/best.pt` (included) if you want to run the YOLO detector
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
# Terminal 1 — backend (http://localhost:8000)
cd backend
python manage.py runserver

# Terminal 2 — frontend (http://localhost:3000)
cd frontend
npm start
```

The first inspection will be slow while PaddleOCR downloads and caches its models.

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
│   │   │   ├── yolo_logo_detector.py # YOLOv8 logo detector (loads best.pt)
│   │   │   ├── ic_verifier.py        # multi-factor genuine/fake verification
│   │   │   └── nexar_service.py      # Nexar web-database fallback
│   │   ├── utils/
│   │   │   ├── marking_parser.py     # part no / suffix / date code / country
│   │   │   ├── database.py
│   │   │   └── image_utils.py
│   │   ├── views.py                  # detect, detect-stream, history, stats, health
│   │   └── urls.py
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
│   ├── train.py
│   └── config/training_config.py
├── trained_models/best.pt            # trained YOLOv8n logo model
├── Dataset/                          # real manufacturer logos (by brand)
├── runs/detect/yolov8_logo_run/      # training artifacts + results.csv
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
