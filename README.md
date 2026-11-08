# Fake-IC-Detection

A web app that checks whether an integrated circuit looks genuine. You upload a photo of a chip, the backend reads the surface marking with OCR, works out the manufacturer, parses the marking, and checks the fields against a database of known parts.

Counterfeit and re-marked ICs are hard to spot by eye. This is a first-pass check: it reads the marking and tests it for consistency. It does not replace electrical or decapsulation testing.

## How it works

The `/api/detect/` request goes through these steps:

1. The image is validated and saved.
2. One PaddleOCR pass reads all the text on the chip. The same text is used to guess the manufacturer (keyword patterns such as `TEXAS INSTRUMENTS`, plus part-number prefixes like `LM`/`SN`/`TPS` for TI) and to extract the marking.
3. `marking_parser.py` splits the marking into part number, suffix, `YYWW` date code and country code.
4. `ic_verifier.py` checks each field:
   - part number against `backend/database/ic_parts_database.json` (six manufacturers: Texas Instruments, STMicroelectronics, NXP, ON Semiconductor, Microchip, Infineon). If the part is not in the local file, it asks the Nexar parts API (needs `NEXAR_ACCESS_TOKEN`; without it that check is skipped)
   - suffix against the valid suffixes for that part
   - date code: not in the future, not older than 20 years, week between 1 and 53
   - country code against the expected manufacturing locations for the part
5. The per-field confidences are averaged. A score of 0.70 or more with no errors returns genuine; anything else returns fake.

The response contains the verdict, the confidence, the per-field results, the manufacturer and the raw OCR text. You can also pass an expected manufacturer and part number, and the response reports whether the detected values match.

`/api/detect-stream/` does the same but streams progress as server-sent events. Inspections are appended to `backend/database/inspections.json`.

The manufacturer in the served pipeline comes from OCR text. A YOLOv8 logo detector is also in the repo (`backend/api/services/yolo_logo_detector.py`, weights in `trained_models/best.pt`), but the detect views import only the OCR detector.

## Stack

- Backend: Django, django-cors-headers, PaddleOCR, Ultralytics YOLOv8, OpenCV, PyTorch
- Frontend: React 18 (Create React App)
- Storage: JSON files, no database

## Setup

```bash
cd backend
pip install -r requirements.txt
python manage.py runserver
```

```bash
cd frontend
npm install
npm start
```

The backend runs on `http://localhost:8000` and the frontend on `http://localhost:3000`. The API URL is hard-coded in `frontend/src/services/api.js`. PaddleOCR downloads its models on first use, so the first inspection is slow.

Example request:

```bash
curl -X POST http://localhost:8000/api/detect/ \
  -F "image=@chip.jpg" \
  -F "expected_manufacturer=Texas Instruments" \
  -F "expected_part_number=LM324"
```

Endpoints under `/api/`: `detect/` (POST), `detect-stream/` (POST), `history/`, `statistics/`, `health/`.

## Logo detector training

`model_training/yolo_logo_train_pipeline.py` builds a synthetic dataset by pasting the logos in `Dataset/` (one folder per manufacturer) onto generated metallic backgrounds with rotation, scaling and opacity changes, writes YOLO labels, and fine-tunes `yolov8n.pt` (60 epochs, image size 640, batch 16 on GPU or 8 on CPU).

The final epoch of `runs/detect/yolov8_logo_run/results.csv` shows precision 0.959, recall 0.961, mAP@50 0.981 and mAP@50-95 0.975. The training and validation images are synthetic, built from a small set of real logos, so these numbers say little about real chip photos. Weights are in `trained_models/best.pt`.

## Project structure

```
backend/
  api/services/   OCR, manufacturer detection, verifier, Nexar client, YOLO detector
  api/utils/      marking parser, image helpers, JSON store
  api/views.py    detect, detect-stream, history, statistics, health
  database/       parts reference data and inspection history
frontend/         React UI (upload, dashboard, expected-value check)
model_training/   YOLO training pipeline and the older EfficientNet scripts
Dataset/          real manufacturer logos used for synthetic data
dataset_synth/    generated YOLO dataset
runs/             Ultralytics training output
trained_models/   best.pt
```

## Limitations

- `model_training/train.py` imports `models/efficientnet_model.py`, which is not in the repo, so the EfficientNet path does not run. The YOLO pipeline is the working one.
- The parts database covers a limited set of parts per manufacturer. Anything not in it and not found on Nexar is marked fake.
- `backend/ic_detection/settings.py` has `DEBUG = True`, `ALLOWED_HOSTS = ['*']` and a development secret key. Change these before deploying.
