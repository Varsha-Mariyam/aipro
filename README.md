# Aadhaar Fraud Detection — 2-Layer System

Short: a two-layer fraud detection pipeline for document images combining algorithmic heuristics (Day 1) and a CNN-based classifier (Day 2).

## Project summary

This repository implements a two-layer fraud detection system for document images. The system first applies algorithmic checks (error-level analysis, EXIF inspection, overlay/edge detection, OCR-based font checks) and then runs a CNN to detect learned visual patterns of forgeries. The outputs are combined into a final decision (APPROVE, MANUAL REVIEW, REJECT) and a JSON report.

## Repository layout

- `app.py` — FastAPI server exposing `/analyze` and `/analyze/quick` endpoints to analyze uploaded images.
- `complete_system.py` — Orchestrator which runs Layer 1 (algorithmic) and Layer 2 (CNN), and implements decision fusion and report saving.
- `fraud_detector.py` — Day 1 algorithmic checks (EL A, EXIF metadata, overlay/edge analysis, OCR/font checks). Exposes `ForgeryDetector` and `BatchProcessor`.
- `cnn_model.py` — CNN model architecture, training and evaluation utilities; saves/loads model at `models/fraud_cnn.h5`.
- `test_complete.py` — Lightweight interactive test driver (single image or batch) that calls `CompleteFraudSystem`.
- `models/` — expected location for `fraud_cnn.h5` (trained Keras model).
- `Images/` — expected dataset layout for training: `Images/train/{authentic,fake}`, `Images/valid/{authentic,fake}`.
- `outputs/` — output folder for reports and plots (create if missing).

## High-level flow

1. Input image provided (local path or uploaded via API).
2. Layer 1 (Algorithmic): `ForgeryDetector.analyze(image_path)` runs several heuristics and returns a structured report with per-technique findings and an aggregated risk level.
3. Layer 2 (CNN): `CompleteFraudSystem.cnn_predict(image_path)` preprocesses the image and runs the Keras model to return a probability and prediction.
4. Decision fusion: `CompleteFraudSystem.make_final_decision(day1_result, day2_result)` combines the two layer outputs into a final decision and confidence score.
5. The system returns the full report (JSON) and may save it to `outputs/`.

## Report format (top-level keys)

- `image` — path or temp filename analyzed.
- `layer1_algorithmic` — Day 1 results including `overall_risk`, `fraud_detected`, `risk_counts`, and `techniques` (ela, metadata, overlay, fonts).
- `layer2_cnn` — CNN result with `available` (bool), `fake_probability`, `authentic_probability`, `prediction`, and `confidence`.
- `final_decision` — `decision` (APPROVE / MANUAL_REVIEW / REJECT), numeric `confidence`, and `reasoning`.

## Requirements

The project uses the following Python packages (suggested):

- numpy
- opencv-python
- pillow
- pytesseract
- tensorflow (or tensorflow-cpu)
- matplotlib
- fastapi
- uvicorn
- python-multipart

Detailed list is provided in `requirements.txt`.

## Setup (Windows)

1. Create a Python 3.9+ virtual environment and activate it.
  
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Install Tesseract OCR for Windows and update the path in `fraud_detector.py` if needed. Default used in code:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

4. Ensure directories exist:

```powershell
mkdir models; mkdir outputs
```

5. Place a trained CNN model (if you have one) at `models/fraud_cnn.h5`.

## How to run

- Quick CLI single test (interactive):

```powershell
python test_complete.py
```

- Run the FastAPI server:

```powershell
uvicorn app:app --reload --port 8000
```

Endpoints:
- `GET /` — basic info
- `GET /health` — component health
- `POST /analyze` — full analysis (multipart file upload)
- `POST /analyze/quick` — quick decision summary

- Train the CNN (if implemented in `cnn_model.py`):

```powershell
python cnn_model.py
```

## Known issues & recommended improvements

- Several absolute paths are hard-coded (image sample path in `test_complete.py`, Tesseract path). Replace with CLI args or a `config.yaml` for portability.
- ELA implementation currently uses a temporary filename; use `tempfile` to avoid collisions.
- EXIF inspection checks a limited set of tags; consider more robust EXIF parsing using `PIL.ExifTags.TAGS`.
- `save_report` expects `outputs/` to exist; code should create the directory if missing.
- Add `logging` instead of prints and add unit tests (mock `ForgeryDetector` and CNN) to validate decision fusion logic.

## Next steps you might want me to do

- Add `requirements.txt` (done) and `README.md` (this file).
- Convert `test_complete.py` to use `argparse` for configurable paths.
- Add unit tests for `CompleteFraudSystem.make_final_decision` with mocks.
- Containerize the app (Dockerfile) including Tesseract.

---

If you want any of the next steps implemented (argparse for test script, unit tests, or Dockerfile), tell me which and I will add them.