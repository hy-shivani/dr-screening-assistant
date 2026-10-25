# Explainable AI-Based Diabetic Retinopathy Screening Assistant

Educational research prototype: fundus image upload, heuristic image-quality checks, optional five-class PyTorch classifier, and Grad-CAM overlay.

**Safety:** This is not a medical device and must not be used for diagnosis or treatment decisions. Outputs require qualified professional review. The app clearly labels placeholder mode when no trained checkpoint exists.

## Stack
- React + Vite frontend
- Node.js + Express upload/API backend
- Python + FastAPI + PyTorch/OpenCV ML service
- DenseNet121 transfer-learning training starter

## Run with Docker
Install Docker Desktop, then from this folder:
```bash
docker compose up --build
```
Open `http://localhost:5173`. Backend health: `http://localhost:4000/health`. ML API docs: `http://localhost:8000/docs`.

Without Docker, run each service in its own terminal:
1. `cd ml-service`; create/activate a Python virtual environment; `pip install -r requirements.txt`; `uvicorn app:app --reload --port 8000`
2. `cd backend`; `npm install`; `npm run dev`
3. `cd frontend`; `npm install`; `npm run dev`

## Train a real model
A dataset and trained weights are not included. Obtain a dataset such as APTOS 2019 only after checking its terms and license. Arrange data as:
```
ml-service/data/aptos/
  train/0 ... train/4
  val/0 ... val/4
```
Each class directory must contain images. Labels: 0 No DR, 1 Mild DR, 2 Moderate DR, 3 Severe DR, 4 Proliferative DR.
```bash
cd ml-service
python train.py --data-dir data/aptos --epochs 5 --batch-size 16
```
The model checkpoint is saved to `model/best_model.pth`. Restart the ML service after training. This starter is not a validated clinical model; use careful patient-level splits, held-out testing, class-imbalance handling and external validation before drawing research conclusions.

## API
`POST /api/screen` accepts `multipart/form-data` with file field `image`. The result includes `mode`, predicted class/label, class probabilities, image-quality heuristic results, heatmap data URL, and a disclaimer. If no checkpoint exists, `mode` is `demo`, predicted class is null and no heatmap is claimed.

## Team split
1. Dataset/model training/evaluation.
2. Preprocessing, image-quality checks, Grad-CAM.
3. Express API and ML service integration.
4. React UI, end-to-end testing, screenshots/documentation.

## Limitations
No dataset or weights are included. Quality checks are simple heuristics, not validated gradability assessment. Grad-CAM is an attribution visualization, not proof of a lesion. U-Net lesion segmentation, clinical validation, authentication, database and deployment are future work.
