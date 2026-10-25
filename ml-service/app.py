import base64, io, os
from pathlib import Path
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from preprocessing import load_image, assess_quality, to_model_array
from gradcam import make_gradcam

app = FastAPI(title="DR Screening ML Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "model" / "best_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
target_layer = None

def load_classifier():
    global model, target_layer
    if not MODEL_PATH.exists():
        model = target_layer = None
        return
    from torchvision import models
    network = models.densenet121(weights=None)
    network.classifier = torch.nn.Linear(network.classifier.in_features, len(LABELS))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    network.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    network.to(DEVICE).eval()
    model, target_layer = network, network.features.denseblock4

@app.on_event("startup")
def startup():
    load_classifier()

def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-service", "mode": "model" if model is not None else "demo", "device": str(DEVICE), "checkpointFound": MODEL_PATH.exists()}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload JPG, PNG, WEBP or BMP.")
    raw = await image.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")
    try:
        original = load_image(raw)
        quality = assess_quality(original)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if model is None:
        return {
            "mode": "demo", "predictedClass": None, "predictedLabel": "Model not trained",
            "probabilities": [], "imageQuality": quality, "heatmapDataUrl": to_data_url(original),
            "heatmapAvailable": False,
            "disclaimer": "DEMO MODE: no trained classifier is installed. No DR prediction or Grad-CAM explanation was generated. Research demo only; not a diagnosis."
        }

    tensor = torch.from_numpy(to_model_array(original)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    try:
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
            predicted = int(torch.argmax(probs).item())
        overlay = make_gradcam(model, tensor, original, target_layer, predicted)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}: {exc}") from exc
    return {
        "mode": "model", "predictedClass": predicted, "predictedLabel": LABELS[predicted],
        "probabilities": [{"label": label, "probability": round(float(probs[i].item()), 6)} for i, label in enumerate(LABELS)],
        "imageQuality": quality, "heatmapDataUrl": to_data_url(overlay), "heatmapAvailable": True,
        "disclaimer": "Research screening prototype only; not a diagnosis. Consult a qualified eye-care professional."
    }
