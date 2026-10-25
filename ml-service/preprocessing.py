from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2

IMAGE_SIZE = 224

def load_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file could not be decoded as an image.") from exc
    if image.width < 64 or image.height < 64:
        raise ValueError("Image is too small. Upload an image at least 64 x 64 pixels.")
    if image.width > 8000 or image.height > 8000:
        raise ValueError("Image dimensions are too large.")
    return image

def assess_quality(image: Image.Image) -> dict:
    """Simple heuristics only; not a validated image-gradability model."""
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    issues = []
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    if blur < 12: issues.append("Image may be blurry (heuristic).")
    if brightness < 35: issues.append("Image may be too dark.")
    elif brightness > 225: issues.append("Image may be overexposed.")
    if contrast < 18: issues.append("Image contrast may be low.")
    return {
        "status": "review" if issues else "acceptable",
        "issues": issues,
        "metrics": {"laplacianVariance": round(blur, 2), "meanBrightness": round(brightness, 2), "contrastStd": round(contrast, 2)},
        "note": "Heuristic checks only; not a validated medical image-quality assessment."
    }

def to_model_array(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std
