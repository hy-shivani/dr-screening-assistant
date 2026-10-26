
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from PIL import Image
from torchvision import transforms

checkpoint_path = Path("model/best_model.pth")

if not checkpoint_path.exists():
    print("Model checkpoint not found!")
    raise SystemExit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 5)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Model loaded successfully!")
print("Labels:", checkpoint["labels"])
print("Best validation accuracy:", round(checkpoint["val_accuracy"] * 100, 2), "%")
print("Best epoch:", checkpoint["epoch"])
print("Device:", device)

image_path = Path("data/aptos/val/0")

image_files = list(image_path.glob("*.png"))

if not image_files:
    print("No image found in class 0 folder.")
    raise SystemExit(1)

image = Image.open(image_files[0]).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model(image).argmax(1).item()

print("Test image:", image_files[0].name)
print("Predicted DR grade:", checkpoint["labels"][prediction])
