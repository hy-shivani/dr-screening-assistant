
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data_dir = Path("data/aptos/val")
checkpoint_path = Path("model/best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=True
)

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 5)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

actual, predicted = [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        actual.extend(labels.tolist())
        predicted.extend(predictions)

names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

print("\nEvaluation accuracy:", round(accuracy_score(actual, predicted) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(
    actual, predicted,
    labels=[0, 1, 2, 3, 4],
    target_names=names,
    zero_division=0
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(actual, predicted, labels=[0, 1, 2, 3, 4]))
