
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


def main():
    parser = argparse.ArgumentParser(
        description="Train DenseNet121 for five-class DR grading"
    )

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="model/best_model.pth")

    args = parser.parse_args()

    # Dataset paths
    root = Path(args.data_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            "Expected data-dir/train and data-dir/val "
            "with class folders 0 through 4."
        )

    # Image preprocessing
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Load datasets
    train_ds = datasets.ImageFolder(
        train_dir, transform=train_tf
    )
    val_ds = datasets.ImageFolder(
        val_dir, transform=val_tf
    )

    expected = [str(i) for i in range(5)]

    if train_ds.classes != expected or val_ds.classes != expected:
        raise SystemExit(
            f"Class folders must be {expected}. "
            f"Got train={train_ds.classes}, val={val_ds.classes}"
        )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Training or validation dataset is empty.")

    print("Training images:", len(train_ds))
    print("Validation images:", len(val_ds))
    print("Classes:", train_ds.classes)

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Load pretrained DenseNet121
    net = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    net.classifier = nn.Linear(
        net.classifier.in_features, 5
    )
    net = net.to(device)

    # Class-weighted loss using ACTUAL training folder counts
    class_counts = torch.bincount(
        torch.tensor(train_ds.targets),
        minlength=5
    ).float()

    if torch.any(class_counts == 0):
        raise SystemExit(
            f"Every class needs training images. Counts: {class_counts.tolist()}"
        )

    class_weights = len(train_ds) / (5 * class_counts)
    class_weights = class_weights.to(device)

    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights.cpu().tolist())

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr
    )

    # Save the best validation checkpoint
    best = -1.0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        net.train()
        total_loss = 0.0

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        # Validation
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)

                predictions = net(x).argmax(dim=1)

                correct += (predictions == y).sum().item()
                total += len(y)

        accuracy = correct / total if total else 0.0
        average_loss = total_loss / len(train_ds)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={average_loss:.4f}, "
            f"val_accuracy={accuracy:.4f}"
        )

        if accuracy > best:
            best = accuracy

            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "labels": LABELS,
                    "val_accuracy": best,
                    "epoch": epoch + 1
                },
                out
            )

            print(f"Saved checkpoint: {out}")

    print(f"\nBest validation accuracy: {best:.4f}")
    print(
        "Validation accuracy alone is not clinical validation."
    )


if __name__ == "__main__":
    main()
