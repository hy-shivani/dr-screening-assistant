import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DenseNet121 for five-class DR grading.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="model/best_model.pth")
    args = parser.parse_args()
    root = Path(args.data_dir)
    train_dir, val_dir = root / "train", root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("Expected data-dir/train and data-dir/val with class folders 0 through 4.")
    train_tf = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    val_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    expected = [str(i) for i in range(5)]
    if train_ds.classes != expected or val_ds.classes != expected:
        raise SystemExit(f"Class folders must be named {expected}; got {train_ds.classes} and {val_ds.classes}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    net.classifier = nn.Linear(net.classifier.in_features, 5)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best = -1.0
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        net.train(); total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y); loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
        net.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                correct += int((net(x).argmax(1) == y).sum().item()); total += len(y)
        acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={total_loss/max(len(train_ds),1):.4f}, val_accuracy={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model_state_dict": net.state_dict(), "labels": LABELS, "val_accuracy": best, "epoch": epoch+1}, out)
            print(f"Saved checkpoint: {out}")
    print(f"Best validation accuracy: {best:.4f}. Validation accuracy alone is not clinical validation.")

if __name__ == "__main__":
    main()
