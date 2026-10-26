
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Dataset paths
csv_path = Path(r"D:\dr-model-training\train.csv")
image_dir = Path(r"D:\dr-model-training\train_images")
output_dir = Path("data/aptos")

# Read labels
df = pd.read_csv(csv_path)

# Keep only images that exist
df["image_path"] = df["id_code"].apply(
    lambda name: image_dir / f"{name}.png"
)
df = df[df["image_path"].apply(lambda p: p.exists())].copy()

print("Images found:", len(df))

# Split: 70% train, 30% remaining
train_df, remaining_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["diagnosis"]
)

# Split remaining data equally into validation and test
val_df, test_df = train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=42,
    stratify=remaining_df["diagnosis"]
)

# Copy images into class folders
for split, split_df in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df)
]:
    for _, row in split_df.iterrows():
        label = str(int(row["diagnosis"]))
        target_dir = output_dir / split / label
        target_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            row["image_path"],
            target_dir / row["image_path"].name
        )

    print(f"{split} images:", len(split_df))

print("Dataset preparation completed!")
print("Output folder:", output_dir.resolve())
