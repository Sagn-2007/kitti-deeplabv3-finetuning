import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

from dataset import KITTIRoadDataset


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# =========================
# Dataset
# =========================

dataset = KITTIRoadDataset(
    root="../road_segmentation/data/data_road/training"
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# =========================
# Model
# =========================

weights = DeepLabV3_ResNet50_Weights.DEFAULT

model = deeplabv3_resnet50(weights=weights)

# Change 21-class COCO output → binary road output
model.classifier[4] = nn.Conv2d(
    256,
    1,
    kernel_size=1
)

model = model.to(device)


# =========================
# Freeze backbone
# =========================

# =========================
# Fine-tune ResNet layer4
# =========================

for param in model.backbone.parameters():
    param.requires_grad = False

for param in model.backbone.layer4.parameters():
    param.requires_grad = True


# =========================
# Loss + Optimizer
# =========================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.backbone.layer4.parameters(),
        "lr": 0.00001
    },
    {
        "params": model.classifier[4].parameters(),
        "lr": 0.001
    }
])
# =========================
# Training
# =========================

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)["out"]

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)


    # =========================
    # Validation IoU
    # =========================

    model.eval()

    intersection_total = 0.0
    union_total = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]

            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities > 0.5).float()

            intersection = (predictions * masks).sum()

            union = ((predictions + masks) > 0).float().sum()

            intersection_total += intersection.item()
            union_total += union.item()

    val_iou = intersection_total / union_total

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val IoU: {val_iou:.4f}"
    )
torch.save(
    model.state_dict(),
    "deeplabv3_kitti.pth"
)
print("Model saved.")