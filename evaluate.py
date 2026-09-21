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

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Validation samples:", len(val_dataset))


# =========================
# Model
# =========================

weights = DeepLabV3_ResNet50_Weights.DEFAULT

model = deeplabv3_resnet50(weights=weights)

model.classifier[4] = nn.Conv2d(
    256,
    1,
    kernel_size=1
)

model.load_state_dict(
    torch.load(
        "deeplabv3_kitti.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================
# Evaluation
# =========================

intersection_total = 0.0
union_total = 0.0
dice_intersection_total = 0.0
prediction_total = 0.0
mask_total = 0.0


with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]

        probabilities = torch.sigmoid(outputs)

        predictions = (probabilities > 0.5).float()

        # IoU
        intersection = (predictions * masks).sum()

        union = ((predictions + masks) > 0).float().sum()

        intersection_total += intersection.item()
        union_total += union.item()

        # Dice
        dice_intersection_total += intersection.item()
        prediction_total += predictions.sum().item()
        mask_total += masks.sum().item()


# =========================
# Metrics
# =========================

iou = intersection_total / union_total

dice = (
    2 * dice_intersection_total
    / (prediction_total + mask_total)
)


print()
print("=========================")
print("DeepLabV3 Evaluation")
print("=========================")
print(f"IoU:  {iou:.4f}")
print(f"Dice: {dice:.4f}")