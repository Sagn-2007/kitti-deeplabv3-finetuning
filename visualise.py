import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

from dataset import KITTIRoadDataset


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Dataset
# =========================

dataset = KITTIRoadDataset(
    root="../road_segmentation/data/data_road/training"
)


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
# Select image
# =========================

image, mask = dataset[40]

input_image = image.unsqueeze(0).to(device)


# =========================
# Prediction
# =========================

with torch.no_grad():

    output = model(input_image)["out"]

    probability = torch.sigmoid(output)

    prediction = (probability > 0.5).float()


# =========================
# Prepare image for display
# =========================

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)

image_display = image.cpu() * std + mean

image_display = image_display.clamp(0, 1)

image_display = image_display.permute(1, 2, 0)


# =========================
# Prepare masks
# =========================

ground_truth = mask.squeeze().cpu()

prediction_display = prediction.squeeze().cpu()


# =========================
# IoU for this image
# =========================

intersection = (prediction_display * ground_truth).sum()

union = (
    (prediction_display + ground_truth) > 0
).float().sum()

iou = intersection / union


# =========================
# Visualization
# =========================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_display)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ground_truth, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(prediction_display, cmap="gray")
plt.title(f"Prediction (IoU: {iou:.4f})")
plt.axis("off")

plt.tight_layout()
plt.tight_layout()

plt.savefig(
    "results/deeplabv3_visualization4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.show()