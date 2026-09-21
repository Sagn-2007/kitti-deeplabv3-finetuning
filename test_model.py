import torch
import torch.nn as nn

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)


# -------------------------
# Load pretrained model
# -------------------------

weights = DeepLabV3_ResNet50_Weights.DEFAULT

model = deeplabv3_resnet50(
    weights=weights
)

# Replace 21-class head with binary head
model.classifier[4] = nn.Conv2d(
    256,
    1,
    kernel_size=1
)

model = model.to(device)

print(model)