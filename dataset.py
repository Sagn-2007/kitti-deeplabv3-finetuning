from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class KITTIRoadDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.image_dir = self.root / "image_2"
        self.mask_dir = self.root / "gt_image_2"

        # Find all road masks
        self.mask_paths = sorted(self.mask_dir.glob("*_road_*.png"))

        # Corresponding image paths
        self.image_paths = []

        for mask_path in self.mask_paths:

            # um_road_000000.png
            #        ↓
            # um_000000.png

            image_name = mask_path.name.replace("_road_", "_")

            image_path = self.image_dir / image_name

            self.image_paths.append(image_path)

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (256, 512),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index])

        image = self.image_transform(image)

        mask = self.mask_transform(mask)

        #Convert RGB mask to a single channel
       # Purple/magenta = road
        mask = (
            (mask[0] == 255) &
            (mask[1] == 0) &
            (mask[2] == 255)
        ).float().unsqueeze(0)

        return image, mask
