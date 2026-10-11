import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms.functional as F

class COCODetectionDataset(Dataset):
    """
    Custom Dataset class for loading COCO formatted military object detection data.
    Includes built-in Horizontal Flip augmentation.
    """
    def __init__(self, images_dir, json_path, use_augmentation=True):
        self.images_dir = images_dir
        self.coco = COCO(json_path)
        self.ids = list(self.coco.imgs.keys())
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # 1. Get image path
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        # 2. Read image and convert to tensor
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img_pil)

        # 3. Read COCO annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            # Exact COCO to Faster R-CNN conversion logic from original code
            x_min, y_min, width, height = ann['bbox']
            x_max = x_min + width
            y_max = y_min + height

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        # Basic Horizontal Flip Augmentation
        if self.use_augmentation and torch.rand(1).item() > 0.5:
            # Flip image
            img_tensor = F.hflip(img_tensor)
            # Flip boxes
            if len(boxes) > 0:
                width = img_tensor.shape[2]
                boxes_t = torch.tensor(boxes)
                boxes_t[:, [0, 2]] = width - boxes_t[:, [2, 0]]
                boxes = boxes_t.tolist()
        
        # Handle edge case: images with no objects
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        # 4. Pack into target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        return img_tensor, target

# Custom collate function for object detection DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))
