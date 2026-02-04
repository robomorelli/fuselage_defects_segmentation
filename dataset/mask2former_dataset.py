import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
import pandas as pd


class Mask2FormerDataset(Dataset):
    """
    Dataset adapter for Mask2Former.
    Converts semantic segmentation masks to instance segmentation format.
    Uses the same structure as KFoldDataframeMulticlassProcessor.
    """

    def __init__(self, data_path, fold, split='train', augmentation=True,
                 normalize=True, crop_size=512):
        self.data_path = data_path
        self.fold = fold
        self.split = split
        self.augmentation = augmentation
        self.normalize = normalize
        self.crop_size = crop_size

        # Images and masks directories (cropped data)
        self.images_dir = os.path.join(data_path, 'images')
        self.masks_dir = os.path.join(data_path, 'masks')

        # Load filenames from CSV in k-fold directory
        kfold_path = os.path.join('data/k-fold', f'fold_{fold}', split)
        csv_path = os.path.join(kfold_path, 'cropped_filenames.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV with filenames
        df = pd.read_csv(csv_path)
        self.images_file_names = df['images'].tolist()
        self.masks_file_names = df['masks'].tolist()

        # Verify files exist
        valid_indices = []
        for i, (img_name, mask_name) in enumerate(zip(self.images_file_names, self.masks_file_names)):
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, mask_name)
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_indices.append(i)
            else:
                print(f"Warning: Missing file - {img_path} or {mask_path}")

        self.images_file_names = [self.images_file_names[i] for i in valid_indices]
        self.masks_file_names = [self.masks_file_names[i] for i in valid_indices]

        assert len(self.images_file_names) > 0, \
            f"No valid samples found!\n  Images dir: {self.images_dir}\n  Masks dir: {self.masks_dir}\n  CSV: {csv_path}"

        print(f"Loaded {len(self.images_file_names)} samples for {split} (fold {fold})")
        print(f"Crop size: {crop_size}x{crop_size}")
        print(f"Images dir: {self.images_dir}")
        print(f"Masks dir: {self.masks_dir}")

        # Setup augmentation
        if augmentation and split == 'train':
            self.transform = A.Compose([
                A.RandomCrop(height=crop_size, width=crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else A.NoOp(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else A.NoOp(),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images_file_names)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.images_file_names[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_file_names[idx])

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (class IDs: 0=background, 1=mark, 2=scratch, 3=drill)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        # Convert semantic mask to instance format
        instances = self._mask_to_instances(mask if isinstance(mask, np.ndarray) else mask.numpy())

        target = {
            'masks': instances['masks'],
            'labels': instances['labels']
        }

        return {
            'image': image,
            'target': target,
            'image_id': idx
        }

    def _mask_to_instances(self, mask):
        """
        Converts semantic mask (H, W) to instance masks.
        Each connected component of each class becomes a separate instance.

        Args:
            mask: numpy array (H, W) with class IDs (0=background, 1=mark, 2=scratch, 3=drill)

        Returns:
            dict with 'masks' (N, H, W) tensor and 'labels' (N,) tensor
        """
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]  # exclude background

        instance_masks = []
        labels = []

        for class_id in unique_classes:
            # Binary mask for this class
            class_mask = (mask == class_id).astype(np.uint8)

            # Find connected components
            num_instances, labels_im = cv2.connectedComponents(class_mask)

            for instance_id in range(1, num_instances):  # skip background (0)
                instance_mask = (labels_im == instance_id).astype(np.float32)

                # Filter out very small instances
                if instance_mask.sum() < 10:
                    continue

                instance_masks.append(torch.from_numpy(instance_mask))
                labels.append(int(class_id))  # use class_id directly (1,2,3)

        # Handle case with no instances
        if len(instance_masks) == 0:
            h, w = mask.shape
            instance_masks = [torch.zeros((h, w), dtype=torch.float32)]
            labels = [0]  # dummy label

        return {
            'masks': torch.stack(instance_masks),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def mask2former_collate_fn(batch):
    """
    Custom collate function for Mask2Former.
    Handles batches of images with variable number of instances.
    """
    images = []
    targets = []
    image_ids = []

    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
        image_ids.append(item['image_id'])

    images = torch.stack(images)

    return {
        'images': images,
        'targets': targets,
        'image_ids': image_ids
    }