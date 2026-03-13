import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

# Fix Qt/xcb errors - MUST BE BEFORE ANY OTHER IMPORTS
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

# Now safe to import matplotlib
import matplotlib
matplotlib.use('Agg')



class SegmentationDatasetFromFolder(Dataset):
    """
    Dataset che legge direttamente da una cartella senza CSV.
    Utile per evaluation/inference su nuovi dati.

    Struttura cartelle attesa:
    data_path/
    ├── images/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── masks/  (opzionale, solo se hai ground truth)
        ├── img1.png
        ├── img2.png
        └── ...
    """

    def __init__(self, data_path, augmentation=False, normalize=True,
                 crop_size=512, has_masks=True):
        """
        Args:
            data_path: Path alla cartella principale contenente images/ (e masks/)
            augmentation: Se applicare data augmentation
            normalize: Se normalizzare con ImageNet stats
            crop_size: Dimensione del crop (solo se augmentation=True)
            has_masks: Se la cartella contiene anche le masks (per evaluation)
        """
        self.data_path = data_path
        self.augmentation = augmentation
        self.normalize = normalize
        self.crop_size = crop_size
        self.has_masks = has_masks

        # Images and masks directories
        self.images_dir = os.path.join(data_path, 'images')
        self.masks_dir = os.path.join(data_path, 'masks') if has_masks else None

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        # Get all image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.images_file_names = sorted([
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if len(self.images_file_names) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        # If has_masks, verify that corresponding masks exist
        if self.has_masks:
            if not os.path.exists(self.masks_dir):
                raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

            valid_images = []
            for img_name in self.images_file_names:
                mask_name = img_name  # assume same filename
                mask_path = os.path.join(self.masks_dir, mask_name)
                if os.path.exists(mask_path):
                    valid_images.append(img_name)
                else:
                    print(f"Warning: No mask found for {img_name}")

            self.images_file_names = valid_images

        print(f"Loaded {len(self.images_file_names)} images from {self.images_dir}")
        if self.has_masks:
            print(f"Masks dir: {self.masks_dir}")
        print(f"Crop size: {crop_size}x{crop_size}")

        # Setup augmentation
        self.transform = self._setup_transforms()

    def _setup_transforms(self):
        """Setup albumentations transforms"""
        transforms_list = []

        if self.augmentation:
            transforms_list.extend([
                A.RandomCrop(height=self.crop_size, width=self.crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.2),
            ])
        else:
            # Per evaluation: center crop o resize
            transforms_list.append(A.CenterCrop(height=self.crop_size, width=self.crop_size))

        # Normalization
        if self.normalize:
            transforms_list.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        transforms_list.append(ToTensorV2())

        return A.Compose(transforms_list)

    def __len__(self):
        return len(self.images_file_names)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.images_file_names[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask if available
        if self.has_masks:
            mask_path = os.path.join(self.masks_dir, self.images_file_names[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
        else:
            # Dummy mask (all zeros)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

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
            'image_id': idx,
            'filename': self.images_file_names[idx]  # utile per salvataggio
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


class SegmentationDataset(Dataset):
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
            transforms_list = [
                A.RandomCrop(height=crop_size, width=crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.2),
            ]

            # Add normalization if requested
            if normalize:
                transforms_list.append(
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                )

            transforms_list.append(ToTensorV2())
            self.transform = A.Compose(transforms_list)

        else:
            transforms_list = [
                A.CenterCrop(height=crop_size, width=crop_size),
            ]

            if normalize:
                transforms_list.append(
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                )

            transforms_list.append(ToTensorV2())
            self.transform = A.Compose(transforms_list)

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


class EvaluationSegmentationDataset(Dataset):
    """
    Dataset SOLO per evaluation con auto-detect CSV.
    Supporta sia file CSV che directory (auto-trova cropped_filenames.csv o full_size_filenames.csv).

    NON usare per training! Usa SegmentationDataset invece.
    """

    def __init__(self, data_path, csv_path=None, normalize=True,
                 from_csv=False, from_folder=False, cropped=True, crop_size=512):
        """
        Args:
            data_path: Root directory containing images/ and masks/ folders
            csv_path: Path to CSV file OR directory containing CSV
            normalize: ImageNet normalization (0 or 1)
            from_csv: Load filenames from CSV
            from_folder: Load all files from folder
            cropped: If True, look for cropped_filenames.csv, else full_size_filenames.csv
            crop_size: Crop size (kept for compatibility, not used)
        """
        self.data_path = Path(data_path)
        self.images_dir = self.data_path / 'images'
        self.masks_dir = self.data_path / 'masks'
        self.csv_path = csv_path
        self.normalize = normalize
        self.from_csv = from_csv
        self.from_folder = from_folder
        self.cropped = cropped

        # Auto-detect CSV file if directory is provided
        if csv_path and Path(csv_path).is_dir():
            csv_filename = 'cropped_filenames.csv' if cropped else 'full_size_filenames.csv'
            self.csv_path = Path(csv_path) / csv_filename
            print(f"Auto-detected CSV: {self.csv_path}")

        # Get filenames
        if from_csv and csv_path:
            self._load_from_csv()
        elif from_folder:
            self._load_from_folder()
        else:
            # Default: try CSV from standard structure
            if csv_path:
                self._load_from_csv()
            else:
                self._load_from_folder()

        # Setup transforms
        if normalize:
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        else:
            self.mean = (0.0, 0.0, 0.0)
            self.std = (1.0, 1.0, 1.0)

        self.transform = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def _load_from_csv(self):
        """Load filenames from CSV"""
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        print(f"Loading filenames from CSV: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Try different column names
        if 'filename' in df.columns:
            self.images_file_names = df['filename'].tolist()
        elif 'images' in df.columns:
            self.images_file_names = df['images'].tolist()
        elif 'image' in df.columns:
            self.images_file_names = df['image'].tolist()
        else:
            # Assume first column contains filenames
            self.images_file_names = df.iloc[:, 0].tolist()

        # Generate mask filenames
        self.masks_file_names = [
            f.replace('.png', '_mask.png').replace('.jpg', '_mask.jpg')
            for f in self.images_file_names
        ]

    def _load_from_folder(self):
        """Load all files from folder"""
        print(f"Loading filenames from folder: {self.images_dir}")

        # Get all image files
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        self.images_file_names = sorted([
            f.name for f in self.images_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ])

        # Generate mask filenames
        self.masks_file_names = sorted([
            f.name for f in self.masks_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ])

    def __len__(self):
        return len(self.images_file_names)

    def __getitem__(self, idx):
        """
        Returns:
            dict with 'images' and 'targets' keys
            Compatible with MMSeg collate function
        """
        # Load image
        img_path = self.images_dir / self.images_file_names[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.masks_dir / self.masks_file_names[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']

        # Handle mask - check if already tensor or numpy
        mask_data = transformed['mask']
        if isinstance(mask_data, torch.Tensor):
            mask_tensor = mask_data.long()
        else:
            mask_tensor = torch.from_numpy(mask_data).long()

        # Convert to instance format
        instances = self._mask_to_instances(mask_tensor.numpy())

        target = {
            'masks': instances['masks'],
            'labels': instances['labels']
        }

        return {
            'images': image_tensor,
            'targets': target
        }

    def _mask_to_instances(self, mask):
        """Convert semantic mask to instance format"""
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]

        instance_masks = []
        labels = []

        for class_id in unique_classes:
            class_mask = (mask == class_id).astype(np.uint8)
            num_instances, labels_im = cv2.connectedComponents(class_mask)

            for instance_id in range(1, num_instances):
                instance_mask = (labels_im == instance_id).astype(np.float32)

                if instance_mask.sum() < 10:
                    continue

                instance_masks.append(torch.from_numpy(instance_mask))
                labels.append(int(class_id))

        # Handle case with no instances
        if len(instance_masks) == 0:
            h, w = mask.shape
            instance_masks = [torch.zeros((h, w), dtype=torch.float32)]
            labels = [0]

        return {
            'masks': torch.stack(instance_masks),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def segmentation_collate_fn(batch):
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


def evaluation_collate_fn(batch):
    """
    Collate function for evaluation dataloader.
    Compatible with MMSeg training code.
    """
    images = torch.stack([item['images'] for item in batch])
    targets = [item['targets'] for item in batch]

    return {
        'images': images,
        'targets': targets
    }