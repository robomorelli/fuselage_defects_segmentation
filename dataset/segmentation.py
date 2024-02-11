from torch.utils.data import Dataset
import os
from PIL import Image
from pathlib import Path
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torchvision.transforms as T

'''
torch transform act naturally on the PIL image
Albumentation is paired with image = cv2.imread("/path/to/image.jpg") - image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
Albumentation transform image and masks contemporarly, toorch.transform need v2 version to do that
torch transform tensor also normalize, albumentation casting to torch does not normalize autatically
'''

'''
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
    ])
'''

processor = SegformerImageProcessor()

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, idxs=None):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(Path(self.root_dir).as_posix(),'images')
        self.masks_dir = os.path.join(Path(self.root_dir).as_posix(), 'masks')
        self.feature_extractor = feature_extractor
        self.indices = idxs


        self.images_file_names = [f for f in os.listdir(self.images_dir) if '.png' in f]
        self.masks_file_names = [f.replace('.', '_mask.') for f in self.image_file_names if '.png' in f]

        if self.indices != None:
            self.images_file_names = [x for ix, x in enumerate(self.images_file_names) if ix in self.indices]
            self.masks_file_names = [x for ix, x in enumerate(self.masks_file_names) if ix in self.indices]

        self.images = sorted(self.images_file_names)
        self.masks = sorted(self.masks_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.images_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.masks_dir, self.masks[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k ,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs


class BinarySegmentationAlb(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, idxs=None, transform=None, test=False, normalize_imagenet=False):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.

        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(Path(self.root_dir).as_posix(), 'images')
        self.masks_dir = os.path.join(Path(self.root_dir).as_posix(), 'masks')
        self.indices = idxs
        self.transform = transform
        self.test = test
        self.normalize_imagenet = normalize_imagenet

        if self.normalize_imagenet:
            self.mean = (0.485, 0.456, 0.406, 0)
            self.std = (0.229, 0.224, 0.225)
        else:
            self.mean = (0.0, 0.0, 0.0)
            self.std = (1.0, 1.0, 1.0)

        self.base_transform = A.Compose(
            [
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

        self.images_file_names = [f for f in os.listdir(self.images_dir) if '.png' in f]
        self.masks_file_names = [f.replace('.', '_mask.') for f in self.images_file_names if '.png' in f]

        if self.indices != None:
            self.images_file_names = [x for ix, x in enumerate(self.images_file_names) if ix in self.indices]
            if not self.test:
                self.masks_file_names = [x for ix, x in enumerate(self.masks_file_names) if ix in self.indices]

        self.images = self.images_file_names
        if not self.test:
            self.masks = self.masks_file_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.images_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not self.test:
            mask = cv2.imread(os.path.join(self.masks_dir, self.masks[idx]))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[:,:,0:1]

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                x = transformed['image']
                y = transformed['mask']
                y = y.permute(2, 0, 1)
                y = y / 255.

            else:
                transformed = self.base_transform(image=image, mask=mask)
                x = transformed['image']
                y = transformed['mask']
                y = y.permute(2, 0, 1)
                y = y / 255.

            return x, y

        else:
            if self.transform is not None:
                transformed = self.transform(image=image)
                x = transformed['image']
            else:
                transformed = self.base_transform(image=image)
                x = transformed['image']
            return x


class BinarySegmentationPil(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, idxs=None, transform=None, test=False, normalize_imagenet=False):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.

        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(Path(self.root_dir).as_posix(), 'images')
        self.masks_dir = os.path.join(Path(self.root_dir).as_posix(), 'masks')
        self.indices = idxs
        self.transform = transform
        self.test = test
        self.normalize_imagenet = normalize_imagenet


        if self.normalize_imagenet:
            self.mean = (0.485, 0.456, 0.406, 0)
            self.std = (0.229, 0.224, 0.225)

            self.make_tensor = T.Compose([
                                     T.ToTensor(),
                        T.Normalize(mean=self.mean, std=self.std)])
        else:
            self.make_tensor = T.Compose([
                T.ToTensor()])

        self.images_file_names = [f for f in os.listdir(self.images_dir) if '.png' in f]
        self.masks_file_names = [f.replace('.', '_mask.') for f in self.images_file_names if '.png' in f]

        if self.indices != None:
            self.images_file_names = [x for ix, x in enumerate(self.images_file_names) if ix in self.indices]
            if not self.test:
                self.masks_file_names = [x for ix, x in enumerate(self.masks_file_names) if ix in self.indices]

        self.images = self.images_file_names
        if not self.test:
            self.masks = self.masks_file_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.images_dir, self.images[idx])).convert('RGB')
        if not self.test:
            mask = Image.open(os.path.join(self.masks_dir, self.masks[idx])).convert("L")

        if self.transform is not None:
            x = self.transform(image)
            if not self.test:
                y = self.transform(mask)
                return x, y

        else:
            x = self.make_tensor(image)
            if not self.test:
                y = self.make_tensor(mask)
                return x, y
