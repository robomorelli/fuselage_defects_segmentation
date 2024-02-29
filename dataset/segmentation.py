import pandas as pd
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
import torch

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

class KFoldDataframe(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, data_path, df_path, df=None, idxs=None, transform=None,
                 test=False, normalize_imagenet=False, cropped=True, from_full_to_crop=False):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.

        """
        self.root_dir = data_path
        self.df_path = df_path
        self.df = df
        self.images_dir = os.path.join(Path(self.root_dir), 'images')
        self.masks_dir = os.path.join(Path(self.root_dir), 'masks')
        self.indices = idxs
        self.transform = transform
        self.test = test
        self.normalize_imagenet = normalize_imagenet
        self.cropped = cropped
        self.from_full_to_crop = from_full_to_crop

        if self.df is None:
            if self.cropped:
                self.df_names = pd.read_csv(os.path.join(self.df_path, "cropped_filenames.csv"))
                self.images_file_names = self.df_names['images']
                self.masks_file_names = self.df_names['masks']
            else:
                self.df_names = pd.read_csv(os.path.join(self.df_path, "full_size_filenames.csv"))

                if self.from_full_to_crop:
                    self.cropped_image_files = os.listdir(self.images_dir)
                    self.df_names = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in range(len(self.df_names)) for crop_fh in
                                  self.cropped_image_files if 'cropped_' + self.df_names['images'].values[idx].split('.')[0]
                                     == '_'.join(crop_fh.split('_')[:-2])]

                    self.images_file_names = [x[0] for x in self.df_names]
                    self.masks_file_names = [x[1] for x in self.df_names]
                else:
                    self.images_file_names = self.df_names['images']
                    self.masks_file_names = self.df_names['masks']

        else:
            self.images_file_names = self.df['images']
            self.masks_file_names = self.df['masks']

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
            ])


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


class KFoldDataframeMulticlass(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, data_path, df_path, df=None, idxs=None, transform=None,
                 test=False, normalize_imagenet=False,
                 cropped=True, from_full_to_crop=False, n_classes = 2,
                 rescale_before_norm=False):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.

        """
        self.root_dir = data_path
        self.df_path = df_path
        self.df = df
        self.images_dir = os.path.join(Path(self.root_dir), 'images')
        self.masks_dir = os.path.join(Path(self.root_dir), 'masks')
        self.indices = idxs
        self.transform = transform
        self.test = test
        self.normalize_imagenet = normalize_imagenet
        self.cropped = cropped
        self.from_full_to_crop = from_full_to_crop
        self.n_classes = n_classes
        self.rescale_before_norm = rescale_before_norm

        if self.df is None:
            if self.cropped:
                self.df_names = pd.read_csv(os.path.join(self.df_path, "cropped_filenames.csv"))
                self.images_file_names = self.df_names['images']
                self.masks_file_names = self.df_names['masks']
            else:
                self.df_names = pd.read_csv(os.path.join(self.df_path, "full_size_filenames.csv"))

                if self.from_full_to_crop:
                    self.cropped_image_files = os.listdir(self.images_dir)
                    self.df_names = [(crop_fh, crop_fh.replace('.', '_mask.')) for idx in range(len(self.df_names)) for crop_fh in
                                  self.cropped_image_files if 'cropped_' + self.df_names['images'].values[idx].split('.')[0]
                                     == '_'.join(crop_fh.split('_')[:-2])]

                    self.images_file_names = [x[0] for x in self.df_names]
                    self.masks_file_names = [x[1] for x in self.df_names]
                else:
                    self.images_file_names = self.df_names['images']
                    self.masks_file_names = self.df_names['masks']

        else:
            self.images_file_names = self.df['images']
            self.masks_file_names = self.df['masks']

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
            ])

        self.mask_base_transform = A.Compose(
            [
                ToTensorV2(),
            ])


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
                if self.rescale_before_norm:
                    y = ((y - 0)/(self.n_classes - 0)) * (255 - 0)
                #y = y / 255.
                #y = y * self.n_classes
                y = y.int()

                channels = [torch.zeros_like(y, dtype=torch.float) for _ in range(self.n_classes)]

                # Assign 1 to each channel where tensor equals the channel index
                for i in range(self.n_classes):
                    channels[i][y == i+1] = 1

                # Stack the channels to form a multi-channel tensor
                multi_channel_y = torch.stack(channels, dim=0)
                multi_channel_y = torch.squeeze(multi_channel_y, 1)

            else:
                transformed = self.base_transform(image=image, mask=mask)
                x = transformed['image']
                y = transformed['mask']
                y = y.permute(2, 0, 1)
                if self.rescale_before_norm:
                    y = ((y - 0)/(self.n_classes - 0)) * (255 - 0)
                #y = y / 255.
                #y = y * self.n_classes
                y = y.int()

                channels = [torch.zeros_like(y, dtype=torch.float) for _ in range(self.n_classes)]

                # Assign 1 to each channel where tensor equals the channel index
                for i in range(self.n_classes):
                    channels[i][y == i+1] = 1

                # Stack the channels to form a multi-channel tensor
                multi_channel_y = torch.stack(channels, dim=0)
                multi_channel_y = torch.squeeze(multi_channel_y, 1)

            return x, y, multi_channel_y

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
