"""Dataset and data-loading utilities for WH-ESPC.

Each sample is one CT scan stored as a folder of axial slice images. A fixed
set of slices is sampled along the axial Z-axis (from lung apex to lung base)
and stacked into a sequence of shape ``[num_frames, C, H, W]``.
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import transforms

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

# ImageNet normalization, matching the pre-trained backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _natural_key(name):
    """Sort key that orders filenames with embedded numbers numerically."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', name)]


class CTSequenceDataset(data.Dataset):
    """Dataset of CT scans, each represented as a sequence of axial slices.

    Args:
        data_path: root directory containing one folder per scan.
        folders: scan folder names (relative to ``data_path``).
        labels: integer class label per scan.
        frames: indices of the slices to sample (into the naturally sorted
            list of slice images of each scan).
        transform: torchvision transform applied to every slice.
    """

    def __init__(self, data_path, folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        # Naturally sorted so that the slice order follows the axial Z-axis.
        file_dir = os.path.join(path, selected_folder)
        names = [f for f in os.listdir(file_dir)
                 if f.lower().endswith(IMAGE_EXTENSIONS)]
        names.sort(key=_natural_key)

        if len(names) <= max(self.frames):
            raise ValueError(
                f"Scan '{selected_folder}' contains {len(names)} slice images, "
                f"but frame index {max(self.frames)} was requested. Adjust the "
                f"--begin-frame/--end-frame/--skip-frame settings.")

        images = []
        for i in self.frames:
            image = Image.open(os.path.join(file_dir, names[i]))
            # Convert single-channel CT slices to RGB for the backbone.
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if use_transform is not None:
                image = use_transform(image)
            images.append(image)
        return torch.stack(images, dim=0)

    def __getitem__(self, index):
        folder = self.folders[index]
        x = self.read_images(self.data_path, folder, self.transform)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


def build_transforms(image_size, augment=False):
    """Build the per-slice transform.

    Args:
        image_size: spatial resolution the slices are resized to.
        augment: if True, apply random data augmentation (training set).
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def balance_by_oversampling(train_list, train_label, seed=42):
    """Balance the training classes by random oversampling with replacement.

    Every class is resampled up to the size of the majority class; the
    duplicated samples are later seen through random data augmentation.
    """
    rng = np.random.default_rng(seed)
    train_label = np.asarray(train_label)
    class_counts = np.bincount(train_label)
    max_count = max(class_counts)

    balanced_list, balanced_label = [], []
    for class_idx in range(len(class_counts)):
        class_indices = np.where(train_label == class_idx)[0]
        balanced_list.extend(train_list[i] for i in class_indices)
        balanced_label.extend([class_idx] * len(class_indices))

        augment_count = max_count - len(class_indices)
        if augment_count > 0:
            resampled = rng.choice(class_indices, size=augment_count, replace=True)
            balanced_list.extend(train_list[i] for i in resampled)
            balanced_label.extend([class_idx] * augment_count)

    return balanced_list, np.asarray(balanced_label)


def build_dataloaders(csv_path, data_dir, id_col="ID", label_col="Label",
                      begin_frame=4, end_frame=40, skip_frame=2,
                      image_size=224, test_size=0.2, seed=42,
                      batch_size=64, num_workers=8, pin_memory=True):
    """Create the training and validation dataloaders.

    The CSV file must contain one row per scan with the scan folder name and
    its class label. The training split is class-balanced by oversampling and
    uses data augmentation; the validation split uses the plain transform.

    Returns:
        train_loader, val_loader
    """
    df = pd.read_csv(csv_path, sep=",", engine="python", encoding="utf-8")
    all_names = [str(v) for v in df[id_col].tolist()]
    all_labels = np.asarray(df[label_col].tolist())

    train_list, val_list, train_label, val_label = train_test_split(
        all_names, all_labels, test_size=test_size, random_state=seed)

    frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    transform = build_transforms(image_size, augment=False)
    transform_aug = build_transforms(image_size, augment=True)

    # Class-balance the training split; oversampled samples are augmented.
    balanced_list, balanced_label = balance_by_oversampling(
        train_list, train_label, seed=seed)

    train_set = CTSequenceDataset(data_dir, balanced_list, balanced_label,
                                  frames, transform=transform_aug)
    val_set = CTSequenceDataset(data_dir, val_list, val_label,
                                frames, transform=transform)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin_memory)
    train_loader = data.DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = data.DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
