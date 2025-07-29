import yaml
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .augmentations import get_augmentations

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class CocoDataset(Dataset):
    def __init__(self, data_dir, preprocessor, split="train", val_size=0.1, random_seed=42):
        """
        Args:
            data_dir (str): Root directory of the dataset (e.g., '/data/')
            preprocessor (callable): Function or object to transform images into tensors (e.g., CLIPProcessor)
            split (str): Dataset split, either 'train' or 'test'
            val_size (float): Fraction of total data to use for the test split (e.g., 0.1 = 10%)
            random_seed (int): Random seed for reproducible train/test split
        """
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.split = split
        self.val_size = val_size
        self.simple_transform, self.train_transform = get_augmentations()
        self.image_paths = []

        coco_dir = os.path.join(data_dir, 'coco2017', 'coco_images')
        splits = ['train2017', 'val2017', 'test2017', 'unlabeled2017']

        for split_name in splits:
            split_dir = os.path.join(coco_dir, split_name)
            if not os.path.exists(split_dir):
                continue
            for root, _, files in os.walk(split_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(root, file))

        total_size = len(self.image_paths)
        indices = list(range(total_size))

        random.Random(random_seed).shuffle(indices)

        split_idx = int(total_size * (1 - self.val_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        if split == "train":
            selected_indices = train_indices
        elif split in ["val", "test"]:
            selected_indices = test_indices
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.image_paths = [self.image_paths[i] for i in selected_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        augmented1 = self.train_transform(image).convert('RGB')
        augmented2 = self.simple_transform(image).convert('RGB')

        tensor1 = self.preprocessor(augmented1)
        tensor2 = self.preprocessor(augmented2)

        return tensor1, tensor2


def get_coco_dataloaders(data_dir, preprocessor, batch_size=32, num_workers=4,
                     val_size=0.1, random_seed=42):
    """
    Creates train and test dataloaders for the COCO dataset with unified train/test split.

    Args:
        data_dir (str): Root path to the dataset.
        preprocessor (callable): Image-to-tensor processor (e.g., CLIPProcessor).
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of subprocesses for data loading.
        val_size (float): Fraction of data to use for test set.
        random_seed (int): Seed for reproducible split.

    Returns:
        dict: {'train': train_loader, 'test': test_loader}
    """
    train_dataset = CocoDataset(
        data_dir=data_dir,
        preprocessor=preprocessor,
        split="train",
        val_size=val_size,
        random_seed=random_seed,
    )

    test_dataset = CocoDataset(
        data_dir=data_dir,
        preprocessor=preprocessor,
        split="test",
        val_size=val_size,
        random_seed=random_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'test': test_loader
    }