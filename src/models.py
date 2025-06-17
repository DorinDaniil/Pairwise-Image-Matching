import json
import yaml
from PIL import Image
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

import torchvision.transforms as transforms
from torchvision.transforms import functional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as transforms
import os
import random

class RandomCompose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, p=0.5):
        self.p = p
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if random.random() > self.p:
                img = t(img)
        return img

class CENTER_CROP(object):
    def __init__(self, sizes=[5, 10]):
        self.sizes = sizes

    def __call__(self, img):
        crop = random.choice(self.sizes)
        w, h = img.size
        h = h * (1 - crop / 100)
        w = w * (1 - crop / 100)
        img = functional.center_crop(img, [h, w])
        return img

class CROP(object):
    def __init__(self, size=5):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        left = random.randint(0, int(w * self.size / 100))
        top = random.randint(0, int(h * self.size / 100))
        width = random.randint(int(w * (1 - self.size / 100) - left), int(w - left - 1))
        height = random.randint(int(h * (1 - self.size / 100) - top), int(h - top - 1))
        img = functional.crop(img, top=top, left=left, height=height, width=width)
        return img

class GAUSSBLUR(object):
    def __init__(self, kernel_sizes=[3, 5]):
        self.kernel_sizes = kernel_sizes

    def __call__(self, img):
        kernel = random.choice(self.kernel_sizes)
        img = functional.gaussian_blur(img, (kernel, kernel))
        return img

class SCALE(object):
    def __init__(self, scale=[0.5, 0.75, 1.5, 2]):
        self.scale = scale

    def __call__(self, img):
        scale = random.choice(self.scale)
        w, h = img.size
        h = int(h * scale)
        w = int(w * scale)
        img = functional.resize(img, size=(h, w))
        return img

class GRAY_OR_CHANNEL(object):
    def __init__(self, channel=[0, 1, 2], p=0.2):
        self.channel = channel
        self.p = p

    def __call__(self, img):
        prob = random.random()
        if prob > (1 - self.p):
            channel = random.choice(self.channel)
            channels = img.split()
            img = channels[channel]
        elif prob < self.p:
            img = functional.to_grayscale(img)
        return img

class RandomRotation90(object):
    """Randomly rotate the image by 90 degrees."""

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        img = functional.rotate(img, angle)
        return img

class Grayscale(object):
    """Convert the image to grayscale."""

    def __call__(self, img):
        img = functional.to_grayscale(img)
        return img

class GaussianNoise(object):
    """Add Gaussian noise to the image with a random std."""

    def __init__(self, mean=0.0):
        self.mean = mean

    def __call__(self, img):
        img_array = np.array(img)
        std = np.random.uniform(0.5, 2.0)  # Randomly select std between 0.5 and 2.0
        noise = np.random.normal(self.mean, std, img_array.shape)
        noisy_img_array = img_array + noise
        noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_img_array)
        return img

def get_transforms():
    # used for both queries and targets in evaluation mode
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # used for targets in train
    simple_transform = RandomCompose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
        GaussianNoise(),
    ])

    # augmentations for queries in train (unlike basic_transform and simple_transform it returns a PIL image)
    # if SimilarityDataset has use_augmentations=True
    train_transform = RandomCompose([
        CENTER_CROP(sizes=[5, 10]),
        CROP(size=5),
        GaussianNoise(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
        GAUSSBLUR(kernel_sizes=[3, 5]),
        SCALE(scale=[0.5, 0.75, 1.5, 2]),
        GRAY_OR_CHANNEL(),
        RandomRotation90(),
        Grayscale(),
        transforms.RandomHorizontalFlip(),  # Added horizontal flip
        transforms.RandomVerticalFlip()     # Added vertical flip
    ])

    return basic_transform, simple_transform, train_transform


class Transform:
    """Transform class for image augmentation."""

    def __init__(self):
        # Define individual transformations
        self.rotate_transform = A.RandomRotate90(p=1.0)
        self.blur_transform = A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 2.0), p=1.0)
        self.gaussnoise_transform = A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
        self.downscale_transform = A.Downscale(scale_min=0.5, scale_max=0.9, interpolation=0, p=1.0)
        self.crop_transform = A.RandomResizedCrop(height=224, width=224, scale=(0.7, 0.9), ratio=(0.75, 1.33), p=1.0)
        self.horizontal_flip = A.HorizontalFlip(p=1.0)
        self.color_jitter = A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=1.0)
        self.to_gray = A.ToGray(p=1.0)

        # Combined transformations
        self.rotate_and_flip = A.Compose([
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=1.0)
        ])

        self.complex_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.1),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5)
        ])

        # Transform for prime
        self.transform_prime = A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.0), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.1),
            A.HorizontalFlip(p=0.1),
            A.ToGray(p=0.1),
            # A.RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0)
        ])

        # Preprocessor
        self.preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        # List of all transformations
        self.all_transforms = [
            self.rotate_transform,
            self.blur_transform,
            self.gaussnoise_transform,
            self.downscale_transform,
            self.color_jitter,
            self.crop_transform,
            self.horizontal_flip,
            self.rotate_and_flip,
            self.complex_transform
        ]

    def __call__(self, image):
        # Randomly select a transformation
        selected_transform = np.random.choice(self.all_transforms)
        augmented = selected_transform(image=image)
        augmented_image = augmented['image']
        augmented_image = Image.fromarray(augmented_image)
        
        # Apply preprocessor
        augmented_image = self.preprocessor(augmented_image)
        return augmented_image

    def prime(self, image):
        # Apply prime transformation
        augmented = self.transform_prime(image=image)
        augmented_image = augmented['image']
        augmented_image = Image.fromarray(augmented_image)

        # Apply preprocessor
        augmented_image = self.preprocessor(augmented_image)
        return augmented_image


class SiamNet(nn.Module):
    def __init__(self):
        super(SiamNet, self).__init__()
        self.extractor = EfficientNet.from_pretrained('efficientnet-b3')
        del self.extractor._fc

        self.dropf = nn.Dropout(p=0.20)
        self.fc1 = nn.Linear(1536,1536)
        self.drop1 = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(1536,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        # Get embeddings for both batches
        emb1 = self.get_q(batch1)
        emb2 = self.get_q(batch2)

        # Create all possible combinations of embeddings
        batch_size1, emb_dim = emb1.size(0), emb1.size(1)
        batch_size2 = emb2.size(0)

        emb1_expanded = emb1.unsqueeze(1).expand(-1, batch_size2, -1)
        emb2_expanded = emb2.unsqueeze(0).expand(batch_size1, -1, -1)

        # dim: batch_size1 x batch_size2 x emb_dim
        combined_embeddings = torch.abs(emb1_expanded - emb2_expanded)
        combined_embeddings = combined_embeddings.view(-1, emb_dim)

        # Get logits
        logits = self.head_model(combined_embeddings)

        # Reshape logits to form a batch_size1 x batch_size2 matrix
        logits = logits.view(batch_size1, batch_size2)

        return logits

    def head_model(self, combined_embeddings):
        r = self.drop1(F.relu(self.fc1(combined_embeddings)))
        r = self.sigmoid(self.fc2(r))
        return r
    
    def forward_original(self, query, target, input_q=None, input_t=None, return_vectors=False):
        b = query.size(0)
        if input_q is None:
            q = self.extractor._conv_stem(query)
            q = self.extractor._bn0(q)
            for k,lay in enumerate(self.extractor._blocks):
                q = lay(q)
            q = self.extractor._conv_head(q)
            q = self.extractor._bn1(q)
            q = self.extractor._avg_pooling(q).view(b,-1)
            q = self.dropf(q)
        else:
            q = input_q

        if input_t is None:
            t = self.extractor._conv_stem(target)
            t = self.extractor._bn0(t)
            for k,lay in enumerate(self.extractor._blocks):
                t = lay(t)
            t = self.extractor._conv_head(t)
            t = self.extractor._bn1(t)
            t = self.extractor._avg_pooling(t).view(b,-1)
            t = self.dropf(t)
        else:
            t = input_t

        r = torch.abs(q-t)
        r = self.drop1(F.relu(self.fc1(r)))
        r = self.sigmoid(self.fc2(r))
        if return_vectors:
            return r, (q, t)
        else:
            return r

    def get_q(self, query):
        b = query.size(0)
        q = self.extractor._conv_stem(query)
        q = self.extractor._bn0(q)
        for k,lay in enumerate(self.extractor._blocks):
            q = lay(q)
        q = self.extractor._conv_head(q)
        q = self.extractor._bn1(q)
        q = self.extractor._avg_pooling(q).view(b,-1)
        q = self.dropf(q)
        return q

def get_siamnet(config):
    device = config['training']['device']
    net = SiamNet()
    if config['model']['weights_dir']:
        net.load_state_dict(torch.load(config['model']['weights_path']))
        print(f"loaded weights from {config['model']['weights_path']}")
        
    if config['model']['reinitialize_fc_layers']:
        net.fc1 = nn.Linear(1536, 1536)
        net.fc2 = nn.Linear(1536, 1)
        print('fully connected layers reinitialized\n')
    
    if config['model']['freeze_extractor_layers']:
        for param in net.parameters():
            param.requires_grad = False
        net.fc1.weight.requires_grad = True
        net.fc1.bias.requires_grad = True
        net.fc2.weight.requires_grad = True
        net.fc2.bias.requires_grad = True
        print('feature extractor layers frozen\n')
        
    net = net.to(device)
    return net