import torchvision.transforms as transforms
from torchvision.transforms import functional
import numpy as np
import albumentations as A
from PIL import Image
import random

class RandomCompose(object):
    """
    Composes several augmentations together with a given probability.

    Args:
        transforms (List[Transform]): List of transforms to compose.
        p (float): Probability of applying each transform.

    Example:
        >>> RandomCompose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ], p=0.5)
    """
    def __init__(self, transforms, p=0.5):
        self.p = p
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if random.random() > self.p:
                img = t(img)
        return img

class CenterCrop(object):
    """
    Randomly crops the image from the center with a random size.

    Args:
        sizes (List[float]): List of possible crop sizes.
    """
    def __init__(self, sizes=[5, 10]):
        self.sizes = sizes

    def __call__(self, img):
        crop = random.choice(self.sizes)
        w, h = img.size
        h = h * (1 - crop / 100)
        w = w * (1 - crop / 100)
        img = functional.center_crop(img, [h, w])
        return img

class Crop(object):
    """
    Randomly crops the image with a random size and position.

    Args:
        size (float): Maximum crop size as a percentage of the image dimensions.
    """
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

class GaussianBlur(object):
    """
    Applies Gaussian blur to the image with a random kernel size.

    Args:
        kernel_sizes (List[int]): List of possible kernel sizes.
    """
    def __init__(self, kernel_sizes=[3, 5]):
        self.kernel_sizes = kernel_sizes

    def __call__(self, img):
        kernel = random.choice(self.kernel_sizes)
        img = functional.gaussian_blur(img, (kernel, kernel))
        return img

class Scale(object):
    """
    Randomly scales the image with a scale factor between 0.3 and 2.
    """
    def __init__(self):
        pass  # No need to initialize with a predefined list of scales

    def __call__(self, img):
        scale = random.uniform(0.3, 2)  # Randomly choose a scale factor between 0.3 and 2
        w, h = img.size
        h = int(h * scale)
        w = int(w * scale)
        img = functional.resize(img, size=(h, w))
        return img

class RandomRotation90(object):
    """
    Randomly rotates the image by 90 degrees.
    """
    def __call__(self, img):
        angle = random.choice([90, 180, 270])
        img = functional.rotate(img, angle)
        return img

class Grayscale(object):
    """
    Converts the image to grayscale.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            img = functional.to_grayscale(img)
        return img

class GaussianNoise(object):
    """
    Adds Gaussian noise to the image with a random standard deviation.

    Args:
        mean (float): Mean of the Gaussian noise.
    """
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

def get_augmentations():
    """
    Returns a set of image transformations for different use cases.

    Returns:
        Tuple[RandomCompose, RandomCompose]: A tuple containing simple, and train transforms.
    """
    # Used for both queries and targets in evaluation mode
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Used for targets in train
    simple_transform = RandomCompose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
        Crop(size=2),
    ])

    # Augmentations for queries in train (unlike basic_transform and simple_transform it returns a PIL image)
    # if SimilarityDataset has use_augmentations=True
    train_transform = RandomCompose([
        CenterCrop(sizes=[5, 10]),
        Crop(size=5),
        GaussianNoise(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.8),
        GaussianBlur(kernel_sizes=[3, 5]),
        Scale(),
        RandomRotation90(),
        Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    return simple_transform, train_transform
