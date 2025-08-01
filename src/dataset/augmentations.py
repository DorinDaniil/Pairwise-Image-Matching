import torchvision.transforms as transforms
import numpy as np
import albumentations as A
import random
import string
from torchvision.transforms import functional
from PIL import Image, ImageDraw, ImageFont

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

class AddTextOverlay(object):
    def __init__(self, 
                 font_size_range=(20, 200), 
                 opacity_range=(0.4, 1.0),
                 text_length_range=(5, 12),
                 special_chars="©®™€$#№@"):
        self.font_size_range = font_size_range
        self.opacity_range = opacity_range
        self.text_length_range = text_length_range
        self.special_chars = special_chars
        self.predefined_texts = ["© 2024", "Sample", "Confidential", "Draft", "TEST", "PRIVATE", "UNCLASSIFIED"]

    def __call__(self, img):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        draw = ImageDraw.Draw(img)
        
        if random.random() < 0.5:
            text = random.choice(self.predefined_texts)  # Вариант 1: готовая фраза
        else:
            chars = string.ascii_letters + string.digits + self.special_chars
            text_length = random.randint(*self.text_length_range)
            text = ''.join(random.choices(chars, k=text_length))
        
        color = random.choice([(255, 255, 255, 255), (0, 0, 0, 255)])
        opacity = int(255 * random.uniform(*self.opacity_range))
        fill_color = (color[0], color[1], color[2], opacity)
        
        font_size = random.randint(*self.font_size_range)
        x = random.randint(10, max(10, img.width - 100))
        y = random.randint(10, max(10, img.height - 50))
        
        draw.text((x, y), text, fill=fill_color)
        return img.convert('RGB')

class AddWatermark(object):
    def __init__(self, opacity_range=(0.3, 0.7), size_ratio_range=(0.05, 0.2)):
        self.opacity_range = opacity_range
        self.size_ratio_range = size_ratio_range

    def __call__(self, img):
        watermark = Image.new("RGBA", (50, 50), (255, 255, 255, 128))
        opacity = random.uniform(*self.opacity_range)
        watermark = watermark.resize((
            int(img.width * random.uniform(*self.size_ratio_range)),
            int(img.height * random.uniform(*self.size_ratio_range))
        ))
        watermark.putalpha(int(255 * opacity))
        
        pos = random.choice([
            (10, 10),
            (img.width - watermark.width - 10, 10),
            (img.width // 2 - watermark.width // 2, img.height // 2 - watermark.height // 2)
        ])
        img = img.convert("RGBA")
        img.paste(watermark, pos, watermark)
        return img.convert("RGB")

class AddColoredSquare(object):
    def __init__(self,
                 size_ratio_range=(0.1, 0.3),
                 opacity_range=(0.7, 1.0)):
        self.size_ratio_range = size_ratio_range
        self.opacity_range = opacity_range

    def __call__(self, img):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        draw = ImageDraw.Draw(img)

        # Square dimensions
        size_ratio = random.uniform(*self.size_ratio_range)
        square_size = int(min(img.width, img.height) * size_ratio)

        # Random position
        x = random.randint(0, img.width - square_size)
        y = random.randint(0, img.height - square_size)

        # Random color with opacity
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        opacity = int(255 * random.uniform(*self.opacity_range))
        fill_color = (color[0], color[1], color[2], opacity)

        # Draw square
        draw.rectangle([x, y, x + square_size, y + square_size], fill=fill_color)

        return img.convert('RGB')

class CenterCrop(object):
    """
    Randomly crops the image from the center with a random size.

    Args:
        sizes (List[float]): List of possible crop sizes.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        crop = random.uniform(1, 5)
        w, h = img.size
        h = int(h * (1 - crop / 100))
        w = int(w * (1 - crop / 100))
        img = functional.center_crop(img, [h, w])
        return img

class Crop(object):
    """
    Randomly crops the image with a random size and position.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        size = random.uniform(1, 10)
        w, h = img.size
        left = random.randint(0, int(w * size / 100))
        top = random.randint(0, int(h * size / 100))
        width = random.randint(int(w * (1 - size / 100) - left), int(w - left - 1))
        height = random.randint(int(h * (1 - size / 100) - top), int(h - top - 1))
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
        if random.random() > 0.2:
            scale = random.uniform(0.3, 2)  # Randomly choose a scale factor between 0.3 and 2
            w, h = img.size
            h = int(h * scale)
            w = int(w * scale)
            img = functional.resize(img, size=(h, w))
        else:
            scale_h = random.uniform(0.3, 2)
            scale_w = random.uniform(0.3, 2)
            w, h = img.size
            h = int(h * scale_h)
            w = int(w * scale_w)
            img = functional.resize(img, size=(h, w))
        return img

class RandomRotation90(object):
    """
    Randomly rotates the image by 90 degrees.
    """
    def __call__(self, img):
        angle = random.choice([90, 180, 270])
        
        img = functional.rotate(
            img, 
            angle, 
            expand=True)
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
    # Used for targets in train
    simple_transform = RandomCompose([
        AddWatermark(),
        AddTextOverlay(),
        Crop(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.4),
        Scale(),
    ])

    # Augmentations for queries in train
    train_transform = RandomCompose([
        AddWatermark(),
        AddColoredSquare(),
        AddTextOverlay(),
        CenterCrop(),
        Crop(),
        GaussianNoise(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=1.),
        GaussianBlur(kernel_sizes=[3, 5]),
        Scale(),
        RandomRotation90(),
        Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    return simple_transform, train_transform