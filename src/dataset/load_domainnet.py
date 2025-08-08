import os
import tensorflow_datasets as tfds
from PIL import Image

def save_domainnet_subset(output_dir, real_count=40, other_count=20):
    """
    Save a subset of DomainNet dataset as JPG images in a single directory.
    
    Args:
        output_dir: Output directory path where images will be saved
        real_count: Number of images to save for 'real' domain (default: 40)
        other_count: Number of images to save for other domains (default: 20)
    """
    os.makedirs(output_dir, exist_ok=True)
    domains = ['real', 'painting', 'clipart', 'quickdraw', 'infograph', 'sketch']
    
    for domain in domains:
        count = real_count if domain == 'real' else other_count
        
        ds = tfds.load(f'domainnet/{domain}', split='train', shuffle_files=True)
        ds = ds.shuffle(buffer_size=10000).take(count)
        
        for i, example in enumerate(tfds.as_numpy(ds)):
            img = example['image']
            pil_img = Image.fromarray(img)
            img_path = os.path.join(output_dir, f'{domain}_image_{i+1:03d}.jpg')
            pil_img.save(img_path, 'JPEG', quality=95)

if __name__ == "__main__":
    output_directory = "Pairwise-Image-Matching/data/domainnet"
    save_domainnet_subset(output_directory, real_count=40, other_count=20)
    print(f"Test subset saved to {output_directory}")