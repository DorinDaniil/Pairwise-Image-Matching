import json
import os
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from ..model import get_siamnet
from ..dataset import load_config


def save_metrics(file_path, metrics_data):
    """Save metrics data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(metrics_data, file, indent=4)

def load_and_preprocess_image(image_path, preprocess):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    return image

class CandidateDataset(Dataset):
    def __init__(self, candidate_data, preprocess):
        self.candidate_data = candidate_data
        self.preprocess = preprocess
        self.paths = list(candidate_data.keys())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = self.candidate_data[path]
        image = load_and_preprocess_image(data['path'], self.preprocess)
        return image

def fpr_benchmark(config_path, return_scores=False):
    config = load_config(config_path)

    dataset_path = config['dataset_path']
    output_metrics_path = config['output_metrics_path']
    model_config_path = config['model_config_path']

    model_config = load_config(model_config_path)
    device = model_config['training']['device']
    checkpoint_path = model_config['model']['initialization_weights_path']
    threshold = config['threshold']

    # Initialize model
    model = get_siamnet(model_config)
    preprocess = model.get_preprocessing()
    model.eval()

    search_originals_path = os.path.join(dataset_path, 'search-originals')
    candidates_path = os.path.join(dataset_path, 'candidates')

    if os.path.exists(output_metrics_path):
        with open(output_metrics_path, 'r') as json_file:
            metrics_data = json.load(json_file)
    else:
        metrics_data = {}

    # Checkpoint name for metrics
    checkpoint_name = os.path.basename(checkpoint_path)
    if checkpoint_name not in metrics_data:
        metrics_data[checkpoint_name] = {}

    all_fprs = []
    all_scores = []

    for original_image_name in tqdm(os.listdir(search_originals_path), desc="Processing originals"):
        original_image_path = os.path.join(search_originals_path, original_image_name)
        original_image = load_and_preprocess_image(original_image_path, preprocess)
        original_batch = original_image.unsqueeze(0).to(device)

        candidate_folder = os.path.join(candidates_path, os.path.splitext(original_image_name)[0])
        candidate_data = {img_name: {'path': os.path.join(candidate_folder, img_name)} for img_name in os.listdir(candidate_folder)}

        # Create dataset and dataloader for candidates
        candidate_dataset = CandidateDataset(candidate_data, preprocess)
        candidate_loader = DataLoader(candidate_dataset, batch_size=4, shuffle=False, num_workers=4)

        similarity_scores = []

        for candidate_images in candidate_loader:
            candidate_images = candidate_images.to(device)
            original_batch_expanded = original_batch.expand(candidate_images.size(0), -1, -1, -1)

            with torch.no_grad():
                scores = model.predict_similarity(original_batch_expanded, candidate_images).cpu().numpy().flatten()

            similarity_scores.extend(scores)
        
        fpr = np.sum(np.array(similarity_scores) > threshold) / len(similarity_scores)
        metrics_data[checkpoint_name][os.path.splitext(original_image_name)[0]] = {'fpr': fpr}
        all_fprs.append(fpr)
        all_scores.extend(similarity_scores)

    mean_fpr = np.mean(all_fprs)
    std_fpr = np.std(all_fprs, ddof=1)
    metrics_data[checkpoint_name]['overall_per_original'] = {'mean_fpr': mean_fpr, 'std_fpr': std_fpr}

    overall_fpr = np.sum(np.array(all_scores) > threshold) / len(all_scores)
    metrics_data[checkpoint_name]['overall_concatenated'] = {'fpr': overall_fpr}

    save_metrics(output_metrics_path, metrics_data)

    print(f"Metrics saved to {output_metrics_path}")
    if return_scores:
        return all_fprs, all_scores