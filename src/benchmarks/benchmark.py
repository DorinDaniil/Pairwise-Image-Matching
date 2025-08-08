import json
import os
import random
import itertools
import torch

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from ..model import get_siamnet
from ..dataset import get_validation_augmentations, load_config


class MetricsManager:
    """Class to handle loading and saving metrics."""

    @staticmethod
    def load_or_create_metrics_json(metrics_json_path):
        """Load existing metrics or create a new empty dictionary."""
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path, 'r') as json_file:
                metrics_data = json.load(json_file)
        else:
            metrics_data = {}
        return metrics_data

    @staticmethod
    def save_metrics(metrics_data, output_metrics_path):
        """Save metrics data to a JSON file."""
        with open(output_metrics_path, 'w') as file:
            json.dump(metrics_data, file, indent=4)


class ImageProcessor:
    """Class to handle image loading and preprocessing."""

    @staticmethod
    def load_and_preprocess_image(image_path, preprocess=None):
        """Load and preprocess an image."""
        image = Image.open(image_path).convert('RGB')
        if preprocess is not None:
            image = preprocess(image)
        return image


class RecallBenchmark:
    """Class to perform recall benchmarking."""

    def __init__(self, config_path, return_scores=False):
        config = load_config(config_path)
        self.dataset_path = config['dataset_path']
        self.output_metrics_path = config['output_metrics_path']
        self.model_config_path = config['model_config_path']
        self.threshold = config['threshold']
        self.return_scores = return_scores

        model_config = load_config(self.model_config_path)
        self.checkpoint_path = model_config['model']['initialization_weights_path']
        self.device = model_config['training']['device']

        self.model = get_siamnet(model_config)
        self.preprocess = self.model.get_preprocessing()
        self.model.eval()

        self.augmentations = get_validation_augmentations()

        if return_scores:
            self.metrics_data = None
        else:
            self.metrics_data = MetricsManager.load_or_create_metrics_json(self.output_metrics_path)
            self.checkpoint_name = os.path.basename(self.checkpoint_path)
            if self.checkpoint_name not in self.metrics_data:
                self.metrics_data[self.checkpoint_name] = {}

    def get_transformed_batch(self, image_path):
        """Generate a batch of transformed images."""
        original_image = ImageProcessor.load_and_preprocess_image(image_path)
        transformed_images = {name: augmentation(original_image) for name, augmentation in self.augmentations.items()}
        return transformed_images

    def save_image_transformations(self, image_path, save_folder):
        """Save original and transformed images to a folder."""
        os.makedirs(save_folder, exist_ok=True)
        original_image = ImageProcessor.load_and_preprocess_image(image_path)
        transformed_batch = self.get_transformed_batch(image_path)

        original_path = os.path.join(save_folder, "original.jpg")
        original_image.save(original_path)

        for aug_name, img in transformed_batch.items():
            safe_name = re.sub(r'[^\w\-]', '_', aug_name) + ".jpg"
            save_path = os.path.join(save_folder, safe_name)
            img.save(save_path)

    def run(self):
        """Run the benchmark and calculate recall metrics."""
        correct_predictions = {aug: 0 for aug in self.augmentations}
        total_predictions = {aug: 0 for aug in self.augmentations}

        image_files = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.lower().endswith(('.png', '.jpg'))]
        all_scores = []
        all_indices = []

        for idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
            original_image = ImageProcessor.load_and_preprocess_image(image_file)
            dataset = AugmentedImageDataset(original_image, self.augmentations, self.preprocess)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

            for original_batch, augmented_batch, indices in dataloader:
                original_batch = original_batch.to(self.device)
                augmented_batch = augmented_batch.to(self.device)

                with torch.no_grad():
                    scores = self.model.predict_similarity(original_batch, augmented_batch).cpu().numpy().flatten()
                    all_scores.extend(scores)
                    all_indices.extend(indices)

        if self.return_scores:
            return all_scores, all_indices

        for idx, score in zip(all_indices, all_scores):
            aug_name = list(self.augmentations.keys())[idx]
            total_predictions[aug_name] += 1
            if score > self.threshold:
                correct_predictions[aug_name] += 1

        self.augmentation_recalls = {
            aug_name: correct / total_predictions[aug_name] if total_predictions[aug_name] > 0 else 0.0
            for aug_name, correct in correct_predictions.items()
        }

        if self.checkpoint_name not in self.metrics_data:
            self.metrics_data[self.checkpoint_name] = {}

        if 'augmentation_recalls' in self.metrics_data[self.checkpoint_name]:
            self.metrics_data[self.checkpoint_name]['augmentation_recalls'].update(self.augmentation_recalls)
        else:
            self.metrics_data[self.checkpoint_name]['augmentation_recalls'] = self.augmentation_recalls

        MetricsManager.save_metrics(self.metrics_data, self.output_metrics_path)
        print(f"Metrics saved to {self.output_metrics_path}")


class DissmatchBenchmark:
    """Class to perform dissimilarity benchmarking."""

    def __init__(self, config_path, batch_size=8, use_random_candidates=False, num_candidates=40, return_scores=False):
        config = load_config(config_path)
        self.dataset_path = config['dataset_path']
        self.output_metrics_path = config['output_metrics_path']
        self.model_config_path = config['model_config_path']
        self.threshold = config['threshold']
        self.return_scores = return_scores

        model_config = load_config(self.model_config_path)
        self.checkpoint_path = model_config['model']['initialization_weights_path']
        self.device = model_config['training']['device']

        self.model = get_siamnet(model_config)
        self.preprocess = self.model.get_preprocessing()
        self.model.eval()

        if return_scores:
            self.metrics_data = None
        else:
            self.metrics_data = MetricsManager.load_or_create_metrics_json(self.output_metrics_path)
            self.checkpoint_name = os.path.basename(self.checkpoint_path)
            if self.checkpoint_name not in self.metrics_data:
                self.metrics_data[self.checkpoint_name] = {}

        self.dataset = RandomCandidateImagePairDataset(self.dataset_path, self.preprocess, num_candidates) if use_random_candidates else ImagePairDataset(self.dataset_path, self.preprocess)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def run(self):
        """Run the benchmark and calculate false positive rate."""
        all_scores = []

        for batch in tqdm(self.dataloader, desc="Processing image pairs"):
            img1_batch, img2_batch = batch
            img1_batch = img1_batch.to(self.device)
            img2_batch = img2_batch.to(self.device)

            with torch.no_grad():
                scores = self.model.predict_similarity(img1_batch, img2_batch).cpu().numpy().flatten()
                all_scores.extend(scores)

        if self.return_scores:
            return all_scores

        false_positives = sum(score > self.threshold for score in all_scores)
        total_pairs = len(self.dataset)
        self.fpr = false_positives / total_pairs if total_pairs > 0 else 0.0

        self.metrics_data[self.checkpoint_name]['false_positive_rate'] = self.fpr
        MetricsManager.save_metrics(self.metrics_data, self.output_metrics_path)
        print(f"Metrics saved to {self.output_metrics_path}")


class RandomCandidateImagePairDataset(Dataset):
    """Dataset class for random candidate image pairs."""

    def __init__(self, dataset_path, preprocess=None, num_candidates=5, seed=42):
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.num_candidates = num_candidates
        self.seed = seed
        self.image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg'))]
        self.image_pairs = self.generate_image_pairs()
        self.imageprocessor = ImageProcessor()

    def generate_image_pairs(self):
        """Generate unique image pairs."""
        image_pairs = set()
        num_images = len(self.image_files)

        for i, img1_path in enumerate(self.image_files):
            random.seed(self.seed + i)
            candidates = [idx for idx in range(num_images) if idx != i]
            random.shuffle(candidates)
            selected_candidates = candidates[:self.num_candidates]

            for candidate_idx in selected_candidates:
                img2_path = self.image_files[candidate_idx]
                pair = tuple(sorted((img1_path, img2_path)))
                image_pairs.add(pair)

        return list(image_pairs)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = self.imageprocessor.load_and_preprocess_image(img1_path, self.preprocess)
        img2 = self.imageprocessor.load_and_preprocess_image(img2_path, self.preprocess)
        return img1, img2


class AugmentedImageDataset(Dataset):
    """Dataset class for augmented images."""

    def __init__(self, original_image, augmentations, preprocess):
        self.original_image = original_image
        self.augmentations = augmentations
        self.preprocess = preprocess
        self.augmented_images = self.apply_augmentations()

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        augmented_image = self.augmented_images[idx].convert('RGB')
        original_image = self.original_image.convert('RGB').copy()
        original_image = self.preprocess(original_image)
        augmented_image = self.preprocess(augmented_image)
        return original_image, augmented_image, idx

    def apply_augmentations(self):
        """Apply augmentations to the original image."""
        return [augmentation(self.original_image) for _, augmentation in self.augmentations.items()]


class ImagePairDataset(Dataset):
    """Dataset class for image pairs."""

    def __init__(self, dataset_path, preprocess=None):
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg'))]
        self.image_pairs = list(itertools.combinations(self.image_files, 2))
        self.imageprocessor = ImageProcessor()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = self.imageprocessor.load_and_preprocess_image(img1_path, self.preprocess)
        img2 = self.imageprocessor.load_and_preprocess_image(img2_path, self.preprocess)
        return img1, img2


def calculate_metrics_with_thresholds(config_path, threshold_start=0.0, threshold_step=0.05):
    """Calculate metrics for different thresholds and save results to a single JSON file."""
    config = load_config(config_path)
    output_metrics_path = config['output_metrics_path']
    checkpoint_path = config['checkpoint_path']

    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)

    benchmark = RecallBenchmark(config_path, return_scores=True)
    dissmatch_benchmark = DissmatchBenchmark(config_path, return_scores=True)

    benchmark_scores, benchmark_indices = benchmark.run()
    dissmatch_scores = dissmatch_benchmark.run()

    thresholds = [threshold_start + i * threshold_step for i in range(int((1 - threshold_start) / threshold_step) + 1)]
    all_results = {}
    model_name = os.path.basename(checkpoint_path)

    for threshold in thresholds:
        correct_predictions = {aug: 0 for aug in benchmark.augmentations}
        total_predictions = {aug: 0 for aug in benchmark.augmentations}

        for score, idx in zip(benchmark_scores, benchmark_indices):
            aug_name = list(benchmark.augmentations.keys())[idx]
            total_predictions[aug_name] += 1
            if score > threshold:
                correct_predictions[aug_name] += 1

        benchmark_recalls = {
            aug_name: correct / total_predictions[aug_name] if total_predictions[aug_name] > 0 else 0.0
            for aug_name, correct in correct_predictions.items()
        }

        false_positives = sum(score > threshold for score in dissmatch_scores)
        total_pairs = len(dissmatch_scores)
        fpr = false_positives / total_pairs if total_pairs > 0 else 0.0

        all_results[f'threshold_{threshold}'] = {
            'benchmark_recalls': benchmark_recalls,
            'dissmatch_fpr': fpr
        }

    existing_metrics = MetricsManager.load_or_create_metrics_json(output_metrics_path)
    existing_metrics[model_name] = all_results
    MetricsManager.save_metrics(existing_metrics, output_metrics_path)

    print(f"All benchmark results saved to {output_metrics_path}")