#!/usr/bin/env python3
import argparse
from src.dataset import get_coco_dataloaders, load_config
from src.model import get_siamnet
from src.train import train_model

def main(data_dir):
    config_path = "configs/train_config.yaml"
    train_config = load_config(config_path)

    net = get_siamnet(train_config)
    preprocessor = net.get_preprocessing()
    dataloaders = get_coco_dataloaders(data_dir,
                                       preprocessor,
                                       batch_size=32,
                                       num_workers=4,
                                       val_size=0.1,
                                       random_seed=42)
    train_model(net, dataloaders['train'], dataloaders['test'], train_config, resume=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Siamese Network.')
    parser.add_argument('--data_dir', type=str, default="/home/jovyan/nkiselev/ddorin/project/Pairwise-Image-Matching/data", help='Full path to the dataset directory')
    args = parser.parse_args()

    main(args.data_dir)