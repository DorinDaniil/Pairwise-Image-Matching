#!/usr/bin/env python3

from src.dataset import get_coco_dataloaders, load_config
from src.model import get_siamnet
from src.train import train_model


def main():
    root = "/home/jovyan/nkiselev/ddorin/project/Pairwise-Image-Matching/data/coco/coco_images"
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

    train_model(model, dataloaders['train'], dataloaders['test'], train_config, resume=False)


if __name__ == "__main__":
    main()