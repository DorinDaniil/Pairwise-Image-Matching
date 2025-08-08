#!/usr/bin/env python3
import argparse
from src.benchmarks import fpr_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model for false positive rate estimation.')
    parser.add_argument('--config_path', type=str, default="configs/fpr_benchmark_config.yaml")
    args = parser.parse_args()
    fpr_benchmark(args.config_path)