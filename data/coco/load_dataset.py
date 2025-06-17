#!/usr/bin/env python3

import os
import requests
import zipfile
from tqdm import tqdm

# Ссылки на файлы
urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/zips/unlabeled2017.zip"
]

# Локальные имена файлов
local_filenames = [
    "train2017.zip",
    "val2017.zip",
    "test2017.zip",
    "unlabeled2017.zip"
]

def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB

    with open(local_filename, 'wb') as file, tqdm(
        desc=local_filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Извлекаем содержимое в указанную директорию
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def main():
    # Получаем путь к текущему скрипту
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Директория для сохранения данных
    data_dir = os.path.join(script_dir, "coco_images")
    os.makedirs(data_dir, exist_ok=True)

    # Загрузка и распаковка файлов
    for url, local_filename in zip(urls, local_filenames):
        zip_path = os.path.join(data_dir, local_filename)
        extract_to = data_dir  # Извлекаем непосредственно в data_dir

        # Загрузка файла
        if not os.path.exists(zip_path):
            print(f"Downloading {local_filename}...")
            download_file(url, zip_path)
            print(f"Downloaded {local_filename}")
        else:
            print(f"{local_filename} already downloaded.")

        # Распаковка файла
        if not os.path.exists(os.path.join(data_dir, local_filename.replace('.zip', ''))):
            print(f"Extracting {local_filename}...")
            unzip_file(zip_path, extract_to)
            print(f"Extracted {local_filename}")

            # Удаление ZIP-файла после распаковки
            os.remove(zip_path)
            print(f"Deleted {local_filename}")
        else:
            print(f"{local_filename} already extracted.")

if __name__ == "__main__":
    main()