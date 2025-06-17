import os
import requests
import zipfile
import json
import shutil


def download_dataset(url: str, local_filename: str) -> None:
    """
    Download a dataset from the given URL and save it locally.

    Args:
        url (str): URL of the dataset to download.
        local_filename (str): Local filename to save the dataset.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': '*/*'
    }
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Dataset {local_filename} successfully downloaded.")


def delete_png_masks(dataset_path: str) -> None:
    """
    Delete all PNG files in the dataset directory.

    Args:
        dataset_path (str): Path to the dataset directory.
    """
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)


def delete_sketch_files(dataset_path: str) -> None:
    """
    Delete files named 'Sketch.jpg' from each category directory.

    Args:
        dataset_path (str): Path to the dataset directory.
    """
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            sketch_file_path = os.path.join(category_path, 'Sketch.jpg')
            if os.path.exists(sketch_file_path):
                print(f"Deleting: {sketch_file_path}")
                os.remove(sketch_file_path)


def extract_archive(archive_path: str, extract_to: str = '.') -> None:
    """
    Extract an archive file (zip, tar, rar) to the specified directory.

    Args:
        archive_path (str): Path to the archive file.
        extract_to (str): Directory to extract the archive to.
    """
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            # Move contents of the extracted folder to the extract_to directory
            for item in os.listdir(os.path.join(extract_to, os.path.splitext(os.path.basename(archive_path))[0])):
                s = os.path.join(extract_to, os.path.splitext(os.path.basename(archive_path))[0], item)
                d = os.path.join(extract_to, item)
                if os.path.isdir(s):
                    shutil.move(s, d)
                else:
                    shutil.move(s, d)
            # Remove the now-empty extracted folder
            os.rmdir(os.path.join(extract_to, os.path.splitext(os.path.basename(archive_path))[0]))
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    elif rarfile.is_rarfile(archive_path):
        with rarfile.RarFile(archive_path, 'r') as rar_ref:
            rar_ref.extractall(extract_to)
    else:
        print(f"Unsupported archive format: {archive_path}")


def create_json_dataset(root_dir: str, output_json: str) -> None:
    """
    Create a JSON dataset from the directory structure.

    Args:
        root_dir (str): Root directory containing class directories.
        output_json (str): Path to the output JSON file.
    """
    dataset = {}

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            src_dir = os.path.join(class_dir, 'Src')
            if os.path.isdir(src_dir):
                images = [os.path.abspath(os.path.join(src_dir, img)) for img
                          in os.listdir(src_dir)
                          if img.lower().endswith(('.png', '.jpg'))]
                dataset[class_name] = images

    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=4)


def main():
    url = "https://mmcheng.net/mftp/Data/THUR15000.zip"
    local_filename = "THUR15000.zip"
    extract_to = "THUR15000"
    output_json = "thur_dataset.json"

    # Step 1: Download the dataset
    download_dataset(url, local_filename)

    # Step 2: Extract the archive
    extract_archive(local_filename, extract_to)

    # Step 3: Delete PNG masks
    delete_png_masks(extract_to)

    # Step 4: Delete 'Sketch.jpg' files from each category directory
    delete_sketch_files(extract_to)

    # Step 5: Create JSON dataset
    create_json_dataset(extract_to, output_json)

    # Step 6: Delete the downloaded archive
    os.remove(local_filename)
    print(f"Archive {local_filename} successfully deleted.")


if __name__ == "__main__":
    main()
