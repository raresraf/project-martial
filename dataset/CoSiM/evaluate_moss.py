
import json
import os
from pathlib import Path
import re
import argparse
from tqdm import tqdm
import mosspy
import logging

# Configure logging for mosspy
logging.basicConfig(level=logging.INFO)

# Mapping of index ranges to languages
LANGUAGE_MAP = {
    "similar": {
        (0, 99): "go",
        (100, 199): "typescript",
        (200, 299): "c",
        (300, 399): "java",
        (400, 499): "rust",
        (500, 599): "c",
        (600, 699): "c", # Cuda is not directly supported, using 'c'
        (700, 1199): "python",
        (1200, 1999): "cc",
    },
    "notsimilar": {
        (0, 99): "go",
        (100, 199): "typescript",
        (200, 299): "c",
        (300, 399): "java",
        (400, 499): "rust",
        (500, 1199): "c", # Mixed, defaulting to C
        (1200, 1999): "cc",
    }
}

def get_language(index, dataset_type):
    """
    Determines the MOSS language based on the pair index and dataset type.
    """
    for (start, end), lang in LANGUAGE_MAP[dataset_type].items():
        if start <= index <= end:
            return lang
    return "text" # Default fallback

def add_files_from_directory(moss, directory_path):
    """
    Recursively adds all files from a directory to the MOSS object, excluding METADATA.json.
    """
    if directory_path.exists():
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file != "METADATA.json":
                    file_path = Path(root) / file
                    if file_path.is_file():
                        moss.addFile(str(file_path))

def load_checkpoint(checkpoint_file):
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_file, data):
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MOSS on the CoSiM dataset.")
    parser.add_argument("moss_user_id", help="Your MOSS user ID.")
    args = parser.parse_args()

    # Create a directory to store the reports
    reports_dir = Path("moss_reports")
    reports_dir.mkdir(exist_ok=True)
    
    checkpoint_file = Path("moss_checkpoint.json")
    results = load_checkpoint(checkpoint_file)

    with open("similar/simdataset.json", "r") as f:
        similar_data = json.load(f)
    with open("notsimilar/notsimdataset.json", "r") as f:
        notsimilar_data = json.load(f)

    print(f"Loaded {len(similar_data)} similar pairs.")
    print(f"Loaded {len(notsimilar_data)} not-similar pairs.")
    
    submission_count = 0

    with tqdm(total=len(similar_data) + len(notsimilar_data), desc="Processing pairs") as pbar:
        for dataset_type, dataset in [("similar", similar_data), ("notsimilar", notsimilar_data)]:
            for key, (uuid1, uuid2) in dataset.items():
                reportkey = key
                if dataset_type == "notsimilar":
                    reportkey = str(int(key) + 2000)
                if reportkey in results:
                    pbar.update(1)
                    continue

                if (reports_dir / f"{reportkey}_report.html").exists():
                    pbar.update(1)
                    continue

                if submission_count >= 90:
                    print("Reached submission limit. Saving checkpoint and exiting.")
                    save_checkpoint(checkpoint_file, results)
                    return
                
                index = int(key)
                lang = get_language(index, dataset_type)
                
                pbar.set_description(f"Processing {dataset_type} pair {key} ({lang})")
                
                moss = mosspy.Moss(args.moss_user_id, lang)
                
                add_files_from_directory(moss, Path("raw") / uuid1)
                add_files_from_directory(moss, Path("raw") / uuid2)

                try:
                    url = moss.send(lambda file_path, display_name: pbar.set_description(f"Uploading {display_name}"))
                    
                    report_path = reports_dir / f"{reportkey}_report.html"
                    moss.saveWebPage(url, str(report_path))
                    
                    results[reportkey] = {
                        "url": url, 
                        "similar": dataset_type == "similar", 
                        "language": lang,
                        "report_path": str(report_path),
                    }
                    
                    submission_count += 1
                    save_checkpoint(checkpoint_file, results)

                except Exception as e:
                    print(f"Error sending to MOSS for pair {key}: {e}")
                
                pbar.update(1)

    print("MOSS evaluation complete. Results saved to moss_results.json")
    save_checkpoint(checkpoint_file, results)

if __name__ == "__main__":
    main()
