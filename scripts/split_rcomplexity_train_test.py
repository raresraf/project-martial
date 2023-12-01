import os
import shutil
import random

def split_files(input_directory, output_directory1, output_directory2):
    # Get a list of all files in the input directory and its subdirectories
    all_files = []
    for root, _, files in os.walk(input_directory):            
        all_files.extend([os.path.join(root, file) for file in files if file.endswith('PROCESSED.RAF')])

    random.shuffle(all_files)  # Shuffle the list to randomize file order

    # Calculate the midpoint to split the files
    midpoint = len(all_files) // 2

    # Divide the list into two parts
    files1 = all_files[:midpoint]
    files2 = all_files[midpoint:]

    # Ensure output directories exist, create them if not
    os.makedirs(output_directory1, exist_ok=True)
    os.makedirs(output_directory2, exist_ok=True)

    # Move files to the output directories
    for file_path in files1:
        file_name = os.path.relpath(file_path, input_directory)
        destination_path = os.path.join(output_directory1, file_name)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(file_path, destination_path)

    for file_path in files2:
        file_name = os.path.relpath(file_path, input_directory)
        destination_path = os.path.join(output_directory2, file_name)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(file_path, destination_path)

if __name__ == "__main__":
    # Replace these paths with your actual directory paths
    input_directory = "/Users/raresraf/code/TheOutputsCodeforces/processed/atomic_perf/"
    output_directory1 = "/Users/raresraf/code/TheOutputsCodeforces/splitted/train/atomic_perf/"
    output_directory2 = "/Users/raresraf/code/TheOutputsCodeforces/splitted/test/atomic_perf/"

    split_files(input_directory, output_directory1, output_directory2)
