import os
import subprocess

def get_directory_size(path):
    try:
        # Using du -sk to get size in KB, then convert to bytes
        output = subprocess.check_output(['du', '-sk', path]).decode('utf-8')
        size_kb = int(output.split('	')[0])
        return size_kb * 1024  # Convert KB to bytes
    except Exception:
        # Return 0 if there's an error (e.g., directory not found)
        return 0

def sort_directories_by_size(input_file, output_file, num_directories=50):
    dir_sizes = []
    with open(input_file, 'r') as f:
        for line in f:
            dir_path = line.strip()
            if dir_path:
                size = get_directory_size(dir_path)
                dir_sizes.append((size, dir_path))
    
    # Sort by size in ascending order
    dir_sizes.sort()

    with open(output_file, 'w') as f:
        for i, (size, path) in enumerate(dir_sizes):
            if i >= num_directories:
                break
            f.write(path + '\n')

    print(f"Smallest {num_directories} directories written to {output_file}")

if __name__ == "__main__":
    input_list_file = "unprocessed_dirs.txt"
    output_list_file = "smallest_50_dirs.txt"
    sort_directories_by_size(input_list_file, output_list_file, 50)
