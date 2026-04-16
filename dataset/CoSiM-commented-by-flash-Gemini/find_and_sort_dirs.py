
import os
import re
import subprocess

def get_dir_size(path):
    """
    Calculates the size of a directory using 'du -s'.
    Returns size in bytes or -1 if an error occurs.
    """
    try:
        # Use 'du -s' to get the summarized disk usage of the directory
        # The output is typically like "123456  /path/to/dir"
        result = subprocess.run(['du', '-s', path], capture_output=True, text=True, check=True)
        size_kb = int(result.stdout.split()[0])
        return size_kb * 1024 # Convert KB to bytes
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting size for {path}: {e}")
        return -1

def find_unprocessed_dirs(root_dir):
    """
    Finds directories that match a UUID pattern and do not contain a .checkpoint file.
    Returns a list of tuples: (directory_path, size_in_bytes).
    """
    unprocessed_dirs = []
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

    # Iterate over all entries in the root directory
    with os.scandir(root_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                dir_name = entry.name
                # Check if the directory name matches the UUID pattern
                if uuid_pattern.match(dir_name):
                    dir_path = entry.path
                    checkpoint_file = os.path.join(dir_path, '.checkpoint')

                    # Check if .checkpoint file exists
                    if not os.path.exists(checkpoint_file):
                        size = get_dir_size(dir_path)
                        if size != -1:
                            unprocessed_dirs.append((dir_path, size))
    return unprocessed_dirs

if __name__ == '__main__':
    root_directory = './'  # Current working directory
    
    # Find all unprocessed directories with their sizes
    unprocessed_directories = find_unprocessed_dirs(root_directory)
    
    # Sort them by size (smallest first)
    unprocessed_directories.sort(key=lambda x: x[1])
    
    # Select the top 50
    top_50_smallest = unprocessed_directories[:50]

    # Print the paths of the selected directories
    for dir_path, size in top_50_smallest:
        print(dir_path)

