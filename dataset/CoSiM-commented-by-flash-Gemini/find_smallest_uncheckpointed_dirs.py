
import os
import subprocess

def get_dir_size(path):
    """Calculates the disk usage of a directory in bytes."""
    try:
        # Use 'du -s' for efficiency, output is in blocks, usually 1KB per block on macOS/Linux
        # We assume 1KB blocks for simplicity, but a more robust solution might check block size.
        result = subprocess.run(['du', '-s', path], capture_output=True, text=True, check=True)
        size_in_blocks = int(result.stdout.split()[0])
        return size_in_blocks * 1024 # Convert to bytes (assuming 1KB blocks)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting size for {path}: {e}")
        return float('inf') # Treat errors as very large to sort them last

def find_smallest_uncheckpointed_dirs(root_dir=".", limit=100):
    """
    Finds the smallest directories that do not contain a .checkpoint file.

    Args:
        root_dir (str): The root directory to search within.
        limit (int): The maximum number of smallest directories to return.

    Returns:
        list: A list of paths to the smallest uncheckpointed directories.
    """
    all_subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    uncheckpointed_dirs = []
    for d in all_subdirs:
        checkpoint_file = os.path.join(d, '.checkpoint')
        if not os.path.exists(checkpoint_file):
            uncheckpointed_dirs.append(d)

    dir_sizes = []
    for d in uncheckpointed_dirs:
        size = get_dir_size(d)
        dir_sizes.append((size, d))

    # Sort by size (smallest first)
    dir_sizes.sort(key=lambda x: x[0])

    # Extract paths of the smallest directories, up to the limit
    smallest_paths = [path for size, path in dir_sizes[:limit]]
    return smallest_paths

if __name__ == "__main__":
    # Remove the './' prefix if it exists to clean up paths for presentation
    current_working_dir = os.getcwd() # Get the absolute path

    smallest_dirs = find_smallest_uncheckpointed_dirs(current_working_dir)
    
    for d in smallest_dirs:
        print(d)
