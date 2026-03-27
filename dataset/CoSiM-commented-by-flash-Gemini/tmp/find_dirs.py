import os

def get_directory_size(path):
    total_size = 0
    if not os.path.exists(path):
        return 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Check if it's a file and not a symlink to avoid errors
            if os.path.isfile(fp) and not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    # Handle cases where file might be inaccessible
                    continue
    return total_size

base_dir = '.' # Current working directory
dirs_to_process = []

# List all subdirectories
for d in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, d)
    if os.path.isdir(dir_path):
        checkpoint_path = os.path.join(dir_path, '.checkpoint')
        if not os.path.exists(checkpoint_path):
            size = get_directory_size(dir_path)
            dirs_to_process.append((size, d))

# Sort by size and get the smallest 100
dirs_to_process.sort()

for size, d in dirs_to_process[:100]:
    print(d)
