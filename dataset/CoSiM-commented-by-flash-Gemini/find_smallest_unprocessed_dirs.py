import os

def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def find_unprocessed_directories(base_path):
    unprocessed_dirs = []
    for entry in os.listdir(base_path):
        dir_path = os.path.join(base_path, entry)
        if os.path.isdir(dir_path) and entry != ".checkpoint":
            checkpoint_file = os.path.join(dir_path, ".checkpoint")
            if not os.path.exists(checkpoint_file):
                size = get_directory_size(dir_path)
                unprocessed_dirs.append((dir_path, size))
    return unprocessed_dirs

if __name__ == "__main__":
    base_directory = "."
    unprocessed_directories = find_unprocessed_directories(base_directory)
    
    # Sort by size (smallest first)
    unprocessed_directories.sort(key=lambda x: x[1])
    
    # Select the smallest 50
    smallest_50_unprocessed = unprocessed_directories[:50]
    
    for dir_path, size in smallest_50_unprocessed:
        print(dir_path)

