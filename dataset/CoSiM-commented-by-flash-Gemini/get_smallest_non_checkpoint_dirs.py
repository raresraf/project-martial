
import os
import sys

def get_dir_size(path):
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Check if it's a symbolic link to avoid issues and cycles
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except FileNotFoundError:
        # Handle cases where a directory might be removed during processing
        pass
    except Exception as e:
        sys.stderr.write(f"Error calculating size for {path}: {e}\n")
    return total_size

def find_smallest_non_checkpoint_dirs(base_dir, num_results=50):
    all_dirs_file = os.path.join(base_dir, "all_uuid_dirs.txt")
    if not os.path.exists(all_dirs_file):
        sys.stderr.write(f"Error: {all_dirs_file} not found.\n")
        return []

    non_checkpoint_dirs = []
    
    with open(all_dirs_file, "r") as f:
        for line in f:
            dir_relative_path = line.strip()
            dir_absolute_path = os.path.join(base_dir, dir_relative_path)
            
            if not os.path.isdir(dir_absolute_path):

                continue

            checkpoint_path = os.path.join(dir_absolute_path, ".checkpoint")
            
            if not os.path.exists(checkpoint_path):
                size = get_dir_size(dir_absolute_path)
                non_checkpoint_dirs.append((dir_absolute_path, size))

    # Sort by size (second element of the tuple)
    non_checkpoint_dirs.sort(key=lambda x: x[1])

    return [d[0] for d in non_checkpoint_dirs[:num_results]]

if __name__ == "__main__":
    current_working_directory = os.getcwd()
    smallest_dirs = find_smallest_non_checkpoint_dirs(current_working_directory, 50)
    
    for d in smallest_dirs:
        print(d)
