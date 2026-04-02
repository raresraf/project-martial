
import os
import sys

def count_files_in_directory(path):
    count = 0
    for root, dirs, files in os.walk(path):
        count += len(files)
    return count

def find_dirs_to_process(base_dir):
    unprocessed_dirs = []
    
    # List all entries in the base directory
    try:
        entries = os.listdir(base_dir)
    except OSError as e:
        print(f"Error listing directory {base_dir}: {e}", file=sys.stderr)
        return []

    for entry in entries:
        dir_path = os.path.join(base_dir, entry)
        if os.path.isdir(dir_path):
            checkpoint_file = os.path.join(dir_path, ".checkpoint")
            if not os.path.exists(checkpoint_file):
                try:
                    file_count = count_files_in_directory(dir_path)
                    unprocessed_dirs.append((dir_path, file_count))
                except OSError as e:
                    print(f"Error processing directory {dir_path}: {e}", file=sys.stderr)
    
    # Sort by file count (size)
    unprocessed_dirs.sort(key=lambda x: x[1])
    
    return [d[0] for d in unprocessed_dirs]

if __name__ == "__main__":
    base_directory = "." 
    if len(sys.argv) > 1:
        base_directory = sys.argv[1]

    dirs = find_dirs_to_process(base_directory)
    
    # Print the top 50 directories or fewer if not enough
    for i, d in enumerate(dirs[:50]):
        print(d)

