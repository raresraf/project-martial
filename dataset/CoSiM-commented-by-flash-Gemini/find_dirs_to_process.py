
import os
import sys
import subprocess

def get_directory_size(path):
    try:
        output = subprocess.check_output(['du', '-sk', path]).decode('utf-8')
        size_kb = int(output.split('\t')[0])
        return size_kb # Return size in KB
    except Exception:
        return 0

def find_dirs_to_process(base_dir):
    unprocessed_dirs = []
    
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
                    dir_size = get_directory_size(dir_path)
                    unprocessed_dirs.append((dir_path, dir_size))
                except OSError as e:
                    print(f"Error processing directory {dir_path}: {e}", file=sys.stderr)
    
    # Sort by directory size (smallest first)
    unprocessed_dirs.sort(key=lambda x: x[1])
    
    return [d[0] for d in unprocessed_dirs]

if __name__ == "__main__":
    base_directory = "." 
    if len(sys.argv) > 1:
        base_directory = sys.argv[1]

    dirs = find_dirs_to_process(base_directory)
    
    for i, d in enumerate(dirs[:50]):
        print(d)

