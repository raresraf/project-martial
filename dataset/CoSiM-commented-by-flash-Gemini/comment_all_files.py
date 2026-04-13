
import os
import sys

# Function to calculate directory size
def get_dir_size(path):
    total_size = 0
    if not os.path.exists(path):
        return total_size
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Avoid re-calculating size of checkpoint files or symlinks
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except FileNotFoundError:
                    # Handle cases where file might be deleted during walk
                    continue
    return total_size

# Step 1: Find all UUID-like directories
def find_uuid_dirs(base_path="."):
    uuid_dirs = []
    for entry in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, entry)):
            # UUIDs are typically 32 hex characters + 4 hyphens
            if len(entry) == 36 and all(c.isalnum() or c == '-' for c in entry):
                uuid_dirs.append(entry)
    return uuid_dirs

# Step 2: Filter out processed directories and calculate sizes
def get_smallest_unprocessed_dirs(uuid_dirs, num_to_select=50):
    unprocessed_dirs_with_size = []
    for dir_name in uuid_dirs:
        checkpoint_path = os.path.join(dir_name, ".checkpoint")
        if not os.path.exists(checkpoint_path):
            size = get_dir_size(dir_name)
            unprocessed_dirs_with_size.append((dir_name, size))

    unprocessed_dirs_with_size.sort(key=lambda x: x[1])
    return [d[0] for d in unprocessed_dirs_with_size[:num_to_select]]

def main():
    base_path = "."
    all_dirs = find_uuid_dirs(base_path)
    print(f"Found {len(all_dirs)} UUID directories.", file=sys.stderr)

    smallest_unprocessed = get_smallest_unprocessed_dirs(all_dirs, num_to_select=50)
    print(f"Identified {len(smallest_unprocessed)} smallest unprocessed directories.", file=sys.stderr)

    # Use a separate mechanism to write the file, or print to stdout and redirect
    # For now, let's try printing to stdout and I'll capture it manually.
    for dir_name in smallest_unprocessed:
        print(dir_name)

if __name__ == "__main__":
    main()
