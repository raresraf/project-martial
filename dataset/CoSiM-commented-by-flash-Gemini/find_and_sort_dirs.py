import os
import subprocess

def get_directory_size(path):
    # Use 'du -sk' to get size in KB, then convert to bytes
    try:
        output = subprocess.check_output(['du', '-sk', path]).decode('utf-8')
        size_kb = int(output.split('	')[0])
        return size_kb * 1024 # Convert KB to bytes
    except Exception:
        return 0

def find_smallest_directories(dir_list_file, num_directories=50):
    sizes = []
    with open(dir_list_file, 'r') as f:
        for line in f:
            dir_path = line.strip()
            if dir_path and os.path.isdir(dir_path):
                size = get_directory_size(dir_path)
                sizes.append((size, dir_path))

    sizes.sort() # Sorts by size (first element of tuple)

    return [d for s, d in sizes[:num_directories]]

if __name__ == "__main__":
    output_file = "unprocessed_dirs.txt"
    with open(output_file, 'w') as outfile:
        # Re-run the find command to capture the full output
        subprocess.run(['find', '.', '-maxdepth', '1', '-type', 'd', '-print0'], stdout=subprocess.PIPE, text=True)
        # Filter in Python to avoid issues with large output and xargs
        find_output = subprocess.check_output(['find', '.', '-maxdepth', '1', '-type', 'd']).decode('utf-8').splitlines()
        for dir_path in find_output:
            dir_path = dir_path.strip()
            if dir_path and os.path.isdir(dir_path) and not os.path.exists(os.path.join(dir_path, '.checkpoint')):
                outfile.write(dir_path + '\n')

    smallest_dirs = find_smallest_directories(output_file, 50)
    
    with open("smallest_50_dirs.txt", 'w') as f:
        for d in smallest_dirs:
            f.write(d + '\n')

    print("Smallest 50 directories (without .checkpoint) written to smallest_50_dirs.txt")
