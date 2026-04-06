
import os
import subprocess

def get_dir_size(path):
    try:
        # Use 'du -s' to get the size of the directory in kilobytes
        result = subprocess.run(['du', '-s', path], capture_output=True, text=True, check=True)
        size_kb = int(result.stdout.split()[0])
        return size_kb
    except Exception as e:
        print(f"Error getting size for {path}: {e}")
        return -1 # Indicate error

def main():
    all_directories_file = 'all_directories.txt'
    output_file = 'smallest_50_dirs_to_process.txt'
    
    unprocessed_dirs = []
    
    with open(all_directories_file, 'r') as f:
        directories = [d.strip() for d in f.readlines()]
    
    for directory in directories:
        checkpoint_path = os.path.join(directory, '.checkpoint')
        if not os.path.exists(checkpoint_path):
            size = get_dir_size(directory)
            if size != -1:
                unprocessed_dirs.append((size, directory))
                
    # Sort by size (smallest first)
    unprocessed_dirs.sort(key=lambda x: x[0])
    
    # Get the top 50
    smallest_50 = unprocessed_dirs[:50]
    
    with open(output_file, 'w') as f:
        for size, directory in smallest_50:
            f.write(f"{directory}\n")
            
    print(f"Successfully identified and saved the 50 smallest unprocessed directories to {output_file}")

if __name__ == "__main__":
    main()
