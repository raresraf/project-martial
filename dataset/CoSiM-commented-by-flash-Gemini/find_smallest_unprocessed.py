import os
import sys

def main():
    unprocessed_dirs = []
    
    # Read the sorted list of directories and their sizes
    with open('all_dirs_with_sizes.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('	')
            if len(parts) < 2:
                continue
            
            size_kb = int(parts[0])
            dir_path = parts[1]
            
            # Construct the path to the checkpoint file
            checkpoint_file = os.path.join(dir_path, '.checkpoint')
            
            # Check if the checkpoint file exists
            if not os.path.exists(checkpoint_file):
                unprocessed_dirs.append((size_kb, dir_path))
                
    # Sort by size (already sorted, but re-sort for safety)
    unprocessed_dirs.sort(key=lambda x: x[0])
    
    # Print the top 100 smallest unprocessed directories
    for i, (size, path) in enumerate(unprocessed_dirs):
        if i >= 100:
            break
        print(f"{size}	{path}")

if __name__ == "__main__":
    main()
