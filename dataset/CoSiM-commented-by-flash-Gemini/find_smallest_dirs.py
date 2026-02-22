import os
import subprocess

def find_smallest_dirs_without_checkpoint(root_dir="."):
    """
    Finds the 50 smallest directories within root_dir that do not contain a .checkpoint file.
    """
    all_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    dirs_to_process = []
    for d in all_dirs:
        dir_path = os.path.join(root_dir, d)
        checkpoint_path = os.path.join(dir_path, ".checkpoint")
        if not os.path.exists(checkpoint_path):
            try:
                # Get disk usage in blocks, then convert to bytes or kilobytes for sorting
                # using -k option with du to get size in 1K blocks, then convert to int
                size_output = subprocess.check_output(['du', '-sk', dir_path]).decode().split('\t')[0]
                size_kb = int(size_output)
                dirs_to_process.append((d, size_kb))
            except Exception as e:
                # print(f"Error getting size for {dir_path}: {e}") # Commented out to reduce output noise
                continue
    
    # Sort by size (smallest first)
    dirs_to_process.sort(key=lambda x: x[1])
    
    # Return the names of the top 50 smallest
    return [d[0] for d in dirs_to_process[:50]]

if __name__ == "__main__":
    smallest_dirs = find_smallest_dirs_without_checkpoint()
    temp_dir = "/Users/raresraf/.gemini/tmp/8f69d9d75cf2f5a11d7fa33d7113d1ad9875dd71f0c57dda23b3a6941c51ce07"
    output_file_path = os.path.join(temp_dir, "smallest_dirs.txt")

    if smallest_dirs:
        with open(output_file_path, "w") as f:
            for directory in smallest_dirs:
                f.write(directory + "\n")
        print(f"Smallest 50 directories without a .checkpoint file written to: {output_file_path}")
    else:
        with open(output_file_path, "w") as f:
            f.write("No directories found to process or all have .checkpoint files.\n")
        print(f"No directories found to process or all have .checkpoint files. Status written to: {output_file_path}")