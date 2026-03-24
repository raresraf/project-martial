import os
import subprocess

base_path = "/Users/trk/project-martial/dataset/CoSiM-commented-by-flash-Gemini"
directories_to_process = []

# Read all subdirectories from the temporary file
with open("/Users/trk/.gemini/tmp/cosim-commented-by-flash-gemini/all_subdirectories.txt", "r") as f:
    all_directories = [line.strip() for line in f if line.strip()]

for directory in all_directories:
    if not directory:
        continue

    checkpoint_file = os.path.join(directory, ".checkpoint")
    
    if not os.path.exists(checkpoint_file):
        # Get directory size
        du_command = f"du -s {directory}"
        du_result = subprocess.run(du_command, shell=True, capture_output=True, text=True)
        
        if du_result.returncode == 0:
            size_str = du_result.stdout.split('	')[0]
            try:
                size_kb = int(size_str) # du -s outputs size in kilobytes
                directories_to_process.append((directory, size_kb))
            except ValueError:
                # Handle cases where size might not be a clean integer
                print(f"Warning: Could not parse size for {directory}: {size_str}")
        else:
            print(f"Error getting size for {directory}: {du_result.stderr}")

# Sort by size
directories_to_process.sort(key=lambda x: x[1])

# Get the smallest 50
smallest_50_dirs = directories_to_process[:50]

print("Smallest 50 directories without a .checkpoint file:")
for directory, size in smallest_50_dirs:
    print(str(size) + "KB	" + directory) # Changed this line for print

# Optionally, write to a file
with open(os.path.join(base_path, "smallest_50_dirs_to_process.txt"), "w") as f:
    for directory, size in smallest_50_dirs:
        f.write(str(size) + "KB	" + directory + "
") # Changed this line for file write
