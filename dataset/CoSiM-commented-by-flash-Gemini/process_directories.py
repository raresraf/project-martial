import os
import subprocess

def get_directory_sizes(directory_list):
    dir_sizes = {}
    for d in directory_list:
        if d == '.' or not d:  # Skip current directory or empty strings
            continue
        try:
            # Use 'du -s' for summarized disk usage of a directory
            # and strip the directory name from the output
            size_output = subprocess.check_output(['du', '-s', d]).decode('utf-8').split('\t')[0]
            dir_sizes[d] = size_output
        except subprocess.CalledProcessError:
            print(f"Error getting size for directory: {d}")
            dir_sizes[d] = "0K" # Assign a default or handle error as needed
    return dir_sizes


def parse_size_to_bytes(size_str):
    """Parses a human-readable size string (e.g., '10K', '2.5M', '1G') to bytes."""
    size_str = size_str.strip().upper()
    if not size_str:
        return 0

    if size_str.endswith('K'):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith('M'):
        return int(float(size_str[:-1]) * 1024**2)
    elif size_str.endswith('G'):
        return int(float(size_str[:-1]) * 1024**3)
    elif size_str.endswith('T'):
        return int(float(size_str[:-1]) * 1024**4)
    else:
        return int(size_str) # Assume bytes if no suffix


def find_unprocessed_and_smallest_directories(all_directories_raw):
    unprocessed_dirs = []
    
    # Clean and filter directories
    all_directories = [d.strip() for d in all_directories_raw.split('\n') if d.strip() and d.strip() != '.']

    for d in all_directories:
        checkpoint_path = os.path.join(d, '.checkpoint')
        if not os.path.exists(checkpoint_path):
            unprocessed_dirs.append(d)

    # Get sizes for unprocessed directories
    dir_sizes_raw = get_directory_sizes(unprocessed_dirs)

    # Convert sizes to bytes for sorting
    dir_sizes_bytes = {d: parse_size_to_bytes(s) for d, s in dir_sizes_raw.items()}

    # Sort by size (smallest first)
    sorted_dirs = sorted(dir_sizes_bytes.items(), key=lambda item: item[1])

    # Get the top 50 smallest directories
    return [d[0] for d in sorted_dirs[:50]]

# Read the full output from the tool
with open('/Users/trk/.gemini/tmp/cosim-commented-by-flash-gemini/tool-outputs/session-e1556269-3389-4f29-a7a6-aaff06f10c1d/run_shell_command_1771965866089_0.txt', 'r') as f:
    full_output = f.read()

# Extract only the relevant lines (actual directories)
# The full_output string contains "Output: .\n" at the beginning, so split and take from the second line.
directory_lines = full_output.split('Output: .\n', 1)
if len(directory_lines) > 1:
    raw_dirs_list = directory_lines[1].split('\n')
    # Filter out empty strings and the '...' truncation indicators
    raw_dirs = [d for d in raw_dirs_list if d.strip() and not d.startswith('...') and not d.startswith('Process Group PGID')]
    
    smallest_dirs = find_unprocessed_and_smallest_directories('\n'.join(raw_dirs))
    
    print("Found 50 smallest unprocessed directories:")
    for d in smallest_dirs:
        print(d)
else:
    print("No directories found or output format unexpected.")

