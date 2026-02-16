import os

def has_checkpoint_recursive(directory):
    """
    @brief Recursively checks for the existence of a '.checkpoint' file within the given directory or any of its subdirectories.
    
    @param directory (str): The path to the directory to search within.
    @return bool: True if a '.checkpoint' file is found, False otherwise.
    """
    for root, _, files in os.walk(directory):
        if '.checkpoint' in files:
            return True
    return False

def find_next_unprocessed_directory(base_path):
    """
    @brief Identifies the next directory within `base_path` that does not contain a '.checkpoint' file.
    This function is crucial for maintaining an iterative processing workflow, allowing the system
    to pick up from where it left off, avoiding redundant work on previously processed directories.

    @param base_path (str): The root directory where subdirectories (representing individual code snippets or tasks) are located.

    @return str: The name of the next unprocessed directory, or None if all directories have been processed.
    """
    # Block Logic: List all immediate subdirectories and sort them for consistent, deterministic processing order.
    # Precondition: `base_path` exists and is accessible.
    # Invariant: `all_dirs` contains only directory names, sorted alphabetically.
    all_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    # Block Logic: Iterate through each subdirectory to determine if it has been processed.
    # Precondition: `all_dirs` contains a list of candidate directories.
    # Invariant: Each directory is checked for a '.checkpoint' file exactly once.
    for d in all_dirs:
        # Block Logic: Check for the existence of a '.checkpoint' file within the current subdirectory or its children.
        # A missing checkpoint file signifies an unprocessed directory.
        # Precondition: `d` is a valid subdirectory name.
        # Invariant: `has_checkpoint_recursive` correctly reports the presence of the checkpoint.
        if not has_checkpoint_recursive(os.path.join(base_path, d)):
            # Functional Utility: Return the name of the first unprocessed directory found.
            return d
    # Functional Utility: If the loop completes, all directories have '.checkpoint' files, indicating full processing.
    return None

# Functional Utility: Set the current working directory as the base path for scanning.
base_path = os.getcwd()
# Functional Utility: Invoke the directory scanner to find the next pending task.
next_dir = find_next_unprocessed_directory(base_path)

# Block Logic: Provide output indicating the status of the dataset processing.
# Precondition: `next_dir` holds the result of the scan (either a directory name or None).
# Invariant: A clear status message is printed to standard output.
if next_dir:
    # Output the name of the directory that requires further action.
    print(next_dir)
else:
    # Indicate successful completion of all processing tasks in the dataset.
    print("All directories processed.")