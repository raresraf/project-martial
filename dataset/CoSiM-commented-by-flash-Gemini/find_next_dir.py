"""
@package find_next_dir
@brief This script is designed to manage an iterative processing workflow for a dataset of code directories.
It identifies the next subdirectory that has not yet been marked as processed by a '.checkpoint' file,
facilitating continuous integration or analysis tasks by ensuring work resumes from the last uncompleted unit.
Algorithm: Directory traversal and checkpoint file detection.
Time Complexity: O(D * F), where D is the number of subdirectories and F is the average time to check for a file's existence within each directory.
Space Complexity: O(D) to store the list of directory names.
"""
import os

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
        # Block Logic: Check for the existence of a '.checkpoint' file within the current subdirectory.
        # A missing checkpoint file signifies an unprocessed directory.
        # Precondition: `d` is a valid subdirectory name.
        # Invariant: `os.path.exists` correctly reports the presence of the checkpoint.
        if not os.path.exists(os.path.join(base_path, d, '.checkpoint')):
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