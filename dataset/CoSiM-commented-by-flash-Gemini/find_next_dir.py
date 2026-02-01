
"""
This script provides utility functions to navigate a dataset of code, identifying the next directory that has not yet been processed.
It is designed to facilitate an iterative workflow for augmenting code with semantic documentation, ensuring that processing resumes from the last unfinished task.
"""
import os

def find_next_unprocessed_directory(base_path):
    """
    Identifies the next directory within `base_path` that does not contain a '.checkpoint' file.
    This function is crucial for maintaining an iterative processing workflow, allowing the system
    to pick up from where it left off, avoiding redundant work on previously processed directories.

    Args:
        base_path (str): The root directory where subdirectories (representing individual code snippets or tasks) are located.

    Returns:
        str: The name of the next unprocessed directory, or None if all directories have been processed.
    """
    # Retrieve all immediate subdirectories within the base_path and sort them to ensure consistent iteration order.
    all_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    # Iterate through each discovered directory to check for the presence of a '.checkpoint' file.
    for d in all_dirs:
        # If a directory does not contain a '.checkpoint' file, it signifies that it is yet to be processed.
        if not os.path.exists(os.path.join(base_path, d, '.checkpoint')):
            return d
    # If all directories contain a '.checkpoint' file, all tasks are considered complete.
    return None

# Define the base path for the dataset, pointing to the location where code directories are stored.
base_path = os.getcwd()
# Determine the next directory requiring processing based on the presence of a '.checkpoint' file.
next_dir = find_next_unprocessed_directory(base_path)
# Conditional output based on whether an unprocessed directory was found.
if next_dir:
    # Print the name of the next directory to be processed.
    print(next_dir)
else:
    # Indicate that all directories have been successfully processed.
    print("All directories processed.")
