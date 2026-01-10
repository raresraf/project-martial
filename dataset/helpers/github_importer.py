import subprocess
import os
import shutil
import argparse # Import the argparse module for command-line arguments

def run_git_command(command, cwd=None, suppress_errors=False):
    """
    Helper function to run Git commands and handle potential errors.
    Args:
        command (list): The Git command as a list of strings (e.g., ["git", "clone", "url"]).
        cwd (str, optional): The current working directory for the command. Defaults to None.
        suppress_errors (bool): If True, do not exit on error, just print and return None.
    Returns:
        str: The stripped stdout of the command, or None if an error occurred and suppressed.
    """
    try:
        print(f"DEBUG: Running command: {' '.join(command)} (in {cwd if cwd else os.getcwd()})")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,  # Raise CalledProcessError for non-zero exit codes
            capture_output=True,
            text=True,   # Decode stdout/stderr as text
            encoding='utf-8' # Specify encoding for broad compatibility
        )
        stdout_output = result.stdout.strip()
        print(f"DEBUG: Command STDOUT:\n{stdout_output if stdout_output else '[No Output]'}")
        return stdout_output
    except subprocess.CalledProcessError as e:
        print(f"\nError running Git command: {' '.join(command)}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        if not suppress_errors:
            print("Exiting due to Git command error.")
            exit(1)
        return None
    except FileNotFoundError:
        print("\nError: Git command not found. Please ensure Git is installed and in your system's PATH.")
        if not suppress_errors:
            exit(1)
        return None

# The find_pr_merge_commit function is no longer needed as we are comparing a single commit and its parent.
# def find_pr_merge_commit(...)

def download_commit_versions(
    repo_url,
    commit_sha,
    old_folder_name="old_commit_code",
    new_folder_name="new_commit_code"
):
    """
    Downloads old and new code versions related to a specific Git commit into separate folders,
    containing only the files that have changed in that commit.

    Args:
        repo_url (str): The full URL of the GitHub repository (e.g., "https://github.com/owner/repo.git").
        commit_sha (str): The SHA hash of the commit to analyze.
        old_folder_name (str): The name for the folder to store the 'before' code.
        new_folder_name (str): The name for the folder to store the 'after' code.
    """
    print(f"\n--- Starting download_commit_versions ---")
    print(f"Repo URL: {repo_url}")
    print(f"Commit SHA: {commit_sha}")
    print(f"Old Folder Name: {old_folder_name}")
    print(f"New Folder Name: {new_folder_name}")


    # Extract repository name for the temporary clone directory
    repo_name_parts = repo_url.split('/')
    if repo_url.endswith('.git'):
        repo_name = repo_name_parts[-1][:-4]
    else:
        repo_name = repo_name_parts[-1]

    # Use a generic temporary clone directory name since it's per repository now, not per PR.
    # We'll rely on git fetch --all to keep it updated.
    temp_clone_dir = f"{repo_name}_temp_clone"
    script_execution_dir = os.getcwd() # Directory where the script is run

    # 1. Clean up existing output folders
    print(f"\n--- Cleaning up previous output folders ---")
    for folder in [old_folder_name, new_folder_name]:
        full_path = os.path.join(script_execution_dir, folder)
        if os.path.exists(full_path):
            print(f"Removing existing directory: {full_path}")
            shutil.rmtree(full_path)

    # 2. Clone the repository into a temporary directory OR update existing clone
    print(f"\n--- Step 1: Cloning/Updating repository ---")
    if os.path.exists(os.path.join(script_execution_dir, temp_clone_dir)):
        print(f"Temporary clone directory '{temp_clone_dir}' already exists.")
        # Ensure the existing clone is up-to-date with all remote refs
        # Using a more robust fetch command here as well
        # print("Fetching all branches and PRs to ensure commit is available...")
        # fetch_all_refs_cmd = [
        #     "git", "fetch", "origin",
        #     "refs/heads/*:refs/remotes/origin/*",
        #     "refs/pull/*/head:refs/remotes/origin/pr/*"
        # ]
        # run_git_command(fetch_all_refs_cmd, cwd=temp_clone_dir, suppress_errors=True) # Suppress errors if some refs fail
        # print("Repository updated successfully.")
    else:
        print(f"Cloning {repo_url} into {temp_clone_dir}...")
        clone_result = run_git_command(["git", "clone", repo_url, temp_clone_dir])
        if clone_result is None:
            return # Exit if cloning failed
        print("Repository cloned successfully.")
        
        print("Fetching all branches and PRs to ensure commit is available...")
        fetch_all_refs_cmd = [
            "git", "fetch", "origin",
            "refs/heads/*:refs/remotes/origin/*",
            "refs/pull/*/head:refs/remotes/origin/pr/*"
        ]
        run_git_command(fetch_all_refs_cmd, cwd=temp_clone_dir)

    # 3. Identify "new" (the commit itself) and "old" (its parent) commit SHAs
    print(f"\n--- Step 2: Identifying Commit SHAs ---")
    
    # The 'new' code is simply the provided commit_sha
    new_commit_hash = run_git_command(["git", "rev-parse", commit_sha], cwd=temp_clone_dir)
    if new_commit_hash is None:
        print(f"Error: Could not resolve commit SHA '{commit_sha}'. It might be invalid or not fetched.")
        # Do not remove temp_clone_dir here, it might be reusable.
        print("Skipping cleanup of temporary clone directory as it might be reused in future runs.")
        return
    
    print(f"Identified 'New Code' Commit: {new_commit_hash}")

    # The 'old' code is the parent of the provided commit_sha
    # Check if it's an initial commit (has no parents)
    parent_commit_output = run_git_command(["git", "rev-parse", f"{new_commit_hash}^1"], cwd=temp_clone_dir, suppress_errors=True)
    
    if parent_commit_output:
        old_commit_hash = parent_commit_output
        print(f"Identified 'Old Code' Commit (Parent of new commit): {old_commit_hash}")
    else:
        # This is an initial commit, it has no 'old' state to compare against.
        old_commit_hash = "4b825dc642cb6eb9a060e54bf8d69288fbee4904" # This is Git's "empty tree" SHA
        print(f"WARNING: Commit '{commit_sha}' appears to be an initial commit (no parent).")
        print(f"Comparing against an empty tree ({old_commit_hash}) to show all files as added.")
        
    print(f"Final 'New Code' Commit: {new_commit_hash}")
    print(f"Final 'Old Code' Commit: {old_commit_hash}")

    if old_commit_hash == new_commit_hash:
        print("\nWARNING: After commit parent resolution, the 'old' and 'new' commit hashes are identical.")
        print("This indicates no changes or a problem in history for this specific commit.")
        print("No files will be copied as there are no detectable differences.")
        print("Skipping cleanup of temporary clone directory as it might be reused in future runs.")
        return
        
    # 4. Get changed files and their status
    print(f"\n--- Step 3: Getting changed files from diff ---")
    diff_output = run_git_command(["git", "diff", "--name-status", old_commit_hash, new_commit_hash], cwd=temp_clone_dir)
    if diff_output is None:
        print("Error getting diff. Skipping cleanup of temporary clone directory as it might be reused in future runs.")
        return

    print(f"DEBUG: Raw git diff --name-status output:\n{diff_output}")

    changed_files = []
    for line in diff_output.splitlines():
        if line:
            parts = line.split('\t')
            status = parts[0]
            
            if len(parts) == 2:
                filepath = parts[1]
                changed_files.append((status, filepath, filepath)) # (status, old_path, new_path)
            elif len(parts) == 3 and (status.startswith('R') or status.startswith('C')):
                # For renames/copies, the format is status<tab>old_path<tab>new_path
                old_filepath_for_rename = parts[1]
                new_filepath_for_rename = parts[2]
                changed_files.append((status, old_filepath_for_rename, new_filepath_for_rename))
            else:
                print(f"WARNING: Unexpected diff output line format: '{line}'")
                continue # Skip malformed lines

    if not changed_files:
        print("No differences found for this commit. No files will be copied.")
        print("Skipping cleanup of temporary clone directory as it might be reused in future runs.")
        return
    
    print(f"Found {len(changed_files)} changed files.")

    # 5. Create new target folders and checkout only changed code
    print(f"\n--- Step 4: Checking out only changed code into separate folders ---")
    os.makedirs(os.path.join(script_execution_dir, old_folder_name), exist_ok=True)
    os.makedirs(os.path.join(script_execution_dir, new_folder_name), exist_ok=True)

    for status, old_filepath, new_filepath in changed_files:
        print(f"  Processing {status}: {old_filepath} (old) / {new_filepath} (new)")
        
        # Prepare target paths for old and new folders
        target_old_file_path = os.path.join(script_execution_dir, old_folder_name, old_filepath)
        target_new_file_path = os.path.join(script_execution_dir, new_folder_name, new_filepath)

        # Create parent directories for the files in the target folders
        os.makedirs(os.path.dirname(target_old_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(target_new_file_path), exist_ok=True)

        # Checkout old version of the file if it exists at the old commit
        # Applies to Modified (M), Deleted (D), Renamed (R), Copied (C) files in their original path
        # Need to specify the `old_filepath` for checkout from the `old_commit_hash`
        if status in ['M', 'D'] or status.startswith('R') or status.startswith('C'):
            checkout_old_file_cmd = ["git", "--work-tree", os.path.join(script_execution_dir, old_folder_name), 
                                     "checkout", old_commit_hash, "--", old_filepath]
            run_git_command(checkout_old_file_cmd, cwd=temp_clone_dir, suppress_errors=True) 

        # Checkout new version of the file if it exists at the new commit
        # Applies to Modified (M), Added (A), Renamed (R), Copied (C) files in their new path
        # Need to specify the `new_filepath` for checkout from the `new_commit_hash`
        if status in ['M', 'A'] or status.startswith('R') or status.startswith('C'):
            checkout_new_file_cmd = ["git", "--work-tree", os.path.join(script_execution_dir, new_folder_name), 
                                     "checkout", new_commit_hash, "--", new_filepath]
            run_git_command(checkout_new_file_cmd, cwd=temp_clone_dir, suppress_errors=True) 

    print("\n--- Download Complete! ---")
    print(f"Old code (only changed files) is located in: {os.path.abspath(old_folder_name)}")
    print(f"New code (only changed files) is located in: {os.path.abspath(new_folder_name)}")

    # 6. Clean up the temporary clone directory
    print("\n--- Note: Temporary clone directory is kept for potential reuse. ---")
    print(f"You can manually remove '{os.path.abspath(temp_clone_dir)}' if no longer needed.")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Download 'before' and 'after' states of a specific Git commit into separate folders.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--repo_url",
        default="https://github.com/kubernetes/kubernetes", # Set default repo URL here
        help="The full URL of the GitHub repository (e.g., https://github.com/octocat/Spoon-Knife.git)"
    )
    parser.add_argument(
        "--commit_sha",
        help="The SHA hash of the commit to analyze (e.g., a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0)"
    )
    parser.add_argument(
        "--old_folder_name",
        default="old_commit_code",
        help="Name for the folder to store the 'before' code (default: old_commit_code)"
    )
    parser.add_argument(
        "--new_folder_name",
        default="new_commit_code",
        help="Name for the folder to store the 'after' code (default: new_commit_code)"
    )

    args = parser.parse_args()

    print("--- Git Commit Code Downloader ---")
    print("This script will download the 'before' and 'after' states of a specific commit into two separate folders,")
    print("containing only the files that were part of the diff (added, modified, or deleted).")
    print("Make sure Git is installed and available in your system's PATH.")

    try:
        download_commit_versions(
            args.repo_url,
            args.commit_sha,
            args.old_folder_name,
            args.new_folder_name
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your inputs and try again.")
