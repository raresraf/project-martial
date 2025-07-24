import subprocess
import os
import shutil

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

def download_pr_versions(
    repo_url,
    pr_number,
    old_folder_name="old_pr_code",
    new_folder_name="new_pr_code",
    base_branch_name="main" # Common default for the base branch
):
    """
    Downloads old and new code versions of a GitHub Pull Request into separate folders,
    containing only the files that have changed in the PR. Handles both open and merged PRs.

    Args:
        repo_url (str): The full URL of the GitHub repository (e.g., "https://github.com/owner/repo.git").
        pr_number (int): The number of the pull request.
        old_folder_name (str): The name for the folder to store the 'old' code.
        new_folder_name (str): The name for the folder to store the 'new' code.
        base_branch_name (str): The name of the base branch the PR is targeting (e.g., "main", "master").
    """
    print(f"\n--- Starting download_pr_versions ---")
    print(f"Repo URL: {repo_url}")
    print(f"PR Number: {pr_number}")
    print(f"Old Folder Name: {old_folder_name}")
    print(f"New Folder Name: {new_folder_name}")
    print(f"Base Branch Name: {base_branch_name}")


    # Extract repository name for the temporary clone directory
    repo_name_parts = repo_url.split('/')
    if repo_url.endswith('.git'):
        repo_name = repo_name_parts[-1][:-4]
    else:
        repo_name = repo_name_parts[-1]

    temp_clone_dir = f"{repo_name}_pr_{pr_number}_temp_clone"
    script_execution_dir = os.getcwd() # Directory where the script is run

    # 1. Clean up existing output folders and temporary clone directory if they exist
    print(f"\n--- Cleaning up previous runs ---")
    for folder in [old_folder_name, new_folder_name, temp_clone_dir]:
        full_path = os.path.join(script_execution_dir, folder)
        if os.path.exists(full_path):
            print(f"Removing existing directory: {full_path}")
            shutil.rmtree(full_path)

    # 2. Clone the repository into a temporary directory
    print(f"\n--- Step 1: Cloning repository ---")
    print(f"Cloning {repo_url} into {temp_clone_dir}...")
    clone_result = run_git_command(["git", "clone", repo_url, temp_clone_dir])
    if clone_result is None:
        return # Exit if cloning failed
    print("Repository cloned successfully.")

    # 3. Fetch necessary refs
    print(f"\n--- Step 2: Fetching Pull Request refs ---")
    
    # Fetch the PR's head ref
    pr_head_ref_name = f"pr-{pr_number}-head"
    fetch_pr_head_command = ["git", "fetch", "origin", f"pull/{pr_number}/head:{pr_head_ref_name}"]
    print(f"Fetching PR #{pr_number} head (ref: {pr_head_ref_name})...")
    fetch_pr_head_result = run_git_command(fetch_pr_head_command, cwd=temp_clone_dir, suppress_errors=True)
    if fetch_pr_head_result is None:
        print(f"Could not fetch PR head for #{pr_number}. It might not exist or the URL/number is incorrect.")
        shutil.rmtree(temp_clone_dir)
        return

    # Fetch the PR's merge ref (if it's a merged PR)
    pr_merge_ref_name = f"pr-{pr_number}-merge"
    fetch_pr_merge_command = ["git", "fetch", "origin", f"pull/{pr_number}/merge:{pr_merge_ref_name}"]
    print(f"Attempting to fetch PR #{pr_number} merge ref (ref: {pr_merge_ref_name})...")
    fetch_pr_merge_result = run_git_command(fetch_pr_merge_command, cwd=temp_clone_dir, suppress_errors=True)
    # Note: suppress_errors=True because this ref might not exist for unmerged PRs

    # Ensure the base branch is also fetched and up-to-date
    print(f"Fetching base branch '{base_branch_name}'...")
    fetch_base_command = ["git", "fetch", "origin", base_branch_name]
    fetch_base_result = run_git_command(fetch_base_command, cwd=temp_clone_dir)
    if fetch_base_result is None:
        print(f"Could not fetch base branch '{base_branch_name}'. Please ensure the base branch name is correct.")
        shutil.rmtree(temp_clone_dir)
        return

    # 4. Get commit SHAs for "new" and "old" code
    print(f"\n--- Step 3: Identifying Commit SHAs ---")
    new_commit_hash = None
    old_commit_hash = None

    # Option 1: Try using the merge commit if available (for merged PRs)
    merge_commit_sha = run_git_command(["git", "rev-parse", pr_merge_ref_name], cwd=temp_clone_dir, suppress_errors=True)
    
    if merge_commit_sha:
        print(f"DEBUG: Found merge commit SHA: {merge_commit_sha}. Assuming merged PR.")
        # First parent is typically the base branch before merge
        old_commit_hash = run_git_command(["git", "rev-parse", f"{merge_commit_sha}^1"], cwd=temp_clone_dir, suppress_errors=True)
        # Second parent is typically the PR branch head
        new_commit_hash = run_git_command(["git", "rev-parse", f"{merge_commit_sha}^2"], cwd=temp_clone_dir, suppress_errors=True)
        
        if not old_commit_hash or not new_commit_hash:
            print("WARNING: Could not determine parents of the merge commit. Falling back to head/merge-base method.")
            old_commit_hash = None # Reset to trigger fallback
            new_commit_hash = None # Reset to trigger fallback
    
    # Option 2: Fallback to head and merge-base (for unmerged PRs or if merge ref failed)
    if not new_commit_hash:
        print("DEBUG: Using head and merge-base for commit identification.")
        new_commit_hash = run_git_command(["git", "rev-parse", pr_head_ref_name], cwd=temp_clone_dir)
        if new_commit_hash is None:
            shutil.rmtree(temp_clone_dir)
            return

        old_commit_hash = run_git_command(["git", "merge-base", new_commit_hash, f"origin/{base_branch_name}"], cwd=temp_clone_dir)
        if old_commit_hash is None:
            print(f"Could not find merge-base for PR #{pr_number} with base branch '{base_branch_name}'.")
            print("This can happen if the base branch name is incorrect or the PR history is unusual.")
            shutil.rmtree(temp_clone_dir)
            return

    print(f"Identified 'New Code' Commit: {new_commit_hash}")
    print(f"Identified 'Old Code' Commit: {old_commit_hash}")

    if old_commit_hash == new_commit_hash:
        print("\nWARNING: The 'old' and 'new' commit hashes are identical. This means there are no changes to diff.")
        print("This often happens for PRs that are fast-forward merged or have no effective changes. No files will be copied.")
        shutil.rmtree(temp_clone_dir)
        return
        
    # 5. Get changed files and their status
    print(f"\n--- Step 4: Getting changed files from diff ---")
    diff_output = run_git_command(["git", "diff", "--name-status", old_commit_hash, new_commit_hash], cwd=temp_clone_dir)
    if diff_output is None:
        shutil.rmtree(temp_clone_dir)
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
        print("No differences found between the old and new code states for this PR. No files will be copied.")
        shutil.rmtree(temp_clone_dir)
        return
    
    print(f"Found {len(changed_files)} changed files.")

    # 6. Create new target folders and checkout only changed code
    print(f"\n--- Step 5: Checking out only changed code into separate folders ---")
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

    # 7. Clean up the temporary clone directory
    print(f"\n--- Cleaning up temporary clone directory ---")
    shutil.rmtree(temp_clone_dir)
    print("Cleanup complete.")

if __name__ == "__main__":
    print("--- GitHub Pull Request Code Downloader ---")
    print("This script will download the 'before' and 'after' states of a PR into two separate folders,")
    print("containing only the files that were part of the diff (added, modified, or deleted).")
    print("Make sure Git is installed and available in your system's PATH.")

    repo_url_input = input("Enter GitHub repository URL (e.g., https://github.com/kubernetes/kubernetes): ").strip() or "https://github.com/kubernetes/kubernetes"
    pr_number_input = input("Enter Pull Request number: ").strip()
    
    # Optional inputs with defaults
    old_folder_input = input(f"Enter name for 'old code' folder (default: old_pr_code): ").strip() or "old_pr_code"
    new_folder_input = input(f"Enter name for 'new code' folder (default: new_pr_code): ").strip() or "new_pr_code"
    base_branch_input = input(f"Enter the base branch name (e.g., main, master, develop - default: master): ").strip() or "master"

    if not repo_url_input or not pr_number_input.isdigit():
        print("\nInvalid input. Please provide a valid repository URL and a numeric PR number.")
    else:
        try:
            pr_num = int(pr_number_input)
            download_pr_versions(repo_url_input, pr_num, old_folder_input, new_folder_input, base_branch_input)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please check your inputs and try again.")
