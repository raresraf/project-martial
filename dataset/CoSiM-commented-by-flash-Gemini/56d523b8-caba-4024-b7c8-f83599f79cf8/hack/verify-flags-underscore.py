#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
@module verify-flags-underscore.py
@brief This script enforces a naming convention for command-line flags in Go source files
within the Kubernetes project.
It checks for flag declarations that contain underscores (`_`) in their names.
Flags found with underscores that are not explicitly whitelisted in
`hack/verify-flags/excluded-flags.txt` will cause the script to fail,
promoting consistent flag naming (typically using hyphens instead of underscores).
"""

from __future__ import print_function

import argparse
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("filenames", help="list of files to check, all files if unspecified", nargs='*')
args = parser.parse_args()

# Cargo culted from http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
def is_binary(pathname):
    """
    Determines if a given file is a binary file.

    This function attempts to read chunks of the file and checks for the
    presence of null bytes ('\0'). Binary files typically contain null
    bytes, whereas text files generally do not.

    Args:
        pathname (str): The path to the file to check.

    Returns:
        bool: True if the file is likely binary, False otherwise.

    Raises:
        EnvironmentError: If the file does not exist or cannot be accessed.
    """
    try:
        with open(pathname, 'r') as f: # Open file in text mode
            CHUNKSIZE = 1024 # Define chunk size for reading
            while 1:
                chunk = f.read(CHUNKSIZE)
                if '\0' in chunk: # found null byte, indicating a binary file
                    return True
                if len(chunk) < CHUNKSIZE:
                    break # Reached end of file
    except:
        # If any error occurs during file reading (e.g., UnicodeDecodeError for non-UTF-8 content,
        # or file not found), assume it's binary or unreadable as text.
        return True

    return False

def get_all_files(rootdir):
    """
    Recursively collects paths to all non-binary files within a specified root directory.
    Certain directories and files are explicitly excluded from the search to optimize
    performance and avoid irrelevant files.

    Args:
        rootdir (str): The root directory from which to start the file collection.

    Returns:
        list: A list of absolute paths to all collected non-binary files.
    """
    all_files = []
    # os.walk generates the file names in a directory tree by walking the tree either top-down or bottom-up.
    for root, dirs, files in os.walk(rootdir):
        # Block Logic: Don't visit certain directories (e.g., build outputs, third-party code).
        # Modifying 'dirs' in-place tells os.walk not to recurse into these directories.
        if 'vendor' in dirs:
            dirs.remove('vendor')
        if 'staging' in dirs:
            dirs.remove('staging')
        if '_output' in dirs:
            dirs.remove('_output')
        if '_gopath' in dirs:
            dirs.remove('third_party')
        if '.git' in dirs:
            dirs.remove('.git')
        if '.make' in dirs:
            dirs.remove('.make')
        if 'third_party' in dirs:
            dirs.remove('third_party')
        
        # Block Logic: Don't process certain files.
        # Removing 'BUILD' from 'files' prevents it from being processed in the current directory.
        if 'BUILD' in files:
           files.remove('BUILD')

        for name in files:
            pathname = os.path.join(root, name)
            # Inline: Skip binary files to prevent errors with text-based processing (e.g., regex matching).
            if is_binary(pathname):
                continue
            all_files.append(pathname)
    return all_files

# Collects all the flags used in golang files and verifies the flags do
# not contain underscore. If any flag needs to be excluded from this check,
# need to add that flag in hack/verify-flags/excluded-flags.txt.
def check_underscore_in_flags(rootdir, files):
    """
    Scans Go source files for flag declarations and verifies that flag names
    do not contain underscores (`_`), unless explicitly whitelisted.

    Args:
        rootdir (str): The root directory of the project, used to locate the exclusion file.
        files (list): A list of file paths to check.

    Raises:
        SystemExit(1): If any flag is found with an underscore and is not in the
                       excluded flags list, the script prints an error message and exits.
    """
    # Block Logic: Load flags that are explicitly excluded from the underscore check.
    # These are flags that are allowed to contain underscores despite the general rule.
    pathname = os.path.join(rootdir, "hack/verify-flags/excluded-flags.txt")
    with open(pathname, 'r') as f:
        excluded_flags = set(f.read().splitlines())
    
    # Defines a list of regular expressions to capture flag names from Go code.
    # Each regex targets a specific Go flag declaration pattern (e.g., flag.StringVar, flag.BoolVar).
    regexs = [ re.compile(r'Var[P]?\([^,]*, "([^"]*)"'), # Catches flag.Var(P) declarations
               re.compile(r'\.String[P]?\("([^"]*)",[^,]+,[^)]+\)'), # Catches flag.String(P)
               re.compile(r'\.Int[P]?\("([^"]*)",[^,]+,[^)]+\)'),    # Catches flag.Int(P)
               re.compile(r'\.Bool[P]?\("([^"]*)",[^,]+,[^)]+\)'),   # Catches flag.Bool(P)
               re.compile(r'\.Duration[P]?\("([^"]*)",[^,]+,[^)]+\)'),# Catches flag.Duration(P)
               re.compile(r'\.StringSlice[P]?\("([^"]*)",[^,]+,[^)]+\)') ] # Catches custom StringSlice flags

    new_excluded_flags = set() # Stores flags found with underscores that are not whitelisted.
    # Block Logic: Iterate through each file, read its content, and apply regexes to find flags.
    for pathname in files:
        # Inline: Only process Go files for flag checks.
        if not pathname.endswith(".go"):
            continue
        with open(pathname, 'r') as f:
            data = f.read()
        
        matches = []
        for regex in regexs:
            # Inline: Find all occurrences of flag names matching the current regex in the file data.
            matches = matches + regex.findall(data)
        
        # Block Logic: Check each found flag for underscores and against the exclusion list.
        for flag in matches:
            # Inline: Skip this flag if it's in the list of explicitly excluded flags.
            if any(x in flag for x in excluded_flags):
                continue
            # Inline: If the flag contains an underscore and is not excluded, add it to the report set.
            if "_" in flag:
                new_excluded_flags.add(flag)
    
    # Block Logic: Report any non-compliant flags and exit with an error code.
    if len(new_excluded_flags) != 0:
        print("Found a flag declared with an _ but which is not explicitly listed as a valid flag name in hack/verify-flags/excluded-flags.txt")
        print("Are you certain this flag should not have been declared with an - instead?")
        l = list(new_excluded_flags) # Convert set to list for sorting.
        l.sort() # Sort the list for consistent output.
        print("%s" % "\n".join(l)) # Print the non-compliant flags.
        sys.exit(1) # Exit with a non-zero status code to indicate failure.

def main():
    """
    Main entry point for the script.
    It parses command-line arguments, determines the set of files to check,
    and then initiates the flag underscore verification process.
    """
    # Inline: Calculate the absolute path to the project root directory.
    rootdir = os.path.dirname(__file__) + "/../"
    rootdir = os.path.abspath(rootdir)

    files = []
    # Block Logic: Determine which files to check.
    # If filenames are provided as arguments, use them; otherwise, gather all relevant files from the project.
    if len(args.filenames) > 0:
        files = args.filenames
    else:
        files = get_all_files(rootdir)

    # Inline: Execute the core logic to check for underscores in flag names.
    check_underscore_in_flags(rootdir, files)

if __name__ == "__main__":
  # This block ensures that `main()` is called only when the script is executed directly,
  # not when it's imported as a module. It also handles the script's exit status.
  sys.exit(main())
