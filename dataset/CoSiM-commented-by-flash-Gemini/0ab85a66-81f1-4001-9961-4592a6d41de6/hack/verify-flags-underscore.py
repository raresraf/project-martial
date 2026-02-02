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
@file verify-flags-underscore.py
@brief This script verifies that command-line flags declared in Go files
adhere to a naming convention disallowing underscores.

Functional Utility: Enforces a consistent code style for command-line flags
within a Go codebase, preventing issues that might arise from different
platforms or tools interpreting flag names with underscores inconsistently.
It helps maintain code quality and readability by ensuring flags use hyphens
instead of underscores.

Algorithm:
1. Argument Parsing: Handles command-line arguments to specify files to check
   or to check all relevant files if none are specified.
2. File Filtering: Recursively collects all non-binary, non-excluded files.
3. Excluded Flags Loading: Reads a list of explicitly allowed flags (that contain
   underscores) from a file.
4. Regex Matching: Uses regular expressions to identify flag declarations in Go files.
5. Underscore Check: For each identified flag, it checks if an underscore is present.
6. Violation Reporting: If a flag with an underscore is found and not in the
   excluded list, it's reported as a violation, and the script exits with an error.

Time Complexity: O(F * (L_read + R_count * L_regex)), where F is the number of
files, L_read is the average file read time, R_count is the number of regexes,
and L_regex is the average regex matching time. In essence, it's proportional
to the total size of scanned Go files.
Space Complexity: O(F_go + L_exclude + S_file), where F_go is the list of Go files,
L_exclude is the list of excluded flags, and S_file is the content of the largest
file read at once.
"""

from __future__ import print_function

import argparse # For parsing command-line arguments.
import os # For interacting with the operating system (e.g., path manipulation, directory traversal).
import re # For regular expression operations.
import sys # For system-specific parameters and functions (e.g., exiting the script).

# Functional Utility: Set up command-line argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("filenames", help="list of files to check, all files if unspecified", nargs='*') # Argument for specifying files.
args = parser.parse_args() # Parse the arguments.

# Cargo culted from http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
def is_binary(pathname):
    """
    @brief Detects if a given file is binary by checking for null bytes.
    Functional Utility: Prevents attempts to apply text-based regex searches on
    binary files, which could lead to errors or incorrect results.
    @param pathname (str): The path to the file.
    @return (bool): `True` if the file is likely binary (contains null bytes or cannot be read), `False` otherwise.
    @raise EnvironmentError: if the file does not exist or cannot be accessed.
    @attention: found @ http://bytes.com/topic/python/answers/21222-determine-file-type-binary-text on 6/08/2010
    @author: Trent Mick <TrentM@ActiveState.com>
    @author: Jorge Orpinel <jorge@orpinel.com>
    """
    try:
        with open(pathname, 'r') as f: # Open file in text mode.
            CHUNKSIZE = 1024 # Read in chunks to efficiently check large files.
            while True: # Block Logic: Iterate through file chunks.
                chunk = f.read(CHUNKSIZE) # Read a chunk.
                if '\0' in chunk: # If null byte is found.
                    return True # File is binary.
                if len(chunk) < CHUNKSIZE: # If chunk is smaller than CHUNKSIZE, end of file reached.
                    break # Done reading.
    except: # Block Logic: Handle errors during file access, assuming binary if an error occurs.
        return True

    return False # No null bytes found, considered text.

def get_all_files(rootdir):
    """
    @brief Recursively collects paths of all non-binary files under a given root directory, excluding specific directories.
    Functional Utility: Prepares a list of files that need to be scanned for flag declarations,
    optimizing the process by ignoring irrelevant or unreadable files.
    @param rootdir (str): The root directory to start scanning from.
    @return (list): A list of absolute file paths (strings).
    """
    all_files = []
    # Block Logic: Walk the directory tree, excluding specified directories for efficiency.
    for root, dirs, files in os.walk(rootdir):
        # Don't visit certain directories (e.g., build artifacts, dependencies, version control).
        if 'vendor' in dirs:
            dirs.remove('vendor')
        if 'staging' in dirs:
            dirs.remove('staging')
        if '_output' in dirs:
            dirs.remove('_output')
        if '_gopath' in dirs:
            dirs.remove('_gopath')
        if 'third_party' in dirs:
            dirs.remove('third_party')
        if '.git' in dirs:
            dirs.remove('.git')
        if '.make' in dirs:
            dirs.remove('.make')
        if 'BUILD' in files: # Remove BUILD file to avoid processing.
           files.remove('BUILD')

        for name in files: # Block Logic: Iterate through files in the current directory.
            pathname = os.path.join(root, name) # Construct full path.
            if is_binary(pathname): # If file is binary, skip it.
                continue
            all_files.append(pathname) # Add non-binary file to the list.
    return all_files

def check_underscore_in_flags(rootdir, files):
    """
    @brief Checks if Go flag declarations contain underscores, enforcing a naming convention.
    Functional Utility: Identifies and reports Go flag declarations that use underscores
    instead of hyphens, excluding a predefined list of allowed exceptions. This helps
    enforce a consistent command-line interface style.
    @param rootdir (str): The root directory of the project, used to locate the excluded flags file.
    @param files (list): A list of file paths to check.
    """
    # preload the 'known' flags which don't follow the - standard
    pathname = os.path.join(rootdir, "hack/verify-flags/excluded-flags.txt") # Path to the file containing excluded flags.
    f = open(pathname, 'r')
    excluded_flags = set(f.read().splitlines()) # Read excluded flags into a set for efficient lookup.
    f.close()

    # Functional Utility: Define regular expressions to capture flag names from Go code.
    # These regexes target common Go flag declaration patterns (e.g., `flag.StringVarP`, `flag.IntP`).
    regexs = [ re.compile('Var[P]?\([^,]*, "([^"]*)"'), # Matches patterns like `Var(