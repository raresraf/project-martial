#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
This script verifies that command-line flags in Go source files do not contain
underscores, as the convention is to use hyphens. It allows for an exclusion
list for flags that are intentionally defined with underscores.
"""

from __future__ import print_function

import argparse
import os
import re
import sys

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Verify that Go command-line flags do not contain underscores."
)
parser.add_argument("filenames", help="List of files to check; if unspecified, all Go files in the project are checked.", nargs='*')
args = parser.parse_args()


# --- Utility Functions ---

def is_binary(pathname):
    """
    Checks if a file is binary by searching for null bytes in its content.
    This function is used to avoid processing non-text files.

    Args:
        pathname (str): The path to the file to check.

    Returns:
        bool: True if the file is determined to be binary, False otherwise.
    """
    # This implementation is a common heuristic for detecting binary files.
    try:
        with open(pathname, 'r') as f:
            CHUNKSIZE = 1024
            while True:
                chunk = f.read(CHUNKSIZE)
                if '\0' in chunk:  # Null byte found, indicating a binary file.
                    return True
                if len(chunk) < CHUNKSIZE:
                    break  # End of file.
    except (IOError, UnicodeDecodeError):
        # If the file cannot be opened in text mode or a decode error occurs,
        # it's likely binary or not readable as standard text.
        return True
    return False


def get_all_files(rootdir):
    """
    Walks a directory tree and collects all non-binary, non-excluded files.

    Args:
        rootdir (str): The root directory to start the walk from.

    Returns:
        list: A list of file paths to be checked.
    """
    all_files = []
    # Pre-defined set of directories to exclude from the file search to improve efficiency.
    excluded_dirs = {'vendor', 'staging', '_output', '_gopath', 'third_party', '.git', '.make'}

    for root, dirs, files in os.walk(rootdir):
        # Modify the list of directories in-place to prevent os.walk from descending into them.
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        if 'BUILD' in files:
           files.remove('BUILD')

        for name in files:
            pathname = os.path.join(root, name)
            if is_binary(pathname):
                continue
            all_files.append(pathname)
    return all_files


# --- Core Logic ---

def check_underscore_in_flags(rootdir, files):
    """
    Scans Go files for command-line flag definitions that contain underscores.

    It reads a list of allowed flags from an exclusion file. If a flag with an
    underscore is found and it is not on the exclusion list, the script will
-   print an error and exit.

    Args:
        rootdir (str): The project's root directory.
        files (list): A list of file paths to check.
    """
    # Load the set of flags that are exempt from the underscore check.
    # This allows for legacy or special-case flags to be ignored.
    exclusion_file = os.path.join(rootdir, "hack/verify-flags/excluded-flags.txt")
    try:
        with open(exclusion_file, 'r') as f:
            excluded_flags = set(f.read().splitlines())
    except IOError:
        print("Warning: Could not find exclusion file: %s" % exclusion_file, file=sys.stderr)
        excluded_flags = set()

    # Regex patterns to find flag definitions in Go code, targeting functions
    # from libraries like pflag or flag (e.g., pflag.StringVar, pflag.Int, etc.).
    # These capture the string literal that defines the flag's name.
    regexs = [ re.compile('Var[P]?\([^,]*", "([^"]*)"'),
               re.compile('.String[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.Int[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.Bool[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.Duration[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.StringSlice[P]?\("([^"]*)",[^,]+,[^)]+"]') ]

    offending_flags = set()
    # --- File Scanning Loop ---
    # Pre-condition: `files` contains a list of file paths to be checked.
    for pathname in files:
        if not pathname.endswith(".go"):
            continue
        
        try:
            with open(pathname, 'r') as f:
                data = f.read()
        except IOError:
            print("Warning: Could not read file: %s" % pathname, file=sys.stderr)
            continue
        
        # --- Regex Matching ---
        # Apply each regex to the file content to find all flag definitions.
        matches = []
        for regex in regexs:
            matches.extend(regex.findall(data))
        
        # --- Flag Verification ---
        # Invariant: After this loop, all flags with underscores are checked against the exclusion list.
        for flag in matches:
            # Skip if the flag is in the exclusion list.
            if any(x in flag for x in excluded_flags):
                continue
            # If a flag contains an underscore and is not excluded, it's an offense.
            if "_" in flag:
                offending_flags.add(flag)

    # --- Result Handling ---
    # Post-condition: If `offending_flags` is not empty, an error is reported.
    if len(offending_flags) != 0:
        print("Error: Found flags declared with an underscore ('_') that are not in the exclusion list.", file=sys.stderr)
        print("Flags should be defined with a hyphen ('-'). If the underscore is intentional, add the flag to 'hack/verify-flags/excluded-flags.txt'.", file=sys.stderr)
        
        sorted_flags = sorted(list(offending_flags))
        print("\nOffending flags:\n%s" % "\n".join(sorted_flags), file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main execution function.
    
    It determines the set of files to check and initiates the verification process.
    """
    # Assume the script is in a 'hack' directory, and the project root is one level up.
    rootdir = os.path.dirname(__file__) + "/../"
    rootdir = os.path.abspath(rootdir)

    # If filenames are provided as arguments, use them. Otherwise, scan the entire project.
    if len(args.filenames) > 0:
        files = args.filenames
    else:
        files = get_all_files(rootdir)

    check_underscore_in_flags(rootdir, files)
    return 0

if __name__ == "__main__":
  sys.exit(main())