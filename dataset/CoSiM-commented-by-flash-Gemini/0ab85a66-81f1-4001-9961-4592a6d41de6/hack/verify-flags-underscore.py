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
@0ab85a66-81f1-4001-9961-4592a6d41de6/hack/verify-flags-underscore.py
@brief Linter for enforcing hyphen-separated command-line flags in Go source files.

Functional Intent: Scans the codebase to ensure that all Go flag declarations 
follow the convention of using hyphens instead of underscores. This promotes 
consistency across CLI tools and avoids potential cross-platform shell issues.

Algorithm:
1. File Discovery: Recursively crawls the project root, skipping build artifacts 
   and version control directories.
2. Content Filtering: Identifies and ignores binary files using null-byte detection.
3. Policy Enforcement:
   - Loads a whitelist of 'excluded' flags allowed to have underscores.
   - Uses regex patterns to find flag definitions (StringVar, Int, etc.).
   - Flag names containing underscores (and not in the whitelist) trigger a failure.

Time Complexity: $O(N \times M)$ where N is the total number of lines in non-binary 
files and M is the number of flag-matching regex patterns.
Space Complexity: $O(F + E)$ where F is the file path list and E is the size 
of the excluded-flags set.
"""

from __future__ import print_function

import argparse
import os
import re
import sys

# Functional Utility: CLI configuration.
# Allows targeted checks on specific files or full-tree sweeps.
parser = argparse.ArgumentParser()
parser.add_argument("filenames", help="list of files to check, all files if unspecified", nargs='*')
args = parser.parse_args()

def is_binary(pathname):
    """
    @brief Heuristic check for binary file types.
    Logic: Samples the file in chunks; if any null byte is encountered, the file 
    is treated as binary to prevent regex mismatch errors.
    @param pathname: Target file path.
    @return: True if the file is likely binary.
    """
    try:
        with open(pathname, 'r') as f:
            CHUNKSIZE = 1024
            while True:
                chunk = f.read(CHUNKSIZE)
                if '\0' in chunk:
                    return True
                if len(chunk) < CHUNKSIZE:
                    break
    except:
        # Block Logic: Error fallback.
        # If the file cannot be read as text, treat it as binary/inaccessible.
        return True

    return False

def get_all_files(rootdir):
    """
    @brief Collects relevant source files for scanning.
    Logic: Performs a directory walk while pruning high-volume/irrelevant 
    directories (vendor, build outputs, git metadata) to optimize performance.
    @param rootdir: Search entry point.
    @return: List of paths to non-binary files.
    """
    all_files = []
    # Block Logic: Tree traversal with pruning.
    # Invariant: Only visits directories not explicitly removed from 'dirs'.
    for root, dirs, files in os.walk(rootdir):
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
        if 'BUILD' in files:
           files.remove('BUILD')

        for name in files:
            pathname = os.path.join(root, name)
            if is_binary(pathname):
                continue
            all_files.append(pathname)
    return all_files

def check_underscore_in_flags(rootdir, files):
    """
    @brief Core validation engine for flag naming conventions.
    Logic: 
    1. Loads the exception list from 'hack/verify-flags/excluded-flags.txt'.
    2. Applies regex patterns to capture flag name arguments from Go source.
    3. Cross-references matches against the exception list and underscore presence.
    @param rootdir: Project root for locating configuration.
    @param files: List of files to analyze.
    """
    # Block Logic: Exception list loading.
    pathname = os.path.join(rootdir, "hack/verify-flags/excluded-flags.txt")
    f = open(pathname, 'r')
    excluded_flags = set(f.read().splitlines())
    f.close()

    # Block Logic: Regex pattern definition.
    # Functional Intent: Targets standard 'flag' package and 'pflag' patterns.
    regexs = [ re.compile('Var[P]?\([^,]*, "([^"]*)"'), # Matches patterns like `Var(
