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
This script is a linting tool used in CI to enforce a naming convention for
command-line flags in Go source code. It scans the codebase for flag
declarations and ensures that flag names use dashes (-) instead of
underscores (_), which is a common convention in Kubernetes.
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
    Uses a heuristic to determine if a file is binary by checking for null bytes.
    This prevents the script from attempting to run regex on compiled binaries,
    images, or other non-text files.

    @raise EnvironmentError: if the file does not exist or cannot be accessed.
    @attention: found @ http://bytes.com/topic/python/answers/21222-determine-file-type-binary-text on 6/08/2010
    @author: Trent Mick <TrentM@ActiveState.com>
    @author: Jorge Orpinel <jorge@orpinel.com>
    """
    try:
        with open(pathname, 'r') as f:
            CHUNKSIZE = 1024
            while 1:
                chunk = f.read(CHUNKSIZE)
                if '\0' in chunk: # found null byte
                    return True
                if len(chunk) < CHUNKSIZE:
                    break # done
    except:
        return True
    return False

def get_all_files(rootdir):
    """
    Walks a directory tree and returns a list of all non-binary files.
    It prunes the search by skipping common directories that do not contain
    source code relevant to this check.
    """
    all_files = []
    for root, dirs, files in os.walk(rootdir):
        # Prune the search space by removing directories that contain vendored
        # code, build artifacts, or other non-source directories.
        for d in ['vendor', 'staging', '_output', '_gopath', 'third_party', '.git', '.make']:
            if d in dirs:
                dirs.remove(d)
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
    Collects all the flags used in golang files and verifies that they do
    not contain underscores, unless they are explicitly excluded.
    """
    # Load the set of flags that are known exceptions to the no-underscore rule.
    pathname = os.path.join(rootdir, "hack/verify-flags/excluded-flags.txt")
    with open(pathname, 'r') as f:
        excluded_flags = set(f.read().splitlines())

    # These regexes are designed to find flag declarations from the popular Go
    # `pflag` and `flag` libraries by matching function calls like `StringVarP`,
    # `Int`, `Bool`, etc., and capturing the flag name string literal.
    regexs = [
        re.compile('Var[P]?\([^,]*, "([^"]*)"'),           # Catches pflag.Var, pflag.VarP
        re.compile('.String[P]?\("([^"]*)",[^,]+,[^)]+\)'),  # Catches .String, .StringP
        re.compile('.Int[P]?\("([^"]*)",[^,]+,[^)]+\)'),      # Catches .Int, .IntP
        re.compile('.Bool[P]?\("([^"]*)",[^,]+,[^)]+\)'),      # Catches .Bool, .BoolP
        re.compile('.Duration[P]?\("([^"]*)",[^,]+,[^)]+\)'), # Catches .Duration, .DurationP
        re.compile('.StringSlice[P]?\("([^"]*)",[^,]+,[^)]+\)') # Catches .StringSlice, .StringSliceP
    ]

    new_excluded_flags = set()
    # Block: Process all relevant files.
    for pathname in files:
        if not pathname.endswith(".go"):
            continue
        try:
            with open(pathname, 'r') as f:
                data = f.read()
        except Exception as e:
            print("Error opening %s: %s" % (pathname, e), file=sys.stderr)
            continue

        # Block: Apply all regexes to find flag declarations in the current file.
        matches = []
        for regex in regexs:
            matches.extend(regex.findall(data))

        # Block: Validate each flag found in the file.
        for flag in matches:
            # Ignore flags that are already in the exclusion list.
            if any(x in flag for x in excluded_flags):
                continue
            # If a flag contains an underscore, it's a violation.
            if "_" in flag:
                new_excluded_flags.add(flag)

    # If any violating flags were found, print them and exit with an error code.
    if len(new_excluded_flags) > 0:
        print("Found a flag declared with an _ but which is not explicitly listed as a valid flag name in hack/verify-flags/excluded-flags.txt", file=sys.stderr)
        print("Are you certain this flag should not have been declared with a - instead?", file=sys.stderr)
        l = sorted(list(new_excluded_flags))
        print("%s" % "\n".join(l), file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function: determines which files to check and runs the verification.
    """
    # Assume the script is in a 'hack' directory and find the repository root.
    rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # If filenames are provided as arguments, check only those.
    # Otherwise, scan the entire repository.
    if len(args.filenames) > 0:
        files = args.filenames
    else:
        files = get_all_files(rootdir)

    check_underscore_in_flags(rootdir, files)
    return 0

if __name__ == "__main__":
  sys.exit(main())
