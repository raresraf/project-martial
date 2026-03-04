#
# gdb helper commands and functions for Linux kernel debugging
#
#  VFS tools
#
# Copyright (c) 2023 Glenn Washburn
# Copyright (c) 2016 Linaro Ltd
#
# Authors:
#  Glenn Washburn <development@efficientek.com>
#  Kieran Bingham <kieran.bingham@linaro.org>
#
# This work is licensed under the terms of the GNU GPL version 2.
#

"""
GDB helpers for analyzing Linux kernel Virtual File System (VFS) structures.

This module provides utility functions and GDB commands to extract and
interpret information about `dentry` and `inode` structures, facilitating
debugging of file system-related issues within the Linux kernel.
"""

import gdb
from linux import utils


def dentry_name(d):
    """
    Functional Utility: Recursively constructs the full path of a given dentry.

    Args:
        d: A GDB value representing a 'struct dentry'.

    Returns:
        A string representing the full path of the dentry.
    """
    parent = d['d_parent']
    # Block Logic: Base case for recursion: If the dentry is its own parent or null, it's the root or invalid.
    # Invariant: 'd' represents a valid 'struct dentry'.
    if parent == d or parent == 0:
        return ""
    p = dentry_name(d['d_parent']) + "/"
    return p + d['d_shortname']['string'].string()

class DentryName(gdb.Function):
    """
    Functional Utility: GDB convenience function to return the full path of a dentry.

    This class registers a GDB command `$lx_dentry_name(PTR)` which, given a pointer
    to a `dentry` structure, returns its absolute path as a string.
    """

    def __init__(self):
        super(DentryName, self).__init__("lx_dentry_name")

    def invoke(self, dentry_ptr):
        """
        Functional Utility: Invokes the `dentry_name` helper function with the provided dentry pointer.
        """
        return dentry_name(dentry_ptr)

DentryName()


dentry_type = utils.CachedType("struct dentry")

class InodeDentry(gdb.Function):
    """
    Functional Utility: GDB convenience function to retrieve the dentry associated with an inode.

    This class registers a GDB command `$lx_i_dentry(PTR)` which, given a pointer
    to an `inode` structure, attempts to find and return a pointer to its associated `dentry` structure.
    """

    def __init__(self):
        super(InodeDentry, self).__init__("lx_i_dentry")

    def invoke(self, inode_ptr):
        """
        Functional Utility: Retrieves the dentry pointer from an inode structure.

        Args:
            inode_ptr: A GDB value representing a 'struct inode'.

        Returns:
            A GDB value representing the associated 'struct dentry *', or an empty string if not found.
        """
        d_u = inode_ptr["i_dentry"]["first"]
        # Block Logic: Checks if the dentry union is populated.
        # Pre-condition: 'inode_ptr' points to a valid 'struct inode'.
        if d_u == 0:
            return ""
        return utils.container_of(d_u, dentry_type.get_type().pointer(), "d_u")

InodeDentry()
