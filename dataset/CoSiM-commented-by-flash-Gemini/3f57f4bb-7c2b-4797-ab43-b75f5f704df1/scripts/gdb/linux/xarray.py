# SPDX-License-Identifier: GPL-2.0
#
#  Xarray helpers
#
# Copyright (c) 2025 Broadcom
#
# Authors:
#  Florian Fainelli <florian.fainelli@broadcom.com>

"""
GDB helpers for analyzing Linux kernel's xarray data structure.

This module provides functions to interpret the raw `void *` entries within an xarray,
distinguishing between internal tags, zero values, and actual node pointers.
It leverages specific bit patterns in the pointer's lower bits to derive semantic meaning,
crucial for debugging and understanding memory management in the Linux kernel.
"""

import gdb

from linux import utils
from linux import constants

def xa_is_internal(entry):
    """
    Functional Utility: Determines if an xarray entry is an internal tag.

    Internal tags are used by the xarray to store metadata or special states
    rather than actual pointers to data.
    """
    ulong_type = utils.get_ulong_type()
    # Inline: Checks the two least significant bits of the entry.
    # A value of '2' (binary '10') indicates an internal entry.
    return ((entry.cast(ulong_type) & 3) == 2)

def xa_mk_internal(v):
    """
    Functional Utility: Constructs an internal xarray entry from a given value.

    This encodes a specific value into an internal tag format used by the xarray.
    """
    # Inline: Shifts the value left by 2 bits and sets the second least significant bit,
    # effectively creating an internal tag.
    return ((v << 2) | 2)

def xa_is_zero(entry):
    """
    Functional Utility: Checks if an xarray entry represents a 'zero' value.

    This is a specific internal tag used to denote an empty or zero-initialized slot.
    """
    ulong_type = utils.get_ulong_type()
    # Inline: Compares the entry to a predefined internal tag for 'zero'.
    return entry.cast(ulong_type) == xa_mk_internal(257)

def xa_is_node(entry):
    """
    Functional Utility: Determines if an xarray entry is a pointer to an xarray node.

    This distinguishes actual nodes (which are memory addresses) from internal tags.
    """
    ulong_type = utils.get_ulong_type()
    # Inline: An entry is considered a node if it's an internal tag (not a direct pointer)
    # AND its numerical value (when cast to ulong) is greater than 4096.
    # This threshold implicitly filters out small integer internal tags that are not node pointers.
    return xa_is_internal(entry) and (entry.cast(ulong_type) > 4096)
