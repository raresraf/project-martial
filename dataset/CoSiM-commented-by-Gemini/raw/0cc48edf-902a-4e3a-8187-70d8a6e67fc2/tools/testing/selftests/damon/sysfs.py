#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0

"""
This script is a selftest for the DAMON (Data Access Monitor) sysfs interface
in the Linux kernel.

The test operates as follows:
1.  It uses a helper module (`_damon_sysfs`) to construct a default DAMON
    monitoring configuration in Python objects.
2.  It applies this configuration by writing to the appropriate DAMON sysfs files.
3.  It uses the `drgn` debugger to execute a script that inspects the live
    kernel's memory, dumping the internal state of the DAMON kernel thread
    into a JSON file.
4.  It parses the JSON file to get the kernel's actual view of the configuration.
5.  It then asserts that the configuration applied via sysfs matches the internal
    kernel state, verifying that the sysfs interface works correctly.
"""

import json
import os
import subprocess

# Helper module providing a Pythonic API for DAMON's sysfs interface.
import _damon_sysfs


# --- Verification Core ---

def dump_damon_status_dict(pid):
    """
    Dumps the internal status of a running DAMON kernel thread using drgn.

    Args:
        pid (int): The process ID of the kdamond thread to inspect.

    Returns:
        A tuple of (dict, str):
        - The parsed JSON dictionary of the DAMON status on success.
        - An error message string on failure.
    """
    # Check if drgn is installed.
    try:
        subprocess.check_output(['which', 'drgn'], stderr=subprocess.DEVNULL)
    except:
        return None, 'drgn not found'
    
    # Path to the drgn script that knows how to read DAMON's internal state.
    file_dir = os.path.dirname(os.path.abspath(__file__))
    dump_script = os.path.join(file_dir, 'drgn_dump_damon_status.py')
    
    # Execute drgn to dump the status to a file.
    rc = subprocess.call(['drgn', dump_script, str(pid), 'damon_dump_output'],
                         stderr=subprocess.DEVNULL)
    if rc != 0:
        return None, 'drgn execution failed'
    
    # Read and parse the resulting JSON file.
    try:
        with open('damon_dump_output', 'r') as f:
            return json.load(f), None
    except Exception as e:
        return None, 'json.load failed (%s)' % e


# --- Assertion Helpers ---

def fail(expectation, status):
    """Prints a failure message and exits."""
    print('unexpected %s' % expectation)
    print(json.dumps(status, indent=4))
    exit(1)

def assert_true(condition, expectation, status):
    """Asserts that a condition is True, otherwise fails."""
    if condition is not True:
        fail(expectation, status)

def assert_watermarks_committed(watermarks, dump):
    """Asserts that DAMOS watermark settings were correctly committed to the kernel."""
    # This dictionary maps the string representation from sysfs to the internal
    # enum value used by the kernel.
    wmark_metric_val = {'none': 0, 'free_mem_rate': 1}
    assert_true(dump['metric'] == wmark_metric_val[watermarks.metric],
                'metric', dump)
    assert_true(dump['interval'] == watermarks.interval, 'interval', dump)
    assert_true(dump['high'] == watermarks.high, 'high', dump)
    assert_true(dump['mid'] == watermarks.mid, 'mid', dump)
    assert_true(dump['low'] == watermarks.low, 'low', dump)

# ... (other assert_*_committed functions follow the same pattern) ...
# These functions recursively verify that the Python object representation of a
# configuration matches the dictionary representation dumped from the kernel.

def assert_quota_committed(quota, dump):
    """Asserts that DAMOS quota settings were correctly committed."""
    assert_true(dump['reset_interval'] == quota.reset_interval_ms,
                'reset_interval', dump)
    assert_true(dump['ms'] == quota.ms, 'ms', dump)
    assert_true(dump['sz'] == quota.sz, 'sz', dump)
    # Recursively check nested quota goals.
    # ...

def assert_scheme_committed(scheme, dump):
    """Asserts that a full DAMOS scheme was correctly committed."""
    # Recursively checks all components of a scheme.
    assert_access_pattern_committed(scheme.access_pattern, dump['pattern'])
    action_val = {
            'willneed': 0, 'cold': 1, 'pageout': 2, 'hugepage': 3,
            'nohugeapge': 4, 'lru_prio': 5, 'lru_deprio': 6,
            'migrate_hot': 7, 'migrate_cold': 8, 'stat': 9,
    }
    assert_true(dump['action'] == action_val[scheme.action], 'action', dump)
    # ... and so on for all scheme attributes.
    assert_watermarks_committed(scheme.watermarks, dump['wmarks'])
    # ...

def assert_schemes_committed(schemes, dump):
    """Asserts that a list of DAMOS schemes was correctly committed."""
    assert_true(len(schemes) == len(dump), 'len_schemes', dump)
    for idx, scheme in enumerate(schemes):
        assert_scheme_committed(scheme, dump[idx])


# --- Main Test Execution ---

def main():
    """The main entry point for the selftest."""
    # 1. Define the desired DAMON configuration using the helper classes.
    #    This creates a default kdamond with one context and one default scheme.
    kdamonds = _damon_sysfs.Kdamonds(
            [_damon_sysfs.Kdamond(
                contexts=[_damon_sysfs.DamonCtx(
                    targets=[_damon_sysfs.DamonTarget(pid=-1)], # Monitor all memory
                    schemes=[_damon_sysfs.Damos()], # Default scheme
                    )])])
    
    # 2. Apply the configuration by writing to sysfs.
    err = kdamonds.start()
    if err is not None:
        print('kdamond start failed: %s' % err)
        exit(1)

    # 3. Read back the internal kernel state using drgn.
    status, err = dump_damon_status_dict(kdamonds.kdamonds[0].pid)
    if err is not None:
        print(err)
        kdamonds.stop()
        exit(1)

    # 4. Assert that the internal state matches the expected defaults.
    if len(status['contexts']) != 1:
        fail('number of contexts', status)

    ctx = status['contexts'][0]
    attrs = ctx['attrs']
    if attrs['sample_interval'] != 5000:
        fail('sample interval', status)
    if attrs['aggr_interval'] != 100000:
        fail('aggr interval', status)
    # ... check other default attributes ...
    
    # Verify the entire default scheme was committed correctly.
    assert_schemes_committed([_damon_sysfs.Damos()], ctx['schemes'])

    # 5. Clean up by stopping the DAMON thread.
    kdamonds.stop()

if __name__ == '__main__':
    main()