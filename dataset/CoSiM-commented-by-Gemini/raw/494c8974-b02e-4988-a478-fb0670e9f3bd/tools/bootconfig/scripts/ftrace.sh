#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
#
# @file ftrace.sh
# @brief A shell script to control and reset the Linux kernel's ftrace subsystem.
#
# This script provides a set of utility functions to manage kernel tracing via
# the tracefs interface. It allows for enabling, disabling, and resetting various
# tracing features like tracers, events, triggers, and filters.

# Resets the main trace buffer.
clear_trace() { # reset trace output
    echo > trace
}

# Stops the ftrace recording mechanism.
disable_tracing() { # stop trace recording
    echo 0 > tracing_on
}

# Starts or resumes the ftrace recording mechanism.
enable_tracing() { # start trace recording
    echo 1 > tracing_on
}

# Resets the current tracer to the default 'nop' (no-op) tracer.
reset_tracer() { # reset the current tracer
    echo nop > current_tracer
}

# Helper function to remove triggers from a given trigger file.
# It parses trigger definitions and prepends '!' to them, which is the
# ftrace syntax for removing a trigger.
reset_trigger_file() {
    # Block Logic: Remove multi-action triggers first (e.g., those with conditions).
    grep -H ':on[^:]*(' $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" >> $file
    done
    # Block Logic: Remove single-action triggers.
    grep -Hv ^# $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" > $file
    done
}

# Resets all event triggers currently set in the system.
reset_trigger() { # reset all current setting triggers
    if [ -d events/synthetic ]; then
        reset_trigger_file events/synthetic/*/trigger
    fi
    reset_trigger_file events/*/*/trigger
}

# Resets all event filters to their default (unfiltered) state.
reset_events_filter() { # reset all current setting filters
    # Block Logic: Finds all filter files that are not set to 'none'
    # and writes '0' to them to disable the filter.
    grep -v ^none events/*/*/filter |
    while read line; do
	echo 0 > `echo $line | cut -f1 -d:`
    done
}

# Resets any filters set on function tracing via set_ftrace_filter.
reset_ftrace_filter() { # reset all triggers in set_ftrace_filter
    if [ ! -f set_ftrace_filter ]; then
      return 0
    fi
    # Block Logic: Reads each line from set_ftrace_filter, parses the
    # filter/trigger, and then writes the command to remove it back to the file.
    echo > set_ftrace_filter
    grep -v '^#' set_ftrace_filter | while read t; do
	tr=`echo $t | cut -d: -f2`
	if [ "$tr" = "" ]; then
	    continue
	fi
	if ! grep -q "$t" set_ftrace_filter; then
		continue;
	fi
	name=`echo $t | cut -d: -f1 | cut -d' ' -f1`
	# Parses different trigger formats (e.g., enable_event, disable_event).
	if [ $tr = "enable_event" -o $tr = "disable_event" ]; then
	    tr=`echo $t | cut -d: -f2-4`
	    limit=`echo $t | cut -d: -f5`
	else
	    tr=`echo $t | cut -d: -f2`
	    limit=`echo $t | cut -d: -f3`
	fi
	if [ "$limit" != "unlimited" ]; then
	    tr="$tr:$limit"
	fi
	# Inline: Prepends '!' to the parsed command to signify removal.
	echo "!$name:$tr" > set_ftrace_filter
    done
}

# Disables all trace events globally.
disable_events() {
    echo 0 > events/enable
}

# Clears all defined synthetic events.
clear_synthetic_events() { # reset all current synthetic events
    # Block Logic: Reads the synthetic_events file and prepends '!' to each
    # line to remove the corresponding synthetic event.
    grep -v ^# synthetic_events |
    while read line; do
        echo "!$line" >> synthetic_events
    done
}

# Resets the entire ftrace subsystem to a clean, default state.
initialize_ftrace() { # Reset ftrace to initial-state
# As the initial state, ftrace will be set to nop tracer,
# no events, no triggers, no filters, no function filters,
# no probes, and tracing on.
    disable_tracing
    reset_tracer
    reset_trigger
    reset_events_filter
    reset_ftrace_filter
    disable_events
    # Block Logic: Clear various ftrace control files to remove specific settings.
    [ -f set_event_pid ] && echo > set_event_pid
    [ -f set_ftrace_pid ] && echo > set_ftrace_pid
    [ -f set_ftrace_notrace ] && echo > set_ftrace_notrace
    [ -f set_graph_function ] && echo | tee set_graph_*
    [ -f stack_trace_filter ] && echo > stack_trace_filter
    [ -f kprobe_events ] && echo > kprobe_events
    [ -f uprobe_events ] && echo > uprobe_events
    [ -f synthetic_events ] && echo > synthetic_events
    [ -f snapshot ] && echo 0 > snapshot
    clear_trace
    enable_tracing
}