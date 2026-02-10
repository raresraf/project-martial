#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
#
# ftrace.sh - A shell script to reset the ftrace tracing subsystem.
#
# This script provides a collection of functions to reset various components
# of the Linux kernel's ftrace infrastructure to a clean, default state.
# It operates by writing to the control files located within the tracefs
# filesystem (usually mounted at /sys/kernel/tracing).

# Clears the main ftrace ring buffer.
clear_trace() { # reset trace output
    echo > trace
}

# Deactivates the recording of trace data globally.
disable_tracing() { # stop trace recording
    echo 0 > tracing_on
}

# Activates the recording of trace data globally.
enable_tracing() { # start trace recording
    echo 1 > tracing_on
}

# Resets the active tracer to 'nop' (no-op), disabling function tracing.
reset_tracer() { # reset the current tracer
    echo nop > current_tracer
}

# Helper function to parse a trigger file and disable all active triggers.
# It prepends '!' to each trigger command to remove it.
reset_trigger_file() {
    # Handles multi-action triggers (e.g., those with parentheses).
    grep -H ':on[^:]*(' $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" >> $file
    done
    # Handles single-action triggers.
    grep -Hv ^# $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" > $file
    done
}

# Disables all currently configured event triggers.
reset_trigger() { # reset all current setting triggers
    # First, reset triggers for any custom synthetic events.
    if [ -d events/synthetic ]; then
        reset_trigger_file events/synthetic/*/trigger
    fi
    # Then, reset triggers for all standard kernel events.
    reset_trigger_file events/*/*/trigger
}

# Removes all filters from all trace events.
reset_events_filter() { # reset all current setting filters
    # Finds all event filter files that are not empty ('none')
    # and resets them by writing '0'.
    grep -v ^none events/*/*/filter |
    while read line; do
	echo 0 > `echo $line | cut -f1 -d:`
    done
}

# Resets the filters for the function tracer.
reset_ftrace_filter() { # reset all triggers in set_ftrace_filter
    if [ ! -f set_ftrace_filter ]; then
      return 0
    fi
    # This complex parsing logic surgically removes each filter entry.
    # A simpler `echo > set_ftrace_filter` would also work but be less precise.
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
	echo "!$name:$tr" > set_ftrace_filter
    done
}

# Globally disables all tracepoint events from being recorded.
disable_events() {
    echo 0 > events/enable
}

# Removes all previously defined synthetic events.
clear_synthetic_events() { # reset all current synthetic events
    grep -v ^# synthetic_events |
    while read line; do
        # Prepends '!' to each line to unregister the event.
        echo "!$line" >> synthetic_events
    done
}

# The main function to completely reset the ftrace subsystem.
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
    # Clear PID filters for event and function tracing.
    [ -f set_event_pid ] && echo > set_event_pid
    [ -f set_ftrace_pid ] && echo > set_ftrace_pid
    # Clear function filters.
    [ -f set_ftrace_notrace ] && echo > set_ftrace_notrace
    [ -f set_graph_function ] && echo | tee set_graph_*
    # Clear other filters and dynamic probes.
    [ -f stack_trace_filter ] && echo > stack_trace_filter
    [ -f kprobe_events ] && echo > kprobe_events
    [ -f uprobe_events ] && echo > uprobe_events
    [ -f synthetic_events ] && echo > synthetic_events
    # Clear the snapshot buffer.
    [ -f snapshot ] && echo 0 > snapshot
    clear_trace
    enable_tracing
}