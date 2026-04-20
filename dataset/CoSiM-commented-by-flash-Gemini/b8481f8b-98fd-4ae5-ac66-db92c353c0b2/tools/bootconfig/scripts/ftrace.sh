#!/bin/sh
# @b8481f8b-98fd-4ae5-ac66-db92c353c0b2/tools/bootconfig/scripts/ftrace.sh
# @brief Utility script for lifecycle management and state restoration of the Linux ftrace subsystem.
#
# Functional Intent: Provides a suite of functions to purge, disable, and reset 
# ftrace components (tracers, events, triggers, filters). It is primarily used 
# during kernel boot sequence debugging and in automated test harnesses to 
# ensure a deterministic baseline tracing state.
#
# Domain: Linux Kernel Tracing, System Observability, Boot Configuration.
#
# SPDX-License-Identifier: GPL-2.0-only

# Functional Utility: Purges the primary trace buffer.
clear_trace() {
    echo > trace
}

# Functional Utility: Globally suspends ftrace activity.
disable_tracing() {
    echo 0 > tracing_on
}

# Functional Utility: Resumes global ftrace recording.
enable_tracing() {
    echo 1 > tracing_on
}

# Functional Utility: Reverts the active tracer to 'nop', effectively disabling function/graph tracking.
reset_tracer() {
    echo nop > current_tracer
}

# Block Logic: Internal helper for dismantling event triggers.
# Logic: Identifies active triggers and issues the negation ('!') command to the 
# respective ftrace control files to safely unregister them.
reset_trigger_file() {
    # Block Logic: Handling of conditional 'on' triggers.
    grep -H ':on[^:]*(' $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" >> $file
    done
    # Block Logic: Handling of generic event triggers.
    grep -Hv ^# $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" > $file
    done
}

# Functional Utility: Aggregates trigger reset logic for all event types.
reset_trigger() {
    if [ -d events/synthetic ]; then
        reset_trigger_file events/synthetic/*/trigger
    fi
    reset_trigger_file events/*/*/trigger
}

# Block Logic: Resets filtering rules for dynamic events.
# Logic: Iterates through all subsystem filters and clears active predicates.
reset_events_filter() {
    grep -v ^none events/*/*/filter |
    while read line; do
	echo 0 > `echo $line | cut -f1 -d:`
    done
}

# Block Logic: Resets function-level tracer filters.
# Pre-condition: 'set_ftrace_filter' must be available in the tracefs hierarchy.
# Logic: Clears the filter list and negation-disables all complex function triggers.
reset_ftrace_filter() {
    if [ ! -f set_ftrace_filter ]; then
      return 0
    fi
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

# Functional Utility: Globally disables all event groups.
disable_events() {
    echo 0 > events/enable
}

# Functional Utility: Dismantles all user-defined synthetic events.
clear_synthetic_events() {
    grep -v ^# synthetic_events |
    while read line; do
        echo "!$line" >> synthetic_events
    done
}

# Block Logic: Orchestrates a comprehensive system-wide ftrace reset.
# Invariant: Upon completion, ftrace is in a clean state with 'nop' tracer 
# and active recording, but with zero events or filters enabled.
initialize_ftrace() {
    disable_tracing
    reset_tracer
    reset_trigger
    reset_events_filter
    reset_ftrace_filter
    disable_events
    
    # Block Logic: Purge persistent configuration state.
    # Logic: Sequentially clears process-level filters, probes, and snapshots.
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
