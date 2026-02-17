#!/bin/sh
# @494c8974-b02e-4988-a478-fb0670e9f3bd/tools/bootconfig/scripts/ftrace.sh
# @brief Manages the Linux ftrace tracing utility, providing functions to clear, enable, disable, and reset various ftrace components.
# This script is designed to bring the ftrace system to a clean, initial state,
# often used for debugging kernel boot processes or setting up specific tracing scenarios.
#
# SPDX-License-Identifier: GPL-2.0-only

clear_trace() {
    # Functional Utility: Resets the content of the trace buffer.
    # This effectively clears all previously recorded trace events.
    echo > trace
}

disable_tracing() {
    # Functional Utility: Stops the ftrace recording mechanism.
    # When tracing is disabled, no new events are written to the trace buffer.
    echo 0 > tracing_on
}

enable_tracing() {
    # Functional Utility: Starts the ftrace recording mechanism.
    # When tracing is enabled, events are written to the trace buffer.
    echo 1 > tracing_on
}

reset_tracer() {
    # Functional Utility: Resets the currently active ftrace tracer to the default 'nop' (no operation) tracer.
    echo nop > current_tracer
}

reset_trigger_file() {
    # Functional Utility: Resets event triggers within specified trigger files.
    # This function is used internally to clear all active triggers for ftrace events.
    # @param $@: List of trigger files to process.
    # Block Logic: Removes 'on' action triggers by prepending '!' to the command and writing to the file.
    grep -H ':on[^:]*(' $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" >> $file
    done
    # Block Logic: Resets all other triggers (excluding comments) by prepending '!' to the command.
    grep -Hv ^# $@ |
    while read line; do
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	file=`echo $line | cut -f1 -d:`
	echo "!$cmd" > $file
    done
}

reset_trigger() {
    # Functional Utility: Resets all current event triggers, including synthetic and regular event triggers.
    if [ -d events/synthetic ]; then
        reset_trigger_file events/synthetic/*/trigger
    fi
    reset_trigger_file events/*/*/trigger
}

reset_events_filter() {
    # Functional Utility: Resets all active filters for ftrace events.
    # This sets all event filters to '0' (disabled), effectively allowing all events to pass if enabled.
    # Block Logic: Iterates through event filter files that are not already set to 'none' and disables them.
    grep -v ^none events/*/*/filter |
    while read line; do
	echo 0 > `echo $line | cut -f1 -d:`
    done
}

reset_ftrace_filter() {
    # Functional Utility: Resets all filters defined in `set_ftrace_filter`.
    # This clears function-specific filters applied to the function tracer.
    # Precondition: `set_ftrace_filter` file must exist.
    if [ ! -f set_ftrace_filter ]; then
      return 0
    fi
    # Block Logic: Clears the existing `set_ftrace_filter` content.
    echo > set_ftrace_filter
    # Block Logic: Iterates through previously set ftrace filters (excluding comments)
    # and disables them by prepending '!' to the filter command.
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

disable_events() {
    # Functional Utility: Disables all ftrace events globally.
    echo 0 > events/enable
}

clear_synthetic_events() {
    # Functional Utility: Clears all currently defined synthetic events.
    grep -v ^# synthetic_events |
    while read line; do
        echo "!$line" >> synthetic_events
    done
}

initialize_ftrace() {
# Functional Utility: Resets the entire ftrace system to an initial, clean state.
# This function is a comprehensive reset, ensuring no tracers, events, triggers,
# or filters are active.
# As the initial state, ftrace will be set to nop tracer,
# no events, no triggers, no filters, no function filters,
# no probes, and tracing on.
    disable_tracing
    reset_tracer
    reset_trigger
    reset_events_filter
    reset_ftrace_filter
    disable_events
    # Block Logic: Clears various ftrace configuration files if they exist.
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
