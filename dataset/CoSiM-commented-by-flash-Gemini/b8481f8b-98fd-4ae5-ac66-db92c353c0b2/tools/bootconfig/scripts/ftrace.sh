# SPDX-License-Identifier: GPL-2.0-only
#
# @file ftrace.sh
# @brief Utility script providing functions to manage and reset ftrace (function tracer) in the Linux kernel.
#
# This script defines a set of bash functions to control various aspects of
# the ftrace debugging and tracing infrastructure. It allows for clearing
# trace buffers, enabling/disabling tracing, resetting tracers, managing
# event triggers and filters, and resetting the entire ftrace system to
# a clean, initial state. These functions are typically used for kernel
# development, debugging, and performance analysis.

# @brief Clears the main ftrace trace buffer.
# Functional Utility: Empties the trace buffer, removing all recorded trace events.
clear_trace() { # reset trace output
    echo > trace
}

# @brief Disables ftrace recording.
# Functional Utility: Stops the ftrace mechanism from capturing new trace events.
disable_tracing() { # stop trace recording
    echo 0 > tracing_on
}

# @brief Enables ftrace recording.
# Functional Utility: Starts the ftrace mechanism to capture trace events.
enable_tracing() { # start trace recording
    echo 1 > tracing_on
}

# @brief Resets the currently active ftrace tracer to 'nop' (no operation).
# Functional Utility: Disables any specific tracer (e.g., function, sched) and sets
# it to a pass-through mode, effectively stopping complex tracing logic.
reset_tracer() { # reset the current tracer
    echo nop > current_tracer
}

# @brief Resets event trigger files by removing existing actions.
#
# This function identifies and disables active triggers within ftrace event files.
# It handles both `on` actions and other actions by prepending '!' to disable them.
# @param $@ List of trigger files to process.
reset_trigger_file() {
    # Block Logic: First, remove 'on' action triggers, which have a specific syntax.
    grep -H ':on[^:]*(' $@ |
    while read line; do
        # Inline: Extract the command part of the trigger.
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	# Inline: Extract the filename part of the trigger.
	file=`echo $line | cut -f1 -d:`
	# Functional Utility: Prepend '!' to the command to disable the trigger.
	echo "!$cmd" >> $file
    done
    # Block Logic: Next, remove other types of triggers (excluding comments).
    grep -Hv ^# $@ |
    while read line; do
        # Inline: Extract the command part of the trigger.
        cmd=`echo $line | cut -f2- -d: | cut -f1 -d"["`
	# Inline: Extract the filename part of the trigger.
	file=`echo $line | cut -f1 -d:`
	# Functional Utility: Prepend '!' to the command to disable the trigger.
	echo "!$cmd" > $file
    done
}

# @brief Resets all active ftrace event triggers.
# Functional Utility: Iterates through synthetic and regular event trigger files
# and calls `reset_trigger_file` to disable all configured triggers.
reset_trigger() { # reset all current setting triggers
    # Block Logic: Resets synthetic event triggers if the directory exists.
    if [ -d events/synthetic ]; then
        reset_trigger_file events/synthetic/*/trigger
    fi
    # Block Logic: Resets regular event triggers.
    reset_trigger_file events/*/*/trigger
}

# @brief Resets all ftrace event filters.
# Functional Utility: Clears any active filters applied to ftrace events,
# allowing all events to be recorded (if enabled).
reset_events_filter() { # reset all current setting filters
    # Block Logic: Identifies active filters by grepping for lines not starting with 'none'
    # and writes '0' to their respective filter files to clear them.
    grep -v ^none events/*/*/filter |
    while read line; do
	# Inline: Extracts the filename part of the filter.
	echo 0 > `echo $line | cut -f1 -d:`
    done
}

# @brief Resets all ftrace function filters.
# Functional Utility: Clears any functions explicitly added or removed from
# tracing via `set_ftrace_filter`.
reset_ftrace_filter() { # reset all triggers in set_ftrace_filter
    # Block Logic: Checks if set_ftrace_filter file exists.
    if [ ! -f set_ftrace_filter ]; then
      return 0
    fi
    # Functional Utility: Clears the primary set_ftrace_filter file.
    echo > set_ftrace_filter
    # Block Logic: Iterates through entries in set_ftrace_filter (excluding comments)
    # and disables them by prepending '!'. This complex logic attempts to parse
    # various trigger formats and disable them.
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

# @brief Disables all ftrace events.
# Functional Utility: Turns off the collection of all trace events.
disable_events() {
    echo 0 > events/enable
}

# @brief Resets all current synthetic ftrace events.
# Functional Utility: Removes dynamically created synthetic events from ftrace.
clear_synthetic_events() { # reset all current synthetic events
    # Block Logic: Reads synthetic event definitions and disables them by writing '!' prefixed lines.
    grep -v ^# synthetic_events |
    while read line; do
        echo "!$line" >> synthetic_events
    done
}

# @brief Resets the entire ftrace system to its initial default state.
#
# This function orchestrates a sequence of calls to other reset functions
# to bring ftrace into a known, clean state. This is crucial for consistent
# debugging and tracing setups.
initialize_ftrace() { # Reset ftrace to initial-state
# Functional Utility: Ensures ftrace starts from a clean slate.
# As the initial state, ftrace will be set to nop tracer,
# no events, no triggers, no filters, no function filters,
# no probes, and tracing on.
    disable_tracing
    reset_tracer
    reset_trigger
    reset_events_filter
    reset_ftrace_filter
    disable_events
    # Block Logic: Resets various ftrace configuration files by emptying them or setting default values.
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