/**
 * @file trace_eprobe.c
 * @brief Implements event probes (eprobes) for the ftrace subsystem in the Linux kernel.
 *
 * Eprobes allow the dynamic creation of new trace events based on existing
 * kernel events. This provides a flexible mechanism to observe and analyze
 * specific data fields of existing trace events, apply filters, and
 * dynamically register/unregister custom tracepoints.
 *
 * Functional Utility:
 * - Dynamically creates custom trace events from existing ftrace events.
 * - Allows extraction and re-expression of data fields from the source event.
 * - Supports filtering of events based on complex conditions.
 * - Integrates with `trace_probe` infrastructure for argument parsing and event management.
 * - Leverages event triggers to tap into the lifecycle of existing trace events.
 *
 * Architectural Intent:
 * - To extend the capabilities of ftrace by allowing users to define
 *   new event types on-the-fly without kernel recompilation.
 * - To provide a powerful debugging and analysis tool for dynamic introspection
 *   of kernel events.
 *
 * Part of this code was copied from kernel/trace/trace_kprobe.c written by
 * Masami Hiramatsu <mhiramat@kernel.org>
 *
 * Copyright (C) 2021, VMware Inc, Steven Rostedt <rostedt@goodmis.org>
 * Copyright (C) 2021, VMware Inc, Tzvetomir Stoyanov tz.stoyanov@gmail.com>
 *
 */
// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/ftrace.h>

#include "trace_dynevent.h"
#include "trace_probe.h"
#include "trace_probe_tmpl.h"
#include "trace_probe_kernel.h"

/**
 * @def EPROBE_EVENT_SYSTEM
 * @brief The default event system name for eprobes.
 *
 * Eprobes are grouped under this system name in the tracefs hierarchy.
 */
#define EPROBE_EVENT_SYSTEM "eprobes"

/**
 * @struct trace_eprobe
 * @brief Represents an event probe (eprobe) instance.
 *
 * This structure holds all necessary information for an eprobe, including
 * its target event, filter string, and underlying `trace_probe` and
 * `dyn_event` structures.
 */
struct trace_eprobe {
	const char *event_system;	/**< @brief The system name of the target tracepoint. */
	const char *event_name;		/**< @brief The name of the target tracepoint. */
	char *filter_str;		/**< @brief Filter string applied to the target event. */
	struct trace_event_call *event;	/**< @brief Pointer to the target `trace_event_call`. */
	struct dyn_event	devent;	/**< @brief Dynamic event base structure. */
	struct trace_probe	tp;		/**< @brief Trace probe base structure. */
};

/**
 * @struct eprobe_data
 * @brief Private data structure passed to eprobe event triggers.
 *
 * This structure links an eprobe trigger back to its `trace_eprobe` instance
 * and the `trace_event_file` it's associated with.
 */
struct eprobe_data {
	struct trace_event_file	*file;	/**< @brief The trace event file this eprobe is attached to. */
	struct trace_eprobe	*ep;	/**< @brief Pointer to the `trace_eprobe` instance. */
};

/**
 * @def for_each_trace_eprobe_tp(ep, _tp)
 * @brief Macro to iterate through `trace_eprobe` instances associated with a `trace_probe`.
 * @param ep The `trace_eprobe` pointer for the current iteration.
 * @param _tp The `trace_probe` whose associated eprobes are to be iterated.
 */
#define for_each_trace_eprobe_tp(ep, _tp) \
	list_for_each_entry(ep, trace_probe_probe_list(_tp), tp.list)

/**
 * @brief Creates a new event probe based on command-line arguments.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @return 0 on success, or a negative errno on failure.
 */
static int __trace_eprobe_create(int argc, const char *argv[]);

/**
 * @brief Cleans up resources allocated for a `trace_eprobe` instance.
 * @param ep Pointer to the `trace_eprobe` instance.
 *
 * This function frees memory associated with the eprobe's event name,
 * event system, filter string, and the eprobe itself.
 */
static void trace_event_probe_cleanup(struct trace_eprobe *ep)
{
	if (!ep)
		return;
	trace_probe_cleanup(&ep->tp);
	kfree(ep->event_name);
	kfree(ep->event_system);
	if (ep->event)
		trace_event_put_ref(ep->event);
	kfree(ep->filter_str);
	kfree(ep);
}

/**
 * @brief Casts a `dyn_event` pointer to a `trace_eprobe` pointer.
 * @param ev Pointer to the `dyn_event` instance.
 * @return Pointer to the embedded `trace_eprobe` instance.
 */
static struct trace_eprobe *to_trace_eprobe(struct dyn_event *ev)
{
	return container_of(ev, struct trace_eprobe, devent);
}

/**
 * @brief Creates a dynamic event for an eprobe from a raw command string.
 * @param raw_command The raw command string to parse.
 * @return 0 on success, or a negative errno on failure.
 */
static int eprobe_dyn_event_create(const char *raw_command)
{
	return trace_probe_create(raw_command, __trace_eprobe_create);
}

/**
 * @brief Shows the details of an eprobe dynamic event in a `seq_file`.
 * @param m Pointer to the `seq_file` for output.
 * @param ev Pointer to the `dyn_event` instance.
 * @return 0 on success.
 */
static int eprobe_dyn_event_show(struct seq_file *m, struct dyn_event *ev)
{
	struct trace_eprobe *ep = to_trace_eprobe(ev);
	int i;

	// Functional Utility: Prints the eprobe's group and event name.
	seq_printf(m, "e:%s/%s", trace_probe_group_name(&ep->tp),
				trace_probe_name(&ep->tp));
	// Functional Utility: Prints the target event's system and name.
	seq_printf(m, " %s.%s", ep->event_system, ep->event_name);

	// Block Logic: Prints each argument of the eprobe.
	for (i = 0; i < ep->tp.nr_args; i++)
		seq_printf(m, " %s=%s", ep->tp.args[i].name, ep->tp.args[i].comm);
	seq_putc(m, '\n');

	return 0;
}

/**
 * @brief Unregisters a `trace_eprobe` instance.
 * @param ep Pointer to the `trace_eprobe` instance.
 * @return 0 on success, -EBUSY if the eprobe is enabled or busy.
 *
 * This function removes the eprobe from the dynamic event list and
 * unlinks its `trace_probe` structure.
 */
static int unregister_trace_eprobe(struct trace_eprobe *ep)
{
	/* If other probes are on the event, just unregister eprobe */
	// Block Logic: Checks if other probes are using the same target event.
	if (trace_probe_has_sibling(&ep->tp))
		goto unreg;

	/* Enabled event can not be unregistered */
	// Block Logic: Returns busy if the probe is currently enabled.
	if (trace_probe_is_enabled(&ep->tp))
		return -EBUSY;

	/* Will fail if probe is being used by ftrace or perf */
	// Block Logic: Returns busy if the trace event call is still in use.
	if (trace_probe_unregister_event_call(&ep->tp))
		return -EBUSY;

unreg:
	dyn_event_remove(&ep->devent); // Functional Utility: Removes the eprobe from dynamic event management.
	trace_probe_unlink(&ep->tp); // Functional Utility: Unlinks the trace probe.

	return 0;
}

/**
 * @brief Releases resources associated with an eprobe dynamic event.
 * @param ev Pointer to the `dyn_event` instance.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function calls `unregister_trace_eprobe` and `trace_event_probe_cleanup`
 * to free all eprobe-related resources.
 */
static int eprobe_dyn_event_release(struct dyn_event *ev)
{
	struct trace_eprobe *ep = to_trace_eprobe(ev);
	int ret = unregister_trace_eprobe(ep);

	// Block Logic: Cleans up only if unregistration is successful.
	if (!ret)
		trace_event_probe_cleanup(ep);
	return ret;
}

/**
 * @brief Checks if an eprobe dynamic event is currently busy (enabled).
 * @param ev Pointer to the `dyn_event` instance.
 * @return True if the eprobe is enabled, false otherwise.
 */
static bool eprobe_dyn_event_is_busy(struct dyn_event *ev)
{
	struct trace_eprobe *ep = to_trace_eprobe(ev);

	return trace_probe_is_enabled(&ep->tp);
}

/**
 * @brief Matches an eprobe dynamic event against a given system, event name, and arguments.
 * @param system The system name to match.
 * @param event The event name to match.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param ev Pointer to the `dyn_event` instance.
 * @return True if the eprobe matches the criteria, false otherwise.
 *
 * This function implements the matching logic for eprobes, allowing flexible
 * identification of eprobes based on their target event and user-defined arguments.
 */
static bool eprobe_dyn_event_match(const char *system, const char *event,
			int argc, const char **argv, struct dyn_event *ev)
{
	struct trace_eprobe *ep = to_trace_eprobe(ev);
	const char *slash;

	/*
	 * We match the following:
	 *  event only		- match all eprobes with event name
	 *  system and event only	- match all system/event probes
	 *  system only		- match all system probes
	 *
	 * The below has the above satisfied with more arguments:
	 *
	 *  attached system/event	- If the arg has the system and event
	 *			  the probe is attached to, match
	 *			  probes with the attachment.
	 *
	 *  If any more args are given, then it requires a full match.
	 */

	/*
	 * If system exists, but this probe is not part of that system
	 * do not match.
	 */
	// Block Logic: If a system is specified, checks if the eprobe's group name matches.
	if (system && strcmp(trace_probe_group_name(&ep->tp), system) != 0)
		return false;

	/* Must match the event name */
	// Block Logic: If an event is specified, checks if the eprobe's name matches.
	if (event[0] != '\0' && strcmp(trace_probe_name(&ep->tp), event) != 0)
		return false;

	/* No arguments match all */
	// Block Logic: If no arguments are provided, it's considered a match.
	if (argc < 1)
		return true;

	/* First argument is the system/event the probe is attached to */

	slash = strchr(argv[0], '/');
	if (!slash)
		slash = strchr(argv[0], '.');
	// Block Logic: Ensures the first argument specifies a system/event in "system.event" or "system/event" format.
	if (!slash)
		return false;

	// Block Logic: Compares the system name and event name with the attached event.
	if (strncmp(ep->event_system, argv[0], slash - argv[0]))
		return false;
	if (strcmp(ep->event_name, slash + 1))
		return false;

	argc--;
	argv++;

	/* If there are no other args, then match */
	// Block Logic: If no more arguments, it's a match.
	if (argc < 1)
		return true;

	// Functional Utility: Matches against additional command arguments.
	return trace_probe_match_command_args(&ep->tp, argc, argv);
}

/**
 * @var eprobe_dyn_event_ops
 * @brief Dynamic event operations for eprobes.
 *
 * This structure defines the callbacks for creating, showing, freeing,
 * and matching eprobe dynamic events.
 */
static struct dyn_event_operations eprobe_dyn_event_ops = {
	.create = eprobe_dyn_event_create,
	.show = eprobe_dyn_event_show,
	.is_busy = eprobe_dyn_event_is_busy,
	.free = eprobe_dyn_event_release,
	.match = eprobe_dyn_event_match,
};

/**
 * @brief Allocates and initializes a new `trace_eprobe` instance.
 * @param group The group name for the eprobe.
 * @param this_event The name for this new eprobe event.
 * @param event The target `trace_event_call` to probe.
 * @param nargs Number of arguments for the eprobe.
 * @return Pointer to the newly allocated `trace_eprobe` on success, or an `ERR_PTR` on failure.
 *
 * This function performs memory allocation and basic initialization
 * for an eprobe, including duplicating relevant strings and initializing
 * its embedded `trace_probe` and `dyn_event` structures.
 */
static struct trace_eprobe *alloc_event_probe(const char *group,
					  const char *this_event,
					  struct trace_event_call *event,
					  int nargs)
{
	struct trace_eprobe *ep;
	const char *event_name;
	const char *sys_name;
	int ret = -ENOMEM;

	// Block Logic: Returns error if target event is invalid.
	if (!event)
		return ERR_PTR(-ENODEV);

	sys_name = event->class->system;
	event_name = trace_event_name(event);

	// Functional Utility: Allocates memory for the eprobe structure.
	ep = kzalloc(struct_size(ep, tp.args, nargs), GFP_KERNEL);
	// Block Logic: Handles memory allocation failure.
	if (!ep) {
		trace_event_put_ref(event); // Functional Utility: Releases reference to the event on failure.
		goto error;
	}
	ep->event = event;
	ep->event_name = kstrdup(event_name, GFP_KERNEL); // Functional Utility: Duplicates event name.
	if (!ep->event_name)
		goto error;
	ep->event_system = kstrdup(sys_name, GFP_KERNEL); // Functional Utility: Duplicates system name.
	if (!ep->event_system)
		goto error;

	ret = trace_probe_init(&ep->tp, this_event, group, false, nargs); // Functional Utility: Initializes the embedded trace probe.
	if (ret < 0)
		goto error;

	dyn_event_init(&ep->devent, &eprobe_dyn_event_ops); // Functional Utility: Initializes the embedded dynamic event.
	return ep;
error:
	trace_event_probe_cleanup(ep); // Functional Utility: Cleans up allocated resources on error.
	return ERR_PTR(ret);
}

/**
 * @brief Defines the fields for an eprobe event call.
 * @param event_call Pointer to the `trace_event_call` for the eprobe.
 * @return 0 on success, -ENOENT if `trace_probe` is not found.
 *
 * This function uses `traceprobe_define_arg_fields` to set up the
 * arguments of the eprobe event based on its `trace_probe` definition.
 */
static int eprobe_event_define_fields(struct trace_event_call *event_call)
{
	struct eprobe_trace_entry_head field;
	struct trace_probe *tp;

	tp = trace_probe_primary_from_call(event_call);
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		return -ENOENT;

	return traceprobe_define_arg_fields(event_call, sizeof(field), tp);
}

/**
 * @var eprobe_fields_array
 * @brief Array of `trace_event_fields` for eprobes.
 *
 * Defines how eprobe event fields are defined, primarily using
 * `eprobe_event_define_fields`.
 */
static struct trace_event_fields eprobe_fields_array[] = {
	{ .type = TRACE_FUNCTION_TYPE,
	  .define_fields = eprobe_event_define_fields },
	{}
};

/* Event entry printers */
/**
 * @brief Printer function for eprobe events in `trace_seq`.
 * @param iter Pointer to the `trace_iterator`.
 * @param flags Print flags.
 * @param event Pointer to the `trace_event`.
 * @return `print_line_t` status.
 *
 * This function formats and prints the eprobe event data, including
 * the eprobe's name, its target event, and its arguments.
 */
static enum print_line_t
print_eprobe_event(struct trace_iterator *iter, int flags,
		   struct trace_event *event)
{
	struct eprobe_trace_entry_head *field;
	struct trace_event_call *pevent;
	struct trace_event *probed_event;
	struct trace_seq *s = &iter->seq;
	struct trace_eprobe *ep;
	struct trace_probe *tp;
	unsigned int type;

	field = (struct eprobe_trace_entry_head *)iter->ent;
	tp = trace_probe_primary_from_call(
		container_of(event, struct trace_event_call, event));
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		goto out;

	ep = container_of(tp, struct trace_eprobe, tp);
	type = ep->event->event.type;

	trace_seq_printf(s, "%s: (", trace_probe_name(tp));

	probed_event = ftrace_find_event(type);
	// Block Logic: Prints the target event's system and name, or its type if not found.
	if (probed_event) {
		pevent = container_of(probed_event, struct trace_event_call, event);
		trace_seq_printf(s, "%s.%s", pevent->class->system,
				 trace_event_name(pevent));
	} else {
		trace_seq_printf(s, "%u", type);
	}

	trace_seq_putc(s, ')');

	// Block Logic: Prints the arguments of the eprobe.
	if (trace_probe_print_args(s, tp->args, tp->nr_args,
			     (u8 *)&field[1], field) < 0)
		goto out;

	trace_seq_putc(s, '\n');
 out:
	return trace_handle_return(s);
}

/**
 * @brief Retrieves a field's value from a trace event record.
 * @param code Pointer to `fetch_insn` for the field.
 * @param rec Pointer to the raw event record.
 * @return The value of the event field.
 *
 * This function handles various field types (string, integer) and their
 * signedness to correctly extract values from a raw trace event record.
 */
static nokprobe_inline unsigned long
get_event_field(struct fetch_insn *code, void *rec)
{
	struct ftrace_event_field *field = code->data;
	unsigned long val;
	void *addr;

	addr = rec + field->offset;

	// Block Logic: Handles string field types.
	if (is_string_field(field)) {
		switch (field->filter_type) {
		case FILTER_DYN_STRING:
			val = (unsigned long)(rec + (*(unsigned int *)addr & 0xffff));
			break;
		case FILTER_RDYN_STRING:
			val = (unsigned long)(addr + (*(unsigned int *)addr & 0xffff));
			break;
		case FILTER_STATIC_STRING:
			val = (unsigned long)addr;
			break;
		case FILTER_PTR_STRING:
			val = (unsigned long)(*(char *)addr);
			break;
		default:
			WARN_ON_ONCE(1);
			return 0;
		}
		return val;
	}

	// Block Logic: Handles integer field types based on size and signedness.
	switch (field->size) {
	case 1:
		if (field->is_signed)
			val = *(char *)addr;
		else
			val = *(unsigned char *)addr;
		break;
	case 2:
		if (field->is_signed)
			val = *(short *)addr;
		else
			val = *(unsigned short *)addr;
		break;
	case 4:
		if (field->is_signed)
			val = *(int *)addr;
		else
			val = *(unsigned int *)addr;
		break;
	default:
		if (field->is_signed)
			val = *(long *)addr;
		else
			val = *(unsigned long *)addr;
		break;
	}
	return val;
}

/**
 * @brief Calculates the dynamic size required for eprobe arguments.
 * @param tp Pointer to the `trace_probe` instance.
 * @param rec Pointer to the raw event record.
 * @return The total dynamic size in bytes.
 *
 * This function iterates through all arguments of the eprobe, and if
 * an argument is dynamic, it calculates its size based on the event record.
 */
static int get_eprobe_size(struct trace_probe *tp, void *rec)
{
	struct fetch_insn *code;
	struct probe_arg *arg;
	int i, len, ret = 0;

	// Block Logic: Iterates through each argument of the trace probe.
	for (i = 0; i < tp->nr_args; i++) {
		arg = tp->args + i;
		// Block Logic: If the argument is dynamic, calculates its size.
		if (arg->dynamic) {
			unsigned long val;

			code = arg->code;
 retry:
			switch (code->op) {
			case FETCH_OP_TP_ARG:
				val = get_event_field(code, rec);
				break;
			case FETCH_NOP_SYMBOL:	/* Ignore a place holder */
				code++;
				goto retry;
			default:
				if (process_common_fetch_insn(code, &val) < 0)
					continue;
			}
			code++;
			len = process_fetch_insn_bottom(code, val, NULL, NULL);
			if (len > 0)
				ret += len;
		}
	}

	return ret;
}

/* Kprobe specific fetch functions */

/* Note that we don't verify it, since the code does not come from user space */
/**
 * @brief Processes a `fetch_insn` to extract data from a record.
 * @param code Pointer to the `fetch_insn` structure.
 * @param rec Pointer to the raw event record.
 * @param edata Pointer to event-specific data.
 * @param dest Destination buffer for the extracted data.
 * @param base Base address for relative offsets.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function is used to execute a sequence of fetch instructions,
 * extracting values from event records. It handles different fetch operations,
 * including those for tracepoint arguments.
 */
static int
process_fetch_insn(struct fetch_insn *code, void *rec, void *edata,
		   void *dest, void *base)
{
	unsigned long val;
	int ret;

 retry:
	switch (code->op) {
	case FETCH_OP_TP_ARG:
		val = get_event_field(code, rec);
		break;
	case FETCH_NOP_SYMBOL:	/* Ignore a place holder */
		code++;
		goto retry;
	default:
		ret = process_common_fetch_insn(code, &val);
		if (ret < 0)
			return ret;
	}
	code++;
	return process_fetch_insn_bottom(code, val, dest, base);
}
NOKPROBE_SYMBOL(process_fetch_insn)

/* eprobe handler */
/**
 * @brief Main handler for eprobe events.
 * @param edata Pointer to `eprobe_data` associated with the eprobe.
 * @param rec Pointer to the raw event record.
 *
 * This function is invoked when the target event for an eprobe occurs.
 * It reserves space in the trace buffer, extracts arguments, and commits
 * the new eprobe event to the buffer.
 */
static inline void
__eprobe_trace_func(struct eprobe_data *edata, void *rec)
{
	struct eprobe_trace_entry_head *entry;
	struct trace_event_call *call = trace_probe_event_call(&edata->ep->tp);
	struct trace_event_buffer fbuffer;
	int dsize;

	// Block Logic: Ensures the event call matches and handles soft-disabled triggers.
	if (WARN_ON_ONCE(call != edata->file->event_call))
		return;

	if (trace_trigger_soft_disabled(edata->file))
		return;

	// Functional Utility: Calculates dynamic size needed for the event.
	dsize = get_eprobe_size(&edata->ep->tp, rec);

	// Functional Utility: Reserves space in the trace event buffer.
	entry = trace_event_buffer_reserve(&fbuffer, edata->file,
					   sizeof(*entry) + edata->ep->tp.size + dsize);

	// Block Logic: Returns if buffer reservation fails.
	if (!entry)
		return;

	entry = fbuffer.entry = ring_buffer_event_data(fbuffer.event);
	// Functional Utility: Stores trace arguments into the reserved buffer space.
	store_trace_args(&entry[1], &edata->ep->tp, rec, NULL, sizeof(*entry), dsize);

	trace_event_buffer_commit(&fbuffer); // Functional Utility: Commits the event to the buffer.
}

/*
 * The event probe implementation uses event triggers to get access to
 * the event it is attached to, but is not an actual trigger. The below
 * functions are just stubs to fulfill what is needed to use the trigger
 * infrastructure.
 */
/**
 * @brief Initializes an eprobe event trigger.
 * @param data Pointer to `event_trigger_data`.
 * @return 0 (always succeeds).
 *
 * This is a stub function for the `event_trigger_ops` interface.
 */
static int eprobe_trigger_init(struct event_trigger_data *data)
{
	return 0;
}

/**
 * @brief Frees an eprobe event trigger.
 * @param data Pointer to `event_trigger_data`.
 *
 * This is a stub function for the `event_trigger_ops` interface.
 */
static void eprobe_trigger_free(struct event_trigger_data *data)
{

}

/**
 * @brief Prints eprobe event trigger information.
 * @param m Pointer to `seq_file`.
 * @param data Pointer to `event_trigger_data`.
 * @return 0 (always succeeds, does not print).
 *
 * This is a stub function for the `event_trigger_ops` interface, as eprobe
 * triggers are not user-parsed directly.
 */
static int eprobe_trigger_print(struct seq_file *m,
				struct event_trigger_data *data)
{
	/* Do not print eprobe event triggers */
	return 0;
}

/**
 * @brief Function callback for an eprobe event trigger.
 * @param data Pointer to `event_trigger_data`.
 * @param buffer Pointer to the trace buffer.
 * @param rec Pointer to the raw event record.
 * @param rbe Pointer to the ring buffer event.
 *
 * This function acts as the entry point from the ftrace trigger
 * infrastructure to the `__eprobe_trace_func` handler.
 */
static void eprobe_trigger_func(struct event_trigger_data *data,
				struct trace_buffer *buffer, void *rec,
				struct ring_buffer_event *rbe)
{
	struct eprobe_data *edata = data->private_data;

	// Block Logic: Returns if the record is invalid.
	if (unlikely(!rec))
		return;

	__eprobe_trace_func(edata, rec);
}

/**
 * @var eprobe_trigger_ops
 * @brief Event trigger operations for eprobes.
 *
 * Defines the callback functions for eprobe triggers.
 */
static const struct event_trigger_ops eprobe_trigger_ops = {
	.trigger		= eprobe_trigger_func,
	.print			= eprobe_trigger_print,
	.init			= eprobe_trigger_init,
	.free			= eprobe_trigger_free,
};

/**
 * @brief Parses an eprobe trigger command.
 * @param cmd_ops Pointer to `event_command`.
 * @param file Pointer to `trace_event_file`.
 * @param glob Glob pattern.
 * @param cmd Command string.
 * @param param_and_filter Parameter and filter string.
 * @return -1 (eprobe triggers are not parsed as direct commands).
 *
 * This is a stub function for the `event_command` interface.
 */
static int eprobe_trigger_cmd_parse(struct event_command *cmd_ops,
				    struct trace_event_file *file,
				    char *glob, char *cmd,
				    char *param_and_filter)
{
	return -1;
}

/**
 * @brief Registers an eprobe trigger function.
 * @param glob Glob pattern.
 * @param data Pointer to `event_trigger_data`.
 * @param file Pointer to `trace_event_file`.
 * @return -1 (eprobe triggers are not registered directly).
 *
 * This is a stub function for the `event_command` interface.
 */
static int eprobe_trigger_reg_func(char *glob,
				   struct event_trigger_data *data,
				   struct trace_event_file *file)
{
	return -1;
}

/**
 * @brief Unregisters an eprobe trigger function.
 * @param glob Glob pattern.
 * @param data Pointer to `event_trigger_data`.
 * @param file Pointer to `trace_event_file`.
 *
 * This is a stub function for the `event_command` interface.
 */
static void eprobe_trigger_unreg_func(char *glob,
				      struct event_trigger_data *data,
				      struct trace_event_file *file)
{

}

/**
 * @brief Retrieves the event trigger operations for an eprobe.
 * @param cmd Command string.
 * @param param Parameter string.
 * @return Pointer to `eprobe_trigger_ops`.
 *
 * This function always returns the static `eprobe_trigger_ops`.
 */
static const struct event_trigger_ops *eprobe_trigger_get_ops(char *cmd,
							      char *param)
{
	return &eprobe_trigger_ops;
}

/**
 * @var event_trigger_cmd
 * @brief Event command structure for eprobe triggers.
 *
 * Defines the interface for eprobe triggers within the event command system.
 */
static struct event_command event_trigger_cmd = {
	.name			= "eprobe",
	.trigger_type		= ETT_EVENT_EPROBE,
	.flags			= EVENT_CMD_FL_NEEDS_REC,
	.parse			= eprobe_trigger_cmd_parse,
	.reg			= eprobe_trigger_reg_func,
	.unreg			= eprobe_trigger_unreg_func,
	.unreg_all		= NULL,
	.get_trigger_ops	= eprobe_trigger_get_ops,
	.set_filter		= NULL,
};

/**
 * @brief Creates a new eprobe trigger.
 * @param ep Pointer to the `trace_eprobe` instance.
 * @param file Pointer to the `trace_event_file` where the trigger will be attached.
 * @return Pointer to a new `event_trigger_data` on success, or an `ERR_PTR` on failure.
 *
 * This function allocates and initializes an `event_trigger_data` structure
 * for the eprobe, setting its private data and optionally creating an event filter.
 */
static struct event_trigger_data *
new_eprobe_trigger(struct trace_eprobe *ep, struct trace_event_file *file)
{
	struct event_trigger_data *trigger;
	struct event_filter *filter = NULL;
	struct eprobe_data *edata;
	int ret;

	edata = kzalloc(sizeof(*edata), GFP_KERNEL);
	trigger = kzalloc(sizeof(*trigger), GFP_KERNEL);
	// Block Logic: Handles memory allocation failure.
	if (!trigger || !edata) {
		ret = -ENOMEM;
		goto error;
	}

	trigger->flags = EVENT_TRIGGER_FL_PROBE;
	trigger->count = -1; // Functional Utility: Infinite count.
	trigger->ops = &eprobe_trigger_ops;

	/*
	 * EVENT PROBE triggers are not registered as commands with
	 * register_event_command(), as they are not controlled by the user
	 * from the trigger file
	 */
	trigger->cmd_ops = &event_trigger_cmd;

	INIT_LIST_HEAD(&trigger->list);

	// Block Logic: If a filter string is provided, creates an event filter.
	if (ep->filter_str) {
		ret = create_event_filter(file->tr, ep->event,
						ep->filter_str, false, &filter);
		if (ret)
			goto error;
	}
	RCU_INIT_POINTER(trigger->filter, filter); // Functional Utility: Initializes RCU pointer for the filter.

	edata->file = file;
	edata->ep = ep;
	trigger->private_data = edata; // Functional Utility: Sets private data to link trigger to eprobe.

	return trigger;
error:
	free_event_filter(filter);
	kfree(edata);
	kfree(trigger);
	return ERR_PTR(ret);
}

/**
 * @brief Enables a `trace_eprobe` by attaching its trigger to the target event.
 * @param ep Pointer to the `trace_eprobe` instance.
 * @param eprobe_file Pointer to the `trace_event_file` for this eprobe.
 * @return 0 on success, -ENOENT if target event file is not found, or an `ERR_PTR` on trigger creation failure.
 *
 * This function finds the target event file, creates an eprobe trigger,
 * adds it to the event's trigger list, and enables the event trigger.
 */
static int enable_eprobe(struct trace_eprobe *ep,
			 struct trace_event_file *eprobe_file)
{
	struct event_trigger_data *trigger;
	struct trace_event_file *file;
	struct trace_array *tr = eprobe_file->tr;

	// Functional Utility: Finds the target trace event file.
	file = find_event_file(tr, ep->event_system, ep->event_name);
	if (!file)
		return -ENOENT;
	trigger = new_eprobe_trigger(ep, eprobe_file); // Functional Utility: Creates a new eprobe trigger.
	if (IS_ERR(trigger))
		return PTR_ERR(trigger);

	list_add_tail_rcu(&trigger->list, &file->triggers); // Functional Utility: Adds the trigger to the event's trigger list.

	trace_event_trigger_enable_disable(file, 1); // Functional Utility: Enables the event trigger.
	update_cond_flag(file); // Functional Utility: Updates conditional flags for the file.

	return 0;
}

/**
 * @var eprobe_funcs
 * @brief Trace event functions for eprobes.
 *
 * Defines the `trace` function (printer) for eprobe events.
 */
static struct trace_event_functions eprobe_funcs = {
	.trace		= print_eprobe_event
};

/**
 * @brief Disables a `trace_eprobe` by removing its trigger from the target event.
 * @param ep Pointer to the `trace_eprobe` instance.
 * @param tr Pointer to the `trace_array`.
 * @return 0 on success, -ENOENT if target event file is not found, -ENODEV if trigger not found.
 *
 * This function finds the target event file, removes the eprobe's trigger
 * from the event's trigger list, disables the event trigger, and frees
 * associated resources.
 */
static int disable_eprobe(struct trace_eprobe *ep,
			  struct trace_array *tr)
{
	struct event_trigger_data *trigger = NULL, *iter;
	struct trace_event_file *file;
	struct event_filter *filter;
	struct eprobe_data *edata;

	// Functional Utility: Finds the target trace event file.
	file = find_event_file(tr, ep->event_system, ep->event_name);
	if (!file)
		return -ENOENT;

	// Block Logic: Iterates through triggers to find the eprobe's specific trigger.
	list_for_each_entry(iter, &file->triggers, list) {
		if (!(iter->flags & EVENT_TRIGGER_FL_PROBE))
			continue;
		edata = iter->private_data;
		if (edata->ep == ep) {
			trigger = iter;
			break;
		}
	}
	// Block Logic: Returns if the trigger is not found.
	if (!trigger)
		return -ENODEV;

	list_del_rcu(&trigger->list); // Functional Utility: Removes the trigger from the list.

	trace_event_trigger_enable_disable(file, 0); // Functional Utility: Disables the event trigger.
	update_cond_flag(file); // Functional Utility: Updates conditional flags.

	/* Make sure nothing is using the edata or trigger */
	tracepoint_synchronize_unregister(); // Functional Utility: Synchronizes to ensure no active users of the unregistering trigger.

	filter = rcu_access_pointer(trigger->filter); // Functional Utility: Retrieves the filter pointer.

	// Functional Utility: Frees filter, private data, and trigger.
	if (filter)
		free_event_filter(filter);
	kfree(edata);
	kfree(trigger);

	return 0;
}

/**
 * @brief Enables a `trace_eprobe` or a group of eprobes associated with a `trace_event_call`.
 * @param call Pointer to the `trace_event_call` for the eprobe.
 * @param file Optional: Pointer to the `trace_event_file` to associate.
 * @return 0 on success, -ENODEV if `trace_probe` is invalid, or a negative errno on failure.
 *
 * This function enables all eprobes linked to the provided `trace_probe`
 * (primary from `call`). It handles both file-associated and profile-flagged
 * enabling.
 */
static int enable_trace_eprobe(struct trace_event_call *call,
			       struct trace_event_file *file)
{
	struct trace_probe *tp;
	struct trace_eprobe *ep;
	bool enabled;
	int ret = 0;
	int cnt = 0;

	tp = trace_probe_primary_from_call(call);
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		return -ENODEV;
	enabled = trace_probe_is_enabled(tp);

	/* This also changes "enabled" state */
	// Block Logic: Associates the trace probe with the file or sets the profile flag.
	if (file) {
		ret = trace_probe_add_file(tp, file);
		if (ret)
			return ret;
	} else
		trace_probe_set_flag(tp, TP_FLAG_PROFILE);

	// Block Logic: If already enabled, just returns.
	if (enabled)
		return 0;

	// Block Logic: Iterates through and enables each eprobe linked to the `trace_probe`.
	for_each_trace_eprobe_tp(ep, tp) {
		ret = enable_eprobe(ep, file);
		if (ret)
			break;
		enabled = true;
		cnt++;
	}

	// Block Logic: If any eprobe enabling failed, rolls back previous successful enables.
	if (ret) {
		/* Failed to enable one of them. Roll back all */
		if (enabled) {
			/*
			 * It's a bug if one failed for something other than memory
			 * not being available but another eprobe succeeded.
			 */
			WARN_ON_ONCE(ret != -ENOMEM);

			for_each_trace_eprobe_tp(ep, tp) {
				disable_eprobe(ep, file->tr);
				if (!--cnt)
					break;
			}
		}
		// Block Logic: Removes file association or profile flag on rollback.
		if (file)
			trace_probe_remove_file(tp, file);
		else
			trace_probe_clear_flag(tp, TP_FLAG_PROFILE);
	}

	return ret;
}

/**
 * @brief Disables a `trace_eprobe` or a group of eprobes associated with a `trace_event_call`.
 * @param call Pointer to the `trace_event_call` for the eprobe.
 * @param file Optional: Pointer to the `trace_event_file` to disassociate.
 * @return 0 on success, -ENODEV if `trace_probe` is invalid, -ENOENT if file link not found.
 *
 * This function disables all eprobes linked to the provided `trace_probe`.
 * It handles both file-associated and profile-flagged disabling.
 */
static int disable_trace_eprobe(struct trace_event_call *call,
				struct trace_event_file *file)
{
	struct trace_probe *tp;
	struct trace_eprobe *ep;

	tp = trace_probe_primary_from_call(call);
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		return -ENODEV;

	// Block Logic: Removes file association or clears profile flag.
	if (file) {
		if (!trace_probe_get_file_link(tp, file))
			return -ENOENT;
		if (!trace_probe_has_single_file(tp))
			goto out;
		trace_probe_clear_flag(tp, TP_FLAG_TRACE);
	} else
		trace_probe_clear_flag(tp, TP_FLAG_PROFILE);

	// Block Logic: If no other probes are enabled, iterates and disables each linked eprobe.
	if (!trace_probe_is_enabled(tp)) {
		for_each_trace_eprobe_tp(ep, tp)
			disable_eprobe(ep, file->tr);
	}

 out:
	if (file)
		/*
		 * Synchronization is done in below function. For perf event,
		 * file == NULL and perf_trace_event_unreg() calls
		 * tracepoint_synchronize_unregister() to ensure synchronize
		 * event. We don't need to care about it.
		 */
		trace_probe_remove_file(tp, file); // Functional Utility: Removes file association.

	return 0;
}

/**
 * @brief Register/unregister callback for eprobe events.
 * @param event Pointer to the `trace_event_call`.
 * @param type Type of registration operation (`TRACE_REG_REGISTER`, `TRACE_REG_UNREGISTER`, etc.).
 * @param data Opaque data (e.g., `trace_event_file`).
 * @return 0 on success.
 *
 * This function acts as the central registration point for eprobes,
 * routing `REGISTER` and `UNREGISTER` requests to `enable_trace_eprobe`
 * and `disable_trace_eprobe` respectively.
 */
static int eprobe_register(struct trace_event_call *event,
			   enum trace_reg type, void *data)
{
	struct trace_event_file *file = data;

	switch (type) {
	case TRACE_REG_REGISTER:
		return enable_trace_eprobe(event, file);
	case TRACE_REG_UNREGISTER:
		return disable_trace_eprobe(event, file);
#ifdef CONFIG_PERF_EVENTS
	case TRACE_REG_PERF_REGISTER:
	case TRACE_REG_PERF_UNREGISTER:
	case TRACE_REG_PERF_OPEN:
	case TRACE_REG_PERF_CLOSE:
	case TRACE_REG_PERF_ADD:
	case TRACE_REG_PERF_DEL:
		return 0;
#endif
	}
	return 0;
}

/**
 * @brief Initializes the `trace_event_call` for an eprobe.
 * @param ep Pointer to the `trace_eprobe` instance.
 *
 * This function sets various flags and callbacks for the eprobe's
 * `trace_event_call`, linking it to the eprobe printing functions.
 */
static inline void init_trace_eprobe_call(struct trace_eprobe *ep)
{
	struct trace_event_call *call = trace_probe_event_call(&ep->tp);

	call->flags = TRACE_EVENT_FL_EPROBE;
	call->event.funcs = &eprobe_funcs;
	call->class->fields_array = eprobe_fields_array;
	call->class->reg = eprobe_register;
}

/**
 * @brief Finds and gets a reference to an existing trace event.
 * @param system The system name of the event.
 * @param event_name The name of the event.
 * @return Pointer to the `trace_event_call` on success, or NULL if not found or reference cannot be obtained.
 *
 * This function iterates through all registered ftrace events to find
 * a matching event by system and name, avoiding other probe types.
 */
static struct trace_event_call *
find_and_get_event(const char *system, const char *event_name)
{
	struct trace_event_call *tp_event;
	const char *name;

	// Block Logic: Iterates through all registered ftrace events.
	list_for_each_entry(tp_event, &ftrace_events, list) {
		/* Skip other probes and ftrace events */
		// Block Logic: Skips events that are not regular ftrace events.
		if (tp_event->flags &
		    (TRACE_EVENT_FL_IGNORE_ENABLE |
		     TRACE_EVENT_FL_KPROBE |
		     TRACE_EVENT_FL_UPROBE |
		     TRACE_EVENT_FL_EPROBE))
			continue;
		// Block Logic: Checks if the system name matches.
		if (!tp_event->class->system ||
		    strcmp(system, tp_event->class->system))
			continue;
		name = trace_event_name(tp_event);
		// Block Logic: Checks if the event name matches.
		if (!name || strcmp(event_name, name))
			continue;
		// Functional Utility: Tries to get a reference to the event.
		if (!trace_event_try_get_ref(tp_event))
			return NULL;
		return tp_event;
	}
	return NULL;
}

/**
 * @brief Updates a `trace_eprobe` argument.
 * @param ep Pointer to the `trace_eprobe` instance.
 * @param argv Array of argument strings.
 * @param i Index of the argument to update.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses and updates a specific argument of the eprobe,
 * handling kernel and trace event-specific flags.
 */
static int trace_eprobe_tp_update_arg(struct trace_eprobe *ep, const char *argv[], int i)
{
	struct traceprobe_parse_context ctx = {
		.event = ep->event,
		.flags = TPARG_FL_KERNEL | TPARG_FL_TEVENT,
	};
	int ret;

	ret = traceprobe_parse_probe_arg(&ep->tp, i, argv[i], &ctx);
	/* Handle symbols "@" */
	// Block Logic: If parsing was successful, updates the argument.
	if (!ret)
		ret = traceprobe_update_arg(&ep->tp.args[i]);

	traceprobe_finish_parse(&ctx); // Functional Utility: Finishes parsing context.
	return ret;
}

/**
 * @brief Parses the filter string for an eprobe.
 * @param ep Pointer to the `trace_eprobe` instance.
 * @param argc Number of arguments for the filter.
 * @param argv Array of filter argument strings.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function reconstructs the filter string from arguments and validates
 * it by attempting to create an event filter.
 */
static int trace_eprobe_parse_filter(struct trace_eprobe *ep, int argc, const char *argv[])
{
	struct event_filter *dummy = NULL;
	int i, ret, len = 0;
	char *p;

	// Block Logic: Returns error if no filter arguments are provided.
	if (argc == 0) {
		trace_probe_log_err(0, NO_EP_FILTER);
		return -EINVAL;
	}

	/* Recover the filter string */
	// Block Logic: Calculates total length of the filter string.
	for (i = 0; i < argc; i++)
		len += strlen(argv[i]) + 1;

	ep->filter_str = kzalloc(len, GFP_KERNEL); // Functional Utility: Allocates memory for the filter string.
	if (!ep->filter_str)
		return -ENOMEM;

	// Block Logic: Concatenates filter arguments into a single string.
	p = ep->filter_str;
	for (i = 0; i < argc; i++) {
		if (i)
			ret = snprintf(p, len, " %s", argv[i]);
		else
			ret = snprintf(p, len, "%s", argv[i]);
		p += ret;
		len -= ret;
	}

	/*
	 * Ensure the filter string can be parsed correctly. Note, this
	 * filter string is for the original event, not for the eprobe.
	 */
	// Functional Utility: Creates a dummy event filter to validate the filter string.
	ret = create_event_filter(top_trace_array(), ep->event, ep->filter_str,
				  true, &dummy);
	free_event_filter(dummy); // Functional Utility: Frees the dummy filter.
	if (ret)
		goto error;

	return 0;
error:
	kfree(ep->filter_str); // Functional Utility: Frees filter string on error.
	ep->filter_str = NULL;
	return ret;
}

/**
 * @brief Main function to create a `trace_eprobe` instance.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses the command-line arguments to define a new eprobe,
 * including its group, name, target event, arguments, and filter.
 */
static int __trace_eprobe_create(int argc, const char *argv[])
{
	/*
	 * Argument syntax:
	 * 	e[:[GRP/][ENAME]] SYSTEM.EVENT [FETCHARGS] [if FILTER]
	 * Fetch args (no space):
	 * 	<name>=$<field>[:TYPE]
	 */
	const char *event = NULL, *group = EPROBE_EVENT_SYSTEM;
	const char *sys_event = NULL, *sys_name = NULL;
	struct trace_event_call *event_call;
	struct trace_eprobe *ep = NULL;
	char buf1[MAX_EVENT_NAME_LEN];
	char buf2[MAX_EVENT_NAME_LEN];
	char gbuf[MAX_EVENT_NAME_LEN];
	int ret = 0, filter_idx = 0;
	int i, filter_cnt;

	// Block Logic: Basic validation of command-line arguments.
	if (argc < 2 || argv[0][0] != 'e')
		return -ECANCELED;

	trace_probe_log_init("event_probe", argc, argv); // Functional Utility: Initializes probe logging.

	// Block Logic: Parses the eprobe's own event name and group.
	event = strchr(&argv[0][1], ':');
	if (event) {
		event++;
		ret = traceprobe_parse_event_name(&event, &group, gbuf,
						  event - argv[0]);
		if (ret)
			goto parse_error;
	}

	trace_probe_log_set_index(1);
	sys_event = argv[1];
	// Block Logic: Parses the target system and event name.
	ret = traceprobe_parse_event_name(&sys_event, &sys_name, buf2, 0);
	if (ret || !sys_event || !sys_name) {
		trace_probe_log_err(0, NO_EVENT_INFO);
		goto parse_error;
	}

	// Block Logic: If eprobe name is not specified, uses target event name.
	if (!event) {
		strscpy(buf1, sys_event, MAX_EVENT_NAME_LEN);
		event = buf1;
	}

	// Block Logic: Scans for "if" keyword to separate arguments from filter.
	for (i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "if")) {
			filter_idx = i + 1;
			filter_cnt = argc - filter_idx;
			argc = i;
			break;
		}
	}

	// Block Logic: Checks for maximum number of arguments.
	if (argc - 2 > MAX_TRACE_ARGS) {
		trace_probe_log_set_index(2);
		trace_probe_log_err(0, TOO_MANY_ARGS);
		ret = -E2BIG;
		goto error;
	}

	scoped_guard(mutex, &event_mutex) { // Functional Utility: Locks `event_mutex` for exclusive access.
		event_call = find_and_get_event(sys_name, sys_event); // Functional Utility: Finds and gets target trace event.
		ep = alloc_event_probe(group, event, event_call, argc - 2); // Functional Utility: Allocates and initializes eprobe.
	}

	// Block Logic: Handles allocation failure for eprobe.
	if (IS_ERR(ep)) {
		ret = PTR_ERR(ep);
		if (ret == -ENODEV)
			trace_probe_log_err(0, BAD_ATTACH_EVENT);
		/* This must return -ENOMEM or missing event, else there is a bug */
		WARN_ON_ONCE(ret != -ENOMEM && ret != -ENODEV);
		ep = NULL;
		goto error;
	}

	// Block Logic: If a filter is specified, parses it.
	if (filter_idx) {
		trace_probe_log_set_index(filter_idx);
		ret = trace_eprobe_parse_filter(ep, filter_cnt, argv + filter_idx);
		if (ret)
			goto parse_error;
	} else
		ep->filter_str = NULL;

	argc -= 2; argv += 2;
	/* parse arguments */
	// Block Logic: Parses and updates eprobe arguments.
	for (i = 0; i < argc; i++) {
		trace_probe_log_set_index(i + 2);
		ret = trace_eprobe_tp_update_arg(ep, argv, i);
		if (ret)
			goto error;
	}
	ret = traceprobe_set_print_fmt(&ep->tp, PROBE_PRINT_EVENT); // Functional Utility: Sets print format for the eprobe.
	if (ret < 0)
		goto error;
	init_trace_eprobe_call(ep); // Functional Utility: Initializes eprobe event call.
	scoped_guard(mutex, &event_mutex) { // Functional Utility: Locks `event_mutex`.
		ret = trace_probe_register_event_call(&ep->tp); // Functional Utility: Registers eprobe event call.
		if (ret) {
			// Block Logic: Handles existing event name.
			if (ret == -EEXIST) {
				trace_probe_log_set_index(0);
				trace_probe_log_err(0, EVENT_EXIST);
			}
			goto error;
		}
		ret = dyn_event_add(&ep->devent, &ep->tp.event->call); // Functional Utility: Adds eprobe as a dynamic event.
		if (ret < 0) {
			trace_probe_unregister_event_call(&ep->tp); // Functional Utility: Unregisters event call on failure.
			goto error;
		}
	}
	trace_probe_log_clear(); // Functional Utility: Clears probe logging.
	return ret;

parse_error:
	ret = -EINVAL; // Functional Utility: Sets error code for parse errors.
error:
	trace_probe_log_clear(); // Functional Utility: Clears probe logging.
	trace_event_probe_cleanup(ep); // Functional Utility: Cleans up allocated resources.
	return ret;
}

/*
 * Register dynevent at core_initcall. This allows kernel to setup eprobe
 * events in postcore_initcall without tracefs.
 */
/**
 * @brief Early initialization function for eprobe dynamic events.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function registers the `eprobe_dyn_event_ops` with the dynamic event
 * system, allowing eprobes to be set up early in the boot process.
 */
static __init int trace_events_eprobe_init_early(void)
{
	int err = 0;

	err = dyn_event_register(&eprobe_dyn_event_ops);
	if (err)
		pr_warn("Could not register eprobe_dyn_event_ops\n");

	return err;
}
core_initcall(trace_events_eprobe_init_early); // Functional Utility: Registers `trace_events_eprobe_init_early` as a core initialization call.