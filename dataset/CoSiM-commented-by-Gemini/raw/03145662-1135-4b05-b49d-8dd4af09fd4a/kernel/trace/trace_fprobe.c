/**
 * @file trace_fprobe.c
 * @brief Implements fprobe-based tracing events for the Linux kernel.
 *
 * This file provides the core logic for dynamically creating ftrace probes
 * as tracing events (fprobes). It allows users to define custom trace events
 * that trigger on function entry and/or exit, or on specific tracepoints,
 * capturing and displaying relevant data.
 *
 * Functional Utility:
 * - Dynamically defines new trace events based on kernel functions or tracepoints.
 * - Supports both function entry (fentry) and function exit (fexit) probing.
 * - Allows extraction of function arguments, return values, stack addresses,
 *   and arbitrary memory locations.
 * - Integrates with the ftrace and perf event subsystems for data collection and reporting.
 * - Provides mechanisms for handling module-specific symbols and tracepoints.
 *
 * Algorithms:
 * - Uses the fprobe framework for low-level function hooking.
 * - Leverages `trace_probe` infrastructure for argument parsing and event management.
 * - Employs `kallsyms` for symbol lookup and address resolution.
 * - Integrates with `dyn_event` for dynamic event registration.
 *
 * Architectural Intent:
 * - To provide a flexible and powerful mechanism for dynamic kernel introspection
 *   and debugging without modifying kernel source code.
 * - To enable users to create highly customized tracing points tailored to
 *   specific analysis needs.
 *
 * Copyright (C) 2022 Google LLC.
 */
// SPDX-License-Identifier: GPL-2.0
#define pr_fmt(fmt)	"trace_fprobe: " fmt
#include <asm/ptrace.h>

#include <linux/fprobe.h>
#include <linux/module.h>
#include <linux/rculist.h>
#include <linux/security.h>
#include <linux/tracepoint.h>
#include <linux/uaccess.h>

#include "trace_dynevent.h"
#include "trace_probe.h"
#include "trace_probe_kernel.h"
#include "trace_probe_tmpl.h"

/**
 * @def FPROBE_EVENT_SYSTEM
 * @brief The default event system name for fprobes.
 *
 * Fprobes are grouped under this system name in the tracefs hierarchy.
 */
#define FPROBE_EVENT_SYSTEM "fprobes"
/**
 * @def TRACEPOINT_EVENT_SYSTEM
 * @brief The event system name for tracepoint-based fprobes.
 */
#define TRACEPOINT_EVENT_SYSTEM "tracepoints"
/**
 * @def RETHOOK_MAXACTIVE_MAX
 * @brief Maximum number of active kretprobe instances.
 *
 * This defines a limit for the number of simultaneously active kretprobe
 * (function return probes) instances.
 */
#define RETHOOK_MAXACTIVE_MAX 4096
/**
 * @def TRACEPOINT_STUB
 * @brief An error pointer indicating a stubbed tracepoint.
 *
 * Used when a tracepoint is referenced but not yet loaded or available.
 */
#define TRACEPOINT_STUB ERR_PTR(-ENOENT)

/**
 * @brief Creates a new fprobe-based trace event.
 * @param raw_command The raw command string for the fprobe.
 * @return 0 on success, or a negative errno on failure.
 */
static int trace_fprobe_create(const char *raw_command);
/**
 * @brief Displays the details of an fprobe dynamic event in a `seq_file`.
 * @param m Pointer to the `seq_file`.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success.
 */
static int trace_fprobe_show(struct seq_file *m, struct dyn_event *ev);
/**
 * @brief Releases resources associated with an fprobe dynamic event.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success, or a negative errno on failure.
 */
static int trace_fprobe_release(struct dyn_event *ev);
/**
 * @brief Checks if an fprobe dynamic event is busy (enabled).
 * @param ev Pointer to the dynamic event.
 * @return True if busy, false otherwise.
 */
static bool trace_fprobe_is_busy(struct dyn_event *ev);
/**
 * @brief Matches an fprobe dynamic event against system, event, and arguments.
 * @param system The system name to match.
 * @param event The event name to match.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param ev Pointer to the dynamic event.
 * @return True if match, false otherwise.
 */
static bool trace_fprobe_match(const char *system, const char *event,
			int argc, const char **argv, struct dyn_event *ev);

/**
 * @var trace_fprobe_ops
 * @brief Dynamic event operations for fprobes.
 *
 * This structure defines the callbacks for creating, showing, freeing,
 * and matching fprobe dynamic events.
 */
static struct dyn_event_operations trace_fprobe_ops = {
	.create = trace_fprobe_create,
	.show = trace_fprobe_show,
	.is_busy = trace_fprobe_is_busy,
	.free = trace_fprobe_release,
	.match = trace_fprobe_match,
};

/*
 * Fprobe event core functions
 */
/**
 * @struct trace_fprobe
 * @brief Represents an fprobe-based tracing event instance.
 *
 * This structure encapsulates all information about a dynamically created
 * fprobe event, including its associated `fprobe`, symbol, tracepoint (if any),
 * module, and `trace_probe` structures.
 */
struct trace_fprobe {
	struct dyn_event	devent;       /**< @brief Dynamic event base structure. */
	struct fprobe		fp;         /**< @brief Fprobe instance for function hooking. */
	const char		*symbol;      /**< @brief The target symbol name. */
	struct tracepoint	*tpoint;     /**< @brief The target tracepoint (if applicable). */
	struct module		*mod;         /**< @brief The module containing the target symbol/tracepoint. */
	struct trace_probe	tp;         /**< @brief Trace probe base structure. */
};

/**
 * @brief Checks if a dynamic event is an fprobe.
 * @param ev Pointer to the dynamic event.
 * @return True if it's an fprobe, false otherwise.
 */
static bool is_trace_fprobe(struct dyn_event *ev)
{
	return ev->ops == &trace_fprobe_ops;
}

/**
 * @brief Casts a `dyn_event` pointer to a `trace_fprobe` pointer.
 * @param ev Pointer to the dynamic event.
 * @return Pointer to the embedded `trace_fprobe` instance.
 */
static struct trace_fprobe *to_trace_fprobe(struct dyn_event *ev)
{
	return container_of(ev, struct trace_fprobe, devent);
}

/**
 * @def for_each_trace_fprobe(pos, dpos)
 * @brief Macro to iterate over the list of `trace_fprobe` instances.
 * @param pos The `trace_fprobe *` for each entry.
 * @param dpos The `dyn_event *` to use as a loop cursor.
 */
#define for_each_trace_fprobe(pos, dpos)		\
	for_each_dyn_event(dpos)			\
		if (is_trace_fprobe(dpos) && (pos = to_trace_fprobe(dpos)))

/**
 * @brief Checks if an fprobe is a return probe.
 * @param tf Pointer to the `trace_fprobe`.
 * @return True if it's a return probe (has an exit handler), false otherwise.
 */
static bool trace_fprobe_is_return(struct trace_fprobe *tf)
{
	return tf->fp.exit_handler != NULL;
}

/**
 * @brief Checks if an fprobe is based on a tracepoint.
 * @param tf Pointer to the `trace_fprobe`.
 * @return True if it's a tracepoint-based fprobe, false otherwise.
 */
static bool trace_fprobe_is_tracepoint(struct trace_fprobe *tf)
{
	return tf->tpoint != NULL;
}

/**
 * @brief Retrieves the symbol name for an fprobe.
 * @param tf Pointer to the `trace_fprobe`.
 * @return The symbol name as a C string, or "unknown" if not set.
 */
static const char *trace_fprobe_symbol(struct trace_fprobe *tf)
{
	return tf->symbol ? tf->symbol : "unknown";
}

/**
 * @brief Checks if an fprobe dynamic event is busy (enabled).
 * @param ev Pointer to the dynamic event.
 * @return True if the fprobe's trace probe is enabled, false otherwise.
 */
static bool trace_fprobe_is_busy(struct dyn_event *ev)
{
	struct trace_fprobe *tf = to_trace_fprobe(ev);

	return trace_probe_is_enabled(&tf->tp);
}

/**
 * @brief Matches the head of a command against an fprobe's symbol and arguments.
 * @param tf Pointer to the `trace_fprobe`.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @return True if the command head matches, false otherwise.
 */
static bool trace_fprobe_match_command_head(struct trace_fprobe *tf,
					    int argc, const char **argv)
{
	char buf[MAX_ARGSTR_LEN + 1];

	// Block Logic: If no arguments, it's a match.
	if (!argc)
		return true;

	// Functional Utility: Formats the fprobe symbol into a buffer.
	snprintf(buf, sizeof(buf), "%s", trace_fprobe_symbol(tf));
	// Block Logic: Compares the buffer with the first argument.
	if (strcmp(buf, argv[0]))
		return false;
	argc--; argv++; // Functional Utility: Advances argument pointer.

	// Functional Utility: Matches against remaining arguments using `trace_probe_match_command_args`.
	return trace_probe_match_command_args(&tf->tp, argc, argv);
}

/**
 * @brief Matches an fprobe dynamic event against system, event, and command arguments.
 * @param system The system name to match.
 * @param event The event name to match.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param ev Pointer to the dynamic event.
 * @return True if the fprobe matches the criteria, false otherwise.
 *
 * This function implements the matching logic for fprobes, checking
 * event name, system, and command head.
 */
static bool trace_fprobe_match(const char *system, const char *event,
			int argc, const char **argv, struct dyn_event *ev)
{
	struct trace_fprobe *tf = to_trace_fprobe(ev);

	// Block Logic: Checks if the event name matches.
	if (event[0] != '\0' && strcmp(trace_probe_name(&tf->tp), event))
		return false;

	// Block Logic: Checks if the system name matches.
	if (system && strcmp(trace_probe_group_name(&tf->tp), system))
		return false;

	// Functional Utility: Matches the command head (symbol and arguments).
	return trace_fprobe_match_command_head(tf, argc, argv);
}

/**
 * @brief Checks if an fprobe is registered with the fprobe framework.
 * @param tf Pointer to the `trace_fprobe`.
 * @return True if the fprobe is registered, false otherwise.
 */
static bool trace_fprobe_is_registered(struct trace_fprobe *tf)
{
	return fprobe_is_registered(&tf->fp);
}

/*
 * Note that we don't verify the fetch_insn code, since it does not come
 * from user space.
 */
/**
 * @brief Processes a `fetch_insn` to extract data from a trace record.
 * @param code Pointer to the `fetch_insn` structure.
 * @param rec Pointer to the raw trace record (which is `ftrace_regs *`).
 * @param edata Pointer to event-specific data.
 * @param dest Destination buffer for the extracted data.
 * @param base Base address for relative offsets.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function handles various fetch operations (`FETCH_OP_STACK`, `FETCH_OP_ARG`, etc.)
 * to extract values from the `ftrace_regs` structure and other contexts.
 */
static int
process_fetch_insn(struct fetch_insn *code, void *rec, void *edata,
		   void *dest, void *base)
{
	struct ftrace_regs *fregs = rec;
	unsigned long val;
	int ret;

retry:
	/* 1st stage: get value from context */
	// Block Logic: Extracts value based on the fetch operation.
	switch (code->op) {
	case FETCH_OP_STACK:
		val = ftrace_regs_get_kernel_stack_nth(fregs, code->param);
		break;
	case FETCH_OP_STACKP:
		val = ftrace_regs_get_stack_pointer(fregs);
		break;
	case FETCH_OP_RETVAL:
		val = ftrace_regs_get_return_value(fregs);
		break;
#ifdef CONFIG_HAVE_FUNCTION_ARG_ACCESS_API
	case FETCH_OP_ARG:
		val = ftrace_regs_get_argument(fregs, code->param);
		break;
	case FETCH_OP_EDATA:
		val = *(unsigned long *)((unsigned long)edata + code->offset);
		break;
#endif
	case FETCH_NOP_SYMBOL: /* Ignore a place holder */
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

/* function entry handler */
/**
 * @brief Internal handler for fentry (function entry) trace events.
 * @param tf Pointer to the `trace_fprobe`.
 * @param entry_ip Instruction pointer at function entry.
 * @param fregs Pointer to ftrace registers.
 * @param trace_file Pointer to the `trace_event_file` for output.
 *
 * This function is responsible for recording an fentry event into the
 * trace buffer, extracting and storing the relevant arguments.
 */
static nokprobe_inline void
__fentry_trace_func(struct trace_fprobe *tf, unsigned long entry_ip,
		    struct ftrace_regs *fregs,
		    struct trace_event_file *trace_file)
{
	struct fentry_trace_entry_head *entry;
	struct trace_event_call *call = trace_probe_event_call(&tf->tp);
	struct trace_event_buffer fbuffer;
	int dsize;

	// Block Logic: Checks for matching event call and soft-disabled triggers.
	if (WARN_ON_ONCE(call != trace_file->event_call))
		return;

	if (trace_trigger_soft_disabled(trace_file))
		return;

	// Functional Utility: Calculates dynamic size needed for the event.
	dsize = __get_data_size(&tf->tp, fregs, NULL);

	// Functional Utility: Reserves space in the trace event buffer.
	entry = trace_event_buffer_reserve(&fbuffer, trace_file,
					   sizeof(*entry) + tf->tp.size + dsize);
	if (!entry)
		return;

	fbuffer.regs = ftrace_get_regs(fregs);
	entry = fbuffer.entry = ring_buffer_event_data(fbuffer.event);
	entry->ip = entry_ip; // Functional Utility: Stores the entry instruction pointer.
	// Functional Utility: Stores trace arguments into the buffer.
	store_trace_args(&entry[1], &tf->tp, fregs, NULL, sizeof(*entry), dsize);

	trace_event_buffer_commit(&fbuffer);
}

/**
 * @brief Dispatcher for fentry trace functions.
 * @param tf Pointer to the `trace_fprobe`.
 * @param entry_ip Instruction pointer at function entry.
 * @param fregs Pointer to ftrace registers.
 *
 * This function iterates through all associated `trace_event_file` links
 * and calls `__fentry_trace_func` for each to record the event.
 */
static void
fentry_trace_func(struct trace_fprobe *tf, unsigned long entry_ip,
		  struct ftrace_regs *fregs)
{
	struct event_file_link *link;

	trace_probe_for_each_link_rcu(link, &tf->tp)
		__fentry_trace_func(tf, entry_ip, fregs, link->file);
}
NOKPROBE_SYMBOL(fentry_trace_func);

/**
 * @brief Stores extracted fprobe entry data into a buffer.
 * @param edata Pointer to the buffer where data will be stored.
 * @param tp Pointer to the `trace_probe`.
 * @param fregs Pointer to ftrace registers.
 *
 * This function processes the `entry_arg` fetch instructions associated
 * with the trace probe to extract and store data from function arguments.
 */
static nokprobe_inline
void store_fprobe_entry_data(void *edata, struct trace_probe *tp, struct ftrace_regs *fregs)
{
	struct probe_entry_arg *earg = tp->entry_arg;
	unsigned long val = 0;
	int i;

	// Block Logic: Returns if no entry arguments are defined.
	if (!earg)
		return;

	// Block Logic: Iterates through entry arguments and stores their values.
	for (i = 0; i < earg->size; i++) {
		struct fetch_insn *code = &earg->code[i];

		switch (code->op) {
		case FETCH_OP_ARG:
			val = ftrace_regs_get_argument(fregs, code->param);
			break;
		case FETCH_OP_ST_EDATA:
			*(unsigned long *)((unsigned long)edata + code->offset) = val;
			break;
		case FETCH_OP_END:
			goto end; // Functional Utility: Exits loop on `FETCH_OP_END`.
		default:
			break;
		}
	}
end:
	return;
}

/* function exit handler */
/**
 * @brief Fprobe entry handler that stores entry data for return probes.
 * @param fp Pointer to the `fprobe` instance.
 * @param entry_ip Instruction pointer at function entry.
 * @param ret_ip Return instruction pointer.
 * @param fregs Pointer to ftrace registers.
 * @param entry_data Pointer to the entry data buffer.
 * @return 0 (always succeeds).
 *
 * This handler is used by return fprobes to capture function arguments
 * at entry and store them for later use by the exit handler.
 */
static int trace_fprobe_entry_handler(struct fprobe *fp, unsigned long entry_ip,
				unsigned long ret_ip, struct ftrace_regs *fregs,
				void *entry_data)
{
	struct trace_fprobe *tf = container_of(fp, struct trace_fprobe, fp);

	// Block Logic: If entry arguments are defined, stores the entry data.
	if (tf->tp.entry_arg)
		store_fprobe_entry_data(entry_data, &tf->tp, fregs);

	return 0;
}
NOKPROBE_SYMBOL(trace_fprobe_entry_handler)

/**
 * @brief Internal handler for fexit (function exit) trace events.
 * @param tf Pointer to the `trace_fprobe`.
 * @param entry_ip Instruction pointer at function entry.
 * @param ret_ip Return instruction pointer.
 * @param fregs Pointer to ftrace registers.
 * @param entry_data Per-entry private data stored at function entry.
 * @param trace_file Pointer to the `trace_event_file` for output.
 *
 * This function is responsible for recording an fexit event into the
 * trace buffer, extracting and storing the relevant arguments, including
 * those captured at function entry.
 */
static nokprobe_inline void
__fexit_trace_func(struct trace_fprobe *tf, unsigned long entry_ip,
		   unsigned long ret_ip, struct ftrace_regs *fregs,
		   void *entry_data, struct trace_event_file *trace_file)
{
	struct fexit_trace_entry_head *entry;
	struct trace_event_buffer fbuffer;
	struct trace_event_call *call = trace_probe_event_call(&tf->tp);
	int dsize;

	// Block Logic: Checks for matching event call and soft-disabled triggers.
	if (WARN_ON_ONCE(call != trace_file->event_call))
		return;

	if (trace_trigger_soft_disabled(trace_file))
		return;

	// Functional Utility: Calculates dynamic size needed for the event.
	dsize = __get_data_size(&tf->tp, fregs, entry_data);

	// Functional Utility: Reserves space in the trace event buffer.
	entry = trace_event_buffer_reserve(&fbuffer, trace_file,
					   sizeof(*entry) + tf->tp.size + dsize);
	if (!entry)
		return;

	fbuffer.regs = ftrace_get_regs(fregs);
	entry = fbuffer.entry = ring_buffer_event_data(fbuffer.event);
	entry->func = entry_ip; // Functional Utility: Stores the function entry instruction pointer.
	entry->ret_ip = ret_ip; // Functional Utility: Stores the return instruction pointer.
	// Functional Utility: Stores trace arguments into the buffer, using entry data.
	store_trace_args(&entry[1], &tf->tp, fregs, entry_data, sizeof(*entry), dsize);

	trace_event_buffer_commit(&fbuffer);
}

/**
 * @brief Dispatcher for fexit trace functions.
 * @param tf Pointer to the `trace_fprobe`.
 * @param entry_ip Instruction pointer at function entry.
 * @param ret_ip Return instruction pointer.
 * @param fregs Pointer to ftrace registers.
 * @param entry_data Per-entry private data stored at function entry.
 *
 * This function iterates through all associated `trace_event_file` links
 * and calls `__fexit_trace_func` for each to record the event.
 */
static void
fexit_trace_func(struct trace_fprobe *tf, unsigned long entry_ip,
		 unsigned long ret_ip, struct ftrace_regs *fregs, void *entry_data)
{
	struct event_file_link *link;

	trace_probe_for_each_link_rcu(link, &tf->tp)
		__fexit_trace_func(tf, entry_ip, ret_ip, fregs, entry_data, link->file);
}
NOKPROBE_SYMBOL(fexit_trace_func);

#ifdef CONFIG_PERF_EVENTS

/**
 * @brief Perf event handler for fentry (function entry) events.
 * @param tf Pointer to the `trace_fprobe`.
 * @param entry_ip Instruction pointer at function entry.
 * @param fregs Pointer to ftrace registers.
 * @return 0 on success.
 *
 * This function is called when an fentry perf event occurs. It allocates
 * a buffer, fills it with event data, and submits it to the perf event subsystem.
 */
static int fentry_perf_func(struct trace_fprobe *tf, unsigned long entry_ip,
			    struct ftrace_regs *fregs)
{
	struct trace_event_call *call = trace_probe_event_call(&tf->tp);
	struct fentry_trace_entry_head *entry;
	struct hlist_head *head;
	int size, __size, dsize;
	struct pt_regs *regs;
	int rctx;

	head = this_cpu_ptr(call->perf_events);
	// Block Logic: Returns if no perf events are registered for this event.
	if (hlist_empty(head))
		return 0;

	// Functional Utility: Calculates sizes for perf buffer allocation.
	dsize = __get_data_size(&tf->tp, fregs, NULL);
	__size = sizeof(*entry) + tf->tp.size + dsize;
	size = ALIGN(__size + sizeof(u32), sizeof(u64));
	size -= sizeof(u32);

	// Functional Utility: Allocates perf trace buffer.
	entry = perf_trace_buf_alloc(size, &regs, &rctx);
	if (!entry)
		return 0;

	regs = ftrace_fill_perf_regs(fregs, regs); // Functional Utility: Fills perf regs structure.

	entry->ip = entry_ip;
	memset(&entry[1], 0, dsize); // Functional Utility: Initializes dynamic data section.
	// Functional Utility: Stores trace arguments and submits to perf.
	store_trace_args(&entry[1], &tf->tp, fregs, NULL, sizeof(*entry), dsize);
	perf_trace_buf_submit(entry, size, rctx, call->event.type, 1, regs,
			      head, NULL);
	return 0;
}
NOKPROBE_SYMBOL(fentry_perf_func);

/**
 * @brief Perf event handler for fexit (function exit) events.
 * @param tf Pointer to the `trace_fprobe`.
 * @param entry_ip Instruction pointer at function entry.
 * @param ret_ip Return instruction pointer.
 * @param fregs Pointer to ftrace registers.
 * @param entry_data Per-entry private data stored at function entry.
 *
 * This function is called when an fexit perf event occurs. It allocates
 * a buffer, fills it with event data, and submits it to the perf event subsystem.
 */
static void
fexit_perf_func(struct trace_fprobe *tf, unsigned long entry_ip,
		 unsigned long ret_ip, struct ftrace_regs *fregs,
		 void *entry_data)
{
	struct trace_event_call *call = trace_probe_event_call(&tf->tp);
	struct fexit_trace_entry_head *entry;
	struct hlist_head *head;
	int size, __size, dsize;
	struct pt_regs *regs;
	int rctx;

	head = this_cpu_ptr(call->perf_events);
	// Block Logic: Returns if no perf events are registered.
	if (hlist_empty(head))
		return;

	// Functional Utility: Calculates sizes for perf buffer allocation.
	dsize = __get_data_size(&tf->tp, fregs, entry_data);
	__size = sizeof(*entry) + tf->tp.size + dsize;
	size = ALIGN(__size + sizeof(u32), sizeof(u64));
	size -= sizeof(u32);

	// Functional Utility: Allocates perf trace buffer.
	entry = perf_trace_buf_alloc(size, &regs, &rctx);
	if (!entry)
		return;

	regs = ftrace_fill_perf_regs(fregs, regs); // Functional Utility: Fills perf regs structure.

	entry->func = entry_ip;
	entry->ret_ip = ret_ip;
	// Functional Utility: Stores trace arguments and submits to perf.
	store_trace_args(&entry[1], &tf->tp, fregs, entry_data, sizeof(*entry), dsize);
	perf_trace_buf_submit(entry, size, rctx, call->event.type, 1, regs,
			      head, NULL);
}
NOKPROBE_SYMBOL(fexit_perf_func);
#endif	/* CONFIG_PERF_EVENTS */

/**
 * @brief Dispatcher for fentry events.
 * @param fp Pointer to the `fprobe` instance.
 * @param entry_ip Instruction pointer at function entry.
 * @param ret_ip Return instruction pointer.
 * @param fregs Pointer to ftrace registers.
 * @param entry_data Per-entry private data.
 * @return An integer status code.
 *
 * This function orchestrates the calls to trace and perf event handlers
 * for function entry events.
 */
static int fentry_dispatcher(struct fprobe *fp, unsigned long entry_ip,
			     unsigned long ret_ip, struct ftrace_regs *fregs,
			     void *entry_data)
{
	struct trace_fprobe *tf = container_of(fp, struct trace_fprobe, fp);
	int ret = 0;

	// Block Logic: If trace flag is set, calls fentry trace function.
	if (trace_probe_test_flag(&tf->tp, TP_FLAG_TRACE))
		fentry_trace_func(tf, entry_ip, fregs);

#ifdef CONFIG_PERF_EVENTS
	// Block Logic: If perf flag is set, calls fentry perf function.
	if (trace_probe_test_flag(&tf->tp, TP_FLAG_PROFILE))
		ret = fentry_perf_func(tf, entry_ip, fregs);
#endif
	return ret;
}
NOKPROBE_SYMBOL(fentry_dispatcher);

/**
 * @brief Dispatcher for fexit events.
 * @param fp Pointer to the `fprobe` instance.
 * @param entry_ip Instruction pointer at function entry.
 * @param ret_ip Return instruction pointer.
 * @param fregs Pointer to ftrace registers.
 * @param entry_data Per-entry private data.
 *
 * This function orchestrates the calls to trace and perf event handlers
 * for function exit events.
 */
static void fexit_dispatcher(struct fprobe *fp, unsigned long entry_ip,
			     unsigned long ret_ip, struct ftrace_regs *fregs,
			     void *entry_data)
{
	struct trace_fprobe *tf = container_of(fp, struct trace_fprobe, fp);

	// Block Logic: If trace flag is set, calls fexit trace function.
	if (trace_probe_test_flag(&tf->tp, TP_FLAG_TRACE))
		fexit_trace_func(tf, entry_ip, ret_ip, fregs, entry_data);
#ifdef CONFIG_PERF_EVENTS
	// Block Logic: If perf flag is set, calls fexit perf function.
	if (trace_probe_test_flag(&tf->tp, TP_FLAG_PROFILE))
		fexit_perf_func(tf, entry_ip, ret_ip, fregs, entry_data);
#endif
}
NOKPROBE_SYMBOL(fexit_dispatcher);

/**
 * @brief Frees resources associated with a `trace_fprobe` instance.
 * @param tf Pointer to the `trace_fprobe` instance.
 *
 * This function cleans up the trace probe, symbol string, and the fprobe
 * structure itself.
 */
static void free_trace_fprobe(struct trace_fprobe *tf)
{
	if (tf) {
		trace_probe_cleanup(&tf->tp);
		kfree(tf->symbol);
		kfree(tf);
	}
}

/* Since alloc_trace_fprobe() can return error, check the pointer is ERR too. */
/**
 * @brief Helper macro for `free_trace_fprobe` to be used with `DEFINE_FREE`.
 */
DEFINE_FREE(free_trace_fprobe, struct trace_fprobe *, if (!IS_ERR_OR_NULL(_T)) free_trace_fprobe(_T))

/*
 * Allocate new trace_probe and initialize it (including fprobe).
 */
/**
 * @brief Allocates and initializes a new `trace_fprobe` instance.
 * @param group The group name for the event.
 * @param event The event name.
 * @param symbol The target symbol name.
 * @param tpoint The target tracepoint (if applicable).
 * @param mod The module containing the target.
 * @param nargs Number of arguments.
 * @param is_return True if it's a return probe, false otherwise.
 * @return Pointer to the newly allocated `trace_fprobe` on success, or an `ERR_PTR` on failure.
 *
 * This function allocates memory for the `trace_fprobe`, duplicates the symbol
 * name, sets up entry/exit handlers, and initializes embedded `trace_probe`
 * and `dyn_event` structures.
 */
static struct trace_fprobe *alloc_trace_fprobe(const char *group,
				       const char *event,
				       const char *symbol,
				       struct tracepoint *tpoint,
				       struct module *mod,
				       int nargs, bool is_return)
{
	struct trace_fprobe *tf __free(free_trace_fprobe) = NULL;
	int ret = -ENOMEM;

	tf = kzalloc(struct_size(tf, tp.args, nargs), GFP_KERNEL);
	if (!tf)
		return ERR_PTR(ret);

	tf->symbol = kstrdup(symbol, GFP_KERNEL);
	if (!tf->symbol)
		return ERR_PTR(-ENOMEM);

	// Block Logic: Sets the appropriate fprobe handler based on `is_return`.
	if (is_return)
		tf->fp.exit_handler = fexit_dispatcher;
	else
		tf->fp.entry_handler = fentry_dispatcher;

	tf->tpoint = tpoint;
	tf->mod = mod;

	ret = trace_probe_init(&tf->tp, event, group, false, nargs);
	if (ret < 0)
		return ERR_PTR(ret);

	dyn_event_init(&tf->devent, &trace_fprobe_ops);
	return_ptr(tf);
}

/**
 * @brief Finds an existing `trace_fprobe` by event and group name.
 * @param event The event name.
 * @param group The group name.
 * @return Pointer to the found `trace_fprobe`, or NULL if not found.
 *
 * This function iterates through all dynamic events and checks if they
 * match the given event and group names.
 */
static struct trace_fprobe *find_trace_fprobe(const char *event,
				      const char *group)
{
	struct dyn_event *pos;
	struct trace_fprobe *tf;

	// Functional Utility: Iterates through all dynamic events.
	for_each_trace_fprobe(tf, pos)
		// Block Logic: Checks for matching event and group names.
		if (strcmp(trace_probe_name(&tf->tp), event) == 0 &&
		    strcmp(trace_probe_group_name(&tf->tp), group) == 0)
			return tf;
	return NULL;
}

/**
 * @brief Enables an `fprobe` instance within a `trace_fprobe`.
 * @param tf Pointer to the `trace_fprobe`.
 * @return 0 on success.
 *
 * This function calls `enable_fprobe` on the embedded `fprobe`
 * if it is registered.
 */
static inline int __enable_trace_fprobe(struct trace_fprobe *tf)
{
	if (trace_fprobe_is_registered(tf))
		enable_fprobe(&tf->fp);

	return 0;
}

/**
 * @brief Disables all `fprobe` instances linked to a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 *
 * This function iterates through all `trace_fprobe` instances associated
 * with the given `trace_probe` and disables their embedded `fprobe`s.
 */
static void __disable_trace_fprobe(struct trace_probe *tp)
{
	struct trace_fprobe *tf;

	list_for_each_entry(tf, trace_probe_probe_list(tp), tp.list) {
		if (!trace_fprobe_is_registered(tf))
			continue;
		disable_fprobe(&tf->fp);
	}
}

/*
 * Enable trace_probe
 * if the file is NULL, enable "perf" handler, or enable "trace" handler.
 */
/**
 * @brief Enables a `trace_fprobe` or a group of fprobes associated with a `trace_event_call`.
 * @param call Pointer to the `trace_event_call` for the fprobe.
 * @param file Optional: Pointer to the `trace_event_file` to associate.
 * @return 0 on success, -ENODEV if `trace_probe` is invalid.
 *
 * This function enables all fprobes linked to the provided `trace_probe`
 * (primary from `call`). It handles both file-associated and profile-flagged
 * enabling.
 */
static int enable_trace_fprobe(struct trace_event_call *call,
			       struct trace_event_file *file)
{
	struct trace_probe *tp;
	struct trace_fprobe *tf;
	bool enabled;
	int ret = 0;

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

	// Block Logic: If not already enabled, iterates through and enables each linked fprobe.
	if (!enabled) {
		list_for_each_entry(tf, trace_probe_probe_list(tp), tp.list) {
			/* TODO: check the fprobe is gone */
			__enable_trace_fprobe(tf);
		}
	}

	return 0;
}

/*
 * Disable trace_probe
 * if the file is NULL, disable "perf" handler, or disable "trace" handler.
 */
/**
 * @brief Disables a `trace_fprobe` or a group of fprobes associated with a `trace_event_call`.
 * @param call Pointer to the `trace_event_call` for the fprobe.
 * @param file Optional: Pointer to the `trace_event_file` to disassociate.
 * @return 0 on success, -ENODEV if `trace_probe` is invalid, -ENOENT if file link not found.
 *
 * This function disables all fprobes linked to the provided `trace_probe`.
 * It handles both file-associated and profile-flagged disabling.
 */
static int disable_trace_fprobe(struct trace_event_call *call,
				struct trace_event_file *file)
{
	struct trace_probe *tp;

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

	// Block Logic: If no other probes are enabled, disables all linked fprobes.
	if (!trace_probe_is_enabled(tp))
		__disable_trace_fprobe(tp);

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

/* Event entry printers */
/**
 * @brief Printer function for fentry (function entry) events in `trace_seq`.
 * @param iter Pointer to the `trace_iterator`.
 * @param flags Print flags.
 * @param event Pointer to the `trace_event`.
 * @return `print_line_t` status.
 *
 * This function formats and prints fentry event data, including the
 * function entry IP and arguments.
 */
static enum print_line_t
print_fentry_event(struct trace_iterator *iter, int flags,
		   struct trace_event *event)
{
	struct fentry_trace_entry_head *field;
	struct trace_seq *s = &iter->seq;
	struct trace_probe *tp;

	field = (struct fentry_trace_entry_head *)iter->ent;
	tp = trace_probe_primary_from_call(
		container_of(event, struct trace_event_call, event));
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		goto out;

	trace_seq_printf(s, "%s: (", trace_probe_name(tp));

	// Functional Utility: Prints the instruction pointer as a symbol.
	if (!seq_print_ip_sym(s, field->ip, flags | TRACE_ITER_SYM_OFFSET))
		goto out;

	trace_seq_putc(s, ')');

	// Block Logic: Prints the arguments of the fprobe.
	if (trace_probe_print_args(s, tp->args, tp->nr_args,
			     (u8 *)&field[1], field) < 0)
		goto out;

	trace_seq_putc(s, '\n');
 out:
	return trace_handle_return(s);
}

/**
 * @brief Printer function for fexit (function exit) events in `trace_seq`.
 * @param iter Pointer to the `trace_iterator`.
 * @param flags Print flags.
 * @param event Pointer to the `trace_event`.
 * @return `print_line_t` status.
 *
 * This function formats and prints fexit event data, including the
 * return IP, function entry IP, and arguments.
 */
static enum print_line_t
print_fexit_event(struct trace_iterator *iter, int flags,
		  struct trace_event *event)
{
	struct fexit_trace_entry_head *field;
	struct trace_seq *s = &iter->seq;
	struct trace_probe *tp;

	field = (struct fexit_trace_entry_head *)iter->ent;
	tp = trace_probe_primary_from_call(
		container_of(event, struct trace_event_call, event));
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		goto out;

	trace_seq_printf(s, "%s: (", trace_probe_name(tp));

	// Functional Utility: Prints the return instruction pointer.
	if (!seq_print_ip_sym(s, field->ret_ip, flags | TRACE_ITER_SYM_OFFSET))
		goto out;

	trace_seq_puts(s, " <-");

	// Functional Utility: Prints the function entry instruction pointer.
	if (!seq_print_ip_sym(s, field->func, flags & ~TRACE_ITER_SYM_OFFSET))
		goto out;

	trace_seq_putc(s, ')');

	// Block Logic: Prints the arguments of the fprobe.
	if (trace_probe_print_args(s, tp->args, tp->nr_args,
			     (u8 *)&field[1], field) < 0)
		goto out;

	trace_seq_putc(s, '\n');

 out:
	return trace_handle_return(s);
}

/**
 * @brief Defines the fields for an fentry (function entry) event call.
 * @param event_call Pointer to the `trace_event_call` for the fentry event.
 * @return 0 on success, -ENOENT if `trace_probe` is not found.
 *
 * This function defines the `ip` field for fentry events.
 */
static int fentry_event_define_fields(struct trace_event_call *event_call)
{
	int ret;
	struct fentry_trace_entry_head field;
	struct trace_probe *tp;

	tp = trace_probe_primary_from_call(event_call);
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		return -ENOENT;

	DEFINE_FIELD(unsigned long, ip, FIELD_STRING_IP, 0); // Functional Utility: Defines the instruction pointer field.

	return traceprobe_define_arg_fields(event_call, sizeof(field), tp);
}

/**
 * @brief Defines the fields for an fexit (function exit) event call.
 * @param event_call Pointer to the `trace_event_call` for the fexit event.
 * @return 0 on success, -ENOENT if `trace_probe` is not found.
 *
 * This function defines the `func` and `ret_ip` fields for fexit events.
 */
static int fexit_event_define_fields(struct trace_event_call *event_call)
{
	int ret;
	struct fexit_trace_entry_head field;
	struct trace_probe *tp;

	tp = trace_probe_primary_from_call(event_call);
	// Block Logic: Ensures `trace_probe` is valid.
	if (WARN_ON_ONCE(!tp))
		return -ENOENT;

	DEFINE_FIELD(unsigned long, func, FIELD_STRING_FUNC, 0); // Functional Utility: Defines the function entry IP field.
	DEFINE_FIELD(unsigned long, ret_ip, FIELD_STRING_RETIP, 0); // Functional Utility: Defines the return IP field.

	return traceprobe_define_arg_fields(event_call, sizeof(field), tp);
}

/**
 * @var fentry_funcs
 * @brief Trace event functions for fentry events.
 *
 * Defines the `trace` function (printer) for fentry events.
 */
static struct trace_event_functions fentry_funcs = {
	.trace		= print_fentry_event
};

/**
 * @var fexit_funcs
 * @brief Trace event functions for fexit events.
 *
 * Defines the `trace` function (printer) for fexit events.
 */
static struct trace_event_functions fexit_funcs = {
	.trace		= print_fexit_event
};

/**
 * @var fentry_fields_array
 * @brief Array of `trace_event_fields` for fentry events.
 *
 * Defines how fentry event fields are defined, primarily using
 * `fentry_event_define_fields`.
 */
static struct trace_event_fields fentry_fields_array[] = {
	{ .type = TRACE_FUNCTION_TYPE,
	  .define_fields = fentry_event_define_fields },
	{}
};

/**
 * @var fexit_fields_array
 * @brief Array of `trace_event_fields` for fexit events.
 *
 * Defines how fexit event fields are defined, primarily using
 * `fexit_event_define_fields`.
 */
static struct trace_event_fields fexit_fields_array[] = {
	{ .type = TRACE_FUNCTION_TYPE,
	  .define_fields = fexit_event_define_fields },
	{}
};

/**
 * @brief Register/unregister callback for fprobe events.
 * @param event Pointer to the `trace_event_call`.
 * @param type Type of registration operation (`TRACE_REG_REGISTER`, `TRACE_REG_UNREGISTER`, etc.).
 * @param data Opaque data (e.g., `trace_event_file`).
 * @return 0 on success.
 *
 * This function acts as the central registration point for fprobes,
 * routing `REGISTER` and `UNREGISTER` requests to `enable_trace_fprobe`
 * and `disable_trace_fprobe` respectively.
 */
static int fprobe_register(struct trace_event_call *event,
			   enum trace_reg type, void *data);

/**
 * @brief Initializes the `trace_event_call` for a `trace_fprobe`.
 * @param tf Pointer to the `trace_fprobe`.
 *
 * This function sets various flags and callbacks for the fprobe's
 * `trace_event_call`, linking it to the appropriate fentry or fexit
 * printing functions and field definitions.
 */
static inline void init_trace_event_call(struct trace_fprobe *tf)
{
	struct trace_event_call *call = trace_probe_event_call(&tf->tp);

	// Block Logic: Configures event call based on whether it's a return probe.
	if (trace_fprobe_is_return(tf)) {
		call->event.funcs = &fexit_funcs;
		call->class->fields_array = fexit_fields_array;
	} else {
		call->event.funcs = &fentry_funcs;
		call->class->fields_array = fentry_fields_array;
	}

	call->flags = TRACE_EVENT_FL_FPROBE; // Functional Utility: Sets the fprobe flag.
	call->class->reg = fprobe_register; // Functional Utility: Sets the registration callback.
}

/**
 * @brief Registers the `trace_event_call` for an fprobe.
 * @param tf Pointer to the `trace_fprobe`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function initializes the event call structure and then
 * registers it with the trace probe subsystem.
 */
static int register_fprobe_event(struct trace_fprobe *tf)
{
	init_trace_event_call(tf); // Functional Utility: Initializes the event call.

	return trace_probe_register_event_call(&tf->tp);
}

/**
 * @brief Unregisters the `trace_event_call` for an fprobe.
 * @param tf Pointer to the `trace_fprobe`.
 * @return 0 on success, or a negative errno on failure.
 */
static int unregister_fprobe_event(struct trace_fprobe *tf)
{
	return trace_probe_unregister_event_call(&tf->tp);
}

/**
 * @brief Registers an fprobe on a tracepoint.
 * @param tf Pointer to the `trace_fprobe`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function registers a tracepoint probe and then an fprobe
 * on the tracepoint's stub function.
 */
static int __regsiter_tracepoint_fprobe(struct trace_fprobe *tf)
{
	struct tracepoint *tpoint = tf->tpoint;
	unsigned long ip = (unsigned long)tpoint->probestub;
	int ret;

	/*
	 * Here, we do 2 steps to enable fprobe on a tracepoint.
	 * At first, put __probestub_##TP function on the tracepoint
	 * and put a fprobe on the stub function.
	 */
	// Functional Utility: Registers a tracepoint probe.
	ret = tracepoint_probe_register_prio_may_exist(tpoint,
								tpoint->probestub, NULL, 0);
	if (ret < 0)
		return ret;
	// Functional Utility: Registers an fprobe on the tracepoint's stub function.
	return register_fprobe_ips(&tf->fp, &ip, 1);
}

/* Internal register function - just handle fprobe and flags */
/**
 * @brief Internal function to register a `trace_fprobe`.
 * @param tf Pointer to the `trace_fprobe`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function handles the security checks, argument updates, and
 * registration with the fprobe framework for both function and
 * tracepoint-based fprobes.
 */
static int __register_trace_fprobe(struct trace_fprobe *tf)
{
	int i, ret;

	/* Should we need new LOCKDOWN flag for fprobe? */
	// Functional Utility: Performs security lockdown check.
	ret = security_locked_down(LOCKDOWN_KPROBES);
	if (ret)
		return ret;

	// Block Logic: Returns error if fprobe is already registered.
	if (trace_fprobe_is_registered(tf))
		return -EINVAL;

	// Block Logic: Updates each trace probe argument.
	for (i = 0; i < tf->tp.nr_args; i++) {
		ret = traceprobe_update_arg(&tf->tp.args[i]);
		if (ret)
			return ret;
	}

	/* Set/clear disabled flag according to tp->flag */
	// Block Logic: Sets or clears the `FPROBE_FL_DISABLED` flag based on `trace_probe`'s enabled state.
	if (trace_probe_is_enabled(&tf->tp))
		tf->fp.flags &= ~FPROBE_FL_DISABLED;
	else
		tf->fp.flags |= FPROBE_FL_DISABLED;

	// Block Logic: Handles tracepoint-based fprobes.
	if (trace_fprobe_is_tracepoint(tf)) {

		/* This tracepoint is not loaded yet */
		// Block Logic: If tracepoint is a stub (not loaded), returns 0.
		if (tf->tpoint == TRACEPOINT_STUB)
			return 0;

		return __regsiter_tracepoint_fprobe(tf);
	}

	/* TODO: handle filter, nofilter or symbol list */
	// Functional Utility: Registers fprobe based on symbol (or filter/nofilter).
	return register_fprobe(&tf->fp, tf->symbol, NULL);
}

/* Internal unregister function - just handle fprobe and flags */
/**
 * @brief Internal function to unregister a `trace_fprobe`.
 * @param tf Pointer to the `trace_fprobe`.
 *
 * This function unregisters the embedded `fprobe` and, if it's a
 * tracepoint-based fprobe, unregisters the tracepoint probe.
 */
static void __unregister_trace_fprobe(struct trace_fprobe *tf)
{
	// Block Logic: If fprobe is registered, unregisters it.
	if (trace_fprobe_is_registered(tf)) {
		unregister_fprobe(&tf->fp);
		memset(&tf->fp, 0, sizeof(tf->fp)); // Functional Utility: Clears the fprobe structure.
		// Block Logic: If tracepoint-based, unregisters the tracepoint probe.
		if (trace_fprobe_is_tracepoint(tf)) {
			tracepoint_probe_unregister(tf->tpoint,
						tf->tpoint->probestub, NULL);
			tf->tpoint = TRACEPOINT_STUB; // Functional Utility: Sets tracepoint to stub.
			tf->mod = NULL; // Functional Utility: Clears module.
		}
	}
}

/* TODO: make this trace_*probe common function */
/**
 * @brief Unregisters a `trace_fprobe` instance.
 * @param tf Pointer to the `trace_fprobe`.
 * @return 0 on success, -EBUSY if the fprobe is enabled or busy.
 *
 * This function removes the fprobe from the dynamic event list, unlinks
 * its trace probe, and unregisters its embedded fprobe.
 */
static int unregister_trace_fprobe(struct trace_fprobe *tf)
{
	/* If other probes are on the event, just unregister fprobe */
	// Block Logic: If other probes are sharing the event, just unlinks the trace probe.
	if (trace_probe_has_sibling(&tf->tp))
		goto unreg;

	/* Enabled event can not be unregistered */
	// Block Logic: Returns busy if the trace probe is enabled.
	if (trace_probe_is_enabled(&tf->tp))
		return -EBUSY;

	/* If there's a reference to the dynamic event */
	// Block Logic: Returns busy if the dynamic event is busy.
	if (trace_event_dyn_busy(trace_probe_event_call(&tf->tp)))
		return -EBUSY;

	/* Will fail if probe is being used by ftrace or perf */
	// Block Logic: Returns busy if the fprobe event is still in use.
	if (unregister_fprobe_event(tf))
		return -EBUSY;

unreg:
	__unregister_trace_fprobe(tf); // Functional Utility: Unregisters the embedded fprobe.
	dyn_event_remove(&tf->devent); // Functional Utility: Removes from dynamic event management.
	trace_probe_unlink(&tf->tp); // Functional Utility: Unlinks the trace probe.

	return 0;
}

/**
 * @brief Checks if two `trace_fprobe` instances target the same fprobe.
 * @param orig Pointer to the original `trace_fprobe`.
 * @param comp Pointer to the comparing `trace_fprobe`.
 * @return True if they have the same fprobe (symbol and arguments), false otherwise.
 *
 * This function compares the symbol name and command arguments to determine
 * if two fprobes are essentially the same.
 */
static bool trace_fprobe_has_same_fprobe(struct trace_fprobe *orig,
					 struct trace_fprobe *comp)
{
	struct trace_probe_event *tpe = orig->tp.event;
	int i;

	// Functional Utility: Iterates through linked fprobes to find a match.
	list_for_each_entry(orig, &tpe->probes, tp.list) {
		// Block Logic: Compares symbol names.
		if (strcmp(trace_fprobe_symbol(orig),
			   trace_fprobe_symbol(comp)))
			continue;

		/*
		 * trace_probe_compare_arg_type() ensured that nr_args and
		 * each argument name and type are same. Let's compare comm.
		 */
		// Block Logic: Compares command arguments.
		for (i = 0; i < orig->tp.nr_args; i++) {
			if (strcmp(orig->tp.args[i].comm,
					   comp->tp.args[i].comm))
				break;
		}

		if (i == orig->tp.nr_args)
			return true;
	}

	return false;
}

/**
 * @brief Appends a new `trace_fprobe` to an existing one.
 * @param tf Pointer to the `trace_fprobe` to append.
 * @param to Pointer to the existing `trace_fprobe` to append to.
 * @return 0 on success, -EEXIST if probe types or argument types mismatch, or a negative errno on failure.
 *
 * This function checks for compatibility between the two fprobes (probe type,
 * argument types) and then appends the new fprobe to the existing one's
 * `trace_probe` list.
 */
static int append_trace_fprobe(struct trace_fprobe *tf, struct trace_fprobe *to)
{
	int ret;

	// Block Logic: Checks for mismatch in probe type (return vs. entry, tracepoint vs. function).
	if (trace_fprobe_is_return(tf) != trace_fprobe_is_return(to) ||
	    trace_fprobe_is_tracepoint(tf) != trace_fprobe_is_tracepoint(to)) {
		trace_probe_log_set_index(0);
		trace_probe_log_err(0, DIFF_PROBE_TYPE);
		return -EEXIST;
	}
	// Functional Utility: Compares argument types.
	ret = trace_probe_compare_arg_type(&tf->tp, &to->tp);
	if (ret) {
		/* Note that argument starts index = 2 */
		trace_probe_log_set_index(ret + 1);
		trace_probe_log_err(0, DIFF_ARG_TYPE);
		return -EEXIST;
	}
	// Block Logic: Checks for duplicate fprobes.
	if (trace_fprobe_has_same_fprobe(to, tf)) {
		trace_probe_log_set_index(0);
		trace_probe_log_err(0, SAME_PROBE);
		return -EEXIST;
	}

	/* Append to existing event */
	ret = trace_probe_append(&tf->tp, &to->tp);
	if (ret)
		return ret;

	// Functional Utility: Registers the appended fprobe.
	ret = __register_trace_fprobe(tf);
	if (ret)
		trace_probe_unlink(&tf->tp);
	else
		dyn_event_add(&tf->devent, trace_probe_event_call(&tf->tp));

	return ret;
}

/**
 * @brief Registers a `trace_fprobe` instance.
 * @param tf Pointer to the `trace_fprobe`.
 * @return 0 on success, -EEXIST if an event with the same name already exists, or a negative errno on failure.
 *
 * This function registers a new fprobe, either by appending it to an existing
 * trace event or by creating a new one. It handles the registration of
 * the underlying `fprobe` and `trace_event_call`.
 */
static int register_trace_fprobe(struct trace_fprobe *tf)
{
	struct trace_fprobe *old_tf;
	int ret;

	guard(mutex)(&event_mutex); // Functional Utility: Locks `event_mutex`.

	// Functional Utility: Checks if an fprobe with the same event and group already exists.
	old_tf = find_trace_fprobe(trace_probe_name(&tf->tp),
				   trace_probe_group_name(&tf->tp));
	// Block Logic: If existing, appends to it.
	if (old_tf)
		return append_trace_fprobe(tf, old_tf);

	/* Register new event */
	ret = register_fprobe_event(tf);
	if (ret) {
		// Block Logic: Handles existing event name.
		if (ret == -EEXIST) {
			trace_probe_log_set_index(0);
			trace_probe_log_err(0, EVENT_EXIST);
		}
		else
			pr_warn("Failed to register probe event(%d)\n", ret);
		return ret;
	}

	/* Register fprobe */
	// Functional Utility: Registers the embedded fprobe.
	ret = __register_trace_fprobe(tf);
	if (ret < 0)
		unregister_fprobe_event(tf); // Functional Utility: Unregisters event on failure.
	else
		dyn_event_add(&tf->devent, trace_probe_event_call(&tf->tp));

	return ret;
}

/**
 * @struct __find_tracepoint_cb_data
 * @brief Private data for tracepoint lookup callbacks.
 *
 * Used to pass search criteria and results between the caller and
 * `for_each_kernel_tracepoint` and `for_each_module_tracepoint`.
 */
struct __find_tracepoint_cb_data {
	const char *tp_name;        /**< @brief The name of the tracepoint to find. */
	struct tracepoint *tpoint;    /**< @brief Pointer to the found tracepoint. */
	struct module *mod;         /**< @brief Pointer to the module (if module tracepoint). */
};

/**
 * @brief Callback function for iterating module tracepoints.
 * @param tp Pointer to the tracepoint.
 * @param mod Pointer to the module owning the tracepoint.
 * @param priv Pointer to `__find_tracepoint_cb_data`.
 *
 * This function finds a tracepoint by name within a module, and optionally
 * gets a reference to the module.
 */
static void __find_tracepoint_module_cb(struct tracepoint *tp, struct module *mod, void *priv)
{
	struct __find_tracepoint_cb_data *data = priv;

	// Block Logic: If tracepoint not found yet and name matches.
	if (!data->tpoint && !strcmp(data->tp_name, tp->name)) {
		/* If module is not specified, try getting module refcount. */
		// Block Logic: If module is not already set and a module is provided.
		if (!data->mod && mod) {
			/* If failed to get refcount, ignore this tracepoint. */
			// Functional Utility: Tries to get a reference to the module.
			if (!try_module_get(mod))
				return;

			data->mod = mod;
		}
		data->tpoint = tp;
	}
}

/**
 * @brief Callback function for iterating kernel tracepoints.
 * @param tp Pointer to the tracepoint.
 * @param priv Pointer to `__find_tracepoint_cb_data`.
 *
 * This function finds a tracepoint by name within the kernel.
 */
static void __find_tracepoint_cb(struct tracepoint *tp, void *priv)
{
	struct __find_tracepoint_cb_data *data = priv;

	// Block Logic: If tracepoint not found yet and name matches.
	if (!data->tpoint && !strcmp(data->tp_name, tp->name))
		data->tpoint = tp;
}

/*
 * Find a tracepoint from kernel and module. If the tracepoint is on the module,
 * the module's refcount is incremented and returned as *@tp_mod. Thus, if it is
 * not NULL, caller must call module_put(*tp_mod) after used the tracepoint.
 */
/**
 * @brief Finds a tracepoint by name, searching both kernel and modules.
 * @param tp_name The name of the tracepoint to find.
 * @param tp_mod Output parameter: Pointer to the module owning the tracepoint (if found in a module).
 * @return Pointer to the found `tracepoint`, or NULL if not found.
 *
 * If the tracepoint is found in a module, its module's reference count is
 * incremented, and the caller is responsible for calling `module_put()`.
 */
static struct tracepoint *find_tracepoint(const char *tp_name,
					  struct module **tp_mod)
{
	struct __find_tracepoint_cb_data data = {
		.tp_name = tp_name,
		.mod = NULL,
	};

	// Functional Utility: Searches kernel tracepoints.
	for_each_kernel_tracepoint(__find_tracepoint_cb, &data);

	// Block Logic: If not found in kernel and modules are enabled, searches module tracepoints.
	if (!data.tpoint && IS_ENABLED(CONFIG_MODULES)) {
		for_each_module_tracepoint(__find_tracepoint_module_cb, &data);
		*tp_mod = data.mod;
	}

	return data.tpoint;
}

#ifdef CONFIG_MODULES
/**
 * @brief Re-enables a `trace_fprobe` that was previously disabled due to module events.
 * @param tf Pointer to the `trace_fprobe`.
 *
 * This function iterates through all fprobes linked to the trace probe
 * and re-enables their embedded fprobes.
 */
static void reenable_trace_fprobe(struct trace_fprobe *tf)
{
	struct trace_probe *tp = &tf->tp;

	list_for_each_entry(tf, trace_probe_probe_list(tp), tp.list) {
		__enable_trace_fprobe(tf);
	}
}

/*
 * Find a tracepoint from specified module. In this case, this does not get the
 * module's refcount. The caller must ensure the module is not freed.
 */
/**
 * @brief Finds a tracepoint by name within a specific module.
 * @param mod Pointer to the module to search within.
 * @param tp_name The name of the tracepoint to find.
 * @return Pointer to the found `tracepoint`, or NULL if not found.
 *
 * This function does not increment the module's reference count; the caller
 * must ensure the module remains loaded.
 */
static struct tracepoint *find_tracepoint_in_module(struct module *mod,
						    const char *tp_name)
{
	struct __find_tracepoint_cb_data data = {
		.tp_name = tp_name,
		.mod = mod,
	};

	// Functional Utility: Iterates through tracepoints in the specified module.
	for_each_tracepoint_in_module(mod, __find_tracepoint_module_cb, &data);
	return data.tpoint;
}

/**
 * @brief Callback function for tracepoint module notifier.
 * @param self Pointer to the `notifier_block`.
 * @param val The notification value (module state).
 * @param data Pointer to `tp_module` (module with tracepoints).
 * @return `NOTIFY_DONE`.
 *
 * This function handles module state changes, specifically for `MODULE_STATE_COMING`
 * and `MODULE_STATE_GOING`, to dynamically register/unregister fprobes on
 * tracepoints within the affected module.
 */
static int __tracepoint_probe_module_cb(struct notifier_block *self,
					unsigned long val, void *data)
{
	struct tp_module *tp_mod = data;
	struct tracepoint *tpoint;
	struct trace_fprobe *tf;
	struct dyn_event *pos;

	if (val != MODULE_STATE_GOING && val != MODULE_STATE_COMING)
		return NOTIFY_DONE;

	mutex_lock(&event_mutex); // Functional Utility: Locks `event_mutex`.
	// Functional Utility: Iterates through all fprobes.
	for_each_trace_fprobe(tf, pos) {
		// Block Logic: Handles `MODULE_STATE_COMING` for tracepoint-based fprobes.
		if (val == MODULE_STATE_COMING && tf->tpoint == TRACEPOINT_STUB) {
			tpoint = find_tracepoint_in_module(tp_mod->mod, tf->symbol);
			if (tpoint) {
				tf->tpoint = tpoint;
				tf->mod = tp_mod->mod;
				// Block Logic: If successful and enabled, re-enables the fprobe.
				if (!WARN_ON_ONCE(__regsiter_tracepoint_fprobe(tf)) &&
				    trace_probe_is_enabled(&tf->tp))
					reenable_trace_fprobe(tf);
			}
		} else if (val == MODULE_STATE_GOING && tp_mod->mod == tf->mod) { // Block Logic: Handles `MODULE_STATE_GOING` for modules.
			unregister_fprobe(&tf->fp); // Functional Utility: Unregisters the embedded fprobe.
			// Block Logic: If tracepoint-based, unregisters the tracepoint probe.
			if (trace_fprobe_is_tracepoint(tf)) {
				tracepoint_probe_unregister(tf->tpoint,
						tf->tpoint->probestub, NULL);
				tf->tpoint = TRACEPOINT_STUB;
				tf->mod = NULL;
			}
		}
	}
	mutex_unlock(&event_mutex); // Functional Utility: Releases `event_mutex`.

	return NOTIFY_DONE;
}

/**
 * @var tracepoint_module_nb
 * @brief Notifier block for tracepoint module state changes.
 *
 * Registers `__tracepoint_probe_module_cb` to be notified of module events
 * affecting tracepoints.
 */
static struct notifier_block tracepoint_module_nb = {
	.notifier_call = __tracepoint_probe_module_cb,
};
#endif /* CONFIG_MODULES */

/**
 * @brief Parses the symbol and return information from command-line arguments.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param symbol Output parameter for the extracted symbol name.
 * @param is_return Output parameter: True if it's a return probe.
 * @param is_tracepoint True if it's a tracepoint-based probe.
 * @return 0 on success, -EINVAL on parsing error, -ENOMEM on memory allocation failure.
 *
 * This function extracts the target symbol name, determines if it's a return
 * probe (based on `%return` suffix or `$retval`), and validates tracepoint names.
 */
static int parse_symbol_and_return(int argc, const char *argv[],
				   char **symbol, bool *is_return,
				   bool is_tracepoint)
{
	char *tmp = strchr(argv[1], '%');
	int i;

	// Block Logic: Checks for `%return` suffix.
	if (tmp) {
		int len = tmp - argv[1];

		if (!is_tracepoint && !strcmp(tmp, "%return")) {
			*is_return = true;
		} else {
			trace_probe_log_err(len, BAD_ADDR_SUFFIX);
			return -EINVAL;
		}
		*symbol = kmemdup_nul(argv[1], len, GFP_KERNEL); // Functional Utility: Duplicates the symbol name.
	} else
		*symbol = kstrdup(argv[1], GFP_KERNEL); // Functional Utility: Duplicates the symbol name.
	if (!*symbol)
		return -ENOMEM;

	// Block Logic: If it's a return probe, returns early.
	if (*is_return)
		return 0;

	// Block Logic: If tracepoint-based, validates the symbol name.
	if (is_tracepoint) {
		tmp = *symbol;
		while (*tmp && (isalnum(*tmp) || *tmp == '_'))
			tmp++;
		if (*tmp) {
			/* find a wrong character. */
			trace_probe_log_err(tmp - *symbol, BAD_TP_NAME);
			kfree(*symbol);
			*symbol = NULL;
			return -EINVAL;
		}
	}

	/* If there is $retval, this should be a return fprobe. */
	// Block Logic: Checks for `$retval` in arguments to infer a return fprobe.
	for (i = 2; i < argc; i++) {
		tmp = strstr(argv[i], "$retval");
		if (tmp && !isalnum(tmp[7]) && tmp[7] != '_') {
			if (is_tracepoint) {
				trace_probe_log_set_index(i);
				trace_probe_log_err(tmp - argv[i], RETVAL_ON_PROBE);
				kfree(*symbol);
				*symbol = NULL;
				return -EINVAL;
			}
			*is_return = true;
			break;
		}
	}
	return 0;
}

/**
 * @brief Helper macro for `module_put` to be used with `DEFINE_FREE`.
 */
DEFINE_FREE(module_put, struct module *, if (_T) module_put(_T))

/**
 * @brief Internal function to create a `trace_fprobe` instance.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses the command-line arguments to define a new fprobe,
 * including its type (fentry/fexit/tracepoint), target symbol/tracepoint,
 * arguments, and dynamically registers it.
 */
static int trace_fprobe_create_internal(int argc, const char *argv[],
					struct traceprobe_parse_context *ctx)
{
	/*
	 * Argument syntax:
	 *  - Add fentry probe:
	 *      f[:[GRP/][EVENT]] [MOD:]KSYM [FETCHARGS]
	 *  - Add fexit probe:
	 *      f[N][:[GRP/][EVENT]] [MOD:]KSYM%return [FETCHARGS]
	 *  - Add tracepoint probe:
	 *      t[:[GRP/][EVENT]] TRACEPOINT [FETCHARGS]
	 *
	 * Fetch args:
	 *  $retval	: fetch return value
	 *  $stack	: fetch stack address
	 *  $stackN	: fetch Nth entry of stack (N:0-)
	 *  $argN	: fetch Nth argument (N:1-)
	 *  $comm       : fetch current task comm
	 *  @ADDR	: fetch memory at ADDR (ADDR should be in kernel)
	 *  @SYM[+|-offs] : fetch memory at SYM +|- offs (SYM is a data symbol)
	 * Dereferencing memory fetch:
	 *  +|-offs(ARG) : fetch memory at ARG +|- offs address.
	 * Alias name of args:
	 *  NAME=FETCHARG : set NAME as alias of FETCHARG.
	 * Type of args:
	 *  FETCHARG:TYPE : use TYPE instead of unsigned long.
	 */
	struct trace_fprobe *tf __free(free_trace_fprobe) = NULL;
	int i, new_argc = 0, ret = 0;
	bool is_return = false;
	char *symbol __free(kfree) = NULL;
	const char *event = NULL, *group = FPROBE_EVENT_SYSTEM;
	const char **new_argv __free(kfree) = NULL;
	char buf[MAX_EVENT_NAME_LEN];
	char gbuf[MAX_EVENT_NAME_LEN];
	char sbuf[KSYM_NAME_LEN];
	char abuf[MAX_BTF_ARGS_LEN];
	char *dbuf __free(kfree) = NULL;
	bool is_tracepoint = false;
	struct module *tp_mod __free(module_put) = NULL;
	struct tracepoint *tpoint = NULL;

	// Block Logic: Basic validation of command-line arguments.
	if ((argv[0][0] != 'f' && argv[0][0] != 't') || argc < 2)
		return -ECANCELED;

	// Block Logic: Determines if it's a tracepoint-based fprobe.
	if (argv[0][0] == 't') {
		is_tracepoint = true;
		group = TRACEPOINT_EVENT_SYSTEM;
	}

	// Block Logic: Parses event name and group from `argv[0]`.
	if (argv[0][1] != '\0') {
		if (argv[0][1] != ':') {
			trace_probe_log_set_index(0);
			trace_probe_log_err(1, BAD_MAXACT);
			return -EINVAL;
		}
		event = &argv[0][2];
	}

	trace_probe_log_set_index(1);

	/* a symbol(or tracepoint) must be specified */
	// Functional Utility: Parses symbol and return information.
	ret = parse_symbol_and_return(argc, argv, &symbol, &is_return, is_tracepoint);
	if (ret < 0)
		return -EINVAL;

	trace_probe_log_set_index(0);
	// Block Logic: Parses event name and group if provided.
	if (event) {
		ret = traceprobe_parse_event_name(&event, &group, gbuf,
							  event - argv[0]);
		if (ret)
			return -EINVAL;
	}

	// Block Logic: If event name not explicitly provided, generates one.
	if (!event) {
		/* Make a new event name */
		if (is_tracepoint)
			snprintf(buf, MAX_EVENT_NAME_LEN, "%s%s",
				 isdigit(*symbol) ? "_" : "", symbol);
		else
			snprintf(buf, MAX_EVENT_NAME_LEN, "%s__%s", symbol,
				 is_return ? "exit" : "entry");
		sanitize_event_name(buf); // Functional Utility: Sanitizes the event name.
		event = buf;
	}

	// Block Logic: Sets context flags for return or fentry.
	if (is_return)
		ctx->flags |= TPARG_FL_RETURN;
	else
		ctx->flags |= TPARG_FL_FENTRY;

	// Block Logic: Handles tracepoint-based fprobes.
	if (is_tracepoint) {
		ctx->flags |= TPARG_FL_TPOINT;
		tpoint = find_tracepoint(symbol, &tp_mod); // Functional Utility: Finds the tracepoint.
		if (tpoint) {
			// Functional Utility: Looks up the symbol name for the tracepoint's probestub.
			ctx->funcname = kallsyms_lookup(
				(unsigned long)tpoint->probestub,
				NULL, NULL, NULL, sbuf);
		} else if (IS_ENABLED(CONFIG_MODULES)) {
				/* This *may* be loaded afterwards */
				tpoint = TRACEPOINT_STUB; // Functional Utility: Sets tracepoint to stub if not found but modules enabled.
				ctx->funcname = symbol;
		} else {
			trace_probe_log_set_index(1);
			trace_probe_log_err(0, NO_TRACEPOINT);
			return -EINVAL;
		}
	} else
		ctx->funcname = symbol;

	argc -= 2; argv += 2;
	// Functional Utility: Expands meta arguments.
	new_argv = traceprobe_expand_meta_args(argc, argv, &new_argc,
							       abuf, MAX_BTF_ARGS_LEN, ctx);
	if (IS_ERR(new_argv))
		return PTR_ERR(new_argv);
	if (new_argv) {
		argc = new_argc;
		argv = new_argv;
	}
	// Block Logic: Checks for maximum number of arguments.
	if (argc > MAX_TRACE_ARGS) {
		trace_probe_log_set_index(2);
		trace_probe_log_err(0, TOO_MANY_ARGS);
		return -E2BIG;
	}

	// Functional Utility: Expands dentry arguments.
	ret = traceprobe_expand_dentry_args(argc, argv, &dbuf);
	if (ret)
		return ret;

	/* setup a probe */
	// Functional Utility: Allocates and initializes `trace_fprobe`.
	tf = alloc_trace_fprobe(group, event, symbol, tpoint, tp_mod,
					argc, is_return);
	if (IS_ERR(tf)) {
		ret = PTR_ERR(tf);
		/* This must return -ENOMEM, else there is a bug */
		WARN_ON_ONCE(ret != -ENOMEM);
		return ret;
	}

	/* parse arguments */
	// Block Logic: Parses each argument for the fprobe.
	for (i = 0; i < argc; i++) {
		trace_probe_log_set_index(i + 2);
		ctx->offset = 0;
		ret = traceprobe_parse_probe_arg(&tf->tp, i, argv[i], ctx);
		if (ret)
			return ret; /* This can be -ENOMEM */
	}

	// Block Logic: If it's a return probe with entry arguments, sets up entry handler and data size.
	if (is_return && tf->tp.entry_arg) {
		tf->fp.entry_handler = trace_fprobe_entry_handler;
		tf->fp.entry_data_size = traceprobe_get_entry_data_size(&tf->tp);
		// Block Logic: Checks if entry data size exceeds maximum.
		if (ALIGN(tf->fp.entry_data_size, sizeof(long)) > MAX_FPROBE_DATA_SIZE) {
			trace_probe_log_set_index(2);
			trace_probe_log_err(0, TOO_MANY_EARGS);
			return -E2BIG;
		}
	}

	// Functional Utility: Sets print format for the trace probe.
	ret = traceprobe_set_print_fmt(&tf->tp,
					is_return ? PROBE_PRINT_RETURN : PROBE_PRINT_NORMAL);
	if (ret < 0)
		return ret;

	ret = register_trace_fprobe(tf); // Functional Utility: Registers the trace fprobe.
	if (ret) {
		// Block Logic: Handles registration errors.
		trace_probe_log_set_index(1);
		if (ret == -EILSEQ)
			trace_probe_log_err(0, BAD_INSN_BNDRY);
		else if (ret == -ENOENT)
			trace_probe_log_err(0, BAD_PROBE_ADDR);
		else if (ret != -ENOMEM && ret != -EEXIST)
			trace_probe_log_err(0, FAIL_REG_PROBE);
		return -EINVAL;
	}

	/* 'tf' is successfully registered. To avoid freeing, assign NULL. */
	tf = NULL; // Functional Utility: Prevents freeing on successful registration.

	return 0;
}

/**
 * @brief Callback function for `trace_probe_create` to create an fprobe.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function initializes the `traceprobe_parse_context` and calls
 * `trace_fprobe_create_internal`.
 */
static int trace_fprobe_create_cb(int argc, const char *argv[])
{
	struct traceprobe_parse_context ctx = {
		.flags = TPARG_FL_KERNEL | TPARG_FL_FPROBE,
	};
	int ret;

	trace_probe_log_init("trace_fprobe", argc, argv); // Functional Utility: Initializes probe logging.
	ret = trace_fprobe_create_internal(argc, argv, &ctx);
	traceprobe_finish_parse(&ctx); // Functional Utility: Finishes parse context.
	trace_probe_log_clear(); // Functional Utility: Clears probe logging.
	return ret;
}

/**
 * @brief Creates a `trace_fprobe` from a raw command string.
 * @param raw_command The raw command string.
 * @return 0 on success, or a negative errno on failure.
 */
static int trace_fprobe_create(const char *raw_command)
{
	return trace_probe_create(raw_command, trace_fprobe_create_cb);
}

/**
 * @brief Releases resources associated with an fprobe dynamic event.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function unregisters the `trace_fprobe` and then frees its resources.
 */
static int trace_fprobe_release(struct dyn_event *ev)
{
	struct trace_fprobe *tf = to_trace_fprobe(ev);
	int ret = unregister_trace_fprobe(tf); // Functional Utility: Unregisters the fprobe.

	// Block Logic: Frees fprobe resources only if unregistration is successful.
	if (!ret)
		free_trace_fprobe(tf);
	return ret;
}

/**
 * @brief Shows the details of an fprobe dynamic event in a `seq_file`.
 * @param m Pointer to the `seq_file`.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success.
 *
 * This function formats and prints the fprobe event details, including
 * its type (f/t), group, event name, symbol, and arguments.
 */
static int trace_fprobe_show(struct seq_file *m, struct dyn_event *ev)
{
	struct trace_fprobe *tf = to_trace_fprobe(ev);
	int i;

	// Block Logic: Prints 't' for tracepoint or 'f' for function probe.
	if (trace_fprobe_is_tracepoint(tf))
		seq_putc(m, 't');
	else
		seq_putc(m, 'f');
	seq_printf(m, ":%s/%s", trace_probe_group_name(&tf->tp),
				trace_probe_name(&tf->tp));

	seq_printf(m, " %s%s", trace_fprobe_symbol(tf),
			       trace_fprobe_is_return(tf) ? "%return" : "");

	// Block Logic: Prints each argument of the trace probe.
	for (i = 0; i < tf->tp.nr_args; i++)
		seq_printf(m, " %s=%s", tf->tp.args[i].name, tf->tp.args[i].comm);
	seq_putc(m, '\n');

	return 0;
}

/*
 * called by perf_trace_init() or __ftrace_set_clr_event() under event_mutex.
 */
/**
 * @brief Register/unregister callback for fprobe events.
 * @param event Pointer to the `trace_event_call`.
 * @param type Type of registration operation (`TRACE_REG_REGISTER`, `TRACE_REG_UNREGISTER`, etc.).
 * @param data Opaque data (e.g., `trace_event_file`).
 * @return 0 on success.
 *
 * This function acts as the central registration point for fprobes,
 * routing `REGISTER` and `UNREGISTER` requests to `enable_trace_fprobe`
 * and `disable_trace_fprobe` respectively. It also handles perf event
 * registration.
 */
static int fprobe_register(struct trace_event_call *event,
			   enum trace_reg type, void *data)
{
	struct trace_event_file *file = data;

	switch (type) {
	case TRACE_REG_REGISTER:
		return enable_trace_fprobe(event, file);
	case TRACE_REG_UNREGISTER:
		return disable_trace_fprobe(event, file);

#ifdef CONFIG_PERF_EVENTS
	case TRACE_REG_PERF_REGISTER:
		return enable_trace_fprobe(event, NULL);
	case TRACE_REG_PERF_UNREGISTER:
		return disable_trace_fprobe(event, NULL);
	case TRACE_REG_PERF_OPEN:
	case TRACE_REG_PERF_CLOSE:
	case TRACE_REG_PERF_ADD:
	case TRACE_REG_PERF_DEL:
		return 0;
#endif
	}
	return 0;
}

/*
 * Register dynevent at core_initcall. This allows kernel to setup fprobe
 * events in postcore_initcall without tracefs.
 */
/**
 * @brief Early initialization function for fprobe tracing events.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function registers the `trace_fprobe_ops` with the dynamic event
 * system and, if modules are enabled, registers a notifier for tracepoint
 * module events. This allows fprobes to be set up early in the boot process.
 */
static __init int init_fprobe_trace_early(void)
{
	int ret;

	ret = dyn_event_register(&trace_fprobe_ops);
	if (ret)
		return ret;

#ifdef CONFIG_MODULES
	ret = register_tracepoint_module_notifier(&tracepoint_module_nb);
	if (ret)
		return ret;
#endif

	return 0;
}
core_initcall(init_fprobe_trace_early); // Functional Utility: Registers `init_fprobe_trace_early` as a core initialization call.