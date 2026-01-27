/**
 * @file trace_kprobe.c
 * @brief Implements Kprobes-based tracing events for the Linux kernel.
 *
 * This file provides the core logic for dynamically creating kprobe and kretprobe
 * tracing events. It allows users to define custom trace events that trigger
 * on function entry (kprobe) or function return (kretprobe), capturing and
 * displaying relevant kernel-internal data.
 *
 * Functional Utility:
 * - Dynamically defines new trace events based on kernel functions (kprobes)
 *   or function returns (kretprobes).
 * - Supports extracting function arguments, return values, stack addresses,
 *   and arbitrary memory locations at the probe point.
 * - Integrates with the ftrace and perf event subsystems for data collection and reporting.
 * - Provides mechanisms for handling module-specific symbols and dynamic event registration.
 * - Allows early definition of kprobe events from the kernel command line.
 *
 * Algorithms:
 * - Uses the kprobes/kretprobes framework for low-level function hooking.
 * - Leverages `trace_probe` infrastructure for argument parsing and event management.
 * - Employs `kallsyms` for symbol lookup and address resolution.
 * - Integrates with `dyn_event` for dynamic event registration.
 *
 * Architectural Intent:
 * - To provide a flexible and powerful mechanism for dynamic kernel introspection
 *   and debugging without modifying kernel source code.
 * - To enable users to create highly customized tracing points for detailed analysis
 *   of kernel behavior.
 *
 * Created by Masami Hiramatsu <mhiramat@redhat.com>
 *
 */
// SPDX-License-Identifier: GPL-2.0
#define pr_fmt(fmt)	"trace_kprobe: " fmt

#include <linux/bpf-cgroup.h>
#include <linux/cleanup.h>
#include <linux/security.h>
#include <linux/module.h>
#include <linux/uaccess.h>
#include <linux/rculist.h>
#include <linux/error-injection.h>

#include <asm/setup.h>  /* for COMMAND_LINE_SIZE */

#include "trace_dynevent.h"
#include "trace_kprobe_selftest.h"
#include "trace_probe.h"
#include "trace_probe_tmpl.h"
#include "trace_probe_kernel.h"

/**
 * @def KPROBE_EVENT_SYSTEM
 * @brief The default event system name for kprobes.
 *
 * Kprobes are grouped under this system name in the tracefs hierarchy.
 */
#define KPROBE_EVENT_SYSTEM "kprobes"
/**
 * @def KRETPROBE_MAXACTIVE_MAX
 * @brief Maximum number of active kretprobe instances allowed.
 *
 * This limits the number of kretprobe instances that can be simultaneously
 * active for a single kretprobe definition.
 */
#define KRETPROBE_MAXACTIVE_MAX 4096

/* Kprobe early definition from command line */
/**
 * @var kprobe_boot_events_buf
 * @brief Buffer to store kprobe event definitions from the kernel command line.
 *
 * Allows kprobe events to be specified and enabled early during boot.
 */
static char kprobe_boot_events_buf[COMMAND_LINE_SIZE] __initdata;

/**
 * @brief Initializes kprobe events from a command-line string.
 * @param str The command-line string containing kprobe event definitions.
 * @return 1 on success.
 *
 * This function parses kprobe event definitions from the kernel command line
 * and disables tracing selftests to prevent conflicts.
 */
static int __init set_kprobe_boot_events(char *str)
{
	strscpy(kprobe_boot_events_buf, str, COMMAND_LINE_SIZE);
	disable_tracing_selftest("running kprobe events");

	return 1;
}
__setup("kprobe_event=", set_kprobe_boot_events); // Functional Utility: Registers this function as a kernel command-line parser. 

/**
 * @brief Creates a new kprobe-based trace event.
 * @param raw_command The raw command string for the kprobe.
 * @return 0 on success, or a negative errno on failure.
 */
static int trace_kprobe_create(const char *raw_command); 
/**
 * @brief Displays the details of a kprobe dynamic event in a `seq_file`.
 * @param m Pointer to the `seq_file`.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success.
 */
static int trace_kprobe_show(struct seq_file *m, struct dyn_event *ev); 
/**
 * @brief Releases resources associated with a kprobe dynamic event.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success, or a negative errno on failure.
 */
static int trace_kprobe_release(struct dyn_event *ev); 
/**
 * @brief Checks if a kprobe dynamic event is busy (enabled).
 * @param ev Pointer to the dynamic event.
 * @return True if busy, false otherwise.
 */
static bool trace_kprobe_is_busy(struct dyn_event *ev); 
/**
 * @brief Matches a kprobe dynamic event against system, event, and arguments.
 * @param system The system name to match.
 * @param event The event name to match.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param ev Pointer to the dynamic event.
 * @return True if match, false otherwise.
 */
static bool trace_kprobe_match(const char *system, const char *event,
			int argc, const char **argv, struct dyn_event *ev);

/**
 * @var trace_kprobe_ops
 * @brief Dynamic event operations for kprobes.
 *
 * This structure defines the callbacks for creating, showing, freeing,
 * and matching kprobe dynamic events.
 */
static struct dyn_event_operations trace_kprobe_ops = {
	.create = trace_kprobe_create,
	.show = trace_kprobe_show,
	.is_busy = trace_kprobe_is_busy,
	.free = trace_kprobe_release,
	.match = trace_kprobe_match,
};

/*
 * Kprobe event core functions
 */
/**
 * @struct trace_kprobe
 * @brief Represents a kprobe-based tracing event instance.
 *
 * This structure encapsulates all information about a dynamically created
 * kprobe or kretprobe event, including its associated `kretprobe`, hit
 * counter, symbol, and `trace_probe` structures.
 */
struct trace_kprobe {
	struct dyn_event	devent;       /**< @brief Dynamic event base structure. */
	struct kretprobe	rp;         /**< @brief Kretprobe instance (contains embedded kprobe). */
	unsigned long __percpu *nhit;		/**< @brief Per-CPU counter for hits. */
	const char		*symbol;       /**< @brief The target symbol name. */
	struct trace_probe	tp;         /**< @brief Trace probe base structure. */
};

/**
 * @brief Checks if a dynamic event is a `trace_kprobe`.
 * @param ev Pointer to the dynamic event.
 * @return True if it's a `trace_kprobe`, false otherwise.
 */
static bool is_trace_kprobe(struct dyn_event *ev)
{
	return ev->ops == &trace_kprobe_ops;
}

/**
 * @brief Casts a `dyn_event` pointer to a `trace_kprobe` pointer.
 * @param ev Pointer to the dynamic event.
 * @return Pointer to the embedded `trace_kprobe` instance.
 */
static struct trace_kprobe *to_trace_kprobe(struct dyn_event *ev)
{
	return container_of(ev, struct trace_kprobe, devent);
}

/**
 * @def for_each_trace_kprobe(pos, dpos)
 * @brief Macro to iterate over the list of `trace_kprobe` instances.
 * @param pos The `trace_kprobe *` for each entry.
 * @param dpos The `dyn_event *` to use as a loop cursor.
 */
#define for_each_trace_kprobe(pos, dpos)	\
	for_each_dyn_event(dpos)		\
		if (is_trace_kprobe(dpos) && (pos = to_trace_kprobe(dpos)))

/**
 * @brief Checks if a `trace_kprobe` is a return probe.
 * @param tk Pointer to the `trace_kprobe`.
 * @return True if it's a return probe (has a kretprobe handler), false otherwise.
 */
static nokprobe_inline bool trace_kprobe_is_return(struct trace_kprobe *tk)
{
	return tk->rp.handler != NULL;
}

/**
 * @brief Retrieves the symbol name for a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return The symbol name as a C string, or "unknown" if not set.
 */
static nokprobe_inline const char *trace_kprobe_symbol(struct trace_kprobe *tk)
{
	return tk->symbol ? tk->symbol : "unknown";
}

/**
 * @brief Retrieves the offset for a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return The kprobe's offset within the symbol.
 */
static nokprobe_inline unsigned long trace_kprobe_offset(struct trace_kprobe *tk)
{
	return tk->rp.kp.offset;
}

/**
 * @brief Checks if a `trace_kprobe` has gone (e.g., its module unloaded).
 * @param tk Pointer to the `trace_kprobe`.
 * @return True if the kprobe has gone, false otherwise.
 */
static nokprobe_inline bool trace_kprobe_has_gone(struct trace_kprobe *tk)
{
	return kprobe_gone(&tk->rp.kp);
}

/**
 * @brief Checks if a `trace_kprobe` is within a specific module.
 * @param tk Pointer to the `trace_kprobe`.
 * @param mod Pointer to the `module` to check against.
 * @return True if the kprobe's symbol is within the module, false otherwise.
 *
 * This function compares the kprobe's symbol name with the module's name
 * to determine if the kprobe belongs to that module.
 */
static nokprobe_inline bool trace_kprobe_within_module(struct trace_kprobe *tk,
					 struct module *mod)
{
	int len = strlen(module_name(mod));
	const char *name = trace_kprobe_symbol(tk);

	return strncmp(module_name(mod), name, len) == 0 && name[len] == ':';
}

#ifdef CONFIG_MODULES
/**
 * @brief Checks if the module targeted by a `trace_kprobe` exists.
 * @param tk Pointer to the `trace_kprobe`.
 * @return True if the module exists, false otherwise.
 *
 * This function extracts the module name from the kprobe's symbol (if
 * specified in "module:symbol" format) and checks for its existence.
 */
static nokprobe_inline bool trace_kprobe_module_exist(struct trace_kprobe *tk)
{
	char *p;
	bool ret;

	if (!tk->symbol)
		return false;
	p = strchr(tk->symbol, ':');
	if (!p)
		return true;
	*p = '\0';
	scoped_guard(rcu)
		ret = !!find_module(tk->symbol);
	*p = ':';

	return ret;
}
#else
// Block Logic: Stub for non-CONFIG_MODULES builds.
static inline bool trace_kprobe_module_exist(struct trace_kprobe *tk)
{
	return false;
}
#endif

/**
 * @brief Checks if a kprobe dynamic event is busy (enabled).
 * @param ev Pointer to the dynamic event.
 * @return True if the kprobe's trace probe is enabled, false otherwise.
 */
static bool trace_kprobe_is_busy(struct dyn_event *ev)
{
	struct trace_kprobe *tk = to_trace_kprobe(ev);

	return trace_probe_is_enabled(&tk->tp);
}

/**
 * @brief Matches the head of a command against a kprobe's symbol and arguments.
 * @param tk Pointer to the `trace_kprobe`.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @return True if the command head matches, false otherwise.
 *
 * This function formats the kprobe's address/symbol and offset into a string
 * and compares it with the first argument, then matches against remaining arguments.
 */
static bool trace_kprobe_match_command_head(struct trace_kprobe *tk,
					 int argc, const char **argv)
{
	char buf[MAX_ARGSTR_LEN + 1];

	if (!argc)
		return true;

	if (!tk->symbol)
		snprintf(buf, sizeof(buf), "0x%p", tk->rp.kp.addr);
	else if (tk->rp.kp.offset)
		snprintf(buf, sizeof(buf), "%s+%u",
			 trace_kprobe_symbol(tk), tk->rp.kp.offset);
	else
		snprintf(buf, sizeof(buf), "%s", trace_kprobe_symbol(tk));
	if (strcmp(buf, argv[0]))
		return false;
	argc--; argv++;

	return trace_probe_match_command_args(&tk->tp, argc, argv);
}

/**
 * @brief Matches a kprobe dynamic event against system, event, and command arguments.
 * @param system The system name to match.
 * @param event The event name to match.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param ev Pointer to the dynamic event.
 * @return True if the kprobe matches the criteria, false otherwise.
 *
 * This function implements the matching logic for kprobes, checking
 * event name, system, and command head.
 */
static bool trace_kprobe_match(const char *system, const char *event,
			int argc, const char **argv, struct dyn_event *ev)
{
	struct trace_kprobe *tk = to_trace_kprobe(ev);

	return (event[0] == '\0' ||
		strcmp(trace_probe_name(&tk->tp), event) == 0) &&
	    (!system || strcmp(trace_probe_group_name(&tk->tp), system) == 0) &&
	    trace_kprobe_match_command_head(tk, argc, argv);
}

/**
 * @brief Retrieves the total number of hits for a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return The total number of hits across all CPUs.
 *
 * This function sums the per-CPU hit counters for the kprobe.
 */
static nokprobe_inline unsigned long trace_kprobe_nhit(struct trace_kprobe *tk)
{
	unsigned long nhit = 0;
	int cpu;

	for_each_possible_cpu(cpu)
		nhit += *per_cpu_ptr(tk->nhit, cpu);

	return nhit;
}

/**
 * @brief Checks if a `trace_kprobe` is registered with the kprobe framework.
 * @param tk Pointer to the `trace_kprobe`.
 * @return True if the kprobe is registered, false otherwise.
 *
 * This function checks the hlist status of the embedded kprobe.
 */
static nokprobe_inline bool trace_kprobe_is_registered(struct trace_kprobe *tk)
{
	return !(list_empty(&tk->rp.kp.list) &&
		 hlist_unhashed(&tk->rp.kp.hlist));
}

/* Return 0 if it fails to find the symbol address */
/**
 * @brief Retrieves the effective address of a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return The effective address of the kprobe, or 0 if symbol lookup fails.
 *
 * This function resolves the kprobe's address, taking into account
 * symbol name and offset if provided.
 */
static nokprobe_inline
unsigned long trace_kprobe_address(struct trace_kprobe *tk)
{
	unsigned long addr;

	if (tk->symbol) {
		addr = (unsigned long)
			kallsyms_lookup_name(trace_kprobe_symbol(tk));
		if (addr)
			addr += tk->rp.kp.offset;
	} else
		addr = (unsigned long)tk->rp.kp.addr;
	return addr;
}

/**
 * @brief Retrieves the primary `trace_kprobe` instance from a `trace_event_call`.
 * @param call Pointer to the `trace_event_call`.
 * @return Pointer to the `trace_kprobe`, or NULL if invalid.
 */
static nokprobe_inline struct trace_kprobe *
trace_kprobe_primary_from_call(struct trace_event_call *call)
{
	struct trace_probe *tp;

	tp = trace_probe_primary_from_call(call);
	if (WARN_ON_ONCE(!tp))
		return NULL;

	return container_of(tp, struct trace_kprobe, tp);
}

/**
 * @brief Checks if a `trace_kprobe` is at a function entry point.
 * @param call Pointer to the `trace_event_call`.
 * @return True if at function entry, false otherwise.
 */
bool trace_kprobe_on_func_entry(struct trace_event_call *call)
{
	struct trace_kprobe *tk = trace_kprobe_primary_from_call(call);

	return tk ? (kprobe_on_func_entry(tk->rp.kp.addr,
			tk->rp.kp.addr ? NULL : tk->rp.kp.symbol_name,
			tk->rp.kp.addr ? 0 : tk->rp.kp.offset) == 0) : false;
}

/**
 * @brief Checks if a `trace_kprobe` is error-injectable.
 * @param call Pointer to the `trace_event_call`.
 * @return True if error-injectable, false otherwise.
 */
bool trace_kprobe_error_injectable(struct trace_event_call *call)
{
	struct trace_kprobe *tk = trace_kprobe_primary_from_call(call);

	return tk ? within_error_injection_list(trace_kprobe_address(tk)) :
	       false;
}

/**
 * @brief Registers a kprobe event.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, or a negative errno on failure.
 */
static int register_kprobe_event(struct trace_kprobe *tk);
/**
 * @brief Unregisters a kprobe event.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, or a negative errno on failure.
 */
static int unregister_kprobe_event(struct trace_kprobe *tk);

/**
 * @brief Kprobe dispatcher function.
 * @param kp Pointer to the `kprobe`.
 * @param regs Pointer to `pt_regs`.
 * @return 0 (or 1 if BPF modified PC).
 *
 * This function is called when a kprobe is hit. It increments the hit counter,
 * records the trace event, and dispatches to perf events if enabled.
 */
static int kprobe_dispatcher(struct kprobe *kp, struct pt_regs *regs);
/**
 * @brief Kretprobe dispatcher function.
 * @param ri Pointer to the `kretprobe_instance`.
 * @param regs Pointer to `pt_regs`.
 * @return 0.
 *
 * This function is called when a kretprobe is hit. It increments the hit counter,
 * records the trace event, and dispatches to perf events if enabled.
 */
static int kretprobe_dispatcher(struct kretprobe_instance *ri,
				struct pt_regs *regs);

/**
 * @brief Frees resources associated with a `trace_kprobe` instance.
 * @param tk Pointer to the `trace_kprobe` instance.
 *
 * This function cleans up the trace probe, symbol string, per-CPU hit counter,
 * and the `trace_kprobe` structure itself.
 */
static void free_trace_kprobe(struct trace_kprobe *tk)
{
	if (tk) {
		trace_probe_cleanup(&tk->tp);
		kfree(tk->symbol);
		free_percpu(tk->nhit);
		kfree(tk);
	}
}

/**
 * @brief Helper macro for `free_trace_kprobe` to be used with `DEFINE_FREE`.
 */
DEFINE_FREE(free_trace_kprobe, struct trace_kprobe *, 
	if (!IS_ERR_OR_NULL(_T)) free_trace_kprobe(_T))

/*
 * Allocate new trace_probe and initialize it (including kprobes).
 */
/**
 * @brief Allocates and initializes a new `trace_kprobe` instance.
 * @param group The group name for the event.
 * @param event The event name.
 * @param addr The direct address for the kprobe.
 * @param symbol The target symbol name.
 * @param offs The offset within the symbol.
 * @param maxactive Maximum active instances for kretprobe.
 * @param nargs Number of arguments.
 * @param is_return True if it's a kretprobe, false otherwise.
 * @return Pointer to the newly allocated `trace_kprobe` on success, or an `ERR_PTR` on failure.
 *
 * This function allocates memory for the `trace_kprobe`, duplicates the symbol
 * name, sets up entry/exit handlers, and initializes embedded `kretprobe`,
 * `trace_probe`, and `dyn_event` structures.
 */
static struct trace_kprobe *alloc_trace_kprobe(const char *group,
					 const char *event,
					 void *addr,
					 const char *symbol,
					 unsigned long offs,
					 int maxactive,
					 int nargs, bool is_return)
{
	struct trace_kprobe *tk __free(free_trace_kprobe) = NULL;
	int ret = -ENOMEM;

	tk = kzalloc(struct_size(tk, tp.args, nargs), GFP_KERNEL);
	if (!tk)
		return ERR_PTR(ret);

	tk->nhit = alloc_percpu(unsigned long);
	if (!tk->nhit)
		return ERR_PTR(ret);

	if (symbol) {
		tk->symbol = kstrdup(symbol, GFP_KERNEL);
		if (!tk->symbol)
			return ERR_PTR(ret);
		tk->rp.kp.symbol_name = tk->symbol;
		tk->rp.kp.offset = offs;
	} else
		tk->rp.kp.addr = addr;

	if (is_return)
		tk->rp.handler = kretprobe_dispatcher;
	else
		tk->rp.kp.pre_handler = kprobe_dispatcher;

	tk->rp.maxactive = maxactive;
	INIT_HLIST_NODE(&tk->rp.kp.hlist);
	INIT_LIST_HEAD(&tk->rp.kp.list);

	ret = trace_probe_init(&tk->tp, event, group, false, nargs);
	if (ret < 0)
		return ERR_PTR(ret);

	dyn_event_init(&tk->devent, &trace_kprobe_ops);
	return_ptr(tk);
}

/**
 * @brief Finds an existing `trace_kprobe` by event and group name.
 * @param event The event name.
 * @param group The group name.
 * @return Pointer to the found `trace_kprobe`, or NULL if not found.
 *
 * This function iterates through all dynamic events and checks if they
 * match the given event and group names.
 */
static struct trace_kprobe *find_trace_kprobe(const char *event,
				      const char *group)
{
	struct dyn_event *pos;
	struct trace_kprobe *tk;

	for_each_trace_kprobe(tk, pos)
		if (strcmp(trace_probe_name(&tk->tp), event) == 0 &&
		    strcmp(trace_probe_group_name(&tk->tp), group) == 0)
			return tk;
	return NULL;
}

/**
 * @brief Enables a `trace_kprobe` instance.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function enables the embedded kprobe or kretprobe if it's registered
 * and not marked as gone.
 */
static inline int __enable_trace_kprobe(struct trace_kprobe *tk)
{
	int ret = 0;

	if (trace_kprobe_is_registered(tk) && !trace_kprobe_has_gone(tk)) {
		if (trace_kprobe_is_return(tk))
			ret = enable_kretprobe(&tk->rp);
		else
			ret = enable_kprobe(&tk->rp.kp);
	}

	return ret;
}

/**
 * @brief Disables all `kprobe` instances linked to a `trace_probe`.
 * @param tp Pointer to the `trace_probe`.
 *
 * This function iterates through all `trace_kprobe` instances associated
 * with the given `trace_probe` and disables their embedded kprobes.
 */
static void __disable_trace_kprobe(struct trace_probe *tp)
{
	struct trace_kprobe *tk;

	list_for_each_entry(tk, trace_probe_probe_list(tp), tp.list) {
		if (!trace_kprobe_is_registered(tk))
			continue;
		if (trace_kprobe_is_return(tk))
			disable_kretprobe(&tk->rp);
		else
			disable_kprobe(&tk->rp.kp);
	}
}

/*
 * Enable trace_probe
 * if the file is NULL, enable "perf" handler, or enable "trace" handler.
 */
/**
 * @brief Enables a `trace_kprobe` or a group of kprobes associated with a `trace_event_call`.
 * @param call Pointer to the `trace_event_call` for the kprobe.
 * @param file Optional: Pointer to the `trace_event_file` to associate.
 * @return 0 on success, -ENODEV if `trace_probe` is invalid.
 *
 * This function enables all kprobes linked to the provided `trace_probe`
 * (primary from `call`). It handles both file-associated and profile-flagged
 * enabling.
 */
static int enable_trace_kprobe(struct trace_event_call *call,
				struct trace_event_file *file)
{
	struct trace_probe *tp;
	struct trace_kprobe *tk;
	bool enabled;
	int ret = 0;

	tp = trace_probe_primary_from_call(call);
	if (WARN_ON_ONCE(!tp))
		return -ENODEV;
	enabled = trace_probe_is_enabled(tp);

	/* This also changes "enabled" state */
	if (file) {
		ret = trace_probe_add_file(tp, file);
		if (ret)
			return ret;
	} else
		trace_probe_set_flag(tp, TP_FLAG_PROFILE);

	if (enabled)
		return 0;

	list_for_each_entry(tk, trace_probe_probe_list(tp), tp.list) {
		if (trace_kprobe_has_gone(tk))
			continue;
		ret = __enable_trace_kprobe(tk);
		if (ret)
			break;
		enabled = true;
	}

	if (ret) {
		/* Failed to enable one of them. Roll back all */
		if (enabled)
			__disable_trace_kprobe(tp);
		if (file)
			trace_probe_remove_file(tp, file);
		else
			trace_probe_clear_flag(tp, TP_FLAG_PROFILE);
	}

	return ret;
}

/*
 * Disable trace_probe
 * if the file is NULL, disable "perf" handler, or disable "trace" handler.
 */
/**
 * @brief Disables a `trace_kprobe` or a group of kprobes associated with a `trace_event_call`.
 * @param call Pointer to the `trace_event_call` for the kprobe.
 * @param file Optional: Pointer to the `trace_event_file` to disassociate.
 * @return 0 on success, -ENODEV if `trace_probe` is invalid, -ENOENT if file link not found.
 *
 * This function disables all kprobes linked to the provided `trace_probe`.
 * It handles both file-associated and profile-flagged disabling.
 */
static int disable_trace_kprobe(struct trace_event_call *call,
				struct trace_event_file *file)
{
	struct trace_probe *tp;

	tp = trace_probe_primary_from_call(call);
	if (WARN_ON_ONCE(!tp))
		return -ENODEV;

	if (file) {
		if (!trace_probe_get_file_link(tp, file))
			return -ENOENT;
		if (!trace_probe_has_single_file(tp))
			goto out;
		trace_probe_clear_flag(tp, TP_FLAG_TRACE);
	} else
		trace_probe_clear_flag(tp, TP_FLAG_PROFILE);

	if (!trace_probe_is_enabled(tp))
		__disable_trace_kprobe(tp);

 out:
	if (file)
		/*
		 * Synchronization is done in below function. For perf event,
		 * file == NULL and perf_trace_event_unreg() calls
		 * tracepoint_synchronize_unregister() to ensure synchronize
		 * event. We don't need to care about it.
		 */
		trace_probe_remove_file(tp, file);

	return 0;
}

#if defined(CONFIG_DYNAMIC_FTRACE) && \
	!defined(CONFIG_KPROBE_EVENTS_ON_NOTRACE)
/**
 * @brief Checks if an address is within a non-traceable function.
 * @param addr The address to check.
 * @return True if within a non-traceable function, false otherwise.
 */
static bool __within_notrace_func(unsigned long addr)
{
	unsigned long offset, size;

	if (!addr || !kallsyms_lookup_size_offset(addr, &size, &offset))
		return false;

	/* Get the entry address of the target function */
	addr -= offset;

	/*
	 * Since ftrace_location_range() does inclusive range check, we need
	 * to subtract 1 byte from the end address.
	 */
	return !ftrace_location_range(addr, addr + size - 1);
}

/**
 * @brief Checks if a `trace_kprobe` is located within a non-traceable function.
 * @param tk Pointer to the `trace_kprobe`.
 * @return True if within a non-traceable function, false otherwise.
 */
static bool within_notrace_func(struct trace_kprobe *tk)
{
	unsigned long addr = trace_kprobe_address(tk);
	char symname[KSYM_NAME_LEN], *p;

	if (!__within_notrace_func(addr))
		return false;

	/* Check if the address is on a suffixed-symbol */
	if (!lookup_symbol_name(addr, symname)) {
		p = strchr(symname, '.');
		if (!p)
			return true;
		*p = '\0';
		addr = (unsigned long)kprobe_lookup_name(symname, 0);
		if (addr)
			return __within_notrace_func(addr);
	}

	return true;
}
#else
// Block Logic: Stub for non-DYNAMIC_FTRACE or KPROBE_EVENTS_ON_NOTRACE builds.
#define within_notrace_func(tk) (false)
#endif

/* Internal register function - just handle k*probes and flags */
/**
 * @brief Internal function to register a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function handles security checks, checks for `notrace` functions,
 * updates arguments, sets kprobe flags, and registers the embedded kprobe
 * or kretprobe.
 */
static int __register_trace_kprobe(struct trace_kprobe *tk)
{
	int i, ret;

	ret = security_locked_down(LOCKDOWN_KPROBES);
	if (ret)
		return ret;

	if (trace_kprobe_is_registered(tk))
		return -EINVAL;

	if (within_notrace_func(tk)) {
		pr_warn("Could not probe notrace function %ps\n",
			(void *)trace_kprobe_address(tk));
		return -EINVAL;
	}

	for (i = 0; i < tk->tp.nr_args; i++) {
		ret = traceprobe_update_arg(&tk->tp.args[i]);
		if (ret)
			return ret;
	}

	/* Set/clear disabled flag according to tp->flag */
	if (trace_probe_is_enabled(&tk->tp))
		tk->rp.kp.flags &= ~KPROBE_FLAG_DISABLED;
	else
		tk->rp.kp.flags |= KPROBE_FLAG_DISABLED;

	if (trace_kprobe_is_return(tk))
		ret = register_kretprobe(&tk->rp);
	else
		ret = register_kprobe(&tk->rp.kp);

	return ret;
}

/* Internal unregister function - just handle k*probes and flags */
/**
 * @brief Internal function to unregister a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 *
 * This function unregisters the embedded kprobe or kretprobe and cleans
 * up its internal state for reuse.
 */
static void __unregister_trace_kprobe(struct trace_kprobe *tk)
{
	if (trace_kprobe_is_registered(tk)) {
		if (trace_kprobe_is_return(tk))
			unregister_kretprobe(&tk->rp);
		else
			unregister_kprobe(&tk->rp.kp);
		/* Cleanup kprobe for reuse and mark it unregistered */
		INIT_HLIST_NODE(&tk->rp.kp.hlist);
		INIT_LIST_HEAD(&tk->rp.kp.list);
		if (tk->rp.kp.symbol_name)
			tk->rp.kp.addr = NULL;
	}
}

/* Unregister a trace_probe and probe_event */
/**
 * @brief Unregisters a `trace_kprobe` instance.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, -EBUSY if the kprobe is enabled or busy.
 *
 * This function removes the kprobe from the dynamic event list, unlinks
 * its trace probe, and unregisters its embedded kprobe.
 */
static int unregister_trace_kprobe(struct trace_kprobe *tk)
{
	/* If other probes are on the event, just unregister kprobe */
	if (trace_probe_has_sibling(&tk->tp))
		goto unreg;

	/* Enabled event can not be unregistered */
	if (trace_probe_is_enabled(&tk->tp))
		return -EBUSY;

	/* If there's a reference to the dynamic event */
	if (trace_event_dyn_busy(trace_probe_event_call(&tk->tp)))
		return -EBUSY;

	/* Will fail if probe is being used by ftrace or perf */
	if (unregister_kprobe_event(tk))
		return -EBUSY;

unreg:
	__unregister_trace_kprobe(tk);
	dyn_event_remove(&tk->devent);
	trace_probe_unlink(&tk->tp);

	return 0;
}

/**
 * @brief Checks if two `trace_kprobe` instances target the same kprobe.
 * @param orig Pointer to the original `trace_kprobe`.
 * @param comp Pointer to the comparing `trace_kprobe`.
 * @return True if they have the same kprobe (symbol, offset, and arguments), false otherwise.
 *
 * This function compares the symbol name, offset, and command arguments to determine
 * if two kprobes are essentially the same.
 */
static bool trace_kprobe_has_same_kprobe(struct trace_kprobe *orig,
					 struct trace_kprobe *comp)
{
	struct trace_probe_event *tpe = orig->tp.event;
	int i;

	list_for_each_entry(orig, &tpe->probes, tp.list) {
		if (strcmp(trace_kprobe_symbol(orig),
			   trace_kprobe_symbol(comp)) ||
		    trace_kprobe_offset(orig) != trace_kprobe_offset(comp))
			continue;

		/*
		 * trace_probe_compare_arg_type() ensured that nr_args and
		 * each argument name and type are same. Let's compare comm.
		 */
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
 * @brief Appends a new `trace_kprobe` to an existing one.
 * @param tk Pointer to the `trace_kprobe` to append.
 * @param to Pointer to the existing `trace_kprobe` to append to.
 * @return 0 on success, -EEXIST if probe types or argument types mismatch, or a negative errno on failure.
 *
 * This function checks for compatibility between the two kprobes (probe type,
 * argument types) and then appends the new kprobe to the existing one's
 * `trace_probe` list.
 */
static int append_trace_kprobe(struct trace_kprobe *tk, struct trace_kprobe *to)
{
	int ret;

	if (trace_kprobe_is_return(tk) != trace_kprobe_is_return(to)) {
		trace_probe_log_set_index(0);
		trace_probe_log_err(0, DIFF_PROBE_TYPE);
		return -EEXIST;
	}
	ret = trace_probe_compare_arg_type(&tk->tp, &to->tp);
	if (ret) {
		/* Note that argument starts index = 2 */
		trace_probe_log_set_index(ret + 1);
		trace_probe_log_err(0, DIFF_ARG_TYPE);
		return -EEXIST;
	}
	if (trace_kprobe_has_same_kprobe(to, tk)) {
		trace_probe_log_set_index(0);
		trace_probe_log_err(0, SAME_PROBE);
		return -EEXIST;
	}

	/* Append to existing event */
	ret = trace_probe_append(&tk->tp, &to->tp);
	if (ret)
		return ret;

	/* Register k*probe */
	ret = __register_trace_kprobe(tk);
	if (ret == -ENOENT && !trace_kprobe_module_exist(tk)) {
		pr_warn("This probe might be able to register after target module is loaded. Continue.\n");
		ret = 0;
	}

	if (ret)
		trace_probe_unlink(&tk->tp);
	else
		dyn_event_add(&tk->devent, trace_probe_event_call(&tk->tp));

	return ret;
}

/* Register a trace_probe and probe_event */
/**
 * @brief Registers a `trace_kprobe` instance.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, -EEXIST if an event with the same name already exists, or a negative errno on failure.
 *
 * This function registers a new kprobe, either by appending it to an existing
 * trace event or by creating a new one. It handles the registration of
 * the underlying kprobe/kretprobe and `trace_event_call`.
 */
static int register_trace_kprobe(struct trace_kprobe *tk)
{
	struct trace_kprobe *old_tk;
	int ret;

	guard(mutex)(&event_mutex);

	old_tk = find_trace_kprobe(trace_probe_name(&tk->tp),
				   trace_probe_group_name(&tk->tp));
	if (old_tk) {
		if (trace_kprobe_is_return(tk) != trace_kprobe_is_return(old_tk)) {
			trace_probe_log_set_index(0);
			trace_probe_log_err(0, DIFF_PROBE_TYPE);
			return -EEXIST;
		}
		return append_trace_kprobe(tk, old_tk);
	}

	/* Register new event */
	ret = register_kprobe_event(tk);
	if (ret) {
		if (ret == -EEXIST) {
			trace_probe_log_set_index(0);
			trace_probe_log_err(0, EVENT_EXIST);
		} else
			pr_warn("Failed to register probe event(%d)\n", ret);
		return ret;
	}

	/* Register k*probe */
	ret = __register_trace_kprobe(tk);
	if (ret == -ENOENT && !trace_kprobe_module_exist(tk)) {
		pr_warn("This probe might be able to register after target module is loaded. Continue.\n");
		ret = 0;
	}

	if (ret < 0)
		unregister_kprobe_event(tk);
	else
		dyn_event_add(&tk->devent, trace_probe_event_call(&tk->tp));

	return ret;
}

#ifdef CONFIG_MODULES
/**
 * @brief Validates a module probe symbol.
 * @param modname The module name.
 * @param symbol The symbol name.
 * @return 0 on success, -EADDRNOTAVAIL if ambiguous, -ENOENT if not found.
 */
static int validate_module_probe_symbol(const char *modname, const char *symbol);

/**
 * @brief Registers a `trace_kprobe` that targets a module.
 * @param mod Pointer to the `module`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, or a negative errno on failure.
 */
static int register_module_trace_kprobe(struct module *mod, struct trace_kprobe *tk)
{
	const char *p;
	int ret = 0;

	p = strchr(trace_kprobe_symbol(tk), ':');
	if (p)
		ret = validate_module_probe_symbol(module_name(mod), p + 1);
	if (!ret)
		ret = __register_trace_kprobe(tk);
	return ret;
}

/* Module notifier call back, checking event on the module */
/**
 * @brief Module callback function to handle module state changes (e.g., loading).
 * @param nb Pointer to the `notifier_block`.
 * @param val The notification value (module state).
 * @param data Pointer to the module.
 * @return `NOTIFY_DONE`.
 *
 * This function is invoked when a module is coming up (`MODULE_STATE_COMING`).
 * It re-registers any trace_kprobes that target this module and were previously
 * deferred due to the module not being loaded.
 */
static int trace_kprobe_module_callback(struct notifier_block *nb,
				       unsigned long val, void *data)
{
	struct module *mod = data;
	struct dyn_event *pos;
	struct trace_kprobe *tk;
	int ret;

	if (val != MODULE_STATE_COMING)
		return NOTIFY_DONE;

	/* Update probes on coming module */
	guard(mutex)(&event_mutex);
	for_each_trace_kprobe(tk, pos) {
		if (trace_kprobe_within_module(tk, mod)) {
			/* Don't need to check busy - this should have gone. */
			__unregister_trace_kprobe(tk);
			ret = register_module_trace_kprobe(mod, tk);
			if (ret)
				pr_warn("Failed to re-register probe %s on %s: %d\n",
					trace_probe_name(&tk->tp),
					module_name(mod), ret);
		}
	}

	return NOTIFY_DONE;
}

/**
 * @var trace_kprobe_module_nb
 * @brief Notifier block for kprobe module state changes.
 *
 * Registers `trace_kprobe_module_callback` to be notified of module events.
 */
static struct notifier_block trace_kprobe_module_nb = {
	.notifier_call = trace_kprobe_module_callback,
	.priority = 2	/* Invoked after kprobe and jump_label module callback */
};
/**
 * @brief Registers the module notifier for kprobes.
 * @return 0 on success.
 */
static int trace_kprobe_register_module_notifier(void)
{
	return register_module_notifier(&trace_kprobe_module_nb);
}
#else
// Block Logic: Stub for non-CONFIG_MODULES builds.
static int trace_kprobe_register_module_notifier(void)
{
	return 0;
}
#endif /* CONFIG_MODULES */

/**
 * @brief Callback function to count symbols.
 * @param data Pointer to the count variable.
 * @param unused Unused parameter.
 * @return 0 to continue.
 *
 * Increments a counter for each symbol encountered.
 */
static int count_symbols(void *data, unsigned long unused)
{
	unsigned int *count = data;

	(*count)++;

	return 0;
}

/**
 * @struct sym_count_ctx
 * @brief Context for counting symbols.
 *
 * Used to pass data to `count_mod_symbols`.
 */
struct sym_count_ctx {
	unsigned int count;	/**< @brief The current count. */
	const char *name;	/**< @brief The symbol name to match. */
};

/**
 * @brief Callback function to count module symbols with a matching name.
 * @param data Pointer to `sym_count_ctx`.
 * @param name Name of the symbol.
 * @param unused Unused parameter.
 * @return 0 to continue.
 *
 * Increments the count if the symbol name matches the target name.
 */
static int count_mod_symbols(void *data, const char *name, unsigned long unused)
{
	struct sym_count_ctx *ctx = data;

	if (strcmp(name, ctx->name) == 0)
		ctx->count++;

	return 0;
}

/**
 * @brief Counts the number of symbols with a given name in kernel and modules.
 * @param mod Module name (NULL for kernel).
 * @param func_name The function name to count.
 * @return The total count of matching symbols.
 *
 * This function uses `kallsyms_on_each_match_symbol` for kernel symbols
 * and `module_kallsyms_on_each_symbol` for module symbols.
 */
static unsigned int number_of_same_symbols(const char *mod, const char *func_name)
{
	struct sym_count_ctx ctx = { .count = 0, .name = func_name };

	if (!mod)
		kallsyms_on_each_match_symbol(count_symbols, func_name, &ctx.count);

	module_kallsyms_on_each_symbol(mod, count_mod_symbols, &ctx);

	return ctx.count;
}

/**
 * @brief Validates a module probe symbol.
 * @param modname The name of the module (or NULL for kernel).
 * @param symbol The name of the symbol.
 * @return 0 on success, -EADDRNOTAVAIL if ambiguous, -ENOENT if not found.
 *
 * This function checks for ambiguity (multiple symbols with the same name)
 * or non-existence of a symbol before registering a probe.
 */
static int validate_module_probe_symbol(const char *modname, const char *symbol)
{
	unsigned int count = number_of_same_symbols(modname, symbol);

	if (count > 1) {
		/*
		 * Users should use ADDR to remove the ambiguity of
		 * using KSYM only.
		 */
		return -EADDRNOTAVAIL;
	} else if (count == 0) {
		/*
		 * We can return ENOENT earlier than when register the
		 * kprobe.
		 */
		return -ENOENT;
	}
	return 0;
}

#ifdef CONFIG_MODULES
/* Return NULL if the module is not loaded or under unloading. */
/**
 * @brief Tries to get a reference to a module by name.
 * @param name The name of the module.
 * @return Pointer to the `module` structure on success, or NULL on failure.
 *
 * This function must be called within an RCU critical section.
 */
static struct module *try_module_get_by_name(const char *name)
{
	struct module *mod;

	guard(rcu)();
	mod = find_module(name);
	if (mod && !try_module_get(mod))
		mod = NULL;
	return mod;
}
#else
#define try_module_get_by_name(name) (NULL)
#endif

/**
 * @brief Validates a kprobe symbol, handling module-prefixed symbols.
 * @param symbol The symbol string (can be "module:symbol").
 * @return 0 on success, or a negative errno on failure.
 *
 * This function extracts module and symbol names, then validates the symbol.
 * It ensures proper cleanup of module references.
 */
static int validate_probe_symbol(char *symbol)
{
	struct module *mod = NULL;
	char *modname = NULL, *p;
	int ret = 0;

	p = strchr(symbol, ':');
	if (p) {
		modname = symbol;
		symbol = p + 1;
		*p = '\0';
		mod = try_module_get_by_name(modname);
		if (!mod)
			goto out;
	}

	ret = validate_module_probe_symbol(modname, symbol);
out:
	if (p)
		*p = ':';
	if (mod)
		module_put(mod);
	return ret;
}

static int trace_kprobe_entry_handler(struct kretprobe_instance *ri,
				      struct pt_regs *regs);

/**
 * @brief Creates a `trace_kprobe` instance from command-line arguments.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @param ctx Pointer to `traceprobe_parse_context`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function parses arguments for `kprobe` and `kretprobe` events,
 * extracts target address/symbol, offset, maxactive, and argument definitions.
 */
static int trace_kprobe_create_internal(int argc, const char *argv[],
					struct traceprobe_parse_context *ctx)
{
	/*
	 * Argument syntax:
	 *  - Add kprobe:
	 *      p[:[GRP/][EVENT]] [MOD:]KSYM[+OFFS]|KADDR [FETCHARGS]
	 *  - Add kretprobe:
	 *      r[MAXACTIVE][:[GRP/][EVENT]] [MOD:]KSYM[+0] [FETCHARGS]
	 *    Or
	 *      p[:[GRP/][EVENT]] [MOD:]KSYM[+0]%return [FETCHARGS]
	 *
	 * Fetch args:
	 *  $retval	: fetch return value
	 *  $stack	: fetch stack address
	 *  $stackN	: fetch Nth of stack (N:0-)
	 *  $comm       : fetch current task comm
	 *  @ADDR	: fetch memory at ADDR (ADDR should be in kernel)
	 *  @SYM[+|-offs] : fetch memory at SYM +|- offs (SYM is a data symbol)
	 *  %REG	: fetch register REG
	 * Dereferencing memory fetch:
	 *  +|-offs(ARG) : fetch memory at ARG +|- offs address.
	 * Alias name of args:
	 *  NAME=FETCHARG : set NAME as alias of FETCHARG.
	 * Type of args:
	 *  FETCHARG:TYPE : use TYPE instead of unsigned long.
	 */
	struct trace_kprobe *tk __free(free_trace_kprobe) = NULL;
	int i, len, new_argc = 0, ret = 0;
	bool is_return = false;
	char *symbol __free(kfree) = NULL;
	char *tmp = NULL;
	const char **new_argv __free(kfree) = NULL;
	const char *event = NULL, *group = KPROBE_EVENT_SYSTEM;
	enum probe_print_type ptype;
	int maxactive = 0;
	long offset = 0;
	void *addr = NULL;
	char buf[MAX_EVENT_NAME_LEN];
	char gbuf[MAX_EVENT_NAME_LEN];
	char abuf[MAX_BTF_ARGS_LEN];
	char *dbuf __free(kfree) = NULL;

	switch (argv[0][0]) {
	case 'r':
		is_return = true;
		break;
	case 'p':
		break;
	default:
		return -ECANCELED;
	}
	if (argc < 2)
		return -ECANCELED;

	event = strchr(&argv[0][1], ':');
	if (event)
		event++;

	if (isdigit(argv[0][1])) {
		if (!is_return) {
			trace_probe_log_err(1, BAD_MAXACT_TYPE);
			return -EINVAL;
		}
		if (event)
			len = event - &argv[0][1] - 1;
		else
			len = strlen(&argv[0][1]);
		if (len > MAX_EVENT_NAME_LEN - 1) {
			trace_probe_log_err(1, BAD_MAXACT);
			return -EINVAL;
		}
		memcpy(buf, &argv[0][1], len);
		buf[len] = '\0';
		ret = kstrtouint(buf, 0, &maxactive);
		if (ret || !maxactive) {
			trace_probe_log_err(1, BAD_MAXACT);
			return -EINVAL;
		}
		/* kretprobes instances are iterated over via a list.
		 * The maximum should stay reasonable.
		 */
		if (maxactive > KRETPROBE_MAXACTIVE_MAX) {
			trace_probe_log_err(1, MAXACT_TOO_BIG);
			return -EINVAL;
		}
	}

	/* try to parse an address. if that fails, try to read the
	 * input as a symbol. */
	if (kstrtoul(argv[1], 0, (unsigned long *)&addr)) {
		trace_probe_log_set_index(1);
		/* Check whether uprobe event specified */
		if (strchr(argv[1], '/') && strchr(argv[1], ':'))
			return -ECANCELED;

		/* a symbol specified */
		symbol = kstrdup(argv[1], GFP_KERNEL);
		if (!symbol)
			return -ENOMEM;

		tmp = strchr(symbol, '%');
		if (tmp) {
			if (!strcmp(tmp, "%return")) {
				*tmp = '\0';
				is_return = true;
			} else {
				trace_probe_log_err(tmp - symbol, BAD_ADDR_SUFFIX);
				return -EINVAL;
			}
		}

		/* TODO: support .init module functions */
		ret = traceprobe_split_symbol_offset(symbol, &offset);
		if (ret || offset < 0 || offset > UINT_MAX) {
			trace_probe_log_err(0, BAD_PROBE_ADDR);
			return -EINVAL;
		}
		ret = validate_probe_symbol(symbol);
		if (ret) {
			if (ret == -EADDRNOTAVAIL)
				trace_probe_log_err(0, NON_UNIQ_SYMBOL);
			else
				trace_probe_log_err(0, BAD_PROBE_ADDR);
			return -EINVAL;
		}
		if (is_return)
			ctx->flags |= TPARG_FL_RETURN;
		ret = kprobe_on_func_entry(NULL, symbol, offset);
		if (ret == 0 && !is_return)
			ctx->flags |= TPARG_FL_FENTRY;
		/* Defer the ENOENT case until register kprobe */
		if (ret == -EINVAL && is_return) {
			trace_probe_log_err(0, BAD_RETPROBE);
			return -EINVAL;
		}
	}

	trace_probe_log_set_index(0);
	if (event) {
		ret = traceprobe_parse_event_name(&event, &group, gbuf,
						  event - argv[0]);
		if (ret)
			return ret;
	}

	if (!event) {
		/* Make a new event name */
		if (symbol)
			snprintf(buf, MAX_EVENT_NAME_LEN, "%c_%s_%ld",
				 is_return ? 'r' : 'p', symbol, offset);
		else
			snprintf(buf, MAX_EVENT_NAME_LEN, "%c_0x%p",
				 is_return ? 'r' : 'p', addr);
		sanitize_event_name(buf);
		event = buf;
	}

	argc -= 2; argv += 2;
	ctx->funcname = symbol;
	new_argv = traceprobe_expand_meta_args(argc, argv, &new_argc,
							       abuf, MAX_BTF_ARGS_LEN, ctx);
	if (IS_ERR(new_argv)) {
		ret = PTR_ERR(new_argv);
		new_argv = NULL;
		return ret;
	}
	if (new_argv) {
		argc = new_argc;
		argv = new_argv;
	}
	if (argc > MAX_TRACE_ARGS) {
		trace_probe_log_set_index(2);
		trace_probe_log_err(0, TOO_MANY_ARGS);
		return -E2BIG;
	}

	ret = traceprobe_expand_dentry_args(argc, argv, &dbuf);
	if (ret)
		return ret;

	/* setup a probe */
	tk = alloc_trace_kprobe(group, event, addr, symbol, offset, maxactive,
					argc, is_return);
	if (IS_ERR(tk)) {
		ret = PTR_ERR(tk);
		/* This must return -ENOMEM, else there is a bug */
		WARN_ON_ONCE(ret != -ENOMEM);
		return ret;	/* We know tk is not allocated */
	}

	/* parse arguments */
	for (i = 0; i < argc; i++) {
		trace_probe_log_set_index(i + 2);
		ctx->offset = 0;
		ret = traceprobe_parse_probe_arg(&tk->tp, i, argv[i], ctx);
		if (ret)
			return ret;	/* This can be -ENOMEM */
	}
	/* entry handler for kretprobe */
	if (is_return && tk->tp.entry_arg) {
		tk->rp.entry_handler = trace_kprobe_entry_handler;
		tk->rp.data_size = traceprobe_get_entry_data_size(&tk->tp);
	}

	ptype = is_return ? PROBE_PRINT_RETURN : PROBE_PRINT_NORMAL;
	ret = traceprobe_set_print_fmt(&tk->tp, ptype);
	if (ret < 0)
		return ret;

	ret = register_trace_kprobe(tk);
	if (ret) {
		trace_probe_log_set_index(1);
		if (ret == -EILSEQ)
			trace_probe_log_err(0, BAD_INSN_BNDRY);
		else if (ret == -ENOENT)
			trace_probe_log_err(0, BAD_PROBE_ADDR);
		else if (ret != -ENOMEM && ret != -EEXIST)
			trace_probe_log_err(0, FAIL_REG_PROBE);
		return ret;
	}
	/*
	 * Here, 'tk' has been registered to the list successfully,
	 * so we don't need to free it.
	 */
	tk = NULL;

	return 0;
}

/**
 * @brief Callback function for `trace_probe_create` to create a kprobe.
 * @param argc Number of arguments.
 * @param argv Array of argument strings.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function initializes the `traceprobe_parse_context` and calls
 * `trace_kprobe_create_internal`.
 */
static int trace_kprobe_create_cb(int argc, const char *argv[])
{
	struct traceprobe_parse_context ctx = { .flags = TPARG_FL_KERNEL };
	int ret;

	trace_probe_log_init("trace_kprobe", argc, argv);

	ret = trace_kprobe_create_internal(argc, argv, &ctx);

	traceprobe_finish_parse(&ctx);
	trace_probe_log_clear();
	return ret;
}

/**
 * @brief Creates a `trace_kprobe` from a raw command string.
 * @param raw_command The raw command string.
 * @return 0 on success, or a negative errno on failure.
 */
static int trace_kprobe_create(const char *raw_command)
{
	return trace_probe_create(raw_command, trace_kprobe_create_cb);
}

/**
 * @brief Creates or deletes a `trace_kprobe` event.
 * @param raw_command The raw command string (starts with '-' for delete).
 * @return 0 on success, or a negative errno on failure.
 *
 * This function acts as a wrapper for `dyn_event_release` and `dyn_event_create`.
 */
static int create_or_delete_trace_kprobe(const char *raw_command)
{
	int ret;

	if (raw_command[0] == '-')
		return dyn_event_release(raw_command, &trace_kprobe_ops);

	ret = dyn_event_create(raw_command, &trace_kprobe_ops);
	return ret == -ECANCELED ? -EINVAL : ret;
}

/**
 * @brief Runs a `trace_kprobe` command.
 * @param cmd Pointer to `dynevent_cmd`.
 * @return 0 on success, or a negative errno on failure.
 */
static int trace_kprobe_run_command(struct dynevent_cmd *cmd)
{
	return create_or_delete_trace_kprobe(cmd->seq.buffer);
}

/**
 * @brief Initializes a kprobe event command object.
 * @param cmd: A pointer to the dynevent_cmd struct representing the new event
 * @param buf: A pointer to the buffer used to build the command
 * @param maxlen: The length of the buffer passed in @buf
 *
 * Initialize a synthetic event command object.  Use this before
 * calling any of the other kprobe_event functions.
 */
void kprobe_event_cmd_init(struct dynevent_cmd *cmd, char *buf, int maxlen)
{
	dynevent_cmd_init(cmd, buf, maxlen, DYNEVENT_TYPE_KPROBE,
			  trace_kprobe_run_command);
}
EXPORT_SYMBOL_GPL(kprobe_event_cmd_init);

/**
 * @brief Generates a kprobe event command from an argument list.
 * @param cmd: A pointer to the dynevent_cmd struct representing the new event
 * @param kretprobe: Is this a return probe?
 * @param name: The name of the kprobe event
 * @param loc: The location of the kprobe event
 * @param ...: Variable number of arg (pairs), one pair for each field
 *
 * NOTE: Users normally won't want to call this function directly, but
 * rather use the kprobe_event_gen_cmd_start() wrapper, which automatically
 * adds a NULL to the end of the arg list.  If this function is used
 * directly, make sure the last arg in the variable arg list is NULL.
 *
 * Generate a kprobe event command to be executed by
 * kprobe_event_gen_cmd_end().  This function can be used to generate the
 * complete command or only the first part of it; in the latter case,
 * kprobe_event_add_fields() can be used to add more fields following this.
 *
 * Unlikely the synth_event_gen_cmd_start(), @loc must be specified. This
 * returns -EINVAL if @loc == NULL.
 *
 * Return: 0 if successful, error otherwise.
 */
int __kprobe_event_gen_cmd_start(struct dynevent_cmd *cmd, bool kretprobe,
				 const char *name, const char *loc, ...)
{
	char buf[MAX_EVENT_NAME_LEN];
	struct dynevent_arg arg;
	va_list args;
	int ret;

	if (cmd->type != DYNEVENT_TYPE_KPROBE)
		return -EINVAL;

	if (!loc)
		return -EINVAL;

	if (kretprobe)
		snprintf(buf, MAX_EVENT_NAME_LEN, "r:kprobes/%s", name);
	else
		snprintf(buf, MAX_EVENT_NAME_LEN, "p:kprobes/%s", name);

	ret = dynevent_str_add(cmd, buf);
	if (ret)
		return ret;

	dynevent_arg_init(&arg, 0);
	arg.str = loc;
	ret = dynevent_arg_add(cmd, &arg, NULL);
	if (ret)
		return ret;

	va_start(args, loc);
	for (;;)
	{
		const char *field;

		field = va_arg(args, const char *);
		if (!field)
			break;

		if (++cmd->n_fields > MAX_TRACE_ARGS) {
			ret = -EINVAL;
			break;
		}

		arg.str = field;
		ret = dynevent_arg_add(cmd, &arg, NULL);
		if (ret)
			break;
	}
	va_end(args);

	return ret;
}
EXPORT_SYMBOL_GPL(__kprobe_event_gen_cmd_start);

/**
 * @brief Adds probe fields to a kprobe command from an argument list.
 * @param cmd: A pointer to the dynevent_cmd struct representing the new event
 * @param ...: Variable number of arg (pairs), one pair for each field
 *
 * NOTE: Users normally won't want to call this function directly, but
 * rather use the kprobe_event_add_fields() wrapper, which
 * automatically adds a NULL to the end of the arg list.  If this
 * function is used directly, make sure the last arg in the variable
 * arg list is NULL.
 *
 * Add probe fields to an existing kprobe command using a variable
 * list of args.  Fields are added in the same order they're listed.
 *
 * Return: 0 if successful, error otherwise.
 */
int __kprobe_event_add_fields(struct dynevent_cmd *cmd, ...)
{
	struct dynevent_arg arg;
	va_list args;
	int ret = 0;

	if (cmd->type != DYNEVENT_TYPE_KPROBE)
		return -EINVAL;

	dynevent_arg_init(&arg, 0);

	va_start(args, cmd);
	for (;;)
	{
		const char *field;

		field = va_arg(args, const char *);
		if (!field)
			break;

		if (++cmd->n_fields > MAX_TRACE_ARGS) {
			ret = -EINVAL;
			break;
		}

		arg.str = field;
		ret = dynevent_arg_add(cmd, &arg, NULL);
		if (ret)
			break;
	}
	va_end(args);

	return ret;
}
EXPORT_SYMBOL_GPL(__kprobe_event_add_fields);

/**
 * @brief Deletes a kprobe event by name.
 * @param name: The name of the kprobe event to delete
 *
 * Delete a kprobe event with the give @name from kernel code rather
 * than directly from the command line.
 *
 * Return: 0 if successful, error otherwise.
 */
int kprobe_event_delete(const char *name)
{
	char buf[MAX_EVENT_NAME_LEN];

	snprintf(buf, MAX_EVENT_NAME_LEN, "-:%s", name);

	return create_or_delete_trace_kprobe(buf);
}
EXPORT_SYMBOL_GPL(kprobe_event_delete);

/**
 * @brief Releases resources associated with a kprobe dynamic event.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function unregisters the `trace_kprobe` and then frees its resources.
 */
static int trace_kprobe_release(struct dyn_event *ev)
{
	struct trace_kprobe *tk = to_trace_kprobe(ev);
	int ret = unregister_trace_kprobe(tk);

	if (!ret)
		free_trace_kprobe(tk);
	return ret;
}

/**
 * @brief Displays the details of a kprobe dynamic event in a `seq_file`.
 * @param m Pointer to the `seq_file`.
 * @param ev Pointer to the dynamic event.
 * @return 0 on success.
 *
 * This function formats and prints the kprobe event details, including
 * its type (p/r), maxactive (for kretprobes), group, event name, symbol,
 * offset, and arguments.
 */
static int trace_kprobe_show(struct seq_file *m, struct dyn_event *ev)
{
	struct trace_kprobe *tk = to_trace_kprobe(ev);
	int i;

	seq_putc(m, trace_kprobe_is_return(tk) ? 'r' : 'p');
	if (trace_kprobe_is_return(tk) && tk->rp.maxactive)
		seq_printf(m, "%d", tk->rp.maxactive);
	seq_printf(m, ":%s/%s", trace_probe_group_name(&tk->tp),
				trace_probe_name(&tk->tp));

	if (!tk->symbol)
		seq_printf(m, " 0x%p", tk->rp.kp.addr);
	else if (tk->rp.kp.offset)
		seq_printf(m, " %s+%u", trace_kprobe_symbol(tk),
			   tk->rp.kp.offset);
	else
		seq_printf(m, " %s", trace_kprobe_symbol(tk));

	for (i = 0; i < tk->tp.nr_args; i++)
		seq_printf(m, " %s=%s", tk->tp.args[i].name, tk->tp.args[i].comm);
	seq_putc(m, '\n');

	return 0;
}

/**
 * @brief `seq_file` show function for displaying kprobes.
 * @param m Pointer to the `seq_file`.
 * @param v Pointer to the current dynamic event.
 * @return 0.
 *
 * This function filters dynamic events to show only `trace_kprobe` events.
 */
static int probes_seq_show(struct seq_file *m, void *v)
{
	struct dyn_event *ev = v;

	if (!is_trace_kprobe(ev))
		return 0;

	return trace_kprobe_show(m, ev);
}

/**
 * @var probes_seq_op
 * @brief `seq_operations` for listing kprobe events.
 */
static const struct seq_operations probes_seq_op = {
	.start  = dyn_event_seq_start,
	.next   = dyn_event_seq_next,
	.stop   = dyn_event_seq_stop,
	.show   = probes_seq_show
};

/**
 * @brief Opens the kprobe events file in tracefs.
 * @param inode Pointer to the inode.
 * @param file Pointer to the file.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function handles security checks and clears existing events if
 * `O_TRUNC` is used during open.
 */
static int probes_open(struct inode *inode, struct file *file)
{
	int ret;

	ret = security_locked_down(LOCKDOWN_TRACEFS);
	if (ret)
		return ret;

	if ((file->f_mode & FMODE_WRITE) && (file->f_flags & O_TRUNC)) {
		ret = dyn_events_release_all(&trace_kprobe_ops);
		if (ret < 0)
			return ret;
	}

	return seq_open(file, &probes_seq_op);
}

/**
 * @brief Writes to the kprobe events file in tracefs.
 * @param file Pointer to the file.
 * @param buffer User buffer to read from.
 * @param count Number of bytes to write.
 * @param ppos Pointer to file offset.
 * @return Number of bytes written, or a negative errno on failure.
 *
 * This function parses and executes kprobe commands from user space.
 */
static ssize_t probes_write(struct file *file, const char __user *buffer,
			    size_t count, loff_t *ppos)
{
	return trace_parse_run_command(file, buffer, count, ppos,
				       create_or_delete_trace_kprobe);
}

/**
 * @var kprobe_events_ops
 * @brief File operations for `/sys/kernel/debug/tracing/kprobe_events`.
 */
static const struct file_operations kprobe_events_ops = {
	.owner          = THIS_MODULE,
	.open           = probes_open,
	.read           = seq_read,
	.llseek         = seq_lseek,
	.release        = seq_release,
	.write		= probes_write,
};

/**
 * @brief Calculates the total number of missed hits for a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return The sum of missed hits for the kprobe and kretprobe.
 */
static unsigned long trace_kprobe_missed(struct trace_kprobe *tk)
{
	return trace_kprobe_is_return(tk) ?
		tk->rp.kp.nmissed + tk->rp.nmissed : tk->rp.kp.nmissed;
}

/* Probes profiling interfaces */
/**
 * @brief `seq_file` show function for displaying kprobe profile data.
 * @param m Pointer to the `seq_file`.
 * @param v Pointer to the current dynamic event.
 * @return 0.
 *
 * This function displays the hit count and missed count for each kprobe.
 */
static int probes_profile_seq_show(struct seq_file *m, void *v)
{
	struct dyn_event *ev = v;
	struct trace_kprobe *tk;
	unsigned long nmissed;

	if (!is_trace_kprobe(ev))
		return 0;

	tk = to_trace_kprobe(ev);
	nmissed = trace_kprobe_missed(tk);
	seq_printf(m, "  %-44s %15lu %15lu\n",
		   trace_probe_name(&tk->tp),
		   trace_kprobe_nhit(tk),
		   nmissed);

	return 0;
}

/**
 * @var profile_seq_op
 * @brief `seq_operations` for displaying kprobe profile data.
 */
static const struct seq_operations profile_seq_op = {
	.start  = dyn_event_seq_start,
	.next   = dyn_event_seq_next,
	.stop   = dyn_event_seq_stop,
	.show   = probes_profile_seq_show
};

/**
 * @brief Opens the kprobe profile file in tracefs.
 * @param inode Pointer to the inode.
 * @param file Pointer to the file.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function handles security checks before opening the profile file.
 */
static int profile_open(struct inode *inode, struct file *file)
{
	int ret;

	ret = security_locked_down(LOCKDOWN_TRACEFS);
	if (ret)
		return ret;

	return seq_open(file, &profile_seq_op);
}

/**
 * @var kprobe_profile_ops
 * @brief File operations for `/sys/kernel/debug/tracing/kprobe_profile`.
 */
static const struct file_operations kprobe_profile_ops = {
	.owner          = THIS_MODULE,
	.open           = profile_open,
	.read           = seq_read,
	.llseek         = seq_lseek,
	.release        = seq_release,
};

/* Note that we don't verify it, since the code does not come from user space */
/**
 * @brief Processes a `fetch_insn` to extract data from a trace record.
 * @param code Pointer to the `fetch_insn` structure.
 * @param rec Pointer to the raw trace record (which is `pt_regs *`).
 * @param edata Pointer to event-specific data.
 * @param dest Destination buffer for the extracted data.
 * @param base Base address for relative offsets.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function handles various fetch operations (`FETCH_OP_REG`, `FETCH_OP_STACK`, etc.)
 * to extract values from the `pt_regs` structure and other contexts.
 */
static int
process_fetch_insn(struct fetch_insn *code, void *rec, void *edata,
		   void *dest, void *base)
{
	struct pt_regs *regs = rec;
	unsigned long val;
	int ret;

retry:
	/* 1st stage: get value from context */
	switch (code->op) {
	case FETCH_OP_REG:
		val = regs_get_register(regs, code->param);
		break;
	case FETCH_OP_STACK:
		val = regs_get_kernel_stack_nth(regs, code->param);
		break;
	case FETCH_OP_STACKP:
		val = kernel_stack_pointer(regs);
		break;
	case FETCH_OP_RETVAL:
		val = regs_return_value(regs);
		break;
#ifdef CONFIG_HAVE_FUNCTION_ARG_ACCESS_API
	case FETCH_OP_ARG:
		val = regs_get_kernel_argument(regs, code->param);
		break;
	case FETCH_OP_EDATA:
		val = *(unsigned long *)((unsigned long)edata + code->offset);
		break;
#endif
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

/* Kprobe handler */
/**
 * @brief Internal handler for kprobe trace events.
 * @param tk Pointer to the `trace_kprobe`.
 * @param regs Pointer to `pt_regs`.
 * @param trace_file Pointer to the `trace_event_file` for output.
 *
 * This function is responsible for recording a kprobe event into the
 * trace buffer, extracting and storing the relevant arguments.
 */
static nokprobe_inline void
__kprobe_trace_func(struct trace_kprobe *tk, struct pt_regs *regs,
		    struct trace_event_file *trace_file)
{
	struct kprobe_trace_entry_head *entry;
	struct trace_event_call *call = trace_probe_event_call(&tk->tp);
	struct trace_event_buffer fbuffer;
	int dsize;

	WARN_ON(call != trace_file->event_call);

	if (trace_trigger_soft_disabled(trace_file))
		return;

	dsize = __get_data_size(&tk->tp, regs, NULL);

	entry = trace_event_buffer_reserve(&fbuffer, trace_file,
					   sizeof(*entry) + tk->tp.size + dsize);
	if (!entry)
		return;

	fbuffer.regs = regs;
	entry->ip = (unsigned long)tk->rp.kp.addr;
	store_trace_args(&entry[1], &tk->tp, regs, NULL, sizeof(*entry), dsize);

	trace_event_buffer_commit(&fbuffer);
}

/**
 * @brief Dispatches kprobe trace events to linked files.
 * @param tk Pointer to the `trace_kprobe`.
 * @param regs Pointer to `pt_regs`.
 *
 * This function iterates through all associated `trace_event_file` links
 * and calls `__kprobe_trace_func` for each to record the event.
 */
static void
kprobe_trace_func(struct trace_kprobe *tk, struct pt_regs *regs)
{
	struct event_file_link *link;

	trace_probe_for_each_link_rcu(link, &tk->tp)
		__kprobe_trace_func(tk, regs, link->file);
}
NOKPROBE_SYMBOL(kprobe_trace_func);

/* Kretprobe handler */

/**
 * @brief Entry handler for kretprobe-based trace kprobes.
 * @param ri Pointer to the `kretprobe_instance`.
 * @param regs Pointer to `pt_regs`.
 * @return 0 (always succeeds).
 *
 * This function is invoked when a kretprobe's target function is entered.
 * It stores arguments into `ri->data` if `entry_arg` is defined.
 */
static int trace_kprobe_entry_handler(struct kretprobe_instance *ri,
			      struct pt_regs *regs)
{
	struct kretprobe *rp = get_kretprobe(ri);
	struct trace_kprobe *tk;

	/*
	 * There is a small chance that get_kretprobe(ri) returns NULL when
	 * the kretprobe is unregister on another CPU between kretprobe's
	 * trampoline_handler and this function.
	 */
	if (unlikely(!rp))
		return -ENOENT;

	tk = container_of(rp, struct trace_kprobe, rp);

	/* store argument values into ri->data as entry data */
	if (tk->tp.entry_arg)
		store_trace_entry_data(ri->data, &tk->tp, regs);

	return 0;
}


/**
 * @brief Internal handler for kretprobe trace events.
 * @param tk Pointer to the `trace_kprobe`.
 * @param ri Pointer to the `kretprobe_instance`.
 * @param regs Pointer to `pt_regs`.
 * @param trace_file Pointer to the `trace_event_file` for output.
 *
 * This function is responsible for recording a kretprobe event into the
 * trace buffer, extracting and storing the relevant arguments, including
 * those captured at function entry.
 */
static nokprobe_inline void
__kretprobe_trace_func(struct trace_kprobe *tk, struct kretprobe_instance *ri,
		       struct pt_regs *regs,
		       struct trace_event_file *trace_file)
{
	struct kretprobe_trace_entry_head *entry;
	struct trace_event_buffer fbuffer;
	struct trace_event_call *call = trace_probe_event_call(&tk->tp);
	int dsize;

	WARN_ON(call != trace_file->event_call);

	if (trace_trigger_soft_disabled(trace_file))
		return;

	dsize = __get_data_size(&tk->tp, regs, ri->data);

	entry = trace_event_buffer_reserve(&fbuffer, trace_file,
					   sizeof(*entry) + tk->tp.size + dsize);
	if (!entry)
		return;

	fbuffer.regs = regs;
	entry->func = (unsigned long)tk->rp.kp.addr;
	entry->ret_ip = get_kretprobe_retaddr(ri);
	store_trace_args(&entry[1], &tk->tp, regs, ri->data, sizeof(*entry), dsize);

	trace_event_buffer_commit(&fbuffer);
}

/**
 * @brief Dispatches kretprobe trace events to linked files.
 * @param tk Pointer to the `trace_kprobe`.
 * @param ri Pointer to the `kretprobe_instance`.
 * @param regs Pointer to `pt_regs`.
 *
 * This function iterates through all associated `trace_event_file` links
 * and calls `__kretprobe_trace_func` for each to record the event.
 */
static void
kretprobe_trace_func(struct trace_kprobe *tk, struct kretprobe_instance *ri,
		     struct pt_regs *regs)
{
	struct event_file_link *link;

	trace_probe_for_each_link_rcu(link, &tk->tp)
		__kretprobe_trace_func(tk, ri, regs, link->file);
}
NOKPROBE_SYMBOL(kretprobe_trace_func);

/* Event entry printers */
/**
 * @brief Printer function for kprobe events in `trace_seq`.
 * @param iter Pointer to the `trace_iterator`.
 * @param flags Print flags.
 * @param event Pointer to the `trace_event`.
 * @return `print_line_t` status.
 *
 * This function formats and prints kprobe event data, including the
 * kprobe address and arguments.
 */
static enum print_line_t
print_kprobe_event(struct trace_iterator *iter, int flags,
		   struct trace_event *event)
{
	struct kprobe_trace_entry_head *field;
	struct trace_seq *s = &iter->seq;
	struct trace_probe *tp;

	field = (struct kprobe_trace_entry_head *)iter->ent;
	tp = trace_probe_primary_from_call(
		container_of(event, struct trace_event_call, event));
	if (WARN_ON_ONCE(!tp))
		goto out;

	trace_seq_printf(s, "%s: (", trace_probe_name(tp));

	if (!seq_print_ip_sym(s, field->ip, flags | TRACE_ITER_SYM_OFFSET))
		goto out;

	trace_seq_putc(s, ')');

	if (trace_probe_print_args(s, tp->args, tp->nr_args,
				     (u8 *)&field[1], field) < 0)
		goto out;

	trace_seq_putc(s, '\n');
 out:
	return trace_handle_return(s);
}

/**
 * @brief Printer function for kretprobe events in `trace_seq`.
 * @param iter Pointer to the `trace_iterator`.
 * @param flags Print flags.
 * @param event Pointer to the `trace_event`.
 * @return `print_line_t` status.
 *
 * This function formats and prints kretprobe event data, including the
 * return IP, function entry IP, and arguments.
 */
static enum print_line_t
print_kretprobe_event(struct trace_iterator *iter, int flags,
		      struct trace_event *event)
{
	struct kretprobe_trace_entry_head *field;
	struct trace_seq *s = &iter->seq;
	struct trace_probe *tp;

	field = (struct kretprobe_trace_entry_head *)iter->ent;
	tp = trace_probe_primary_from_call(
		container_of(event, struct trace_event_call, event));
	if (WARN_ON_ONCE(!tp))
		goto out;

	trace_seq_printf(s, "%s: (", trace_probe_name(tp));

	if (!seq_print_ip_sym(s, field->ret_ip, flags | TRACE_ITER_SYM_OFFSET))
		goto out;

	trace_seq_puts(s, " <- ");

	if (!seq_print_ip_sym(s, field->func, flags & ~TRACE_ITER_SYM_OFFSET))
		goto out;

	trace_seq_putc(s, ')');

	if (trace_probe_print_args(s, tp->args, tp->nr_args,
				     (u8 *)&field[1], field) < 0)
		goto out;

	trace_seq_putc(s, '\n');
 out:
	return trace_handle_return(s);
}


/**
 * @brief Defines the fields for a kprobe event call.
 * @param event_call Pointer to the `trace_event_call` for the kprobe.
 * @return 0 on success, -ENOENT if `trace_probe` is not found.
 *
 * This function defines the `ip` field for kprobe events.
 */
static int kprobe_event_define_fields(struct trace_event_call *event_call)
{
	int ret;
	struct kprobe_trace_entry_head field;
	struct trace_probe *tp;

	tp = trace_probe_primary_from_call(event_call);
	if (WARN_ON_ONCE(!tp))
		return -ENOENT;

	DEFINE_FIELD(unsigned long, ip, FIELD_STRING_IP, 0);

	return traceprobe_define_arg_fields(event_call, sizeof(field), tp);
}

/**
 * @brief Defines the fields for a kretprobe event call.
 * @param event_call Pointer to the `trace_event_call` for the kretprobe.
 * @return 0 on success, -ENOENT if `trace_probe` is not found.
 *
 * This function defines the `func` and `ret_ip` fields for kretprobe events.
 */
static int kretprobe_event_define_fields(struct trace_event_call *event_call)
{
	int ret;
	struct kretprobe_trace_entry_head field;
	struct trace_probe *tp;

	tp = trace_probe_primary_from_call(event_call);
	if (WARN_ON_ONCE(!tp))
		return -ENOENT;

	DEFINE_FIELD(unsigned long, func, FIELD_STRING_FUNC, 0);
	DEFINE_FIELD(unsigned long, ret_ip, FIELD_STRING_RETIP, 0);

	return traceprobe_define_arg_fields(event_call, sizeof(field), tp);
}

#ifdef CONFIG_PERF_EVENTS

/* Kprobe profile handler */
/**
 * @brief Perf event handler for kprobe events.
 * @param tk Pointer to the `trace_kprobe`.
 * @param regs Pointer to `pt_regs`.
 * @return 0 on success, 1 if BPF modified PC.
 *
 * This function is called when a kprobe perf event occurs. It handles
 * BPF program execution, allocates a buffer, fills it with event data,
 * and submits it to the perf event subsystem.
 */
static int
kprobe_perf_func(struct trace_kprobe *tk, struct pt_regs *regs)
{
	struct trace_event_call *call = trace_probe_event_call(&tk->tp);
	struct kprobe_trace_entry_head *entry;
	struct hlist_head *head;
	int size, __size, dsize;
	int rctx;

	if (bpf_prog_array_valid(call)) {
		unsigned long orig_ip = instruction_pointer(regs);
		int ret;

		ret = trace_call_bpf(call, regs);

		/*
		 * We need to check and see if we modified the pc of the
		 * pt_regs, and if so return 1 so that we don't do the
		 * single stepping.
		 */
		if (orig_ip != instruction_pointer(regs))
			return 1;
		if (!ret)
			return 0;
	}

	head = this_cpu_ptr(call->perf_events);
	if (hlist_empty(head))
		return 0;

	dsize = __get_data_size(&tk->tp, regs, NULL);
	__size = sizeof(*entry) + tk->tp.size + dsize;
	size = ALIGN(__size + sizeof(u32), sizeof(u64));
	size -= sizeof(u32);

	entry = perf_trace_buf_alloc(size, NULL, &rctx);
	if (!entry)
		return 0;

	entry->ip = (unsigned long)tk->rp.kp.addr;
	memset(&entry[1], 0, dsize);
	store_trace_args(&entry[1], &tk->tp, regs, NULL, sizeof(*entry), dsize);
	perf_trace_buf_submit(entry, size, rctx, call->event.type, 1, regs,
			      head, NULL);
	return 0;
}
NOKPROBE_SYMBOL(kprobe_perf_func);

/* Kretprobe profile handler */
/**
 * @brief Perf event handler for kretprobe events.
 * @param tk Pointer to the `trace_kprobe`.
 * @param ri Pointer to the `kretprobe_instance`.
 * @param regs Pointer to `pt_regs`.
 *
 * This function is called when a kretprobe perf event occurs. It handles
 * BPF program execution, allocates a buffer, fills it with event data,
 * and submits it to the perf event subsystem.
 */
static void
kretprobe_perf_func(struct trace_kprobe *tk, struct kretprobe_instance *ri,
		    struct pt_regs *regs)
{
	struct trace_event_call *call = trace_probe_event_call(&tk->tp);
	struct kretprobe_trace_entry_head *entry;
	struct hlist_head *head;
	int size, __size, dsize;
	int rctx;

	if (bpf_prog_array_valid(call) && !trace_call_bpf(call, regs))
		return;

	head = this_cpu_ptr(call->perf_events);
	if (hlist_empty(head))
		return;

	dsize = __get_data_size(&tk->tp, regs, ri->data);
	__size = sizeof(*entry) + tk->tp.size + dsize;
	size = ALIGN(__size + sizeof(u32), sizeof(u64));
	size -= sizeof(u32);

	entry = perf_trace_buf_alloc(size, NULL, &rctx);
	if (!entry)
		return;

	entry->func = (unsigned long)tk->rp.kp.addr;
	entry->ret_ip = get_kretprobe_retaddr(ri);
	store_trace_args(&entry[1], &tk->tp, regs, ri->data, sizeof(*entry), dsize);
	perf_trace_buf_submit(entry, size, rctx, call->event.type, 1, regs,
			      head, NULL);
}
NOKPROBE_SYMBOL(kretprobe_perf_func);

/**
 * @brief Retrieves kprobe information for BPF.
 * @param event Pointer to the `perf_event`.
 * @param fd_type Output parameter for BPF FD type.
 * @param symbol Output parameter for symbol name.
 * @param probe_offset Output parameter for probe offset.
 * @param probe_addr Output parameter for probe address.
 * @param missed Output parameter for missed hits.
 * @param perf_type_tracepoint True if perf event type is tracepoint.
 * @return 0 on success, -EINVAL on failure.
 *
 * This function extracts relevant kprobe information for BPF programs,
 * distinguishing between kprobes and kretprobes.
 */
int bpf_get_kprobe_info(const struct perf_event *event, u32 *fd_type,
			const char **symbol, u64 *probe_offset,
			 u64 *probe_addr, unsigned long *missed,
			 bool perf_type_tracepoint)
{
	const char *pevent = trace_event_name(event->tp_event);
	const char *group = event->tp_event->class->system;
	struct trace_kprobe *tk;

	if (perf_type_tracepoint)
		tk = find_trace_kprobe(pevent, group);
	else
		tk = trace_kprobe_primary_from_call(event->tp_event);
	if (!tk)
		return -EINVAL;

	*fd_type = trace_kprobe_is_return(tk) ? BPF_FD_TYPE_KRETPROBE
			      : BPF_FD_TYPE_KPROBE;
	*probe_offset = tk->rp.kp.offset;
	*probe_addr = kallsyms_show_value(current_cred()) ?
		      (unsigned long)tk->rp.kp.addr : 0;
	*symbol = tk->symbol;
	if (missed)
		*missed = trace_kprobe_missed(tk);
	return 0;
}
#endif	/* CONFIG_PERF_EVENTS */

/*
 * called by perf_trace_init() or __ftrace_set_clr_event() under event_mutex.
 *
 * kprobe_trace_self_tests_init() does enable_trace_probe/disable_trace_probe
 * lockless, but we can't race with this __init function.
 */
/**
 * @brief Register/unregister callback for kprobe events.
 * @param event Pointer to the `trace_event_call`.
 * @param type Type of registration operation (`TRACE_REG_REGISTER`, `TRACE_REG_UNREGISTER`, etc.).
 * @param data Opaque data (e.g., `trace_event_file`).
 * @return 0 on success.
 *
 * This function acts as the central registration point for kprobes,
 * routing `REGISTER` and `UNREGISTER` requests to `enable_trace_kprobe`
 * and `disable_trace_kprobe` respectively. It also handles perf event
 * registration.
 */
static int kprobe_register(struct trace_event_call *event,
			   enum trace_reg type, void *data)
{
	struct trace_event_file *file = data;

	switch (type) {
	case TRACE_REG_REGISTER:
		return enable_trace_kprobe(event, file);
	case TRACE_REG_UNREGISTER:
		return disable_trace_kprobe(event, file);

#ifdef CONFIG_PERF_EVENTS
	case TRACE_REG_PERF_REGISTER:
		return enable_trace_kprobe(event, NULL);
	case TRACE_REG_PERF_UNREGISTER:
		return disable_trace_kprobe(event, NULL);
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
 * @brief Kprobe dispatcher function.
 * @param kp Pointer to the `kprobe`.
 * @param regs Pointer to `pt_regs`.
 * @return 0 (or 1 if BPF modified PC).
 *
 * This function is called when a kprobe is hit. It increments the hit counter,
 * records the trace event, and dispatches to perf events if enabled.
 */
static int kprobe_dispatcher(struct kprobe *kp, struct pt_regs *regs)
{
	struct trace_kprobe *tk = container_of(kp, struct trace_kprobe, rp.kp);
	int ret = 0;

	raw_cpu_inc(*tk->nhit);

	if (trace_probe_test_flag(&tk->tp, TP_FLAG_TRACE))
		kprobe_trace_func(tk, regs);
#ifdef CONFIG_PERF_EVENTS
	if (trace_probe_test_flag(&tk->tp, TP_FLAG_PROFILE))
		ret = kprobe_perf_func(tk, regs);
#endif
	return ret;
}
NOKPROBE_SYMBOL(kprobe_dispatcher);

/**
 * @brief Kretprobe dispatcher function.
 * @param ri Pointer to the `kretprobe_instance`.
 * @param regs Pointer to `pt_regs`.
 * @return 0.
 *
 * This function is called when a kretprobe is hit. It increments the hit counter,
 * records the trace event, and dispatches to perf events if enabled.
 */
static int
kretprobe_dispatcher(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	struct kretprobe *rp = get_kretprobe(ri);
	struct trace_kprobe *tk;

	/*
	 * There is a small chance that get_kretprobe(ri) returns NULL when
	 * the kretprobe is unregister on another CPU between kretprobe's
	 * trampoline_handler and this function.
	 */
	if (unlikely(!rp))
		return 0;

	tk = container_of(rp, struct trace_kprobe, rp);
	raw_cpu_inc(*tk->nhit);

	if (trace_probe_test_flag(&tk->tp, TP_FLAG_TRACE))
		kretprobe_trace_func(tk, ri, regs);
#ifdef CONFIG_PERF_EVENTS
	if (trace_probe_test_flag(&tk->tp, TP_FLAG_PROFILE))
		kretprobe_perf_func(tk, ri, regs);
#endif
	return 0;	/* We don't tweak kernel, so just return 0 */
}
NOKPROBE_SYMBOL(kretprobe_dispatcher);

/**
 * @var kretprobe_funcs
 * @brief Trace event functions for kretprobe events.
 *
 * Defines the `trace` function (printer) for kretprobe events.
 */
static struct trace_event_functions kretprobe_funcs = {
	.trace	= print_kretprobe_event
};

/**
 * @var kprobe_funcs
 * @brief Trace event functions for kprobe events.
 *
 * Defines the `trace` function (printer) for kprobe events.
 */
static struct trace_event_functions kprobe_funcs = {
	.trace	= print_kprobe_event
};

/**
 * @var kretprobe_fields_array
 * @brief Array of `trace_event_fields` for kretprobe events.
 *
 * Defines how kretprobe event fields are defined, primarily using
 * `kretprobe_event_define_fields`.
 */
static struct trace_event_fields kretprobe_fields_array[] = {
	{ .type = TRACE_FUNCTION_TYPE,
	  .define_fields = kretprobe_event_define_fields },
	{}
};

/**
 * @var kprobe_fields_array
 * @brief Array of `trace_event_fields` for kprobe events.
 *
 * Defines how kprobe event fields are defined, primarily using
 * `kprobe_event_define_fields`.
 */
static struct trace_event_fields kprobe_fields_array[] = {
	{ .type = TRACE_FUNCTION_TYPE,
	  .define_fields = kprobe_event_define_fields },
	{}
};

/**
 * @brief Initializes the `trace_event_call` for a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 *
 * This function sets various flags and callbacks for the kprobe's
 * `trace_event_call`, linking it to the appropriate kprobe or kretprobe
 * printing functions and field definitions.
 */
static inline void init_trace_event_call(struct trace_kprobe *tk)
{
	struct trace_event_call *call = trace_probe_event_call(&tk->tp);

	if (trace_kprobe_is_return(tk)) {
		call->event.funcs = &kretprobe_funcs;
		call->class->fields_array = kretprobe_fields_array;
	} else {
		call->event.funcs = &kprobe_funcs;
		call->class->fields_array = kprobe_fields_array;
	}

	call->flags = TRACE_EVENT_FL_KPROBE;
	call->class->reg = kprobe_register;
}

/**
 * @brief Registers the `trace_event_call` for a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function initializes the event call structure and then
 * registers it with the trace probe subsystem.
 */
static int register_kprobe_event(struct trace_kprobe *tk)
{
	init_trace_event_call(tk);

	return trace_probe_register_event_call(&tk->tp);
}

/**
 * @brief Unregisters the `trace_event_call` for a `trace_kprobe`.
 * @param tk Pointer to the `trace_kprobe`.
 * @return 0 on success, or a negative errno on failure.
 */
static int unregister_kprobe_event(struct trace_kprobe *tk)
{
	return trace_probe_unregister_event_call(&tk->tp);
}

#ifdef CONFIG_PERF_EVENTS

/* create a trace_kprobe, but don't add it to global lists */
/**
 * @brief Creates a local (non-global) `trace_kprobe` for perf events.
 * @param func Symbol name for the kprobe.
 * @param addr Direct address for the kprobe.
 * @param offs Offset within the symbol.
 * @param is_return True if it's a kretprobe, false otherwise.
 * @return Pointer to the `trace_event_call` on success, or an `ERR_PTR` on failure.
 *
 * This function creates a `trace_kprobe` but does not add it to the global
 * dynamic event lists, making it suitable for internal perf event use.
 */
struct trace_event_call *
create_local_trace_kprobe(char *func, void *addr, unsigned long offs,
			  bool is_return)
{
	enum probe_print_type ptype;
	struct trace_kprobe *tk __free(free_trace_kprobe) = NULL;
	int ret;
	char *event;

	if (func) {
		ret = validate_probe_symbol(func);
		if (ret)
			return ERR_PTR(ret);
	}

	/*
	 * local trace_kprobes are not added to dyn_event, so they are never
	 * searched in find_trace_kprobe(). Therefore, there is no concern of
	 * duplicated name here.
	 */
	event = func ? func : "DUMMY_EVENT";

	tk = alloc_trace_kprobe(KPROBE_EVENT_SYSTEM, event, (void *)addr,
					func, offs, 0 /* maxactive */, 0 /* nargs */,
					is_return);

	if (IS_ERR(tk)) {
		pr_info("Failed to allocate trace_probe.(%d)\n",
			(int)PTR_ERR(tk));
		return ERR_CAST(tk);
	}

	init_trace_event_call(tk);

	ptype = trace_kprobe_is_return(tk) ?
		PROBE_PRINT_RETURN : PROBE_PRINT_NORMAL;
	if (traceprobe_set_print_fmt(&tk->tp, ptype) < 0)
		return ERR_PTR(-ENOMEM);

	ret = __register_trace_kprobe(tk);
	if (ret < 0)
		return ERR_PTR(ret);

	return trace_probe_event_call(&(no_free_ptr(tk)->tp));
}

/**
 * @brief Destroys a local (non-global) `trace_kprobe`.
 * @param event_call Pointer to the `trace_event_call` for the kprobe.
 *
 * This function unregisters the embedded kprobe and frees its resources.
 */
void destroy_local_trace_kprobe(struct trace_event_call *event_call)
{
	struct trace_kprobe *tk;

	tk = trace_kprobe_primary_from_call(event_call);
	if (unlikely(!tk))
		return;

	if (trace_probe_is_enabled(&tk->tp)) {
		WARN_ON(1);
		return;
	}

	__unregister_trace_kprobe(tk);

	free_trace_kprobe(tk);
}
#endif /* CONFIG_PERF_EVENTS */

/**
 * @brief Enables kprobe events specified on the kernel command line during boot.
 *
 * This function iterates through all registered `trace_kprobe` events and
 * enables those that were specified on the command line.
 */
static __init void enable_boot_kprobe_events(void)
{
	struct trace_array *tr = top_trace_array();
	struct trace_event_file *file;
	struct trace_kprobe *tk;
	struct dyn_event *pos;

	guard(mutex)(&event_mutex);
	for_each_trace_kprobe(tk, pos) {
		list_for_each_entry(file, &tr->events, list)
			if (file->event_call == trace_probe_event_call(&tk->tp))
				trace_event_enable_disable(file, 1, 0);
	}
}

/**
 * @brief Sets up kprobe events specified on the kernel command line.
 *
 * This function parses the `kprobe_boot_events_buf` from the command line,
 * creating or deleting kprobe events as specified, and then enables them.
 */
static __init void setup_boot_kprobe_events(void)
{
	char *p, *cmd = kprobe_boot_events_buf;
	int ret;

	strreplace(kprobe_boot_events_buf, ',', ' ');

	while (cmd && *cmd != '\0') {
		p = strchr(cmd, ';');
		if (p)
			*p++ = '\0';

		ret = create_or_delete_trace_kprobe(cmd);
		if (ret)
			pr_warn("Failed to add event(%d): %s\n", ret, cmd);

		cmd = p;
	}

	enable_boot_kprobe_events();
}

/*
 * Register dynevent at core_initcall. This allows kernel to setup kprobe
 * events in postcore_initcall without tracefs.
 */
/**
 * @brief Early initialization function for kprobe tracing events.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function registers the `trace_kprobe_ops` with the dynamic event
 * system and sets up a module notifier for kprobes, allowing them to be
 * set up early in the boot process.
 */
static __init int init_kprobe_trace_early(void)
{
	int ret;

	ret = dyn_event_register(&trace_kprobe_ops);
	if (ret)
		return ret;

	ret = trace_kprobe_register_module_notifier();
	if (ret)
		return ret;

	/* Setup boot kprobe_events after trace_kprobe_ops is registered. */
	if (kprobe_boot_events_buf[0] != '\0')
		setup_boot_kprobe_events();

	return 0;
}
core_initcall(init_kprobe_trace_early);

/* Make a tracefs interface for controlling probe points */
/**
 * @brief Initialization function for the kprobe tracing interface in tracefs.
 * @return 0 on success.
 *
 * This function creates the `kprobe_events` and `kprobe_profile` files
 * in tracefs, allowing users to dynamically create and manage kprobe events.
 */
static __init int init_kprobe_trace(void)
{
	int ret;

	ret = tracing_init_dentry();
	if (ret)
		return 0;

	/* Event list interface */
	trace_create_file("kprobe_events", TRACE_MODE_WRITE,
			  NULL, NULL, &kprobe_events_ops);

	/* Profile interface */
	trace_create_file("kprobe_profile", TRACE_MODE_READ,
			  NULL, NULL, &kprobe_profile_ops);

	setup_boot_kprobe_events();

	return 0;
}
fs_initcall(init_kprobe_trace);


#ifdef CONFIG_FTRACE_STARTUP_TEST
static __init struct trace_event_file *
find_trace_probe_file(struct trace_kprobe *tk, struct trace_array *tr)
{
	struct trace_event_file *file;

	list_for_each_entry(file, &tr->events, list)
		if (file->event_call == trace_probe_event_call(&tk->tp))
			return file;

	return NULL;
}

/*
 * Nobody but us can call enable_trace_kprobe/disable_trace_kprobe at this
 * stage, we can do this lockless.
 */
static __init int kprobe_trace_self_tests_init(void)
{
	int ret, warn = 0;
	int (*target)(int, int, int, int, int, int);
	struct trace_kprobe *tk;
	struct trace_event_file *file;

	if (tracing_is_disabled())
		return -ENODEV;

	if (tracing_selftest_disabled)
		return 0;

	target = kprobe_trace_selftest_target;

	pr_info("Testing kprobe tracing: ");

	ret = create_or_delete_trace_kprobe("p:testprobe kprobe_trace_selftest_target $stack $stack0 +0($stack)");
	if (WARN_ONCE(ret, "error on probing function entry.")) {
		warn++;
	} else {
		/* Enable trace point */
		tk = find_trace_kprobe("testprobe", KPROBE_EVENT_SYSTEM);
		if (WARN_ONCE(tk == NULL, "error on probing function entry.")) {
			warn++;
		} else {
			file = find_trace_probe_file(tk, top_trace_array());
			if (WARN_ONCE(file == NULL, "error on getting probe file.")) {
				warn++;
			} else
				enable_trace_kprobe(
						trace_probe_event_call(&tk->tp), file);
		}
	}

	ret = create_or_delete_trace_kprobe("r:testprobe2 kprobe_trace_selftest_target $retval");
	if (WARN_ONCE(ret, "error on probing function return.")) {
		warn++;
	} else {
		/* Enable trace point */
		tk = find_trace_kprobe("testprobe2", KPROBE_EVENT_SYSTEM);
		if (WARN_ONCE(tk == NULL, "error on getting 2nd new probe.")) {
			warn++;
		} else {
			file = find_trace_probe_file(tk, top_trace_array());
			if (WARN_ONCE(file == NULL, "error on getting probe file.")) {
				warn++;
			} else
				enable_trace_kprobe(
						trace_probe_event_call(&tk->tp), file);
		}
	}

	if (warn)
		goto end;

	ret = target(1, 2, 3, 4, 5, 6);

	/*
	 * Not expecting an error here, the check is only to prevent the
	 * optimizer from removing the call to target() as otherwise there
	 * are no side-effects and the call is never performed.
	 */
	if (ret != 21)
		warn++;

	/* Disable trace points before removing it */
	tk = find_trace_kprobe("testprobe", KPROBE_EVENT_SYSTEM);
	if (WARN_ONCE(tk == NULL, "error on getting test probe.")) {
		warn++;
	} else {
		if (WARN_ONCE(trace_kprobe_nhit(tk) != 1,
				 "incorrect number of testprobe hits."))
			warn++;

		file = find_trace_probe_file(tk, top_trace_array());
		if (WARN_ONCE(file == NULL, "error on getting probe file.")) {
			warn++;
		} else
			disable_trace_kprobe(
					trace_probe_event_call(&tk->tp), file);
	}

	tk = find_trace_kprobe("testprobe2", KPROBE_EVENT_SYSTEM);
	if (WARN_ONCE(tk == NULL, "error on getting 2nd test probe.")) {
		warn++;
	} else {
		if (WARN_ONCE(trace_kprobe_nhit(tk) != 1,
				 "incorrect number of testprobe2 hits."))
			warn++;

		file = find_trace_probe_file(tk, top_trace_array());
		if (WARN_ONCE(file == NULL, "error on getting probe file.")) {
			warn++;
		} else
			disable_trace_kprobe(
					trace_probe_event_call(&tk->tp), file);
	}

	ret = create_or_delete_trace_kprobe("-:testprobe");
	if (WARN_ONCE(ret, "error on deleting a probe."))
		warn++;

	ret = create_or_delete_trace_kprobe("-:testprobe2");
	if (WARN_ONCE(ret, "error on deleting a probe."))
		warn++;


end:
	/*
	 * Wait for the optimizer work to finish. Otherwise it might fiddle
	 * with probes in already freed __init text.
	 */
	wait_for_kprobe_optimizer();
	if (warn)
		pr_cont("NG: Some tests are failed. Please check them.\n");
	else
		pr_cont("OK\n");
	return 0;
}

late_initcall(kprobe_trace_self_tests_init);

#endif