/**
 * @file fprobe.h
 * @brief Provides a framework for ftrace-based probes in the Linux kernel.
 *
 * This header defines the necessary data structures and API for implementing
 * function entry/exit probes using the ftrace infrastructure. It allows
 * dynamic instrumentation of kernel functions for debugging, profiling,
 * and monitoring without modifying the original source code.
 *
 * The fprobe mechanism is designed to be lightweight and flexible, supporting
 * various use cases from simple function call logging to more complex
 * state-tracking during function execution. It leverages `ftrace` for the
 * underlying hooking mechanism and provides callback-based interfaces for
 * probe handlers.
 *
 * Key Structures:
 * - `struct fprobe`: The main structure to define a probe, including callbacks
 *   and internal state.
 * - `struct fprobe_hlist_node`, `struct fprobe_hlist`: Internal structures
 *   for managing probe addresses within hash tables for efficient lookup.
 *
 * Functional Utility:
 * - Dynamic kernel function instrumentation.
 * - Execution of custom code on function entry and/or exit.
 * - Support for filtering target functions.
 * - Management of per-probe private data.
 *
 * Time/Space Complexity:
 * - Probe registration/unregistration involves updating ftrace data structures
 *   and hash tables.
 * - Probe hit overhead is minimal, typically involving a function call and
 *   some data access per entry/exit.
 *
 * @see Documentation/trace/fprobe.rst in the Linux kernel source for more details.
 */
/* SPDX-License-Identifier: GPL-2.0 */
/* Simple ftrace probe wrapper */
#ifndef _LINUX_FPROBE_H
#define _LINUX_FPROBE_H

#include <linux/compiler.h>
#include <linux/ftrace.h>
#include <linux/rcupdate.h>
#include <linux/refcount.h>
#include <linux/slab.h>

struct fprobe;
/**
 * @typedef fprobe_entry_cb
 * @brief Callback function type for fprobe on function entry.
 * @param fp Pointer to the `fprobe` instance.
 * @param entry_ip The instruction pointer at function entry.
 * @param ret_ip The return instruction pointer.
 * @param regs Pointer to the ftrace registers structure.
 * @param entry_data Pointer to per-entry private data.
 * @return An integer status code, typically 0 for success.
 *
 * This callback is invoked when a probed function is entered.
 * It provides context about the function call and allows for
 * custom actions or data collection.
 */
typedef int (*fprobe_entry_cb)(struct fprobe *fp, unsigned long entry_ip,
			       unsigned long ret_ip, struct ftrace_regs *regs,
			       void *entry_data);

/**
 * @typedef fprobe_exit_cb
 * @brief Callback function type for fprobe on function exit.
 * @param fp Pointer to the `fprobe` instance.
 * @param entry_ip The instruction pointer at function entry.
 * @param ret_ip The return instruction pointer.
 * @param regs Pointer to the ftrace registers structure.
 * @param entry_data Pointer to per-entry private data.
 *
 * This callback is invoked when a probed function is exited.
 * It can be used to clean up `entry_data` or perform actions
 * related to function completion.
 */
typedef void (*fprobe_exit_cb)(struct fprobe *fp, unsigned long entry_ip,
			       unsigned long ret_ip, struct ftrace_regs *regs,
			       void *entry_data);

/**
 * @struct fprobe_hlist_node
 * @brief Address based hash list node for fprobe.
 *
 * This structure is used internally to manage probe addresses within
 * a hash table, facilitating efficient lookup of `fprobe` instances
 * based on the probed function's address.
 */
struct fprobe_hlist_node {
	struct hlist_node	hlist;	/**< @brief The hlist node for address search hash table. */
	unsigned long		addr;	/**< @brief One of the probing addresses of @fp. */
	struct fprobe		*fp;	/**< @brief The fprobe which owns this. */
};

/**
 * @struct fprobe_hlist
 * @brief Hash list nodes for fprobe.
 *
 * This structure aggregates `fprobe_hlist_node` instances for a single
 * `fprobe` and includes RCU (Read-Copy Update) mechanisms for safe
 * deferred release in concurrent environments.
 */
struct fprobe_hlist {
	struct hlist_node		hlist;		/**< @brief The hlist node for existence checking hash table. */
	struct rcu_head			rcu;		/**< @brief rcu_head for RCU deferred release. */
	struct fprobe			*fp;		/**< @brief The fprobe which owns this fprobe_hlist. */
	int				size;		/**< @brief The size of @array. */
	struct fprobe_hlist_node	array[] __counted_by(size); /**< @brief The fprobe_hlist_node for each address to probe. */
};

/**
 * @struct fprobe
 * @brief Ftrace based probe main structure.
 *
 * This is the core structure used to define and manage a function probe.
 * It holds configuration, state, and pointers to the entry and exit handlers.
 */
struct fprobe {
	unsigned long		nmissed;	/**< @brief The counter for missing events. */
	unsigned int		flags;		/**< @brief The status flag (e.g., FPROBE_FL_DISABLED). */
	size_t			entry_data_size;/**< @brief The size of per-entry private data storage. */

	fprobe_entry_cb entry_handler;		/**< @brief The callback function for function entry. */
	fprobe_exit_cb  exit_handler;		/**< @brief The callback function for function exit. */

	struct fprobe_hlist	*hlist_array;	/**< @brief The fprobe_hlist for fprobe search from IP hash table. */
};

/**
 * @def FPROBE_FL_DISABLED
 * @brief Flag indicating that this fprobe is soft-disabled.
 *
 * When this flag is set, the fprobe's handlers will not be invoked,
 * but the underlying ftrace hooks remain in place.
 */
#define FPROBE_FL_DISABLED	1

/**
 * @def FPROBE_FL_KPROBE_SHARED
 * @brief Flag indicating that this fprobe handler will be shared with kprobes.
 *
 * This flag must be set before registering the fprobe to ensure proper
 * interoperability with kprobe mechanisms.
 */
#define FPROBE_FL_KPROBE_SHARED	2

/**
 * @brief Checks if an fprobe is disabled.
 * @param fp Pointer to the `fprobe` instance.
 * @return True if the fprobe is disabled, false otherwise.
 */
static inline bool fprobe_disabled(struct fprobe *fp)
{
	return (fp) ? fp->flags & FPROBE_FL_DISABLED : false;
}

/**
 * @brief Checks if an fprobe is shared with kprobes.
 * @param fp Pointer to the `fprobe` instance.
 * @return True if the fprobe is shared with kprobes, false otherwise.
 */
static inline bool fprobe_shared_with_kprobes(struct fprobe *fp)
{
	return (fp) ? fp->flags & FPROBE_FL_KPROBE_SHARED : false;
}

#ifdef CONFIG_FPROBE
/**
 * @brief Registers an fprobe for functions matching specified filters.
 * @param fp Pointer to the `fprobe` instance to register.
 * @param filter Glob-style filter string for function names to probe.
 * @param notfilter Glob-style filter string for function names to exclude.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function registers an fprobe, activating its entry and exit callbacks
 * for functions whose names match the `filter` but not the `notfilter`.
 */
int register_fprobe(struct fprobe *fp, const char *filter, const char *notfilter);

/**
 * @brief Registers an fprobe for specific instruction pointers (addresses).
 * @param fp Pointer to the `fprobe` instance to register.
 * @param addrs Array of function addresses (instruction pointers) to probe.
 * @param num Number of addresses in the `addrs` array.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function registers an fprobe to specifically target a list of
 * kernel function addresses.
 */
int register_fprobe_ips(struct fprobe *fp, unsigned long *addrs, int num);

/**
 * @brief Registers an fprobe for functions identified by symbol names.
 * @param fp Pointer to the `fprobe` instance to register.
 * @param syms Array of symbol names (strings) of functions to probe.
 * @param num Number of symbol names in the `syms` array.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function registers an fprobe to specifically target functions
 * identified by their symbolic names.
 */
int register_fprobe_syms(struct fprobe *fp, const char **syms, int num);

/**
 * @brief Unregisters an fprobe.
 * @param fp Pointer to the `fprobe` instance to unregister.
 * @return 0 on success, or a negative errno on failure.
 *
 * This function removes the fprobe, deactivating its callbacks and
 * cleaning up associated resources.
 */
int unregister_fprobe(struct fprobe *fp);

/**
 * @brief Checks if an fprobe is currently registered.
 * @param fp Pointer to the `fprobe` instance.
 * @return True if the fprobe is registered, false otherwise.
 */
bool fprobe_is_registered(struct fprobe *fp);
#else /* CONFIG_FPROBE */
// Block Logic: Provides stub implementations when CONFIG_FPROBE is not enabled.
static inline int register_fprobe(struct fprobe *fp, const char *filter, const char *notfilter)
{
	return -EOPNOTSUPP;
}
static inline int register_fprobe_ips(struct fprobe *fp, unsigned long *addrs, int num)
{
	return -EOPNOTSUPP;
}
static inline int register_fprobe_syms(struct fprobe *fp, const char **syms, int num)
{
	return -EOPNOTSUPP;
}
static inline int unregister_fprobe(struct fprobe *fp)
{
	return -EOPNOTSUPP;
}
static inline bool fprobe_is_registered(struct fprobe *fp)
{
	return false;
}
#endif /* CONFIG_FPROBE */

/**
 * @brief Disables an fprobe.
 * @param fp The fprobe to be disabled.
 *
 * This will soft-disable @fp. Note that this doesn't remove the ftrace
 * hooks from the function entry; it only prevents the callbacks from executing.
 */
static inline void disable_fprobe(struct fprobe *fp)
{
	if (fp)
		fp->flags |= FPROBE_FL_DISABLED;
}

/**
 * @brief Enables an fprobe.
 * @param fp The fprobe to be enabled.
 *
 * This will soft-enable @fp, allowing its callbacks to be invoked again
 * when probed functions are hit.
 */
static inline void enable_fprobe(struct fprobe *fp)
{
	if (fp)
		fp->flags &= ~FPROBE_FL_DISABLED;
}

/**
 * @def FPROBE_DATA_SIZE_BITS
 * @brief Number of bits used to encode the entry data size.
 *
 * This macro defines the bit width used to store the size of the
 * per-entry private data.
 */
#define FPROBE_DATA_SIZE_BITS		4

/**
 * @def MAX_FPROBE_DATA_SIZE_WORD
 * @brief Maximum size of per-entry data in words.
 *
 * This macro calculates the maximum number of `long` words that can
 * be allocated for per-entry private data, derived from `FPROBE_DATA_SIZE_BITS`.
 */
#define MAX_FPROBE_DATA_SIZE_WORD	((1L << FPROBE_DATA_SIZE_BITS) - 1)

/**
 * @def MAX_FPROBE_DATA_SIZE
 * @brief Maximum size of per-entry data in bytes.
 *
 * This macro defines the maximum total size in bytes for the per-entry
 * private data that can be associated with an fprobe.
 */
#define MAX_FPROBE_DATA_SIZE		(MAX_FPROBE_DATA_SIZE_WORD * sizeof(long))

#endif
