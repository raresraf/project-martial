/**
 * @file x_tables.c
 * @brief Backend core for Netfilter's x_tables framework.
 *
 * This file implements the core functionality for the Netfilter x_tables framework,
 * which provides a generic table infrastructure for packet filtering and manipulation.
 * It serves as the backend for protocol-specific implementations like ip_tables,
 * ip6_tables, and arp_tables. The module handles registration of matches and targets,
 * management of table instances per network namespace, and compatibility layers for
 * userspace interactions.
 *
 * Functional Utility: Provides a flexible and extensible framework for defining and
 * applying packet filtering rules in the Linux kernel. It abstracts common functionalities
 * allowing protocol-specific modules to plug in their matching and targeting logic.
 *
 * Architecture:
 * - `xt_af`: Stores lists of registered matches and targets per address family,
 *            along with compatibility data.
 * - `xt_template`: Used for registering table initialization functions that
 *                  create table instances for new network namespaces.
 * - `xt_pernet`: Stores lists of active tables per network namespace.
 * - Extensive use of mutexes for protecting shared lists and data structures.
 * - Support for per-CPU counters for efficient packet accounting.
 * - Compatibility functions for handling differences in userspace and kernel data structures.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * x_tables core - Backend for {ip,ip6,arp}_tables
 *
 * Copyright (C) 2006-2006 Harald Welte <laforge@netfilter.org>
 * Copyright (C) 2006-2012 Patrick McHardy <kaber@trash.net>
 *
 * Based on existing ip_tables code which is
 *   Copyright (C) 1999 Paul `Rusty' Russell & Michael J. Neuling
 *   Copyright (C) 2000-2005 Netfilter Core Team <coreteam@netfilter.org>
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt ///< Macro to prepend module name to printk messages.
#include <linux/kernel.h> ///< Standard kernel definitions and logging.
#include <linux/module.h> ///< Module loading and unloading functionality.
#include <linux/socket.h> ///< Socket definitions.
#include <linux/net.h> ///< Networking core definitions.
#include <linux/proc_fs.h> ///< /proc filesystem functionality.
#include <linux/seq_file.h> ///< Sequence file interface for /proc.
#include <linux/string.h> ///< String manipulation functions.
#include <linux/vmalloc.h> ///< Virtual memory allocation functions.
#include <linux/mutex.h> ///< Mutex synchronization primitives.
#include <linux/mm.h> ///< Memory management definitions.
#include <linux/slab.h> ///< Slab allocator for kernel memory.
#include <linux/audit.h> ///< Audit logging interface.
#include <linux/user_namespace.h> ///< User namespace definitions.
#include <net/net_namespace.h> ///< Network namespace definitions.
#include <net/netns/generic.h> ///< Generic network namespace helpers.

#include <linux/netfilter/x_tables.h> ///< Netfilter x_tables specific definitions.
#include <linux/netfilter_arp.h> ///< ARP Netfilter definitions.
#include <linux/netfilter_ipv4/ip_tables.h> ///< IPv4 Netfilter ip_tables definitions.
#include <linux/netfilter_ipv6/ip6_tables.h> ///< IPv6 Netfilter ip6_tables definitions.
#include <linux/netfilter_arp/arp_tables.h> ///< ARP Netfilter arp_tables definitions.

MODULE_LICENSE("GPL"); ///< Sets the module's license to GPL.
MODULE_AUTHOR("Harald Welte <laforge@netfilter.org>"); ///< Sets the module's author.
MODULE_DESCRIPTION("{ip,ip6,arp,eb}_tables backend module"); ///< Sets the module's description.

#define XT_PCPU_BLOCK_SIZE 4096 ///< Size of a per-CPU block for counters, in bytes.
#define XT_MAX_TABLE_SIZE	(512 * 1024 * 1024) ///< Maximum allowed size for an x_tables table.

/**
 * @struct xt_template
 * @brief Represents a template for a Netfilter table.
 * Functional Utility: Allows protocol-specific modules to register a function that
 * initializes their tables within a given network namespace, facilitating dynamic
 * table creation.
 */
struct xt_template {
	struct list_head list; ///< Linked list node for template list.

	/* called when table is needed in the given netns */
	int (*table_init)(struct net *net); ///< Function pointer to initialize the table for a specific network namespace.

	struct module *me; ///< Pointer to the module owning this template.

	/* A unique name... */
	char name[XT_TABLE_MAXNAMELEN]; ///< Unique name of the table template.
};

/**
 * @brief Array of list heads for `xt_template` structures, indexed by Netfilter protocol family.
 * Functional Utility: Stores lists of registered table templates, allowing x_tables to find
 * and initialize tables for different protocol families.
 */
static struct list_head xt_templates[NFPROTO_NUMPROTO];

/**
 * @struct xt_pernet
 * @brief Per-network namespace data for x_tables.
 * Functional Utility: Stores the lists of active Netfilter tables (`xt_table`) for each
 * network namespace, ensuring isolation of table configurations between namespaces.
 */
struct xt_pernet {
	struct list_head tables[NFPROTO_NUMPROTO]; ///< Linked list heads for active tables, indexed by protocol family.
};

/**
 * @struct compat_delta
 * @brief Describes an offset difference for compatibility.
 * Functional Utility: Used during compatibility translations between 32-bit userspace
 * and 64-bit kernel structures to adjust memory offsets.
 */
struct compat_delta {
	unsigned int offset; /* offset in kernel */ ///< Offset within the kernel structure.
	int delta; /* delta in 32bit user land */ ///< Difference in offset for 32-bit userspace.
};

/**
 * @struct xt_af
 * @brief Per-address family data for x_tables.
 * Functional Utility: Stores lists of registered matches and targets for each
 * Netfilter protocol family, along with compatibility data structures.
 */
struct xt_af {
	struct mutex mutex; ///< Mutex to protect access to `match` and `target` lists.
	struct list_head match; ///< Linked list of registered `xt_match` structures.
	struct list_head target; ///< Linked list of registered `xt_target` structures.
#ifdef CONFIG_NETFILTER_XTABLES_COMPAT
	struct mutex compat_mutex; ///< Mutex to protect compatibility data.
	struct compat_delta *compat_tab; ///< Array of compatibility deltas.
	unsigned int number; /* number of slots in compat_tab[] */ ///< Total slots in `compat_tab`.
	unsigned int cur; /* number of used slots in compat_tab[] */ ///< Used slots in `compat_tab`.
#endif
};

static unsigned int xt_pernet_id __read_mostly; ///< Net namespace ID for x_tables per-net data.
static struct xt_af *xt __read_mostly; ///< Array of `xt_af` structures, one per protocol family.

/**
 * @brief Array of string prefixes for Netfilter protocol families.
 * Functional Utility: Used to construct module names (e.g., "ipt_match_name") for `request_module`.
 */
static const char *const xt_prefix[NFPROTO_NUMPROTO] = {
	[NFPROTO_UNSPEC] = "x", ///< Unspecified protocol family.
	[NFPROTO_IPV4]   = "ip", ///< IPv4 protocol family.
	[NFPROTO_ARP]    = "arp", ///< ARP protocol family.
	[NFPROTO_BRIDGE] = "eb", ///< Ethernet bridge protocol family.
	[NFPROTO_IPV6]   = "ip6", ///< IPv6 protocol family.
};

/* Registration hooks for targets. */
/**
 * @brief Registers a single Netfilter target.
 * Functional Utility: Adds a new `xt_target` definition to the list of available targets
 * for its specified address family, protected by a mutex.
 *
 * @param target Pointer to the `xt_target` structure to register.
 * @return 0 on success.
 */
int xt_register_target(struct xt_target *target)
{
	u_int8_t af = target->family; ///< Get address family from target.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex for the address family's target list.
	list_add(&target->list, &xt[af].target); ///< Add target to the list.
	mutex_unlock(&xt[af].mutex); ///< Release mutex.
	return 0;
}
EXPORT_SYMBOL(xt_register_target); ///< Export this function for use by other kernel modules.

/**
 * @brief Unregisters a single Netfilter target.
 * Functional Utility: Removes an `xt_target` definition from the list of available targets
 * for its specified address family, protected by a mutex.
 *
 * @param target Pointer to the `xt_target` structure to unregister.
 */
void
xt_unregister_target(struct xt_target *target)
{
	u_int8_t af = target->family; ///< Get address family from target.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex.
	list_del(&target->list); ///< Remove target from the list.
	mutex_unlock(&xt[af].mutex); ///< Release mutex.
}
EXPORT_SYMBOL(xt_unregister_target); ///< Export this function.

/**
 * @brief Registers multiple Netfilter targets.
 * Functional Utility: Iteratively registers an array of `xt_target` structures.
 * If an error occurs during registration, already registered targets are unregistered.
 *
 * @param target Pointer to the first `xt_target` in the array.
 * @param n Number of targets in the array.
 * @return 0 on success, or a negative errno on failure.
 */
int
xt_register_targets(struct xt_target *target, unsigned int n)
{
	unsigned int i;
	int err = 0;

	for (i = 0; i < n; i++) { ///< Block Logic: Iterate through the array of targets.
		err = xt_register_target(&target[i]); ///< Register each target.
		if (err)
			goto err; ///< If error, jump to cleanup.
	}
	return err;

err: ///< Error handling: unregister already registered targets.
	if (i > 0)
		xt_unregister_targets(target, i);
	return err;
}
EXPORT_SYMBOL(xt_register_targets); ///< Export this function.

/**
 * @brief Unregisters multiple Netfilter targets.
 * Functional Utility: Iteratively unregisters an array of `xt_target` structures.
 *
 * @param target Pointer to the first `xt_target` in the array.
 * @param n Number of targets in the array.
 */
void
xt_unregister_targets(struct xt_target *target, unsigned int n)
{
	while (n-- > 0) ///< Block Logic: Iterate backwards, unregistering each target.
		xt_unregister_target(&target[n]);
}
EXPORT_SYMBOL(xt_unregister_targets); ///< Export this function.

/**
 * @brief Registers a single Netfilter match.
 * Functional Utility: Adds a new `xt_match` definition to the list of available matches
 * for its specified address family, protected by a mutex.
 *
 * @param match Pointer to the `xt_match` structure to register.
 * @return 0 on success.
 */
int xt_register_match(struct xt_match *match)
{
	u_int8_t af = match->family; ///< Get address family from match.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex.
	list_add(&match->list, &xt[af].match); ///< Add match to the list.
	mutex_unlock(&xt[af].mutex); ///< Release mutex.
	return 0;
}
EXPORT_SYMBOL(xt_register_match); ///< Export this function.

/**
 * @brief Unregisters a single Netfilter match.
 * Functional Utility: Removes an `xt_match` definition from the list of available matches
 * for its specified address family, protected by a mutex.
 *
 * @param match Pointer to the `xt_match` structure to unregister.
 */
void
xt_unregister_match(struct xt_match *match)
{
	u_int8_t af = match->family; ///< Get address family from match.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex.
	list_del(&match->list); ///< Remove match from the list.
	mutex_unlock(&xt[af].mutex); ///< Release mutex.
}
EXPORT_SYMBOL(xt_unregister_match); ///< Export this function.

/**
 * @brief Registers multiple Netfilter matches.
 * Functional Utility: Iteratively registers an array of `xt_match` structures.
 * If an error occurs during registration, already registered matches are unregistered.
 *
 * @param match Pointer to the first `xt_match` in the array.
 * @param n Number of matches in the array.
 * @return 0 on success, or a negative errno on failure.
 */
int
xt_register_matches(struct xt_match *match, unsigned int n)
{
	unsigned int i;
	int err = 0;

	for (i = 0; i < n; i++) { ///< Block Logic: Iterate through the array of matches.
		err = xt_register_match(&match[i]); ///< Register each match.
		if (err)
			goto err; ///< If error, jump to cleanup.
	}
	return err;

err: ///< Error handling: unregister already registered matches.
	if (i > 0)
		xt_unregister_matches(match, i);
	return err;
}
EXPORT_SYMBOL(xt_register_matches); ///< Export this function.

/**
 * @brief Unregisters multiple Netfilter matches.
 * Functional Utility: Iteratively unregisters an array of `xt_match` structures.
 *
 * @param match Pointer to the first `xt_match` in the array.
 * @param n Number of matches in the array.
 */
void
xt_unregister_matches(struct xt_match *match, unsigned int n)
{
	while (n-- > 0) ///< Block Logic: Iterate backwards, unregistering each match.
		xt_unregister_match(&match[n]);
}
EXPORT_SYMBOL(xt_unregister_matches); ///< Export this function.


/*
 * These are weird, but module loading must not be done with mutex
 * held (since they will register), and we have to have a single
 * function to use.
 */

/**
 * @brief Finds a Netfilter match by name and revision, and takes a module reference.
 * Functional Utility: Searches for a registered `xt_match` module. If found and the
 * revision matches, it increments the module's reference count. It also attempts
 * to find family-independent matches if not found in the specified family.
 *
 * @param af Address family (e.g., NFPROTO_IPV4).
 * @param name Name of the match to find.
 * @param revision Revision of the match.
 * @return Pointer to the `xt_match` on success (with module ref held), or an `ERR_PTR` on error.
 */
struct xt_match *xt_find_match(u8 af, const char *name, u8 revision)
{
	struct xt_match *m;
	int err = -ENOENT;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL); // Return error if name is too long.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex for the match list.
	list_for_each_entry(m, &xt[af].match, list) { ///< Block Logic: Iterate through registered matches.
		if (strcmp(m->name, name) == 0) { ///< If name matches.
			if (m->revision == revision) { ///< If revision matches.
				if (try_module_get(m->me)) { ///< Try to get a module reference.
					mutex_unlock(&xt[af].mutex); ///< Release mutex.
					return m; ///< Return the found match.
				}
			} else
				err = -EPROTOTYPE; /* Found something, but revision mismatch. */
		}
	}
	mutex_unlock(&xt[af].mutex); ///< Release mutex.

	if (af != NFPROTO_UNSPEC)
		/* Try searching again in the family-independent list */
		return xt_find_match(NFPROTO_UNSPEC, name, revision); // Recursive call for family-independent list.

	return ERR_PTR(err); // Return error if not found.
}
EXPORT_SYMBOL(xt_find_match); ///< Export this function.

/**
 * @brief Requests and finds a Netfilter match, performing module autoloading if necessary.
 * Functional Utility: This function is a wrapper around `xt_find_match` that will
 * attempt to autoload a kernel module if the match is not found.
 *
 * @param nfproto Netfilter protocol family.
 * @param name Name of the match.
 * @param revision Revision of the match.
 * @return Pointer to the `xt_match` on success, or an `ERR_PTR` on error.
 */
struct xt_match *
xt_request_find_match(uint8_t nfproto, const char *name, uint8_t revision)
{
	struct xt_match *match;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL); // Return error if name is too long.

	match = xt_find_match(nfproto, name, revision); ///< Try to find match directly.
	if (IS_ERR(match)) { ///< If not found and it's an error.
		request_module("%st_%s", xt_prefix[nfproto], name); ///< Request module autoload.
		match = xt_find_match(nfproto, name, revision); ///< Try again after request.
	}

	return match;
}
EXPORT_SYMBOL_GPL(xt_request_find_match); ///< Export this function (GPL-only).

/**
 * @brief Finds a Netfilter target by name and revision, and takes a module reference.
 * Functional Utility: Searches for a registered `xt_target` module. If found and the
 * revision matches, it increments the module's reference count. It also attempts
 * to find family-independent targets if not found in the specified family.
 *
 * @param af Address family (e.g., NFPROTO_IPV4).
 * @param name Name of the target to find.
 * @param revision Revision of the target.
 * @return Pointer to the `xt_target` on success (with module ref held), or an `ERR_PTR` on error.
 */
static struct xt_target *xt_find_target(u8 af, const char *name, u8 revision)
{
	struct xt_target *t;
	int err = -ENOENT;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL); // Return error if name too long.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex.
	list_for_each_entry(t, &xt[af].target, list) { ///< Block Logic: Iterate through registered targets.
		if (strcmp(t->name, name) == 0) { ///< If name matches.
			if (t->revision == revision) { ///< If revision matches.
				if (try_module_get(t->me)) { ///< Try to get module reference.
					mutex_unlock(&xt[af].mutex); ///< Release mutex.
					return t; ///< Return found target.
				}
			} else
				err = -EPROTOTYPE; /* Found something, but revision mismatch. */
		}
	}
	mutex_unlock(&xt[af].mutex); ///< Release mutex.

	if (af != NFPROTO_UNSPEC)
		/* Try searching again in the family-independent list */
		return xt_find_target(NFPROTO_UNSPEC, name, revision); // Recursive call for family-independent list.

	return ERR_PTR(err); // Return error if not found.
}

/**
 * @brief Requests and finds a Netfilter target, performing module autoloading if necessary.
 * Functional Utility: This function is a wrapper around `xt_find_target` that will
 * attempt to autoload a kernel module if the target is not found.
 *
 * @param af Address family.
 * @param name Name of the target.
 * @param revision Revision of the target.
 * @return Pointer to the `xt_target` on success, or an `ERR_PTR` on error.
 */
struct xt_target *xt_request_find_target(u8 af, const char *name, u8 revision)
{
	struct xt_target *target;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL); // Return error if name too long.

	target = xt_find_target(af, name, revision); ///< Try to find target directly.
	if (IS_ERR(target)) { ///< If not found and it's an error.
		request_module("%st_%s", xt_prefix[af], name); ///< Request module autoload.
		target = xt_find_target(af, name, revision); ///< Try again after request.
	}

	return target;
}
EXPORT_SYMBOL_GPL(xt_request_find_target); ///< Export this function (GPL-only).


/**
 * @brief Copies x_tables object metadata (size, name, revision) from kernel to userspace.
 * Functional Utility: Facilitates communication between kernel Netfilter modules and
 * userspace tools by transferring object identification and size information.
 *
 * @param psize Userspace pointer to store the size.
 * @param size Kernel size of the object.
 * @param pname Userspace pointer to store the name.
 * @param name Kernel name of the object.
 * @param prev Userspace pointer to store the revision.
 * @param rev Kernel revision of the object.
 * @return 0 on success, or a negative errno (`-EFAULT`) if `copy_to_user` fails.
 */
static int xt_obj_to_user(u16 __user *psize, u16 size,
			  void __user *pname, const char *name,
			  u8 __user *prev, u8 rev)
{
	if (put_user(size, psize)) ///< Copy size to userspace.
		return -EFAULT;
	if (copy_to_user(pname, name, strlen(name) + 1)) ///< Copy name to userspace.
		return -EFAULT;
	if (put_user(rev, prev)) ///< Copy revision to userspace.
		return -EFAULT;

	return 0;
}

#define XT_OBJ_TO_USER(U, K, TYPE, C_SIZE)				\
	xt_obj_to_user(&U->u.TYPE##_size, C_SIZE ? : K->u.TYPE##_size,	\
		       U->u.user.name, K->u.kernel.TYPE->name,		\
		       &U->u.user.revision, K->u.kernel.TYPE->revision) ///< Macro to simplify `xt_obj_to_user` calls.

/**
 * @brief Copies x_tables data from kernel to userspace, handling alignment and padding.
 * Functional Utility: Transfers the actual data payload of a Netfilter match or target
 * from kernel memory to userspace, ensuring proper handling of user-specified size
 * and kernel alignment requirements.
 *
 * @param dst Userspace destination pointer.
 * @param src Kernel source pointer.
 * @param usersize Size requested by userspace.
 * @param size Actual kernel size.
 * @param aligned_size Aligned kernel size.
 * @return 0 on success, or a negative errno (`-EFAULT`) if `copy_to_user` or `clear_user` fails.
 */
int xt_data_to_user(void __user *dst, const void *src,
		    int usersize, int size, int aligned_size)
{
	usersize = usersize ? : size; ///< Use usersize if provided, else kernel size.
	if (copy_to_user(dst, src, usersize)) ///< Copy data to userspace.
		return -EFAULT;
	if (usersize != aligned_size && ///< Block Logic: If usersize is less than aligned size, clear remaining.
	    clear_user(dst + usersize, aligned_size - usersize))
		return -EFAULT;

	return 0;
}
EXPORT_SYMBOL_GPL(xt_data_to_user); ///< Export this function (GPL-only).

#define XT_DATA_TO_USER(U, K, TYPE)					\
	xt_data_to_user(U->data, K->data,				\
			K->u.kernel.TYPE->usersize,			\
			K->u.kernel.TYPE->TYPE##size,			\
			XT_ALIGN(K->u.kernel.TYPE->TYPE##size)) ///< Macro to simplify `xt_data_to_user` calls for matches/targets.

/**
 * @brief Copies an `xt_entry_match` structure from kernel to userspace.
 * Functional Utility: Helper function for `xt_data_to_user` to transfer match metadata and data.
 *
 * @param m Kernel `xt_entry_match` structure.
 * @param u Userspace `xt_entry_match` structure.
 * @return 0 on success, or a negative errno if copy operations fail.
 */
int xt_match_to_user(const struct xt_entry_match *m,
		     struct xt_entry_match __user *u)
{
	return XT_OBJ_TO_USER(u, m, match, 0) || ///< Copy match object metadata.
	       XT_DATA_TO_USER(u, m, match); ///< Copy match data.
}
EXPORT_SYMBOL_GPL(xt_match_to_user); ///< Export this function (GPL-only).

/**
 * @brief Copies an `xt_entry_target` structure from kernel to userspace.
 * Functional Utility: Helper function for `xt_data_to_user` to transfer target metadata and data.
 *
 * @param t Kernel `xt_entry_target` structure.
 * @param u Userspace `xt_entry_target` structure.
 * @return 0 on success, or a negative errno if copy operations fail.
 */
int xt_target_to_user(const struct xt_entry_target *t,
		      struct xt_entry_target __user *u)
{
	return XT_OBJ_TO_USER(u, t, target, 0) || ///< Copy target object metadata.
	       XT_DATA_TO_USER(u, t, target); ///< Copy target data.
}
EXPORT_SYMBOL_GPL(xt_target_to_user); ///< Export this function (GPL-only).

/**
 * @brief Checks the revision of a Netfilter match for a specific address family.
 * Functional Utility: Used internally to determine if a match with a given name
 * and revision exists, and to find the highest available revision. This is crucial
 * for versioning Netfilter extensions.
 *
 * @param af Address family.
 * @param name Name of the match.
 * @param revision Requested revision.
 * @param bestp Pointer to store the best (highest) revision found.
 * @return 1 if the exact revision is found, 0 otherwise.
 */
static int match_revfn(u8 af, const char *name, u8 revision, int *bestp)
{
	const struct xt_match *m;
	int have_rev = 0; ///< Flag indicating if the exact revision is found.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex for match list.
	list_for_each_entry(m, &xt[af].match, list) { ///< Block Logic: Iterate through registered matches.
		if (strcmp(m->name, name) == 0) { ///< If name matches.
			if (m->revision > *bestp) ///< Update best revision.
				*bestp = m->revision;
			if (m->revision == revision)
				have_rev = 1; ///< Exact revision found.
		}
	}
	mutex_unlock(&xt[af].mutex); ///< Release mutex.

	if (af != NFPROTO_UNSPEC && !have_rev)
		return match_revfn(NFPROTO_UNSPEC, name, revision, bestp); // Recursive call for family-independent matches.

	return have_rev;
}

/**
 * @brief Checks the revision of a Netfilter target for a specific address family.
 * Functional Utility: Used internally to determine if a target with a given name
 * and revision exists, and to find the highest available revision. This is crucial
 * for versioning Netfilter extensions.
 *
 * @param af Address family.
 * @param name Name of the target.
 * @param revision Requested revision.
 * @param bestp Pointer to store the best (highest) revision found.
 * @return 1 if the exact revision is found, 0 otherwise.
 */
static int target_revfn(u8 af, const char *name, u8 revision, int *bestp)
{
	const struct xt_target *t;
	int have_rev = 0; ///< Flag indicating if the exact revision is found.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex for target list.
	list_for_each_entry(t, &xt[af].target, list) { ///< Block Logic: Iterate through registered targets.
		if (strcmp(t->name, name) == 0) { ///< If name matches.
			if (t->revision > *bestp) ///< Update best revision.
				*bestp = t->revision;
			if (t->revision == revision)
				have_rev = 1; ///< Exact revision found.
		}
	}
	mutex_unlock(&xt[af].mutex); ///< Release mutex.

	if (af != NFPROTO_UNSPEC && !have_rev)
		return target_revfn(NFPROTO_UNSPEC, name, revision, bestp); // Recursive call for family-independent targets.

	return have_rev;
}

/**
 * @brief Finds the revision of a Netfilter extension (match or target).
 * Functional Utility: Determines if a given revision of a Netfilter extension exists
 * and what the highest available revision is. It sets an error code based on the findings.
 *
 * @param af Address family.
 * @param name Name of the extension.
 * @param revision Requested revision.
 * @param target Boolean flag: 1 for target, 0 for match.
 * @param err Pointer to an integer to store the error code or best revision.
 * @return 1 if the exact revision is found, 0 if no such extension exists, negative errno otherwise.
 */
int xt_find_revision(u8 af, const char *name, u8 revision, int target,
		     int *err)
{
	int have_rev, best = -1; ///< `best` stores the highest revision found.

	if (target == 1)
		have_rev = target_revfn(af, name, revision, &best); ///< Call target revision function.
	else
		have_rev = match_revfn(af, name, revision, &best); ///< Call match revision function.

	/* Nothing at all?  Return 0 to try loading module. */
	if (best == -1) { ///< Block Logic: If no extension was found.
		*err = -ENOENT; ///< Set error to No Entry.
		return 0; ///< Return 0, indicating module autoload might be needed.
	}

	*err = best; ///< Store the best revision.
	if (!have_rev)
		*err = -EPROTONOSUPPORT; ///< Set error if requested revision not found.
	return 1; ///< Return 1, indicating extension found (possibly wrong revision).
}
EXPORT_SYMBOL_GPL(xt_find_revision); ///< Export this function (GPL-only).

/**
 * @brief Converts hook masks to a human-readable string representation.
 * Functional Utility: Translates Netfilter hook bitmasks into a slash-separated
 * string of hook names (e.g., "PREROUTING/INPUT"), aiding in debugging and logging.
 *
 * @param buf Buffer to store the resulting string.
 * @param size Size of the buffer.
 * @param mask Hook mask (bitmask of NF_HOOK_XXX flags).
 * @param nfproto Netfilter protocol family.
 * @return Pointer to the buffer.
 */
static char *
textify_hooks(char *buf, size_t size, unsigned int mask, uint8_t nfproto)
{
	static const char *const inetbr_names[] = { ///< Hook names for IPv4/bridge families.
		"PREROUTING", "INPUT", "FORWARD",
		"OUTPUT", "POSTROUTING", "BROUTING",
	};
	static const char *const arp_names[] = { ///< Hook names for ARP family.
		"INPUT", "FORWARD", "OUTPUT",
	};
	const char *const *names; ///< Pointer to array of names.
	unsigned int i, max;
	char *p = buf; ///< Current position in buffer.
	bool np = false; ///< Flag for needing a slash separator.
	int res;

	names = (nfproto == NFPROTO_ARP) ? arp_names : inetbr_names; ///< Select appropriate names array.
	max   = (nfproto == NFPROTO_ARP) ? ARRAY_SIZE(arp_names) :
	                                   ARRAY_SIZE(inetbr_names); ///< Set max number of hooks.
	*p = '\0'; ///< Initialize buffer.
	for (i = 0; i < max; ++i) { ///< Block Logic: Iterate through possible hook numbers.
		if (!(mask & (1 << i)))
			continue; ///< If hook bit not set, skip.
		res = snprintf(p, size, "%s%s", np ? "/" : "", names[i]); ///< Print hook name, with separator.
		if (res > 0) {
			size -= res; ///< Update remaining size.
			p += res; ///< Advance buffer pointer.
		}
		np = true; ///< Next hook will need a separator.
	}

	return buf;
}

/**
 * @brief Checks that a given name is suitable for creating a file in `/proc`.
 * Functional Utility: Validates string format for procfs entries to prevent
 * security vulnerabilities (e.g., path traversal) and ensure proper naming conventions.
 *
 * @param name File name candidate.
 * @param size Length of the buffer.
 * @return 0 if the name is usable, or a negative errno on failure.
 */
int xt_check_proc_name(const char *name, unsigned int size)
{
	if (name[0] == '\0')
		return -EINVAL; ///< Name cannot be empty.

	if (strnlen(name, size) == size)
		return -ENAMETOOLONG; ///< Name is not NUL-terminated or too long.

	if (strcmp(name, ".") == 0 || ///< Block Logic: Check for special names or path separators.
	    strcmp(name, "..") == 0 ||
	    strchr(name, '/'))
		return -EINVAL; ///< Invalid characters or special names.

	return 0;
}
EXPORT_SYMBOL(xt_check_proc_name); ///< Export this function.

/**
 * @brief Checks the validity of a Netfilter match structure.
 * Functional Utility: Performs various checks on an `xt_match` structure, including
 * size validation, table applicability, hook mask compatibility, and protocol
 * matching, ensuring that the match is correctly configured.
 *
 * @param par Pointer to `xt_mtchk_param` containing match check parameters.
 * @param size Actual size of the match data.
 * @param proto Protocol (e.g., IPPROTO_TCP).
 * @param inv_proto Inverse protocol flag.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_check_match(struct xt_mtchk_param *par,
		   unsigned int size, u16 proto, bool inv_proto)
{
	int ret;

	// Block Logic: Check match size alignment.
	if (XT_ALIGN(par->match->matchsize) != size &&
	    par->match->matchsize != -1) {
		/*
		 * ebt_among is exempt from centralized matchsize checking
		 * because it uses a dynamic-size data set.
		 */
		pr_err_ratelimited("%s_tables: %s.%u match: invalid size %u (kernel) != (user) %u\n",
				   xt_prefix[par->family], par->match->name,
				   par->match->revision,
				   XT_ALIGN(par->match->matchsize), size);
		return -EINVAL;
	}
	// Block Logic: Check table applicability.
	if (par->match->table != NULL &&
	    strcmp(par->match->table, par->table) != 0) {
		pr_info_ratelimited("%s_tables: %s match: only valid in %s table, not %s\n",
				    xt_prefix[par->family], par->match->name,
				    par->match->table, par->table);
		return -EINVAL;
	}
	// Block Logic: Check hook mask compatibility.
	if (par->match->hooks && (par->hook_mask & ~par->match->hooks) != 0) {
		char used[64], allow[64];

		pr_info_ratelimited("%s_tables: %s match: used from hooks %s, but only valid from %s\n",
				    xt_prefix[par->family], par->match->name,
				    textify_hooks(used, sizeof(used),
						  par->hook_mask, par->family),
				    textify_hooks(allow, sizeof(allow),
						  par->match->hooks,
						  par->family));
		return -EINVAL;
	}
	// Block Logic: Check protocol matching.
	if (par->match->proto && (par->match->proto != proto || inv_proto)) {
		pr_info_ratelimited("%s_tables: %s match: only valid for protocol %u\n",
				    xt_prefix[par->family], par->match->name,
				    par->match->proto);
		return -EINVAL;
	}
	// Block Logic: Call match-specific checkentry function.
	if (par->match->checkentry != NULL) {
		ret = par->match->checkentry(par);
		if (ret < 0)
			return ret;
		else if (ret > 0)
			/* Flag up potential errors. */
			return -EIO;
	}
	return 0;
}
EXPORT_SYMBOL_GPL(xt_check_match); ///< Export this function (GPL-only).

/**
 * @brief Validates that matches within an entry end before the start of the target.
 *
 * @param match Beginning of `xt_entry_match` data.
 * @param target Beginning of the rule's target (alleged end of matches).
 * @param alignment Alignment requirement of match structures.
 *
 * Validates that all matches add up to the beginning of the target,
 * and that each match covers at least the base structure size.
 *
 * Return: 0 on success, negative errno on failure.
 */
static int xt_check_entry_match(const char *match, const char *target,
				const size_t alignment)
{
	const struct xt_entry_match *pos;
	int length = target - match; // Calculate total length of matches.

	if (length == 0) /* no matches */
		return 0;

	pos = (struct xt_entry_match *)match; ///< Initialize position to start of matches.
	do { ///< Block Logic: Iterate through each match structure.
		if ((unsigned long)pos % alignment)
			return -EINVAL; ///< Check alignment.

		if (length < (int)sizeof(struct xt_entry_match))
			return -EINVAL; ///< Check minimum size.

		if (pos->u.match_size < sizeof(struct xt_entry_match))
			return -EINVAL; ///< Check match data size.

		if (pos->u.match_size > length)
			return -EINVAL; ///< Check if match size exceeds remaining length.

		length -= pos->u.match_size; ///< Subtract current match size from remaining length.
		pos = ((void *)((char *)(pos) + (pos)->u.match_size)); ///< Advance to next match.
	} while (length > 0); ///< Continue until all matches are processed.

	return 0;
}

/**
 * @brief Checks hook entry points and underflow points for sanity.
 * Functional Utility: Validates the configuration of hook entry points and underflow
 * points within an `xt_table_info` structure, ensuring that they are correctly set
 * and sorted. This is critical for preventing infinite loops or incorrect packet processing.
 *
 * @param info `xt_table_info` to check.
 * @param valid_hooks Hook entry points that we can enter from.
 * @return 0 on success, or a negative errno on failure (`-EINVAL`) for configuration errors.
 */
int xt_check_table_hooks(const struct xt_table_info *info, unsigned int valid_hooks)
{
	const char *err = "unsorted underflow"; ///< Default error message.
	unsigned int i, max_uflow, max_entry;
	bool check_hooks = false;

	BUILD_BUG_ON(ARRAY_SIZE(info->hook_entry) != ARRAY_SIZE(info->underflow)); ///< Compile-time check for array sizes.

	max_entry = 0;
	max_uflow = 0;

	for (i = 0; i < ARRAY_SIZE(info->hook_entry); i++) { ///< Block Logic: Iterate through all possible hook entries.
		if (!(valid_hooks & (1 << i)))
			continue; ///< Skip if hook is not valid.

		if (info->hook_entry[i] == 0xFFFFFFFF)
			return -EINVAL; ///< Hook entry point must be valid.
		if (info->underflow[i] == 0xFFFFFFFF)
			return -EINVAL; ///< Underflow point must be valid.

		if (check_hooks) { ///< Block Logic: After the first valid hook, check sorting and duplicates.
			if (max_uflow > info->underflow[i]) {
				goto error; ///< Unsorted underflow.
            }
			if (max_uflow == info->underflow[i]) {
				err = "duplicate underflow";
				goto error; ///< Duplicate underflow.
			}
			if (max_entry > info->hook_entry[i]) {
				err = "unsorted entry";
				goto error; ///< Unsorted entry.
			}
			if (max_entry == info->hook_entry[i]) {
				err = "duplicate entry";
				goto error; ///< Duplicate entry.
			}
		}
		max_entry = info->hook_entry[i]; ///< Update max entry.
		max_uflow = info->underflow[i]; ///< Update max underflow.
		check_hooks = true; ///< Start checking sorting from next iteration.
	}

	return 0;
error: ///< Error handling for hook configuration issues.
	pr_err_ratelimited("%s at hook %d\n", err, i);
	return -EINVAL;
}
EXPORT_SYMBOL(xt_check_table_hooks); ///< Export this function.

/**
 * @brief Checks if a Netfilter verdict is valid.
 * Functional Utility: Determines if an integer verdict value corresponds to a
 * valid Netfilter action (e.g., NF_ACCEPT, NF_DROP, NF_QUEUE, XT_RETURN).
 *
 * @param verdict The integer verdict value.
 * @return `true` if the verdict is valid, `false` otherwise.
 */
static bool verdict_ok(int verdict)
{
	if (verdict > 0)
		return true; // Positive verdicts are valid (direct jump offsets).

	if (verdict < 0) {
		int v = -verdict - 1; // Translate negative verdicts to NF_HOOK_* values.

		if (verdict == XT_RETURN)
			return true; // XT_RETURN is a valid verdict.

		switch (v) { ///< Block Logic: Check specific standard Netfilter actions.
		case NF_ACCEPT: return true;
		case NF_DROP: return true;
		case NF_QUEUE: return true;
		default:
			break;
		}

		return false; // Invalid negative verdict.
	}

	return false; // Zero verdict is not valid in this context.
}

/**
 * @brief Checks if an error target is correctly formatted.
 * Functional Utility: Validates the size and content of an error target message,
 * ensuring it fits within the allocated buffer and is null-terminated.
 *
 * @param usersize Userspace reported size of the target.
 * @param kernsize Kernel expected size of the target.
 * @param msg Error message string.
 * @param msglen Length of the error message buffer.
 * @return `true` if the error target is well-formed, `false` otherwise.
 */
static bool error_tg_ok(unsigned int usersize, unsigned int kernsize,
			const char *msg, unsigned int msglen)
{
	return usersize == kernsize && strnlen(msg, msglen) < msglen; ///< Check size match and null-termination.
}

#ifdef CONFIG_NETFILTER_XTABLES_COMPAT
/**
 * @brief Adds an offset to the compatibility table.
 * Functional Utility: Records an offset and delta for translating kernel memory
 * addresses to userspace 32-bit addresses, crucial for compatibility with older tools.
 *
 * @param af Address family.
 * @param offset Kernel offset.
 * @param delta Delta to apply for userspace.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_compat_add_offset(u_int8_t af, unsigned int offset, int delta)
{
	struct xt_af *xp = &xt[af]; ///< Get per-AF data.

	WARN_ON(!mutex_is_locked(&xt[af].compat_mutex)); ///< Precondition: Mutex must be locked.

	if (WARN_ON(!xp->compat_tab))
		return -ENOMEM; ///< Compatibility table not allocated.

	if (xp->cur >= xp->number)
		return -EINVAL; ///< Table is full.

	if (xp->cur)
		delta += xp->compat_tab[xp->cur - 1].delta; ///< Accumulate delta.
	xp->compat_tab[xp->cur].offset = offset; ///< Store kernel offset.
	xp->compat_tab[xp->cur].delta = delta; ///< Store delta.
	xp->cur++; ///< Increment current count.
	return 0;
}
EXPORT_SYMBOL_GPL(xt_compat_add_offset); ///< Export this function (GPL-only).

/**
 * @brief Flushes (frees) the compatibility offset table.
 * Functional Utility: Deallocates the memory used by the `compat_tab`,
 * clearing all stored compatibility offsets.
 *
 * @param af Address family.
 */
void xt_compat_flush_offsets(u_int8_t af)
{
	WARN_ON(!mutex_is_locked(&xt[af].compat_mutex)); ///< Precondition: Mutex must be locked.

	if (xt[af].compat_tab) { ///< Block Logic: If table exists, free it.
		vfree(xt[af].compat_tab);
		xt[af].compat_tab = NULL;
		xt[af].number = 0;
		xt[af].cur = 0;
	}
}
EXPORT_SYMBOL_GPL(xt_compat_flush_offsets); ///< Export this function (GPL-only).

/**
 * @brief Calculates the compatibility jump offset for a given kernel offset.
 * Functional Utility: Performs a binary search on the `compat_tab` to find the
 * accumulated delta that needs to be applied to a kernel offset to get its
 * userspace 32-bit equivalent.
 * Algorithm: Binary search.
 * Time Complexity: O(log N) where N is the number of entries in `compat_tab`.
 *
 * @param af Address family.
 * @param offset Kernel offset.
 * @return The calculated jump delta.
 */
int xt_compat_calc_jump(u_int8_t af, unsigned int offset)
{
	struct compat_delta *tmp = xt[af].compat_tab; ///< Pointer to compatibility table.
	int mid, left = 0, right = xt[af].cur - 1; ///< Binary search indices.

	/**
	 * Block Logic: Perform binary search to find the correct delta.
	 * Invariant: `left` and `right` define the search range.
	 */
	while (left <= right) {
		mid = (left + right) >> 1; // Calculate middle index.
		if (offset > tmp[mid].offset)
			left = mid + 1; // Search in right half.
		else if (offset < tmp[mid].offset)
			right = mid - 1; // Search in left half.
		else
			return mid ? tmp[mid - 1].delta : 0; // Found exact offset, return previous delta.
	}
	return left ? tmp[left - 1].delta : 0; // Return delta from closest entry.
}
EXPORT_SYMBOL_GPL(xt_compat_calc_jump); ///< Export this function (GPL-only).

/**
 * @brief Initializes the compatibility offset table.
 * Functional Utility: Allocates memory for the `compat_tab` which will store
 * offset deltas for compatibility translation.
 *
 * @param af Address family.
 * @param number Number of entries to allocate.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_compat_init_offsets(u8 af, unsigned int number)
{
	size_t mem;

	WARN_ON(!mutex_is_locked(&xt[af].compat_mutex)); ///< Precondition: Mutex must be locked.

	if (!number || number > (INT_MAX / sizeof(struct compat_delta)))
		return -EINVAL; ///< Invalid number of entries.

	if (WARN_ON(xt[af].compat_tab))
		return -EINVAL; ///< Table already initialized.

	mem = sizeof(struct compat_delta) * number; ///< Calculate memory size.
	if (mem > XT_MAX_TABLE_SIZE)
		return -ENOMEM; ///< Memory request too large.

	xt[af].compat_tab = vmalloc(mem); ///< Allocate memory.
	if (!xt[af].compat_tab)
		return -ENOMEM; ///< Memory allocation failed.

	xt[af].number = number; ///< Store total number of entries.
	xt[af].cur = 0; ///< Initialize current count.

	return 0;
}
EXPORT_SYMBOL(xt_compat_init_offsets); ///< Export this function.

/**
 * @brief Calculates the compatibility offset for a Netfilter match.
 * Functional Utility: Determines the size difference between the kernel and
 * userspace (compat) representations of a match structure, needed for adjusting
 * offsets during compatibility translations.
 *
 * @param match Pointer to the `xt_match` structure.
 * @return The calculated offset difference.
 */
int xt_compat_match_offset(const struct xt_match *match)
{
	u_int16_t csize = match->compatsize ? : match->matchsize; ///< Use compatsize if provided, else matchsize.
	return XT_ALIGN(match->matchsize) - COMPAT_XT_ALIGN(csize); ///< Calculate difference.
}
EXPORT_SYMBOL_GPL(xt_compat_match_offset); ///< Export this function (GPL-only).

/**
 * @brief Converts an `xt_entry_match` from userspace (compat) to kernel format.
 * Functional Utility: Translates a match structure from a userspace 32-bit
 * format to the kernel's native format, handling size adjustments and
 * data copying.
 *
 * @param m Kernel `xt_entry_match` structure.
 * @param dstptr Pointer to the destination for the converted match.
 * @param size Pointer to the total size, which will be adjusted by the offset.
 */
void xt_compat_match_from_user(struct xt_entry_match *m, void **dstptr,
				unsigned int *size)
{
	const struct xt_match *match = m->u.kernel.match; ///< Get kernel match pointer.
	struct compat_xt_entry_match *cm = (struct compat_xt_entry_match *)m; ///< Cast to compat structure.
	int off = xt_compat_match_offset(match); ///< Calculate offset.
	u_int16_t msize = cm->u.user.match_size; ///< Get userspace match size.
	char name[sizeof(m->u.user.name)]; ///< Buffer for name.

	m = *dstptr; ///< Set `m` to destination pointer.
	memcpy(m, cm, sizeof(*cm)); ///< Copy base structure.
	if (match->compat_from_user) ///< Block Logic: If custom `compat_from_user` function exists.
		match->compat_from_user(m->data, cm->data); ///< Call custom function.
	else
		memcpy(m->data, cm->data, msize - sizeof(*cm)); ///< Copy data directly.

	msize += off; ///< Adjust match size by offset.
	m->u.user.match_size = msize; ///< Update userspace match size.
	strscpy(name, match->name, sizeof(name)); ///< Copy name.
	module_put(match->me); ///< Release module reference.
	strscpy_pad(m->u.user.name, name, sizeof(m->u.user.name)); ///< Copy name with padding.

	*size += off; ///< Adjust total size.
	*dstptr += msize; ///< Advance destination pointer.
}
EXPORT_SYMBOL_GPL(xt_compat_match_from_user); ///< Export this function (GPL-only).

#define COMPAT_XT_DATA_TO_USER(U, K, TYPE, C_SIZE)			\
	xt_data_to_user(U->data, K->data,				\
			K->u.kernel.TYPE->usersize,			\
			C_SIZE,						\
			COMPAT_XT_ALIGN(C_SIZE)) ///< Macro to simplify `xt_data_to_user` calls for compatibility.

/**
 * @brief Converts an `xt_entry_match` from kernel to userspace (compat) format.
 * Functional Utility: Translates a match structure from the kernel's native format
 * to a userspace 32-bit format, handling size adjustments and data copying.
 *
 * @param m Kernel `xt_entry_match` structure.
 * @param dstptr Pointer to the destination for the converted match.
 * @param size Pointer to the total size, which will be adjusted by the offset.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_compat_match_to_user(const struct xt_entry_match *m,
			    void __user **dstptr, unsigned int *size)
{
	const struct xt_match *match = m->u.kernel.match; ///< Get kernel match pointer.
	struct compat_xt_entry_match __user *cm = *dstptr; ///< Cast to compat userspace structure.
	int off = xt_compat_match_offset(match); ///< Calculate offset.
	u_int16_t msize = m->u.user.match_size - off; ///< Calculate userspace match size.

	if (XT_OBJ_TO_USER(cm, m, match, msize)) ///< Copy match object metadata.
		return -EFAULT;

	if (match->compat_to_user) { ///< Block Logic: If custom `compat_to_user` function exists.
		if (match->compat_to_user((void __user *)cm->data, m->data))
			return -EFAULT;
	} else {
		if (COMPAT_XT_DATA_TO_USER(cm, m, match, msize - sizeof(*cm))) ///< Copy data using compat macro.
			return -EFAULT;
	}

	*size -= off; ///< Adjust total size.
	*dstptr += msize; ///< Advance destination pointer.
	return 0;
}
EXPORT_SYMBOL_GPL(xt_compat_match_to_user); ///< Export this function (GPL-only).

/* non-compat version may have padding after verdict */
/**
 * @struct compat_xt_standard_target
 * @brief Compatibility structure for a standard Netfilter target.
 * Functional Utility: Represents a standard target (like ACCEPT, DROP) in a
 * compatibility format, used for translating between kernel and userspace structures.
 */
struct compat_xt_standard_target {
	struct compat_xt_entry_target t; ///< Compatibility target entry.
	compat_uint_t verdict; ///< The target verdict.
};

/**
 * @struct compat_xt_error_target
 * @brief Compatibility structure for a Netfilter error target.
 * Functional Utility: Represents an error target (which indicates an error condition)
 * in a compatibility format, used for translating between kernel and userspace structures.
 */
struct compat_xt_error_target {
	struct compat_xt_entry_target t; ///< Compatibility target entry.
	char errorname[XT_FUNCTION_MAXNAMELEN]; ///< Name of the error.
};

/**
 * @brief Checks compatibility entry offsets for validity.
 * Functional Utility: Validates the layout and sizes of Netfilter entry structures
 * during compatibility translations, ensuring that matches and targets are correctly
 * aligned and sized for both kernel and userspace (compat) representations.
 *
 * @param base Base pointer to the compatibility entry.
 * @param elems Pointer to the first match element.
 * @param target_offset Offset to the target structure.
 * @param next_offset Offset to the next rule.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_compat_check_entry_offsets(const void *base, const char *elems,
				  unsigned int target_offset,
				  unsigned int next_offset)
{
	long size_of_base_struct = elems - (const char *)base; ///< Size of the base structure.
	const struct compat_xt_entry_target *t;
	const char *e = base;

	if (target_offset < size_of_base_struct)
		return -EINVAL; ///< Target starts within the base struct, which is invalid.

	if (target_offset + sizeof(*t) > next_offset)
		return -EINVAL; ///< Target extends beyond the end of the rule.

	t = (void *)(e + target_offset); ///< Pointer to the target.
	if (t->u.target_size < sizeof(*t))
		return -EINVAL; ///< Target size is too small.

	if (target_offset + t->u.target_size > next_offset)
		return -EINVAL; ///< Target size exceeds rule boundary.

	// Block Logic: Check specific target types (STANDARD, ERROR) for validity.
	if (strcmp(t->u.user.name, XT_STANDARD_TARGET) == 0) {
		const struct compat_xt_standard_target *st = (const void *)t;

		if (COMPAT_XT_ALIGN(target_offset + sizeof(*st)) != next_offset)
			return -EINVAL; ///< Check alignment for standard target.

		if (!verdict_ok(st->verdict))
			return -EINVAL; ///< Check if verdict is valid.
	} else if (strcmp(t->u.user.name, XT_ERROR_TARGET) == 0) {
		const struct compat_xt_error_target *et = (const void *)t;

		if (!error_tg_ok(t->u.target_size, sizeof(*et),
				 et->errorname, sizeof(et->errorname)))
			return -EINVAL; ///< Check error target validity.
	}

	/* compat_xt_entry match has less strict alignment requirements,
	 * otherwise they are identical.  In case of padding differences
	 * we need to add compat version of xt_check_entry_match.
	 */
	BUILD_BUG_ON(sizeof(struct compat_xt_entry_match) != sizeof(struct xt_entry_match)); ///< Compile-time check for structure size.

	return xt_check_entry_match(elems, base + target_offset,
				    __alignof__(struct compat_xt_entry_match)); ///< Check match entries.
}
EXPORT_SYMBOL(xt_compat_check_entry_offsets); ///< Export this function.
#endif /* CONFIG_NETFILTER_XTABLES_COMPAT */

/**
 * @brief Validates Netfilter entry offsets for correctness.
 *
 * @param base Pointer to the arp/ip/ip6t_entry structure.
 * @param elems Pointer to the first `xt_entry_match` structure.
 * @param target_offset Offset from `base` to the target structure.
 * @param next_offset Offset from `base` to the next rule (also total size of current rule).
 *
 * Functional Utility: Ensures that the layout of a Netfilter rule (including matches
 * and the target) is consistent and correctly aligned, preventing memory corruption
 * and incorrect rule processing. It checks that matches and the target fit within
 * the allocated space and are properly aligned.
 *
 * Return: 0 on success, negative errno on failure.
 */
int xt_check_entry_offsets(const void *base,
			   const char *elems,
			   unsigned int target_offset,
			   unsigned int next_offset)
{
	long size_of_base_struct = elems - (const char *)base; ///< Size of the base entry struct.
	const struct xt_entry_target *t;
	const char *e = base;

	/* target start is within the ip/ip6/arpt_entry struct */
	if (target_offset < size_of_base_struct)
		return -EINVAL; ///< Target cannot start within the base structure.

	if (target_offset + sizeof(*t) > next_offset)
		return -EINVAL; ///< Target structure cannot extend beyond the rule boundary.

	t = (void *)(e + target_offset); ///< Pointer to the target structure.
	if (t->u.target_size < sizeof(*t))
		return -EINVAL; ///< Target size must be at least the base structure size.

	if (target_offset + t->u.target_size > next_offset)
		return -EINVAL; ///< Target data cannot extend beyond the rule boundary.

	// Block Logic: Check specific target types (STANDARD, ERROR) for validity.
	if (strcmp(t->u.user.name, XT_STANDARD_TARGET) == 0) {
		const struct xt_standard_target *st = (const void *)t;

		if (XT_ALIGN(target_offset + sizeof(*st)) != next_offset)
			return -EINVAL; ///< Check alignment for standard target.

		if (!verdict_ok(st->verdict))
			return -EINVAL; ///< Check if verdict is valid.
	} else if (strcmp(t->u.user.name, XT_ERROR_TARGET) == 0) {
		const struct xt_error_target *et = (const void *)t;

		if (!error_tg_ok(t->u.target_size, sizeof(*et),
				 et->errorname, sizeof(et->errorname)))
			return -EINVAL; ///< Check error target validity.
	}

	return xt_check_entry_match(elems, base + target_offset,
				    __alignof__(struct xt_entry_match)); ///< Check match entries.
}
EXPORT_SYMBOL(xt_check_entry_offsets); ///< Export this function.

/**
 * @brief Allocates an array to store rule head offsets.
 * Functional Utility: Provides memory for an array of `unsigned int` to hold
 * offsets of rule entries within a table, optimizing memory access for rule traversal.
 *
 * @param size Number of entries to allocate.
 * @return Pointer to the allocated array on success, or `NULL` on failure.
 */
unsigned int *xt_alloc_entry_offsets(unsigned int size)
{
	if (size > XT_MAX_TABLE_SIZE / sizeof(unsigned int))
		return NULL; ///< Allocation size too large.

	return kvcalloc(size, sizeof(unsigned int), GFP_KERNEL); ///< Allocate zeroed memory.

}
EXPORT_SYMBOL(xt_alloc_entry_offsets); ///< Export this function.

/**
 * @brief Checks if a target offset is a valid jump offset within an array of offsets.
 * Functional Utility: Performs a binary search on a sorted array of rule offsets
 * to quickly determine if a given target offset corresponds to a valid jump destination.
 * Algorithm: Binary search.
 * Time Complexity: O(log N) where N is the number of offsets.
 *
 * @param offsets Array containing all valid rule start offsets of a rule blob.
 * @param target The jump target to search for.
 * @param size Number of entries in `offsets`.
 * @return `true` if `target` is found in `offsets`, `false` otherwise.
 */
bool xt_find_jump_offset(const unsigned int *offsets,
			 unsigned int target, unsigned int size)
{
	int m, low = 0, hi = size; ///< Binary search indices.

	while (hi > low) { ///< Block Logic: Perform binary search.
		m = (low + hi) / 2u; // Calculate middle index.

		if (offsets[m] > target)
			hi = m; // Search in left half.
		else if (offsets[m] < target)
			low = m + 1; // Search in right half.
		else
			return true; // Found exact target.
	}

	return false; // Target not found.
}
EXPORT_SYMBOL(xt_find_jump_offset); ///< Export this function.

/**
 * @brief Checks the validity of a Netfilter target structure.
 * Functional Utility: Performs various checks on an `xt_target` structure, including
 * size validation, table applicability, hook mask compatibility, and protocol
 * matching, ensuring that the target is correctly configured.
 *
 * @param par Pointer to `xt_tgchk_param` containing target check parameters.
 * @param size Actual size of the target data.
 * @param proto Protocol.
 * @param inv_proto Inverse protocol flag.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_check_target(struct xt_tgchk_param *par,
		    unsigned int size, u16 proto, bool inv_proto)
{
	int ret;

	// Block Logic: Check target size alignment.
	if (XT_ALIGN(par->target->targetsize) != size) {
		pr_err_ratelimited("%s_tables: %s.%u target: invalid size %u (kernel) != (user) %u\n",
				   xt_prefix[par->family], par->target->name,
				   par->target->revision,
				   XT_ALIGN(par->target->targetsize), size);
		return -EINVAL;
	}
	// Block Logic: Check table applicability.
	if (par->target->table != NULL &&
	    strcmp(par->target->table, par->table) != 0) {
		pr_info_ratelimited("%s_tables: %s target: only valid in %s table, not %s\n",
				    xt_prefix[par->family], par->target->name,
				    par->target->table, par->table);
		return -EINVAL;
	}
	// Block Logic: Check hook mask compatibility.
	if (par->target->hooks && (par->hook_mask & ~par->target->hooks) != 0) {
		char used[64], allow[64];

		pr_info_ratelimited("%s_tables: %s target: used from hooks %s, but only usable from %s\n",
				    xt_prefix[par->family], par->target->name,
				    textify_hooks(used, sizeof(used),
						  par->hook_mask, par->family),
				    textify_hooks(allow, sizeof(allow),
						  par->target->hooks,
						  par->family));
		return -EINVAL;
	}
	// Block Logic: Check protocol matching.
	if (par->target->proto && (par->target->proto != proto || inv_proto)) {
		pr_info_ratelimited("%s_tables: %s target: only valid for protocol %u\n",
				    xt_prefix[par->family], par->target->name,
				    par->target->proto);
		return -EINVAL;
	}
	// Block Logic: Call target-specific checkentry function.
	if (par->target->checkentry != NULL) {
		ret = par->target->checkentry(par);
		if (ret < 0)
			return ret;
		else if (ret > 0)
			/* Flag up potential errors. */
			return -EIO;
	}
	return 0;
}
EXPORT_SYMBOL_GPL(xt_check_target); ///< Export this function (GPL-only).

/**
 * @brief Copies counters and metadata from userspace (sockptr_t).
 * Functional Utility: Transfers Netfilter counter data and associated metadata
 * from userspace memory into kernel space, handling compatibility for 32/64-bit
 * systems. It allocates memory for the counters and copies the data.
 *
 * @param arg Source `sockptr` to userspace memory.
 * @param len Alleged size of userspace memory.
 * @param info Pointer to `xt_counters_info` to store metadata.
 * @return Pointer to kernel memory containing counters (caller must `vfree`),
 *         or an `ERR_PTR` on error.
 */
void *xt_copy_counters(sockptr_t arg, unsigned int len,
		       struct xt_counters_info *info)
{
	size_t offset;
	void *mem;
	u64 size;

#ifdef CONFIG_NETFILTER_XTABLES_COMPAT ///< Block Logic: Handle compatibility for 32-bit userspace.
	if (in_compat_syscall()) {
		/* structures only differ in size due to alignment */
		struct compat_xt_counters_info compat_tmp;

		if (len <= sizeof(compat_tmp))
			return ERR_PTR(-EINVAL); ///< Invalid length.

		len -= sizeof(compat_tmp);
		if (copy_from_sockptr(&compat_tmp, arg, sizeof(compat_tmp)) != 0)
			return ERR_PTR(-EFAULT); ///< Failed to copy from userspace.

		memcpy(info->name, compat_tmp.name, sizeof(info->name) - 1); ///< Copy name.
		info->num_counters = compat_tmp.num_counters; ///< Copy number of counters.
		offset = sizeof(compat_tmp); ///< Set offset.
	} else
#endif
	{ ///< Block Logic: Handle native 64-bit userspace.
		if (len <= sizeof(*info))
			return ERR_PTR(-EINVAL); ///< Invalid length.

		len -= sizeof(*info);
		if (copy_from_sockptr(info, arg, sizeof(*info)) != 0)
			return ERR_PTR(-EFAULT); ///< Failed to copy from userspace.

		offset = sizeof(*info); ///< Set offset.
	}
	info->name[sizeof(info->name) - 1] = '\0'; ///< Ensure null-termination of name.

	size = sizeof(struct xt_counters); ///< Calculate size of a single counter.
	size *= info->num_counters; ///< Calculate total size for all counters.

	if (size != (u64)len)
		return ERR_PTR(-EINVAL); ///< Size mismatch.

	mem = vmalloc(len); ///< Allocate kernel virtual memory.
	if (!mem)
		return ERR_PTR(-ENOMEM); ///< Memory allocation failed.

	if (copy_from_sockptr_offset(mem, arg, offset, len) == 0) ///< Copy counter data.
		return mem;

	vfree(mem); ///< Free memory on copy failure.
	return ERR_PTR(-EFAULT);
}
EXPORT_SYMBOL_GPL(xt_copy_counters); ///< Export this function (GPL-only).

#ifdef CONFIG_NETFILTER_XTABLES_COMPAT
/**
 * @brief Calculates the compatibility offset for a Netfilter target.
 * Functional Utility: Determines the size difference between the kernel and
 * userspace (compat) representations of a target structure, needed for adjusting
 * offsets during compatibility translations.
 *
 * @param target Pointer to the `xt_target` structure.
 * @return The calculated offset difference.
 */
int xt_compat_target_offset(const struct xt_target *target)
{
	u_int16_t csize = target->compatsize ? : target->targetsize; ///< Use compatsize if provided, else targetsize.
	return XT_ALIGN(target->targetsize) - COMPAT_XT_ALIGN(csize); ///< Calculate difference.
}
EXPORT_SYMBOL_GPL(xt_compat_target_offset); ///< Export this function (GPL-only).

/**
 * @brief Converts an `xt_entry_target` from userspace (compat) to kernel format.
 * Functional Utility: Translates a target structure from a userspace 32-bit
 * format to the kernel's native format, handling size adjustments and
 * data copying.
 *
 * @param t Kernel `xt_entry_target` structure.
 * @param dstptr Pointer to the destination for the converted target.
 * @param size Pointer to the total size, which will be adjusted by the offset.
 */
void xt_compat_target_from_user(struct xt_entry_target *t, void **dstptr,
				unsigned int *size)
{
	const struct xt_target *target = t->u.kernel.target; ///< Get kernel target pointer.
	struct compat_xt_entry_target *ct = (struct compat_xt_entry_target *)t; ///< Cast to compat structure.
	int off = xt_compat_target_offset(target); ///< Calculate offset.
	u_int16_t tsize = ct->u.user.target_size; ///< Get userspace target size.
	char name[sizeof(t->u.user.name)]; ///< Buffer for name.

	t = *dstptr; ///< Set `t` to destination pointer.
	memcpy(t, ct, sizeof(*ct)); ///< Copy base structure.
	if (target->compat_from_user) ///< Block Logic: If custom `compat_from_user` function exists.
		target->compat_from_user(t->data, ct->data); ///< Call custom function.
	else
		unsafe_memcpy(t->data, ct->data, tsize - sizeof(*ct),
			      /* UAPI 0-sized destination */); ///< Copy data directly.

	tsize += off; ///< Adjust target size by offset.
	t->u.user.target_size = tsize; ///< Update userspace target size.
	strscpy(name, target->name, sizeof(name)); ///< Copy name.
	module_put(target->me); ///< Release module reference.
	strscpy_pad(t->u.user.name, name, sizeof(t->u.user.name)); ///< Copy name with padding.

	*size += off; ///< Adjust total size.
	*dstptr += tsize; ///< Advance destination pointer.
}
EXPORT_SYMBOL_GPL(xt_compat_target_from_user); ///< Export this function (GPL-only).

/**
 * @brief Converts an `xt_entry_target` from kernel to userspace (compat) format.
 * Functional Utility: Translates a target structure from the kernel's native format
 * to a userspace 32-bit format, handling size adjustments and data copying.
 *
 * @param t Kernel `xt_entry_target` structure.
 * @param dstptr Pointer to the destination for the converted target.
 * @param size Pointer to the total size, which will be adjusted by the offset.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_compat_target_to_user(const struct xt_entry_target *t,
			     void __user **dstptr, unsigned int *size)
{
	const struct xt_target *target = t->u.kernel.target; ///< Get kernel target pointer.
	struct compat_xt_entry_target __user *ct = *dstptr; ///< Cast to compat userspace structure.
	int off = xt_compat_target_offset(target); ///< Calculate offset.
	u_int16_t tsize = t->u.user.target_size - off; ///< Calculate userspace target size.

	if (XT_OBJ_TO_USER(ct, t, target, tsize)) ///< Copy target object metadata.
		return -EFAULT;

	if (target->compat_to_user) { ///< Block Logic: If custom `compat_to_user` function exists.
		if (target->compat_to_user((void __user *)ct->data, t->data))
			return -EFAULT;
	} else {
		if (COMPAT_XT_DATA_TO_USER(ct, t, target, tsize - sizeof(*ct))) ///< Copy data using compat macro.
			return -EFAULT;
	}

	*size -= off; ///< Adjust total size.
	*dstptr += tsize; ///< Advance destination pointer.
	return 0;
}
EXPORT_SYMBOL_GPL(xt_compat_target_to_user); ///< Export this function (GPL-only).
#endif

/**
 * @brief Allocates and initializes an `xt_table_info` structure.
 * Functional Utility: Provides memory for table information, including a flexible
 * array member for rule data, ensuring it is zeroed out.
 *
 * @param size Size of the flexible array member (rule data).
 * @return Pointer to the allocated `xt_table_info` on success, or `NULL` on failure.
 */
struct xt_table_info *xt_alloc_table_info(unsigned int size)
{
	struct xt_table_info *info = NULL;
	size_t sz = sizeof(*info) + size; ///< Calculate total size including flexible array.

	if (sz < sizeof(*info) || sz >= XT_MAX_TABLE_SIZE)
		return NULL; ///< Invalid size requested.

	info = kvmalloc(sz, GFP_KERNEL_ACCOUNT); ///< Allocate kernel virtual memory.
	if (!info)
		return NULL; ///< Memory allocation failed.

	memset(info, 0, sizeof(*info)); ///< Zero out the allocated memory.
	info->size = size; ///< Store the size.
	return info;
}
EXPORT_SYMBOL(xt_alloc_table_info); ///< Export this function.

/**
 * @brief Frees an `xt_table_info` structure and associated resources.
 * Functional Utility: Deallocates memory used by the table information, including
 * the jumpstack if it was allocated.
 *
 * @param info Pointer to the `xt_table_info` to free.
 */
void xt_free_table_info(struct xt_table_info *info)
{
	int cpu;

	if (info->jumpstack != NULL) { ///< Block Logic: If jumpstack was allocated, free it per-CPU.
		for_each_possible_cpu(cpu)
			kvfree(info->jumpstack[cpu]);
		kvfree(info->jumpstack); ///< Free the jumpstack array itself.
	}

	kvfree(info); ///< Free the `xt_table_info` structure.
}
EXPORT_SYMBOL(xt_free_table_info); ///< Export this function.

/**
 * @brief Finds an `xt_table` by name within a network namespace.
 * Functional Utility: Searches for an active `xt_table` instance given its name
 * and address family within a specific network namespace.
 *
 * @param net Pointer to the `net` structure (network namespace).
 * @param af Address family.
 * @param name Name of the table to find.
 * @return Pointer to the `xt_table` if found, or `NULL` otherwise.
 */
struct xt_table *xt_find_table(struct net *net, u8 af, const char *name)
{
	struct xt_pernet *xt_net = net_generic(net, xt_pernet_id); ///< Get per-net data.
	struct xt_table *t;

	mutex_lock(&xt[af].mutex); ///< Acquire mutex for table list.
	list_for_each_entry(t, &xt_net->tables[af], list) { ///< Block Logic: Iterate through tables in the network namespace.
		if (strcmp(t->name, name) == 0) { ///< If name matches.
			mutex_unlock(&xt[af].mutex); ///< Release mutex.
			return t; ///< Return found table.
		}
	}
	mutex_unlock(&xt[af].mutex); ///< Release mutex.
	return NULL;
}
EXPORT_SYMBOL(xt_find_table); ///< Export this function.

/**
 * @brief Finds an `xt_table` by name, acquiring a mutex and taking a module reference.
 * Functional Utility: This function searches for an `xt_table`. If not found, it attempts
 * to find and initialize the table using registered templates, potentially causing module
 * autoloading. It ensures the table's module reference count is incremented.
 *
 * @param net Pointer to the `net` structure.
 * @param af Address family.
 * @param name Name of the table.
 * @return Pointer to the `xt_table` on success (with module ref held), or an `ERR_PTR` on error.
 */
struct xt_table *xt_find_table_lock(struct net *net, u_int8_t af,
				    const char *name)
{
	struct xt_pernet *xt_net = net_generic(net, xt_pernet_id); ///< Get per-net data.
	struct module *owner = NULL;
	struct xt_template *tmpl;
	struct xt_table *t;

	mutex_lock(&xt[af].mutex); ///< Acquire mutex for table list.
	list_for_each_entry(t, &xt_net->tables[af], list)
		if (strcmp(t->name, name) == 0 && try_module_get(t->me)) ///< Block Logic: Search for existing table.
			return t; ///< Return if found and ref acquired.

	/* Table doesn't exist in this netns, check larval list */
	list_for_each_entry(tmpl, &xt_templates[af], list) { ///< Block Logic: Search in templates.
		int err;

		if (strcmp(tmpl->name, name))
			continue; ///< Skip if template name doesn't match.
		if (!try_module_get(tmpl->me)) ///< Try to get template module reference.
			goto out; ///< If cannot get ref, goto out.

		owner = tmpl->me; ///< Store owner module.

		mutex_unlock(&xt[af].mutex); ///< Release mutex before calling `table_init` (may register new items).
		err = tmpl->table_init(net); ///< Initialize table using template.
		if (err < 0) {
			module_put(owner); ///< Release module reference on error.
			return ERR_PTR(err);
		}

		mutex_lock(&xt[af].mutex); ///< Re-acquire mutex.
		break; ///< Break from template loop.
	}

	/* and once again: */
	list_for_each_entry(t, &xt_net->tables[af], list) ///< Block Logic: Search again for the newly created table.
		if (strcmp(t->name, name) == 0 && owner == t->me)
			return t; ///< Return if found and owned by the template module.

	module_put(owner); ///< Release module reference if not found after init.
 out:
	mutex_unlock(&xt[af].mutex); ///< Release mutex.
	return ERR_PTR(-ENOENT); ///< Return No Entry error.
}
EXPORT_SYMBOL_GPL(xt_find_table_lock); ///< Export this function (GPL-only).

/**
 * @brief Unlocks the mutex associated with an `xt_table`.
 * Functional Utility: Releases the mutex that protects the table's internal structures,
 * typically after a modification or lookup operation has completed.
 *
 * @param table Pointer to the `xt_table`.
 */
void xt_table_unlock(struct xt_table *table)
{
	mutex_unlock(&xt[table->af].mutex); ///< Release mutex for the table's address family.
}
EXPORT_SYMBOL_GPL(xt_table_unlock); ///< Export this function (GPL-only).

#ifdef CONFIG_NETFILTER_XTABLES_COMPAT
/**
 * @brief Locks the compatibility mutex for a given address family.
 * Functional Utility: Provides mutual exclusion for accessing compatibility
 * data structures, preventing race conditions during 32/64-bit translation.
 *
 * @param af Address family.
 */
void xt_compat_lock(u_int8_t af)
{
	mutex_lock(&xt[af].compat_mutex); ///< Acquire compatibility mutex.
}
EXPORT_SYMBOL_GPL(xt_compat_lock); ///< Export this function (GPL-only).

/**
 * @brief Unlocks the compatibility mutex for a given address family.
 * Functional Utility: Releases the mutex for compatibility data structures.
 *
 * @param af Address family.
 */
void xt_compat_unlock(u_int8_t af)
{
	mutex_unlock(&xt[af].compat_mutex); ///< Release compatibility mutex.
}
EXPORT_SYMBOL_GPL(xt_compat_unlock); ///< Export this function (GPL-only).
#endif

DEFINE_PER_CPU(seqcount_t, xt_recseq); ///< Per-CPU sequence counter for x_tables rule traversal.
EXPORT_PER_CPU_SYMBOL_GPL(xt_recseq); ///< Export this symbol (GPL-only).

struct static_key xt_tee_enabled __read_mostly; ///< Static key to optimize `TEE` target checks.
EXPORT_SYMBOL_GPL(xt_tee_enabled); ///< Export this symbol (GPL-only).

/**
 * @brief Allocates jumpstack memory for an `xt_table_info` structure.
 * Functional Utility: Provides per-CPU memory for storing jump targets during rule
 * traversal, especially for rulesets involving jumps (e.g., to user-defined chains)
 * or `TEE` targets which require preserving callchains.
 *
 * @param i Pointer to the `xt_table_info` structure.
 * @return 0 on success, or `-ENOMEM` if memory allocation fails.
 */
static int xt_jumpstack_alloc(struct xt_table_info *i)
{
	unsigned int size;
	int cpu;

	size = sizeof(void **) * nr_cpu_ids; ///< Calculate size for array of per-CPU jumpstack pointers.
	if (size > PAGE_SIZE)
		i->jumpstack = kvzalloc(size, GFP_KERNEL); ///< Allocate virtual zeroed memory if large.
	else
		i->jumpstack = kzalloc(size, GFP_KERNEL); ///< Allocate kernel zeroed memory.
	if (i->jumpstack == NULL)
		return -ENOMEM; ///< Memory allocation failed.

	/* ruleset without jumps -- no stack needed */
	if (i->stacksize == 0)
		return 0;

	/* Jumpstack needs to be able to record two full callchains, one
	 * from the first rule set traversal, plus one table reentrancy
	 * via -j TEE without clobbering the callchain that brought us to
	 * TEE target.
	 *
	 * This is done by allocating two jumpstacks per cpu, on reentry
	 * the upper half of the stack is used.
	 *
	 * see the jumpstack setup in ipt_do_table() for more details.
	 */
	size = sizeof(void *) * i->stacksize * 2u; ///< Calculate size for two callchains.
	for_each_possible_cpu(cpu) { ///< Block Logic: Allocate jumpstack for each possible CPU.
		i->jumpstack[cpu] = kvmalloc_node(size, GFP_KERNEL,
			cpu_to_node(cpu)); ///< Allocate node-local memory.
		if (i->jumpstack[cpu] == NULL)
			/*
			 * Freeing will be done later on by the callers. The
			 * chain is: xt_replace_table -> __do_replace ->
			 * do_replace -> xt_free_table_info.
			 */
			return -ENOMEM; ///< Memory allocation failed for a CPU.
	}

	return 0;
}

/**
 * @brief Allocates `xt_counters` memory.
 * Functional Utility: Provides zeroed memory for an array of `xt_counters`
 * structures, used for packet and byte counting per rule.
 *
 * @param counters Number of counters to allocate.
 * @return Pointer to the allocated `xt_counters` array on success, or `NULL` on failure.
 */
struct xt_counters *xt_counters_alloc(unsigned int counters)
{
	struct xt_counters *mem;

	if (counters == 0 || counters > INT_MAX / sizeof(*mem))
		return NULL; ///< Invalid number of counters.

	counters *= sizeof(*mem); ///< Calculate total memory size.
	if (counters > XT_MAX_TABLE_SIZE)
		return NULL; ///< Memory request too large.

	return vzalloc(counters); ///< Allocate zeroed virtual memory.
}
EXPORT_SYMBOL(xt_counters_alloc); ///< Export this function.

/**
 * @brief Replaces the info structure of an existing `xt_table`.
 * Functional Utility: Atomically replaces the `xt_table_info` associated with an
 * `xt_table` with a new one. This is a critical operation for updating Netfilter
 * rulesets dynamically. It handles jumpstack allocation and ensures proper
 * synchronization across CPUs.
 *
 * @param table Pointer to the `xt_table` whose info is to be replaced.
 * @param num_counters Number of counters in the old table info.
 * @param newinfo Pointer to the new `xt_table_info` structure.
 * @param error Pointer to an integer to store error code.
 * @return Pointer to the old `xt_table_info` on success, or `NULL` on failure.
 */
struct xt_table_info *
xt_replace_table(struct xt_table *table,
	      unsigned int num_counters,
	      struct xt_table_info *newinfo,
	      int *error)
{
	struct xt_table_info *private;
	unsigned int cpu;
	int ret;

	ret = xt_jumpstack_alloc(newinfo); ///< Allocate jumpstack for new info.
	if (ret < 0) {
		*error = ret;
		return NULL;
	}

	/* Do the substitution. */
	local_bh_disable(); ///< Disable bottom halves to prevent preemption.
	private = table->private; ///< Get current (old) table info.

	/* Check inside lock: is the old number correct? */
	if (num_counters != private->number) { ///< Block Logic: Verify consistency of counter count.
		pr_debug("num_counters != table->private->number (%u/%u)\n",
			 num_counters, private->number);
		local_bh_enable(); ///< Re-enable bottom halves.
		*error = -EAGAIN; ///< Return error for mismatch.
		return NULL;
	}

	newinfo->initial_entries = private->initial_entries; ///< Copy initial entries count.
	/*
	 * Ensure contents of newinfo are visible before assigning to
	 * private.
	 */
	smp_wmb(); ///< Write memory barrier to ensure `newinfo` is fully written before `table->private` is updated.
	table->private = newinfo; ///< Atomically replace table info.

	/* make sure all cpus see new ->private value */
	smp_mb(); ///< Full memory barrier to ensure all CPUs see the new value.

	/*
	 * Even though table entries have now been swapped, other CPU's
	 * may still be using the old entries...
	 */
	local_bh_enable(); ///< Re-enable bottom halves.

	/* ... so wait for even xt_recseq on all cpus */
	for_each_possible_cpu(cpu) { ///< Block Logic: Wait for all CPUs to finish current rule traversal.
		seqcount_t *s = &per_cpu(xt_recseq, cpu);
		u32 seq = raw_read_seqcount(s);

		if (seq & 1) { ///< If sequence count is odd (meaning traversal is in progress).
			do {
				cond_resched(); ///< Conditionally reschedule.
				cpu_relax(); ///< Hint for CPU to relax.
			} while (seq == raw_read_seqcount(s)); ///< Wait until sequence count changes (traversal completes).
		}
	}

	audit_log_nfcfg(table->name, table->af, private->number,
			!private->number ? AUDIT_XT_OP_REGISTER :
					   AUDIT_XT_OP_REPLACE,
			GFP_KERNEL); ///< Log audit event.
	return private; ///< Return old table info for later freeing.
}
EXPORT_SYMBOL_GPL(xt_replace_table); ///< Export this function (GPL-only).

/**
 * @brief Registers a new `xt_table` instance.
 * Functional Utility: Adds a new Netfilter table to the system. This involves
 * duplicating the table structure, initializing it, and atomically swapping
 * in the provided table information. It checks for name conflicts and ensures
 * proper locking.
 *
 * @param net Pointer to the `net` structure.
 * @param input_table Pointer to the template `xt_table` structure.
 * @param bootstrap Pointer to a bootstrap `xt_table_info`.
 * @param newinfo Pointer to the new `xt_table_info` structure to install.
 * @return Pointer to the newly registered `xt_table` on success, or an `ERR_PTR` on error.
 */
struct xt_table *xt_register_table(struct net *net,
				   const struct xt_table *input_table,
				   struct xt_table_info *bootstrap,
				   struct xt_table_info *newinfo)
{
	struct xt_pernet *xt_net = net_generic(net, xt_pernet_id); ///< Get per-net data.
	struct xt_table *t, *table;
	int ret;

	/* Don't add one object to multiple lists. */
	table = kmemdup(input_table, sizeof(struct xt_table), GFP_KERNEL); ///< Duplicate input table.
	if (!table) {
		ret = -ENOMEM;
		goto out;
	}

	mutex_lock(&xt[table->af].mutex); ///< Acquire mutex.
	/* Don't autoload: we'd eat our tail... */
	list_for_each_entry(t, &xt_net->tables[table->af], list) { ///< Block Logic: Check for name conflicts.
		if (strcmp(t->name, table->name) == 0) {
			ret = -EEXIST;
			goto unlock;
		}
	}

	/* Simplifies replace_table code. */
	table->private = bootstrap; ///< Set bootstrap table info.

	if (!xt_replace_table(table, 0, newinfo, &ret)) ///< Block Logic: Replace table with new info.
		goto unlock;

	private = table->private;
	pr_debug("table->private->number = %u\n", private->number);

	/* save number of initial entries */
	private->initial_entries = private->number; ///< Save initial entries count.

	list_add(&table->list, &xt_net->tables[table->af]); ///< Add table to list.
	mutex_unlock(&xt[table->af].mutex); ///< Release mutex.
	return table;

unlock: ///< Error handling: unlock and free table.
	mutex_unlock(&xt[table->af].mutex);
	kfree(table);
out: ///< Error handling: return error pointer.
	return ERR_PTR(ret);
}
EXPORT_SYMBOL_GPL(xt_register_table); ///< Export this function (GPL-only).

/**
 * @brief Unregisters an existing `xt_table` instance.
 * Functional Utility: Removes a Netfilter table from the system, freeing its
 * associated resources and logging an audit event.
 *
 * @param table Pointer to the `xt_table` to unregister.
 * @return Pointer to the old `xt_table_info` (which caller must `kfree`).
 */
void *xt_unregister_table(struct xt_table *table)
{
	struct xt_table_info *private;

	mutex_lock(&xt[table->af].mutex); ///< Acquire mutex.
	private = table->private; ///< Get private info.
	list_del(&table->list); ///< Remove table from list.
	mutex_unlock(&xt[table->af].mutex); ///< Release mutex.
	audit_log_nfcfg(table->name, table->af, private->number,
			AUDIT_XT_OP_UNREGISTER, GFP_KERNEL); ///< Log audit event.
	kfree(table->ops); ///< Free table operations.
	kfree(table); ///< Free table structure.

	return private; ///< Return old private info.
}
EXPORT_SYMBOL_GPL(xt_unregister_table); ///< Export this function (GPL-only).

#ifdef CONFIG_PROC_FS
/**
 * @brief Starts sequence file traversal for Netfilter tables.
 * Functional Utility: Initializes the iteration over Netfilter tables for a
 * given address family, used by `/proc` entries.
 *
 * @param seq Pointer to the `seq_file` structure.
 * @param pos Pointer to the file position.
 * @return Pointer to the first element in the sequence, or `NULL`.
 */
static void *xt_table_seq_start(struct seq_file *seq, loff_t *pos)
{
	u8 af = (unsigned long)pde_data(file_inode(seq->file)); ///< Get address family from inode data.
	struct net *net = seq_file_net(seq); ///< Get network namespace.
	struct xt_pernet *xt_net;

	xt_net = net_generic(net, xt_pernet_id); ///< Get per-net data.

	mutex_lock(&xt[af].mutex); ///< Acquire mutex.
	return seq_list_start(&xt_net->tables[af], *pos); ///< Start list traversal.
}

/**
 * @brief Gets the next element in sequence file traversal for Netfilter tables.
 * Functional Utility: Advances the iteration over Netfilter tables for `/proc` entries.
 *
 * @param seq Pointer to the `seq_file` structure.
 * @param v Current element pointer.
 * @param pos Pointer to the file position.
 * @return Pointer to the next element in the sequence, or `NULL`.
 */
static void *xt_table_seq_next(struct seq_file *seq, void *v, loff_t *pos)
{
	u8 af = (unsigned long)pde_data(file_inode(seq->file)); ///< Get address family.
	struct net *net = seq_file_net(seq); ///< Get network namespace.
	struct xt_pernet *xt_net;

	xt_net = net_generic(net, xt_pernet_id); ///< Get per-net data.

	return seq_list_next(v, &xt_net->tables[af], pos); ///< Get next element.
}

/**
 * @brief Stops sequence file traversal for Netfilter tables.
 * Functional Utility: Releases resources (mutex) held during iteration.
 *
 * @param seq Pointer to the `seq_file` structure.
 * @param v Last element pointer.
 */
static void xt_table_seq_stop(struct seq_file *seq, void *v)
{
	u_int8_t af = (unsigned long)pde_data(file_inode(seq->file)); ///< Get address family.

	mutex_unlock(&xt[af].mutex); ///< Release mutex.
}

/**
 * @brief Shows the name of an `xt_table` in a sequence file.
 * Functional Utility: Formats and prints the name of a Netfilter table to the
 * `/proc` sequence file.
 *
 * @param seq Pointer to the `seq_file` structure.
 * @param v Pointer to the `xt_table`.
 * @return 0 on success.
 */
static int xt_table_seq_show(struct seq_file *seq, void *v)
{
	struct xt_table *table = list_entry(v, struct xt_table, list); ///< Get table from list entry.

	if (*table->name)
		seq_printf(seq, "%s\n", table->name); ///< Print table name.
	return 0;
}

/**
 * @brief Sequence operations for `/proc/net/ip_tables_names` (and similar).
 * Functional Utility: Defines the `seq_operations` structure for iterating and
 * displaying Netfilter table names in the `/proc` filesystem.
 */
static const struct seq_operations xt_table_seq_ops = {
	.start	= xt_table_seq_start, ///< Start function.
	.next	= xt_table_seq_next, ///< Next function.
	.stop	= xt_table_seq_stop, ///< Stop function.
	.show	= xt_table_seq_show, ///< Show function.
};

/*
 * Traverse state for ip{,6}_{tables,matches} for helping crossing
 * the multi-AF mutexes.
 */
/**
 * @struct nf_mttg_trav
 * @brief Traversal state for Netfilter match/target sequence files.
 * Functional Utility: Manages the state required to iterate through lists of
 * Netfilter matches or targets across different address families, correctly
 * handling mutex locking.
 */
struct nf_mttg_trav {
	struct list_head *head, *curr; ///< Head and current pointer for list traversal.
	uint8_t class; ///< Traversal class/state.
};

enum {
	MTTG_TRAV_INIT, ///< Initial state.
	MTTG_TRAV_NFP_UNSPEC, ///< Unspecified protocol family traversal.
	MTTG_TRAV_NFP_SPEC, ///< Specific protocol family traversal.
	MTTG_TRAV_DONE, ///< Traversal complete.
};

/**
 * @brief Gets the next element in sequence file traversal for Netfilter matches/targets.
 * Functional Utility: Advances the iteration through match/target lists, handling
 * mutex locking and transitions between unspecified and specific protocol families.
 *
 * @param seq Pointer to the `seq_file` structure.
 * @param v Current element pointer.
 * @param ppos Pointer to the file position.
 * @param is_target `true` if traversing targets, `false` for matches.
 * @return Pointer to the next element in the sequence, or `NULL`.
 */
static void *xt_mttg_seq_next(struct seq_file *seq, void *v, loff_t *ppos,
    bool is_target)
{
	static const uint8_t next_class[] = { ///< State transition table.
		[MTTG_TRAV_NFP_UNSPEC] = MTTG_TRAV_NFP_SPEC,
		[MTTG_TRAV_NFP_SPEC]   = MTTG_TRAV_DONE,
	};
	uint8_t nfproto = (unsigned long)pde_data(file_inode(seq->file)); ///< Get protocol family.
	struct nf_mttg_trav *trav = seq->private; ///< Get traversal state.

	if (ppos != NULL)
		++(*ppos); ///< Increment file position.

	switch (trav->class) { ///< Block Logic: State machine for traversal.
	case MTTG_TRAV_INIT:
		trav->class = MTTG_TRAV_NFP_UNSPEC; ///< Transition to UNSPEC state.
		mutex_lock(&xt[NFPROTO_UNSPEC].mutex); ///< Acquire mutex for UNSPEC family.
		trav->head = trav->curr = is_target ?
			&xt[NFPROTO_UNSPEC].target : &xt[NFPROTO_UNSPEC].match; ///< Set head and current pointers.
 		break;
	case MTTG_TRAV_NFP_UNSPEC:
		trav->curr = trav->curr->next; ///< Move to next element.
		if (trav->curr != trav->head)
			break; ///< If not end of list, break.
		mutex_unlock(&xt[NFPROTO_UNSPEC].mutex); ///< Release UNSPEC mutex.
		mutex_lock(&xt[nfproto].mutex); ///< Acquire mutex for specific family.
		trav->head = trav->curr = is_target ?
			&xt[nfproto].target : &xt[nfproto].match; ///< Set head and current pointers.
		trav->class = next_class[trav->class]; ///< Transition to next state.
		break;
	case MTTG_TRAV_NFP_SPEC:
		trav->curr = trav->curr->next; ///< Move to next element.
		if (trav->curr != trav->head)
			break; ///< If not end of list, break.
		fallthrough;
	default:
		return NULL; ///< End of traversal.
	}
	return trav;
}

/**
 * @brief Starts sequence file traversal for Netfilter matches/targets.
 * Functional Utility: Initializes the iteration over Netfilter matches or targets,
 * skipping a specified number of initial entries.
 *
 * @param seq Pointer to the `seq_file` structure.
 * @param pos Pointer to the file position (number of entries to skip).
 * @param is_target `true` if traversing targets, `false` for matches.
 * @return Pointer to the traversal state.
 */
static void *xt_mttg_seq_start(struct seq_file *seq, loff_t *pos,
    bool is_target)
{
	struct nf_mttg_trav *trav = seq->private; ///< Get traversal state.
	unsigned int j;

	trav->class = MTTG_TRAV_INIT; ///< Initialize class.
	for (j = 0; j < *pos; ++j) ///< Block Logic: Skip `pos` initial entries.
		if (xt_mttg_seq_next(seq, NULL, NULL, is_target) == NULL)
			return NULL; ///< If end reached while skipping, return NULL.
	return trav;
}

/**
 * @brief Stops sequence file traversal for Netfilter matches/targets.
 * Functional Utility: Releases mutexes held during iteration for matches/targets.
 *
 * @param seq Pointer to the `seq_file` structure.
 * @param v Last element pointer.
 */
static void xt_mttg_seq_stop(struct seq_file *seq, void *v)
{
	uint8_t nfproto = (unsigned long)pde_data(file_inode(seq->file)); ///< Get protocol family.
	struct nf_mttg_trav *trav = seq->private; ///< Get traversal state.

	switch (trav->class) { ///< Block Logic: Release mutex based on current traversal phase.
	case MTTG_TRAV_NFP_UNSPEC:
		mutex_unlock(&xt[NFPROTO_UNSPEC].mutex); ///< Release UNSPEC mutex.
		break;
	case MTTG_TRAV_NFP_SPEC:
		mutex_unlock(&xt[nfproto].mutex); ///< Release specific AF mutex.
		break;
	}
}

/**
 * @brief Starts sequence file traversal for Netfilter matches.
 * Functional Utility: Entry point for `seq_operations` to begin iterating over matches.
 */
static void *xt_match_seq_start(struct seq_file *seq, loff_t *pos)
{
	return xt_mttg_seq_start(seq, pos, false); ///< Delegate to general start function for matches.
}

/**
 * @brief Gets the next element in sequence file traversal for Netfilter matches.
 * Functional Utility: Entry point for `seq_operations` to get the next match.
 */
static void *xt_match_seq_next(struct seq_file *seq, void *v, loff_t *ppos)
{
	return xt_mttg_seq_next(seq, v, ppos, false); ///< Delegate to general next function for matches.
}

/**
 * @brief Shows the name of an `xt_match` in a sequence file.
 * Functional Utility: Prints the name of a Netfilter match to the `/proc` sequence file.
 */
static int xt_match_seq_show(struct seq_file *seq, void *v)
{
	const struct nf_mttg_trav *trav = seq->private; ///< Get traversal state.
	const struct xt_match *match;

	switch (trav->class) { ///< Block Logic: Show match name based on traversal phase.
	case MTTG_TRAV_NFP_UNSPEC:
	case MTTG_TRAV_NFP_SPEC:
		if (trav->curr == trav->head)
			return 0; ///< If end of list, return.
		match = list_entry(trav->curr, struct xt_match, list); ///< Get match from list entry.
		if (*match->name)
			seq_printf(seq, "%s\n", match->name); ///< Print match name.
	}
	return 0;
}

/**
 * @brief Sequence operations for `/proc/net/ip_tables_matches` (and similar).
 * Functional Utility: Defines the `seq_operations` structure for iterating and
 * displaying Netfilter match names in the `/proc` filesystem.
 */
static const struct seq_operations xt_match_seq_ops = {
	.start	= xt_match_seq_start, ///< Start function.
	.next	= xt_match_seq_next, ///< Next function.
	.stop	= xt_mttg_seq_stop, ///< Stop function.
	.show	= xt_match_seq_show, ///< Show function.
};

/**
 * @brief Starts sequence file traversal for Netfilter targets.
 * Functional Utility: Entry point for `seq_operations` to begin iterating over targets.
 */
static void *xt_target_seq_start(struct seq_file *seq, loff_t *pos)
{
	return xt_mttg_seq_start(seq, pos, true); ///< Delegate to general start function for targets.
}

/**
 * @brief Gets the next element in sequence file traversal for Netfilter targets.
 * Functional Utility: Entry point for `seq_operations` to get the next target.
 */
static void *xt_target_seq_next(struct seq_file *seq, void *v, loff_t *ppos)
{
	return xt_mttg_seq_next(seq, v, ppos, true); ///< Delegate to general next function for targets.
}

/**
 * @brief Shows the name of an `xt_target` in a sequence file.
 * Functional Utility: Prints the name of a Netfilter target to the `/proc` sequence file.
 */
static int xt_target_seq_show(struct seq_file *seq, void *v)
{
	const struct nf_mttg_trav *trav = seq->private; ///< Get traversal state.
	const struct xt_target *target;

	switch (trav->class) { ///< Block Logic: Show target name based on traversal phase.
	case MTTG_TRAV_NFP_UNSPEC:
	case MTTG_TRAV_NFP_SPEC:
		if (trav->curr == trav->head)
			return 0; ///< If end of list, return.
		target = list_entry(trav->curr, struct xt_target, list); ///< Get target from list entry.
		if (*target->name)
			seq_printf(seq, "%s\n", target->name); ///< Print target name.
	}
	return 0;
}

/**
 * @brief Sequence operations for `/proc/net/ip_tables_targets` (and similar).
 * Functional Utility: Defines the `seq_operations` structure for iterating and
 * displaying Netfilter target names in the `/proc` filesystem.
 */
static const struct seq_operations xt_target_seq_ops = {
	.start	= xt_target_seq_start, ///< Start function.
	.next	= xt_target_seq_next, ///< Next function.
	.stop	= xt_mttg_seq_stop, ///< Stop function.
	.show	= xt_target_seq_show, ///< Show function.
};

#define FORMAT_TABLES	"_tables_names" ///< Suffix for table names procfs entry.
#define	FORMAT_MATCHES	"_tables_matches" ///< Suffix for match names procfs entry.
#define FORMAT_TARGETS 	"_tables_targets" ///< Suffix for target names procfs entry.

#endif /* CONFIG_PROC_FS */

/**
 * @brief Allocates and initializes `nf_hook_ops` structures for a Netfilter table.
 * Functional Utility: Creates an array of `nf_hook_ops` entries based on the
 * `valid_hooks` mask of an `xt_table`. These `nf_hook_ops` are then registered
 * with the Netfilter core to attach the table's hook function to various Netfilter
 * hook points.
 *
 * @param table Pointer to the `xt_table` with metadata for hook setup.
 * @param fn Hook function pointer (`nf_hookfn`) to be used for all hooks.
 * @return Pointer to the allocated `nf_hook_ops` array on success, or `ERR_PTR` on failure.
 */
struct nf_hook_ops *
xt_hook_ops_alloc(const struct xt_table *table, nf_hookfn *fn)
{
	unsigned int hook_mask = table->valid_hooks; ///< Get valid hooks bitmask.
	uint8_t i, num_hooks = hweight32(hook_mask); ///< Count number of set bits in mask.
	uint8_t hooknum;
	struct nf_hook_ops *ops;

	if (!num_hooks)
		return ERR_PTR(-EINVAL); ///< No hooks specified.

	ops = kcalloc(num_hooks, sizeof(*ops), GFP_KERNEL); ///< Allocate zeroed memory for hook ops array.
	if (ops == NULL)
		return ERR_PTR(-ENOMEM); ///< Memory allocation failed.

	for (i = 0, hooknum = 0; i < num_hooks && hook_mask != 0;
	     hook_mask >>= 1, ++hooknum) { ///< Block Logic: Iterate through hook mask to find set bits.
		if (!(hook_mask & 1))
			continue; ///< Skip if hook bit is not set.
		ops[i].hook     = fn; ///< Assign hook function.
		ops[i].pf       = table->af; ///< Assign protocol family.
		ops[i].hooknum  = hooknum; ///< Assign hook number.
		ops[i].priority = table->priority; ///< Assign priority.
		++i; ///< Increment index for ops array.
	}

	return ops;
}
EXPORT_SYMBOL_GPL(xt_hook_ops_alloc); ///< Export this function (GPL-only).

/**
 * @brief Registers an `xt_template` for a new table.
 * Functional Utility: Adds a new table initialization template to the global list.
 * This template allows for lazy creation of tables in network namespaces as they are needed.
 *
 * @param table Pointer to the template `xt_table` structure.
 * @param table_init Function pointer to initialize the table.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_register_template(const struct xt_table *table,
			 int (*table_init)(struct net *net))
{
	int ret = -EEXIST, af = table->af; ///< Default return error, get address family.
	struct xt_template *t;

	mutex_lock(&xt[af].mutex); ///< Acquire mutex for template list.

	list_for_each_entry(t, &xt_templates[af], list) { ///< Block Logic: Check for duplicate template names.
		if (WARN_ON_ONCE(strcmp(table->name, t->name) == 0))
			goto out_unlock; ///< Duplicate name found, warn and exit.
	}

	ret = -ENOMEM;
	t = kzalloc(sizeof(*t), GFP_KERNEL); ///< Allocate memory for the template.
	if (!t)
		goto out_unlock; ///< Memory allocation failed.

	BUILD_BUG_ON(sizeof(t->name) != sizeof(table->name)); ///< Compile-time check for name buffer size.

	strscpy(t->name, table->name, sizeof(t->name)); ///< Copy template name.
	t->table_init = table_init; ///< Assign initialization function.
	t->me = table->me; ///< Assign owning module.
	list_add(&t->list, &xt_templates[af]); ///< Add template to list.
	ret = 0; ///< Success.
out_unlock: ///< Error handling: release mutex.
	mutex_unlock(&xt[af].mutex);
	return ret;
}
EXPORT_SYMBOL_GPL(xt_register_template); ///< Export this function (GPL-only).

/**
 * @brief Unregisters an `xt_template`.
 * Functional Utility: Removes a previously registered table initialization template
 * from the global list, typically during module unload.
 *
 * @param table Pointer to the template `xt_table` structure to unregister.
 */
void xt_unregister_template(const struct xt_table *table)
{
	struct xt_template *t;
	int af = table->af;

	mutex_lock(&xt[af].mutex); ///< Acquire mutex.
	list_for_each_entry(t, &xt_templates[af], list) { ///< Block Logic: Iterate through templates to find a match.
		if (strcmp(table->name, t->name))
			continue; ///< Skip if name doesn't match.

		list_del(&t->list); ///< Remove template from list.
		mutex_unlock(&xt[af].mutex); ///< Release mutex.
		kfree(t); ///< Free template memory.
		return;
	}

	mutex_unlock(&xt[af].mutex); ///< Release mutex.
	WARN_ON_ONCE(1); ///< Warn if template was not found.
}
EXPORT_SYMBOL_GPL(xt_unregister_template); ///< Export this function (GPL-only).

/**
 * @brief Initializes Netfilter protocol-specific entries in `/proc`.
 * Functional Utility: Creates `/proc` filesystem entries for Netfilter tables,
 * matches, and targets for a given address family, allowing userspace to inspect
 * their names and status.
 *
 * @param net Pointer to the `net` structure.
 * @param af Address family.
 * @return 0 on success, or a negative errno on failure.
 */
int xt_proto_init(struct net *net, u_int8_t af)
{
#ifdef CONFIG_PROC_FS ///< Block Logic: Only compile if /proc filesystem is enabled.
	char buf[XT_FUNCTION_MAXNAMELEN];
	struct proc_dir_entry *proc;
	kuid_t root_uid;
	kgid_t root_gid;
#endif

	if (af >= ARRAY_SIZE(xt_prefix))
		return -EINVAL; ///< Invalid address family.


#ifdef CONFIG_PROC_FS
	root_uid = make_kuid(net->user_ns, 0); ///< Get root UID for current user namespace.
	root_gid = make_kgid(net->user_ns, 0); ///< Get root GID for current user namespace.

	strscpy(buf, xt_prefix[af], sizeof(buf)); ///< Copy protocol prefix.
	strlcat(buf, FORMAT_TABLES, sizeof(buf)); ///< Concatenate with table format string.
	proc = proc_create_net_data(buf, 0440, net->proc_net, &xt_table_seq_ops, ///< Create /proc entry for tables.
			sizeof(struct seq_net_private),
			(void *)(unsigned long)af);
	if (!proc)
		goto out; ///< If creation fails, goto error cleanup.
	if (uid_valid(root_uid) && gid_valid(root_gid))
		proc_set_user(proc, root_uid, root_gid); ///< Set ownership.

	strscpy(buf, xt_prefix[af], sizeof(buf));
	strlcat(buf, FORMAT_MATCHES, sizeof(buf));
	proc = proc_create_seq_private(buf, 0440, net->proc_net, ///< Create /proc entry for matches.
			&xt_match_seq_ops, sizeof(struct nf_mttg_trav),
			(void *)(unsigned long)af);
	if (!proc)
		goto out_remove_tables; ///< If creation fails, goto cleanup.
	if (uid_valid(root_uid) && gid_valid(root_gid))
		proc_set_user(proc, root_uid, root_gid); ///< Set ownership.

	strscpy(buf, xt_prefix[af], sizeof(buf));
	strlcat(buf, FORMAT_TARGETS, sizeof(buf));
	proc = proc_create_seq_private(buf, 0440, net->proc_net, ///< Create /proc entry for targets.
			 &xt_target_seq_ops, sizeof(struct nf_mttg_trav),
			 (void *)(unsigned long)af);
	if (!proc)
		goto out_remove_matches; ///< If creation fails, goto cleanup.
	if (uid_valid(root_uid) && gid_valid(root_gid))
		proc_set_user(proc, root_uid, root_gid); ///< Set ownership.
#endif

	return 0;

#ifdef CONFIG_PROC_FS
out_remove_matches: ///< Error cleanup: remove matches proc entry.
	strscpy(buf, xt_prefix[af], sizeof(buf));
	strlcat(buf, FORMAT_MATCHES, sizeof(buf));
	remove_proc_entry(buf, net->proc_net);

out_remove_tables: ///< Error cleanup: remove tables proc entry.
	strscpy(buf, xt_prefix[af], sizeof(buf));
	strlcat(buf, FORMAT_TABLES, sizeof(buf));
	remove_proc_entry(buf, net->proc_net);
out: ///< Generic error cleanup.
	return -1;
#endif
}
EXPORT_SYMBOL_GPL(xt_proto_init); ///< Export this function (GPL-only).

/**
 * @brief Cleans up Netfilter protocol-specific entries in `/proc`.
 * Functional Utility: Removes `/proc` filesystem entries created by `xt_proto_init`
 * during module unload or network namespace destruction.
 *
 * @param net Pointer to the `net` structure.
 * @param af Address family.
 */
void xt_proto_fini(struct net *net, u_int8_t af)
{
#ifdef CONFIG_PROC_FS ///< Block Logic: Only compile if /proc filesystem is enabled.
	char buf[XT_FUNCTION_MAXNAMELEN];

	strscpy(buf, xt_prefix[af], sizeof(buf));
	strlcat(buf, FORMAT_TABLES, sizeof(buf));
	remove_proc_entry(buf, net->proc_net); ///< Remove tables proc entry.

	strscpy(buf, xt_prefix[af], sizeof(buf));
	strlcat(buf, FORMAT_TARGETS, sizeof(buf));
	remove_proc_entry(buf, net->proc_net); ///< Remove targets proc entry.

	strscpy(buf, xt_prefix[af], sizeof(buf));
	strlcat(buf, FORMAT_MATCHES, sizeof(buf));
	remove_proc_entry(buf, net->proc_net); ///< Remove matches proc entry.
#endif /*CONFIG_PROC_FS*/
}
EXPORT_SYMBOL_GPL(xt_proto_fini); ///< Export this function (GPL-only).

/**
 * @brief Allocates per-CPU counters for x_tables rules.
 * Functional Utility: Provides optimized per-CPU memory allocation for packet
 * and byte counters, enhancing performance by reducing cache line contention.
 * Counters are allocated in blocks (e.g., 4KB) for efficiency.
 *
 * @param state Pointer to `xt_percpu_counter_alloc_state` for managing allocation state.
 * @param counter Pointer to `xt_counters` struct to store the per-CPU counter address.
 * @return `true` on success, `false` on error (e.g., memory allocation failure).
 */
bool xt_percpu_counter_alloc(struct xt_percpu_counter_alloc_state *state,
			     struct xt_counters *counter)
{
	BUILD_BUG_ON(XT_PCPU_BLOCK_SIZE < (sizeof(*counter) * 2)); ///< Compile-time check for block size.

	if (nr_cpu_ids <= 1)
		return true; ///< No need for per-CPU allocation on single-CPU systems.

	if (!state->mem) { ///< Block Logic: If no memory block allocated yet.
		state->mem = __alloc_percpu(XT_PCPU_BLOCK_SIZE,
					    XT_PCPU_BLOCK_SIZE); ///< Allocate new per-CPU block.
		if (!state->mem)
			return false; ///< Memory allocation failed.
	}
	counter->pcnt = (__force unsigned long)(state->mem + state->off); ///< Assign per-CPU counter address.
	state->off += sizeof(*counter); ///< Advance offset in the block.
	if (state->off > (XT_PCPU_BLOCK_SIZE - sizeof(*counter))) { ///< Block Logic: If block is nearly full.
		state->mem = NULL; ///< Reset block pointer.
		state->off = 0; ///< Reset offset.
	}
	return true;
}
EXPORT_SYMBOL_GPL(xt_percpu_counter_alloc); ///< Export this function (GPL-only).

/**
 * @brief Frees per-CPU counters allocated by `xt_percpu_counter_alloc`.
 * Functional Utility: Deallocates the per-CPU memory blocks used for counters.
 * This function handles freeing entire blocks when the first counter within a
 * block is passed for freeing, assuming all counters allocated within the same
 * block are freed together.
 *
 * @param counters Pointer to the `xt_counters` struct whose per-CPU counter is to be freed.
 */
void xt_percpu_counter_free(struct xt_counters *counters)
{
	unsigned long pcnt = counters->pcnt; ///< Get per-CPU counter address.

	if (nr_cpu_ids > 1 && (pcnt & (XT_PCPU_BLOCK_SIZE - 1)) == 0) ///< Block Logic: If per-CPU block start address.
		free_percpu((void __percpu *)pcnt); ///< Free the entire per-CPU block.
}
EXPORT_SYMBOL_GPL(xt_percpu_counter_free); ///< Export this function (GPL-only).

/**
 * @brief Per-network namespace initialization for x_tables.
 * Functional Utility: Initializes the list heads for Netfilter tables within a new
 * network namespace. This is called when a new network namespace is created.
 *
 * @param net Pointer to the new `net` structure (network namespace).
 * @return 0 on success.
 */
static int __net_init xt_net_init(struct net *net)
{
	struct xt_pernet *xt_net = net_generic(net, xt_pernet_id); ///< Get per-net data for the new namespace.
	int i;

	for (i = 0; i < NFPROTO_NUMPROTO; i++) ///< Block Logic: Initialize list heads for all protocol families.
		INIT_LIST_HEAD(&xt_net->tables[i]);
	return 0;
}

/**
 * @brief Per-network namespace exit for x_tables.
 * Functional Utility: Cleans up when a network namespace is destroyed, verifying that
 * all Netfilter tables have been unregistered.
 *
 * @param net Pointer to the `net` structure being destroyed.
 */
static void __net_exit xt_net_exit(struct net *net)
{
	struct xt_pernet *xt_net = net_generic(net, xt_pernet_id); ///< Get per-net data.
	int i;

	for (i = 0; i < NFPROTO_NUMPROTO; i++) ///< Block Logic: Check if all table lists are empty.
		WARN_ON_ONCE(!list_empty(&xt_net->tables[i])); ///< Warn if any table list is not empty (resource leak).
}

/**
 * @brief `pernet_operations` structure for x_tables.
 * Functional Utility: Defines the callback functions and metadata for integrating
 * x_tables with the Linux kernel's network namespace management.
 */
static struct pernet_operations xt_net_ops = {
	.init = xt_net_init, ///< Initialization callback for new network namespaces.
	.exit = xt_net_exit, ///< Exit callback for destroying network namespaces.
	.id   = &xt_pernet_id, ///< Pointer to the generic network namespace ID.
	.size = sizeof(struct xt_pernet), ///< Size of the per-net data structure.
};

/**
 * @brief Module initialization function for x_tables.
 * Functional Utility: Sets up the global `xt_af` structures, initializes mutexes,
 * list heads, and registers with the per-network namespace subsystem.
 * This is the entry point for the module when it is loaded into the kernel.
 *
 * @return 0 on success, or a negative errno on failure.
 */
static int __init xt_init(void)
{
	unsigned int i;
	int rv;

	for_each_possible_cpu(i) { ///< Block Logic: Initialize sequence counters for all possible CPUs.
		seqcount_init(&per_cpu(xt_recseq, i));
	}

	xt = kcalloc(NFPROTO_NUMPROTO, sizeof(struct xt_af), GFP_KERNEL); ///< Allocate memory for `xt_af` array.
	if (!xt)
		return -ENOMEM; ///< Memory allocation failed.

	for (i = 0; i < NFPROTO_NUMPROTO; i++) { ///< Block Logic: Initialize mutexes and list heads for each protocol family.
		mutex_init(&xt[i].mutex); ///< Initialize main mutex.
#ifdef CONFIG_NETFILTER_XTABLES_COMPAT ///< Block Logic: Initialize compatibility mutex if enabled.
		mutex_init(&xt[i].compat_mutex);
		xt[i].compat_tab = NULL;
#endif
		INIT_LIST_HEAD(&xt[i].target); ///< Initialize target list head.
		INIT_LIST_HEAD(&xt[i].match); ///< Initialize match list head.
		INIT_LIST_HEAD(&xt_templates[i]); ///< Initialize template list head.
	}
	rv = register_pernet_subsys(&xt_net_ops); ///< Register with per-net subsystem.
	if (rv < 0)
		kfree(xt); ///< Free `xt` on registration failure.
	return rv;
}

/**
 * @brief Module exit function for x_tables.
 * Functional Utility: Cleans up resources allocated by `xt_init` during module unload,
 * including unregistering from the per-network namespace subsystem and freeing memory.
 * This is the exit point for the module when it is unloaded from the kernel.
 */
static void __exit xt_fini(void)
{
	unregister_pernet_subsys(&xt_net_ops); ///< Unregister from per-net subsystem.
	kfree(xt); ///< Free `xt` memory.
}

module_init(xt_init); ///< Registers `xt_init` as the module's initialization function.
module_exit(xt_fini); ///< Registers `xt_fini` as the module's exit function.