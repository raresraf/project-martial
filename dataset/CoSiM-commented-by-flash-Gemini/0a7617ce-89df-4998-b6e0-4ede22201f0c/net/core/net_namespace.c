/**
 * @file net_namespace.c
 * @brief Core implementation for Linux kernel network namespace management.
 *
 * This file provides the foundational infrastructure for network namespaces
 * in the Linux kernel. It manages the creation, destruction, and isolation
 * of network resources for different network namespaces. Key functionalities
 * include registration and deregistration of per-net operations, allocation
 * of network namespace IDs (nsids), and handling netlink messages for nsid management.
 *
 * Functional Utility: Enables network virtualization, allowing multiple isolated
 * network stacks to coexist on a single system. This is fundamental for
 * container technologies and network segmentation.
 *
 * Algorithm:
 * - Pernet Operations: A linked list (`pernet_list`) of `pernet_operations`
 *   structures allows different kernel subsystems (e.g., IPv4, IPv6, Netfilter)
 *   to register initialization and exit functions. These functions are called
 *   when a network namespace is created or destroyed.
 * - NSID Management: Network namespace IDs (nsids) are allocated and managed
 *   using an `idr` (IDR tree) to provide unique identifiers for peer network
 *   namespaces. Netlink messages are used to communicate nsid assignments
 *   and deletions to userspace.
 * - Reference Counting: Network namespaces are reference counted, ensuring they
 *   are not prematurely destroyed while still in use. A `cleanup_net` workqueue
 *   handles asynchronous destruction.
 * - Synchronization: Extensive use of `rwsem` (`net_rwsem`, `pernet_ops_rwsem`)
 *   and `spin_lock_bh` (`net->nsid_lock`) for protecting shared data structures
 *   and ensuring thread-safe access to network namespace information.
 *
 * Kernel Architecture Details:
 * - VFS Integration: Exposes network namespaces through the /proc filesystem
 *   via `proc_ns_operations`.
 * - Netlink: Utilizes the Netlink socket family for userspace communication
 *   regarding nsid assignments.
 * - RCU: Employs Read-Copy-Update (RCU) for managing certain lists and pointers,
 *   allowing concurrent readers without locks.
 */
// SPDX-License-Identifier: GPL-2.0-only
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt ///< Macro to prepend module name to printk messages.

#include <linux/workqueue.h> ///< Workqueue definitions for asynchronous task execution.
#include <linux/rtnetlink.h> ///< RTNetlink definitions for network configuration.
#include <linux/cache.h> ///< Cacheline definitions for optimization.
#include <linux/slab.h> ///< Slab allocator for kernel memory.
#include <linux/list.h> ///< Linked list definitions.
#include <linux/delay.h> ///< Delay functions.
#include <linux/sched.h> ///< Scheduler definitions, task_struct.
#include <linux/idr.h> ///< IDR (ID allocator) definitions.
#include <linux/rculist.h> ///< RCU-protected linked list definitions.
#include <linux/nsproxy.h> ///< Namespace proxy definitions.
#include <linux/fs.h> ///< Filesystem definitions.
#include <linux/proc_ns.h> ///< Proc filesystem namespace definitions.
#include <linux/file.h> ///< File definitions.
#include <linux/export.h> ///< Symbol export macros.
#include <linux/user_namespace.h> ///< User namespace definitions.
#include <linux/net_namespace.h> ///< Core network namespace definitions.
#include <linux/sched/task.h> ///< Task scheduling definitions.
#include <linux/uidgid.h> ///< UID/GID management.
#include <linux/cookie.h> ///< Cookie definitions.
#include <linux/proc_fs.h> ///< Proc filesystem definitions.

#include <net/sock.h> ///< Socket definitions.
#include <net/netlink.h> ///< Netlink socket definitions.
#include <net/net_namespace.h> ///< Net-namespace specific network definitions.
#include <net/netns/generic.h> ///< Generic per-net namespace data.

/*
 *	Our network namespace constructor/destructor lists
 */

static LIST_HEAD(pernet_list); ///< Head of a linked list for generic pernet operations.
static struct list_head *first_device = &pernet_list; ///< Pointer to the first element in `pernet_list`.

LIST_HEAD(net_namespace_list); ///< Head of a linked list for all active network namespaces.
EXPORT_SYMBOL_GPL(net_namespace_list); ///< Export this symbol for GPL-licensed modules.

/* Protects net_namespace_list. Nests iside rtnl_lock() */
DECLARE_RWSEM(net_rwsem); ///< Read-write semaphore to protect `net_namespace_list`.
EXPORT_SYMBOL_GPL(net_rwsem); ///< Export this symbol.

#ifdef CONFIG_KEYS
static struct key_tag init_net_key_domain = { .usage = REFCOUNT_INIT(1) }; ///< Key domain for the initial network namespace.
#endif

struct net init_net; ///< The initial (root) network namespace.
EXPORT_SYMBOL(init_net); ///< Export this symbol.

static bool init_net_initialized; ///< Flag indicating if the initial network namespace has been set up.
/*
 * pernet_ops_rwsem: protects: pernet_list, net_generic_ids,
 * init_net_initialized and first_device pointer.
 * This is internal net namespace object. Please, don't use it
 * outside.
 */
DECLARE_RWSEM(pernet_ops_rwsem); ///< Read-write semaphore to protect pernet operations related data.

#define MIN_PERNET_OPS_ID	\
	((sizeof(struct net_generic) + sizeof(void *) - 1) / sizeof(void *)) ///< Minimum ID for per-net operations.

#define INITIAL_NET_GEN_PTRS	13 /* +1 for len +2 for rcu_head */ ///< Initial size for generic per-net pointers array.

static unsigned int max_gen_ptrs = INITIAL_NET_GEN_PTRS; ///< Current maximum size for generic per-net pointers array.

DEFINE_COOKIE(net_cookie); ///< Defines a cookie for network namespace identification.

/**
 * @brief Allocates a new `net_generic` structure.
 * Functional Utility: Provides dynamically sized storage for per-net private
 * data used by various subsystems. The size is determined by `max_gen_ptrs`.
 *
 * @return Pointer to the newly allocated `net_generic` structure, or `NULL` on failure.
 */
static struct net_generic *net_alloc_generic(void)
{
	unsigned int gen_ptrs = READ_ONCE(max_gen_ptrs); ///< Get current max generic pointers count.
	unsigned int generic_size;
	struct net_generic *ng;

	generic_size = offsetof(struct net_generic, ptr[gen_ptrs]); ///< Calculate total size needed.

	ng = kzalloc(generic_size, GFP_KERNEL); ///< Allocate zeroed kernel memory.
	if (ng)
		ng->s.len = gen_ptrs; ///< Store the length.

	return ng;
}

/**
 * @brief Assigns data to a specific generic per-net pointer ID.
 * Functional Utility: Extends the `net_generic` array if needed to accommodate
 * a new `id` and then stores the `data` pointer at that `id`. This mechanism
 * allows subsystems to have private data associated with each network namespace.
 *
 * @param net The network namespace to modify.
 * @param id The ID for the generic per-net pointer.
 * @param data The data pointer to assign.
 * @return 0 on success, or `-ENOMEM` if memory allocation fails.
 */
static int net_assign_generic(struct net *net, unsigned int id, void *data)
{
	struct net_generic *ng, *old_ng;

	BUG_ON(id < MIN_PERNET_OPS_ID); ///< Precondition: ID must be greater than or equal to minimum.

	old_ng = rcu_dereference_protected(net->gen,
					   lockdep_is_held(&pernet_ops_rwsem)); ///< Get old generic data, protected by RCU.
	if (old_ng->s.len > id) { ///< Block Logic: If current array is large enough.
		old_ng->ptr[id] = data; ///< Assign data directly.
		return 0;
	}

	ng = net_alloc_generic(); ///< Allocate a new, larger `net_generic` structure.
	if (!ng)
		return -ENOMEM;

	/*
	 * Some synchronisation notes:
	 *
	 * The net_generic explores the net->gen array inside rcu
	 * read section. Besides once set the net->gen->ptr[x]
	 * pointer never changes (see rules in netns/generic.h).
	 *
	 * That said, we simply duplicate this array and schedule
	 * the old copy for kfree after a grace period.
	 */

	// Block Logic: Copy existing generic data to the new, larger structure.
	memcpy(&ng->ptr[MIN_PERNET_OPS_ID], &old_ng->ptr[MIN_PERNET_OPS_ID],
	       (old_ng->s.len - MIN_PERNET_OPS_ID) * sizeof(void *));
	ng->ptr[id] = data; ///< Assign the new data.

	rcu_assign_pointer(net->gen, ng); ///< Atomically update `net->gen` to point to the new structure.
	kfree_rcu(old_ng, s.rcu); ///< Schedule old structure for RCU-delayed free.
	return 0;
}

/**
 * @brief Initializes a single pernet operation for a network namespace.
 * Functional Utility: Calls the `init` callback of a `pernet_operations`
 * structure for a given network namespace. If the operation has an ID and
 * size, it also allocates and assigns generic private data for it.
 *
 * @param ops Pointer to the `pernet_operations` structure.
 * @param net The network namespace to initialize the operation for.
 * @return 0 on success, or a negative errno on failure.
 */
static int ops_init(const struct pernet_operations *ops, struct net *net)
{
	struct net_generic *ng;
	int err = -ENOMEM;
	void *data = NULL;

	if (ops->id) { ///< Block Logic: If operation has a generic ID.
		data = kzalloc(ops->size, GFP_KERNEL); ///< Allocate private data.
		if (!data)
			goto out;

		err = net_assign_generic(net, *ops->id, data); ///< Assign data to generic array.
		if (err)
			goto cleanup;
	}
	err = 0;
	if (ops->init)
		err = ops->init(net); ///< Call the subsystem's init function.
	if (!err)
		return 0;

	if (ops->id) { ///< Block Logic: On error, clean up allocated generic data.
		ng = rcu_dereference_protected(net->gen,
					       lockdep_is_held(&pernet_ops_rwsem));
		ng->ptr[*ops->id] = NULL; ///< Clear generic pointer.
	}

cleanup: ///< Cleanup allocated private data.
	kfree(data);

out:
	return err;
}

/**
 * @brief Calls the `pre_exit` callback for a pernet operation on a list of network namespaces.
 * Functional Utility: Executes the `pre_exit` hook for a specific `pernet_operations`
 * module across all network namespaces provided in the `net_exit_list`. This is
 * typically used for cleanup that needs to happen before network devices are removed.
 *
 * @param ops Pointer to the `pernet_operations` structure.
 * @param net_exit_list List of network namespaces to process.
 */
static void ops_pre_exit_list(const struct pernet_operations *ops,
			      struct list_head *net_exit_list)
{
	struct net *net;

	if (ops->pre_exit) { ///< Block Logic: If `pre_exit` callback is defined.
		list_for_each_entry(net, net_exit_list, exit_list)
			ops->pre_exit(net); ///< Call `pre_exit` for each network namespace.
	}
}

/**
 * @brief Calls the `exit_rtnl` callback for a pernet operation on a list of network namespaces.
 * Functional Utility: Executes the `exit_rtnl` hook for a specific `pernet_operations`
 * module across all network namespaces provided in the `net_exit_list`, under the `rtnl_lock`.
 * This is used for cleanup that involves Netlink operations or network devices.
 *
 * @param ops_list The main list of `pernet_operations`.
 * @param ops Pointer to the current `pernet_operations` structure.
 * @param net_exit_list List of network namespaces to process.
 */
static void ops_exit_rtnl_list(const struct list_head *ops_list,
			       const struct pernet_operations *ops,
			       struct list_head *net_exit_list)
{
	const struct pernet_operations *saved_ops = ops;
	LIST_HEAD(dev_kill_list); ///< List to collect devices to be unregistered.
	struct net *net;

	rtnl_lock(); ///< Acquire RTNetlink lock.

	list_for_each_entry(net, net_exit_list, exit_list) { ///< Block Logic: Iterate through network namespaces.
		__rtnl_net_lock(net); ///< Acquire per-net RTNetlink lock.

		ops = saved_ops;
		// Block Logic: Iterate backwards through operations, calling `exit_rtnl`.
		list_for_each_entry_continue_reverse(ops, ops_list, list) {
			if (ops->exit_rtnl)
				ops->exit_rtnl(net, &dev_kill_list); ///< Call `exit_rtnl` and collect devices.
		}

		__rtnl_net_unlock(net); ///< Release per-net RTNetlink lock.
	}

	unregister_netdevice_many(&dev_kill_list); ///< Unregister collected devices.

	rtnl_unlock(); ///< Release RTNetlink lock.
}

/**
 * @brief Calls the `exit` callback for a pernet operation on a list of network namespaces.
 * Functional Utility: Executes the `exit` hook for a specific `pernet_operations`
 * module across all network namespaces provided in the `net_exit_list`. This is
 * used for general cleanup during network namespace destruction.
 *
 * @param ops Pointer to the `pernet_operations` structure.
 * @param net_exit_list List of network namespaces to process.
 */
static void ops_exit_list(const struct pernet_operations *ops,
			  struct list_head *net_exit_list)
{
	if (ops->exit) { ///< Block Logic: If `exit` callback is defined.
		struct net *net;

		list_for_each_entry(net, net_exit_list, exit_list) {
			ops->exit(net); ///< Call `exit` for each network namespace.
			cond_resched(); ///< Conditionally reschedule to prevent CPU hogging.
		}
	}

	if (ops->exit_batch)
		ops->exit_batch(net_exit_list); ///< Call `exit_batch` if defined.
}

/**
 * @brief Frees generic pernet data for a list of network namespaces.
 * Functional Utility: Deallocates the private generic data associated with a
 * `pernet_operations` module for each network namespace in the `net_exit_list`.
 *
 * @param ops Pointer to the `pernet_operations` structure.
 * @param net_exit_list List of network namespaces to process.
 */
static void ops_free_list(const struct pernet_operations *ops,
			  struct list_head *net_exit_list)
{
	struct net *net;

	if (ops->id) { ///< Block Logic: If operation has a generic ID.
		list_for_each_entry(net, net_exit_list, exit_list)
			kfree(net_generic(net, *ops->id)); ///< Free generic data for each network namespace.
	}
}

/**
 * @brief Undoes a list of pernet operations on a set of network namespaces.
 * Functional Utility: This is a comprehensive cleanup function that iterates
 * through a list of `pernet_operations` in reverse order of registration.
 * It calls `pre_exit`, `exit_rtnl`, `exit`, and `free` callbacks for each
 * network namespace that is being destroyed.
 *
 * @param ops_list The master list of `pernet_operations`.
 * @param ops Pointer to the `pernet_operations` (start point for undo, or `NULL` to start from beginning).
 * @param net_exit_list List of network namespaces to undo operations for.
 * @param expedite_rcu If true, uses expedited RCU synchronization.
 */
static void ops_undo_list(const struct list_head *ops_list,
			  const struct pernet_operations *ops,
			  struct list_head *net_exit_list,
			  bool expedite_rcu)
{
	const struct pernet_operations *saved_ops;
	bool hold_rtnl = false;

	if (!ops)
		ops = list_entry(ops_list, typeof(*ops), list); ///< If `ops` is NULL, start from the first element of `ops_list`.

	saved_ops = ops; // Save the starting point.

	// Block Logic: Call `pre_exit` callbacks in reverse order of registration.
	list_for_each_entry_continue_reverse(ops, ops_list, list) {
		hold_rtnl |= !!ops->exit_rtnl; // Check if `exit_rtnl` needs to be called.
		ops_pre_exit_list(ops, net_exit_list);
	}

	/* Another CPU might be rcu-iterating the list, wait for it.
	 * This needs to be before calling the exit() notifiers, so the
	 * rcu_barrier() after ops_undo_list() isn't sufficient alone.
	 * Also the pre_exit() and exit() methods need this barrier.
	 */
	if (expedite_rcu)
		synchronize_rcu_expedited(); ///< Expedited RCU synchronization.
	else
		synchronize_rcu(); ///< Standard RCU synchronization.

	if (hold_rtnl)
		ops_exit_rtnl_list(ops_list, saved_ops, net_exit_list); ///< Call `exit_rtnl` callbacks.

	ops = saved_ops;
	// Block Logic: Call `exit` callbacks in reverse order of registration.
	list_for_each_entry_continue_reverse(ops, ops_list, list)
		ops_exit_list(ops, net_exit_list);

	ops = saved_ops;
	// Block Logic: Call `free` callbacks in reverse order of registration.
	list_for_each_entry_continue_reverse(ops, ops_list, list)
		ops_free_list(ops, net_exit_list);
}

/**
 * @brief Undoes a single pernet operation on a set of network namespaces.
 * Functional Utility: A wrapper around `ops_undo_list` for handling cleanup
 * of a single `pernet_operations` structure.
 *
 * @param ops Pointer to the `pernet_operations` structure to undo.
 * @param net_exit_list List of network namespaces to undo operations for.
 */
static void ops_undo_single(struct pernet_operations *ops,
			    struct list_head *net_exit_list)
{
	LIST_HEAD(ops_list); ///< Temporary list for the single operation.

	list_add(&ops->list, &ops_list); ///< Add the single operation to the list.
	ops_undo_list(&ops_list, NULL, net_exit_list, false); ///< Call undo list.
	list_del(&ops->list); ///< Remove from temporary list.
}

/* should be called with nsid_lock held */
/**
 * @brief Allocates a network ID (`nsid`) for a peer network namespace.
 * Functional Utility: Assigns a unique integer ID to a `peer` network namespace
 * within the context of the `net` network namespace. This ID is used for
 * communication and identification between different network namespaces.
 *
 * @param net The network namespace in which the ID is being allocated.
 * @param peer The peer network namespace to assign an ID to.
 * @param reqid If `>= 0`, a specific ID is requested; otherwise, an ID is allocated automatically.
 * @return The allocated ID, or a negative errno on failure.
 */
static int alloc_netid(struct net *net, struct net *peer, int reqid)
{
	int min = 0, max = 0;

	if (reqid >= 0) { ///< Block Logic: If a specific ID is requested.
		min = reqid;
		max = reqid + 1;
	}

	return idr_alloc(&net->netns_ids, peer, min, max, GFP_ATOMIC); ///< Allocate ID from IDR tree.
}

/* This function is used by idr_for_each(). If net is equal to peer, the
 * function returns the id so that idr_for_each() stops. Because we cannot
 * returns the id 0 (idr_for_each() will not stop), we return the magic value
 * NET_ID_ZERO (-1) for it.
 */
#define NET_ID_ZERO -1 ///< Magic value returned by `net_eq_idr` when ID is 0.
/**
 * @brief Callback for `idr_for_each` to find the ID of a peer network namespace.
 * Functional Utility: Used during iteration of an IDR tree to compare a given
 * `net` structure with a `peer` structure. If they are equal, it returns the
 * ID associated with the `peer`.
 *
 * @param id The current ID being iterated.
 * @param net The network namespace being checked.
 * @param peer The peer network namespace to find.
 * @return The ID of the peer if found (or `NET_ID_ZERO` if ID is 0), otherwise 0.
 */
static int net_eq_idr(int id, void *net, void *peer)
{
	if (net_eq(net, peer)) ///< Functional Utility: Compare network namespaces.
		return id ? : NET_ID_ZERO; ///< Return ID or `NET_ID_ZERO` for ID 0.
	return 0;
}

/* Must be called from RCU-critical section or with nsid_lock held */
/**
 * @brief Internal function to find the nsid of a peer network namespace.
 * Functional Utility: Searches the `net->netns_ids` IDR tree to find the
 * ID (nsid) assigned to a `peer` network namespace within the context of `net`.
 *
 * @param net The network namespace context.
 * @param peer The peer network namespace to find the ID for.
 * @return The nsid of the peer, or `NETNSA_NSID_NOT_ASSIGNED` if not found.
 */
static int __peernet2id(const struct net *net, struct net *peer)
{
	int id = idr_for_each(&net->netns_ids, net_eq_idr, peer); ///< Iterate IDR to find peer's ID.

	/* Magic value for id 0. */
	if (id == NET_ID_ZERO)
		return 0;
	if (id > 0)
		return id;

	return NETNSA_NSID_NOT_ASSIGNED;
}

/**
 * @brief Notifies Netlink listeners about network ID changes.
 * Functional Utility: Sends a Netlink message (`RTM_NEWNSID` or `RTM_DELNSID`)
 * to interested userspace applications (e.g., `iproute2`) whenever a network
 * namespace ID is assigned or removed.
 *
 * @param net The network namespace generating the notification.
 * @param cmd The Netlink command (e.g., `RTM_NEWNSID`).
 * @param id The network ID being notified.
 * @param portid The Netlink port ID of the sender.
 * @param nlh Original Netlink message header (if a response).
 * @param gfp GFP flags for memory allocation.
 */
static void rtnl_net_notifyid(struct net *net, int cmd, int id, u32 portid,
			      struct nlmsghdr *nlh, gfp_t gfp);
/**
 * @brief Allocates an nsid for a peer network namespace if one doesn't already exist.
 * Functional Utility: Ensures that a `peer` network namespace has an nsid
 * assigned within the `net` network namespace. If an ID is not assigned,
 * it allocates one and notifies userspace via Netlink.
 *
 * @param net The network namespace allocating the ID.
 * @param peer The peer network namespace to get/allocate an ID for.
 * @param gfp GFP flags for memory allocation.
 * @return The allocated nsid, or `NETNSA_NSID_NOT_ASSIGNED` on failure.
 */
int peernet2id_alloc(struct net *net, struct net *peer, gfp_t gfp)
{
	int id;

	if (refcount_read(&net->ns.count) == 0)
		return NETNSA_NSID_NOT_ASSIGNED; ///< Return if network namespace is dying.

	spin_lock_bh(&net->nsid_lock); ///< Acquire spinlock for nsid operations.
	id = __peernet2id(net, peer); ///< Try to find existing ID.
	if (id >= 0) {
		spin_unlock_bh(&net->nsid_lock); ///< Release spinlock.
		return id; ///< Return existing ID.
	}

	/* When peer is obtained from RCU lists, we may race with
	 * its cleanup. Check whether it's alive, and this guarantees
	 * we never hash a peer back to net->netns_ids, after it has
	 * just been idr_remove()'d from there in cleanup_net().
	 */
	if (!maybe_get_net(peer)) { ///< Functional Utility: Check if peer is still alive and get a reference.
		spin_unlock_bh(&net->nsid_lock); ///< Release spinlock.
		return NETNSA_NSID_NOT_ASSIGNED;
	}

	id = alloc_netid(net, peer, -1); ///< Allocate a new ID.
	spin_unlock_bh(&net->nsid_lock); ///< Release spinlock.

	put_net(peer); ///< Release temporary reference to peer.
	if (id < 0)
		return NETNSA_NSID_NOT_ASSIGNED;

	rtnl_net_notifyid(net, RTM_NEWNSID, id, 0, NULL, gfp); ///< Notify userspace.

	return id;
}
EXPORT_SYMBOL_GPL(peernet2id_alloc); ///< Export this symbol (GPL-only).

/**
 * @brief Retrieves the nsid of a peer network namespace if already assigned.
 * Functional Utility: Provides a read-only way to check if a `peer` network
 * namespace has an nsid assigned within `net` without allocating a new one.
 * It's RCU-protected for concurrent access.
 *
 * @param net The network namespace context.
 * @param peer The peer network namespace to query.
 * @return The nsid of the peer if assigned, or `NETNSA_NSID_NOT_ASSIGNED` if not.
 */
int peernet2id(const struct net *net, struct net *peer)
{
	int id;

	rcu_read_lock(); ///< Acquire RCU read lock.
	id = __peernet2id(net, peer); ///< Call internal function to find ID.
	rcu_read_unlock(); ///< Release RCU read lock.

	return id;
}
EXPORT_SYMBOL(peernet2id); ///< Export this symbol.

/**
 * @brief Checks if a peer network namespace has an nsid assigned.
 * Functional Utility: A convenience function to check for the existence of an
 * assigned nsid for a `peer` network namespace within `net`.
 *
 * @param net The network namespace context.
 * @param peer The peer network namespace to check.
 * @return `true` if an nsid is assigned, `false` otherwise.
 */
bool peernet_has_id(const struct net *net, struct net *peer)
{
	return peernet2id(net, peer) >= 0;
}

/**
 * @brief Retrieves a network namespace by its nsid within a given network namespace.
 * Functional Utility: Looks up a `net` structure using its assigned `id` within
 * the context of another `net` namespace. This is used to resolve nsids to actual
 * network namespace objects.
 *
 * @param net The network namespace context.
 * @param id The nsid to look up.
 * @return Pointer to the `net` structure if found and valid, or `NULL`.
 */
struct net *get_net_ns_by_id(const struct net *net, int id)
{
	struct net *peer;

	if (id < 0)
		return NULL;

	rcu_read_lock(); ///< Acquire RCU read lock.
	peer = idr_find(&net->netns_ids, id); ///< Find peer in IDR tree.
	if (peer)
		peer = maybe_get_net(peer); ///< Check if peer is still alive and get a reference.
	rcu_read_unlock(); ///< Release RCU read lock.

	return peer;
}
EXPORT_SYMBOL_GPL(get_net_ns_by_id); ///< Export this symbol (GPL-only).

/**
 * @brief Initializes sysctl parameters for a network namespace.
 * Functional Utility: Sets default values for various network-related sysctl
 * parameters (e.g., `somaxconn`, `optmem_max`, `txrehash`, `tstamp_allow_data`)
 * for a newly created network namespace.
 *
 * @param net The network namespace to initialize.
 */
static __net_init void preinit_net_sysctl(struct net *net)
{
	net->core.sysctl_somaxconn = SOMAXCONN; ///< Default backlog for listen sockets.
	/* Limits per socket sk_omem_alloc usage.
	 * TCP zerocopy regular usage needs 128 KB.
	 */
	net->core.sysctl_optmem_max = 128 * 1024; ///< Maximum socket option memory buffer.
	net->core.sysctl_txrehash = SOCK_TXREHASH_ENABLED; ///< TCP transmit rehash enabled.
	net->core.sysctl_tstamp_allow_data = 1; ///< Allow timestamp data.
}

/* init code that must occur even if setup_net() is not called. */
/**
 * @brief Pre-initializes a network namespace structure.
 * Functional Utility: Sets up basic fields and data structures for a new `net`
 * object, including reference counts, random mix for hashing, user namespace
 * association, IDR for nsids, and mutexes. This is called early in the lifecycle.
 *
 * @param net The network namespace to pre-initialize.
 * @param user_ns The user namespace associated with this network namespace.
 */
static __net_init void preinit_net(struct net *net, struct user_namespace *user_ns)
{
	refcount_set(&net->passive, 1); ///< Initialize passive reference count.
	refcount_set(&net->ns.count, 1); ///< Initialize active reference count.
	ref_tracker_dir_init(&net->refcnt_tracker, 128, "net refcnt"); ///< Initialize reference count tracker.
	ref_tracker_dir_init(&net->notrefcnt_tracker, 128, "net notrefcnt"); ///< Initialize non-reference count tracker.

	get_random_bytes(&net->hash_mix, sizeof(u32)); ///< Initialize hash mix for network devices.
	net->dev_base_seq = 1; ///< Initialize device base sequence number.
	net->user_ns = user_ns; ///< Associate user namespace.

	idr_init(&net->netns_ids); ///< Initialize IDR tree for nsids.
	spin_lock_init(&net->nsid_lock); ///< Initialize spinlock for nsid.
	mutex_init(&net->ipv4.ra_mutex); ///< Initialize mutex for IPv4 RA.

#ifdef CONFIG_DEBUG_NET_SMALL_RTNL ///< Debugging for small RTNetlink.
	mutex_init(&net->rtnl_mutex);
	lock_set_cmp_fn(&net->rtnl_mutex, rtnl_net_lock_cmp_fn, NULL);
#endif

	INIT_LIST_HEAD(&net->ptype_all); ///< Initialize list for all packet types.
	INIT_LIST_HEAD(&net->ptype_specific); ///< Initialize list for specific packet types.
	preinit_net_sysctl(net); ///< Initialize sysctl parameters.
}

/*
 * setup_net runs the initializers for the network namespace object.
 */
/**
 * @brief Sets up a network namespace by calling all registered pernet operations.
 * Functional Utility: Iterates through the `pernet_list` and calls the `ops_init`
 * function for each registered `pernet_operations` structure. This brings up
 * all network-related subsystems within the new network namespace.
 *
 * @param net The network namespace to set up.
 * @return 0 on success, or a negative errno on failure.
 */
static __net_init int setup_net(struct net *net)
{
	/* Must be called with pernet_ops_rwsem held */
	const struct pernet_operations *ops;
	LIST_HEAD(net_exit_list); ///< List to collect network namespaces for exit.
	int error = 0;

	preempt_disable(); ///< Disable preemption for cookie generation.
	net->net_cookie = gen_cookie_next(&net_cookie); ///< Generate a unique cookie for the net namespace.
	preempt_enable(); ///< Re-enable preemption.

	// Block Logic: Iterate through registered pernet operations and initialize them.
	list_for_each_entry(ops, &pernet_list, list) {
		error = ops_init(ops, net); ///< Initialize current operation.
		if (error < 0)
			goto out_undo; ///< If error, jump to undo.
	}
	down_write(&net_rwsem); ///< Acquire write lock for `net_namespace_list`.
	list_add_tail_rcu(&net->list, &net_namespace_list); ///< Add network namespace to global list.
	up_write(&net_rwsem); ///< Release write lock.
out:
	return error;

out_undo: ///< Error handling: undo already initialized operations.
	/* Walk through the list backwards calling the exit functions
	 * for the pernet modules whose init functions did not fail.
	 */
	list_add(&net->exit_list, &net_exit_list); ///< Add current net to exit list.
	ops_undo_list(&pernet_list, ops, &net_exit_list, false); ///< Undo operations.
	rcu_barrier(); ///< Wait for RCU grace period.
	goto out;
}

#ifdef CONFIG_NET_NS ///< Conditional compilation: Only if network namespaces are enabled.
/**
 * @brief Increments the count of network namespaces for a user namespace.
 * Functional Utility: Increases a counter that tracks the number of network
 * namespaces owned by a specific user namespace, enforcing limits.
 *
 * @param ns The user namespace.
 * @return Pointer to `ucounts` on success, or `NULL` if limit reached.
 */
static struct ucounts *inc_net_namespaces(struct user_namespace *ns)
{
	return inc_ucount(ns, current_euid(), UCOUNT_NET_NAMESPACES); ///< Increment user namespace count.
}

/**
 * @brief Decrements the count of network namespaces for a user namespace.
 * Functional Utility: Decreases the counter tracking network namespaces owned
 * by a user namespace.
 *
 * @param ucounts Pointer to the `ucounts` structure.
 */
static void dec_net_namespaces(struct ucounts *ucounts)
{
	dec_ucount(ucounts, UCOUNT_NET_NAMESPACES); ///< Decrement user namespace count.
}

static struct kmem_cache *net_cachep __ro_after_init; ///< Kmem cache for `net` structures.
static struct workqueue_struct *netns_wq; ///< Workqueue for network namespace cleanup.

/**
 * @brief Allocates a new `net` (network namespace) structure.
 * Functional Utility: Creates and initializes a new `net` structure,
 * including allocation of `net_generic` data and key domain if CONFIG_KEYS is enabled.
 *
 * @return Pointer to the newly allocated `net` structure, or `NULL` on failure.
 */
static struct net *net_alloc(void)
{
	struct net *net = NULL;
	struct net_generic *ng;

	ng = net_alloc_generic(); ///< Allocate generic per-net data.
	if (!ng)
		goto out;

	net = kmem_cache_zalloc(net_cachep, GFP_KERNEL); ///< Allocate `net` structure from slab cache.
	if (!net)
		goto out_free;

#ifdef CONFIG_KEYS ///< Block Logic: If CONFIG_KEYS is enabled, allocate key domain.
	net->key_domain = kzalloc(sizeof(struct key_tag), GFP_KERNEL);
	if (!net->key_domain)
		goto out_free_2;
	refcount_set(&net->key_domain->usage, 1);
#endif

	rcu_assign_pointer(net->gen, ng); ///< Assign generic data, RCU-protected.
out:
	return net;

#ifdef CONFIG_KEYS
out_free_2: ///< Error handling: free `net` if key domain allocation fails.
	kmem_cache_free(net_cachep, net);
	net = NULL;
#endif
out_free: ///< Error handling: free `net_generic` if `net` allocation fails.
	kfree(ng);
	goto out;
}

static LLIST_HEAD(defer_free_list); ///< List for network namespaces to be freed later.

/**
 * @brief Completes the freeing of network namespace structures.
 * Functional Utility: Processes a list of network namespaces that have been
 * marked for deferred freeing and deallocates their memory.
 */
static void net_complete_free(void)
{
	struct llist_node *kill_list;
	struct net *net, *next;

	/* Get the list of namespaces to free from last round. */
	kill_list = llist_del_all(&defer_free_list); ///< Atomically get all nodes from the list.

	llist_for_each_entry_safe(net, next, kill_list, defer_free_list)
		kmem_cache_free(net_cachep, net); ///< Free each network namespace structure.

}

/**
 * @brief Decrements the passive reference count of a network namespace.
 * Functional Utility: When the passive reference count reaches zero, it
 * indicates that the network namespace is no longer actively referenced
 * and its memory can be freed (after an RCU grace period).
 *
 * @param net The network namespace whose passive reference count to decrement.
 */
void net_passive_dec(struct net *net)
{
	if (refcount_dec_and_test(&net->passive)) { ///< Block Logic: If passive refcount drops to zero.
		kfree(rcu_access_pointer(net->gen)); ///< Free generic per-net data.

		/* There should not be any trackers left there. */
		ref_tracker_dir_exit(&net->notrefcnt_tracker); ///< Exit non-reference tracker.

		/* Wait for an extra rcu_barrier() before final free. */
		llist_add(&net->defer_free_list, &defer_free_list); ///< Add to deferred free list.
	}
}

/**
 * @brief Drops a network namespace reference (void* wrapper).
 * Functional Utility: A generic wrapper to decrement the passive reference
 * count of a network namespace, primarily used for cleanup contexts that
 * expect a `void *` pointer.
 *
 * @param p Pointer to the network namespace (as `void *`).
 */
void net_drop_ns(void *p)
{
	struct net *net = (struct net *)p;

	if (net)
		net_passive_dec(net); ///< Call `net_passive_dec` if `net` is valid.
}

/**
 * @brief Copies a network namespace, potentially creating a new one.
 * Functional Utility: Creates a new network namespace if `CLONE_NEWNET` flag
 * is set, otherwise increments the reference count of the `old_net` namespace.
 * This is used during process cloning or `unshare` system calls.
 *
 * @param flags Cloning flags, especially `CLONE_NEWNET`.
 * @param user_ns The user namespace of the new network namespace.
 * @param old_net The network namespace to copy from (or reference if not new).
 * @return Pointer to the new or referenced `net` structure, or `ERR_PTR` on failure.
 */
struct net *copy_net_ns(unsigned long flags,
			struct user_namespace *user_ns, struct net *old_net)
{
	struct ucounts *ucounts;
	struct net *net;
	int rv;

	if (!(flags & CLONE_NEWNET))
		return get_net(old_net); ///< If not creating new netns, just get reference to old one.

	ucounts = inc_net_namespaces(user_ns); ///< Increment count for user namespace.
	if (!ucounts)
		return ERR_PTR(-ENOSPC); ///< Return error if resource limit reached.

	net = net_alloc(); ///< Allocate new network namespace structure.
	if (!net) {
		rv = -ENOMEM;
		goto dec_ucounts;
	}

	preinit_net(net, user_ns); ///< Pre-initialize the new network namespace.
	net->ucounts = ucounts; ///< Assign user namespace ucounts.
	get_user_ns(user_ns); ///< Get a reference to the user namespace.

	rv = down_read_killable(&pernet_ops_rwsem); ///< Acquire read lock on pernet operations.
	if (rv < 0)
		goto put_userns;

	rv = setup_net(net); ///< Set up the new network namespace with pernet operations.

	up_read(&pernet_ops_rwsem); ///< Release read lock.

	if (rv < 0) {
put_userns: ///< Error handling: put user namespace reference.
#ifdef CONFIG_KEYS
		key_remove_domain(net->key_domain); ///< Remove key domain if allocated.
#endif
		put_user_ns(user_ns); ///< Put user namespace reference.
		net_passive_dec(net); ///< Decrement passive reference count.
dec_ucounts: ///< Error handling: decrement ucounts.
		dec_net_namespaces(ucounts);
		return ERR_PTR(rv);
	}
	return net;
}

/**
 * @brief Retrieves sysfs ownership UID/GID for a network namespace.
 * Functional Utility: Provides the kernel UID/GID pair corresponding to the
 * root user within the user namespace associated with a given network namespace.
 * This is used for setting ownership of sysfs objects related to the network namespace.
 *
 * @param net The network namespace (can be `NULL`).
 * @param uid Pointer to `kuid_t` to store the UID.
 * @param gid Pointer to `kgid_t` to store the GID.
 */
void net_ns_get_ownership(const struct net *net, kuid_t *uid, kgid_t *gid)
{
	if (net) { ///< Block Logic: If `net` is valid.
		kuid_t ns_root_uid = make_kuid(net->user_ns, 0); ///< Get UID of root in net's user namespace.
		kgid_t ns_root_gid = make_kgid(net->user_ns, 0); ///< Get GID of root in net's user namespace.

		if (uid_valid(ns_root_uid))
			*uid = ns_root_uid; ///< Assign valid UID.

		if (gid_valid(ns_root_gid))
			*gid = ns_root_gid; ///< Assign valid GID.
	} else { ///< Block Logic: If `net` is `NULL`, use global root UID/GID.
		*uid = GLOBAL_ROOT_UID;
		*gid = GLOBAL_ROOT_GID;
	}
}
EXPORT_SYMBOL_GPL(net_ns_get_ownership); ///< Export this symbol (GPL-only).

/**
 * @brief Unhashes nsids associated with a dying network namespace.
 * Functional Utility: Iterates through all active network namespaces and removes
 * any nsids that refer to the `net` network namespace (which is being destroyed)
 * from their respective IDR trees. It also notifies userspace via Netlink.
 *
 * @param net The network namespace that is being destroyed.
 * @param last The last network namespace in `net_namespace_list`.
 */
static void unhash_nsid(struct net *net, struct net *last)
{
	struct net *tmp;
	/* This function is only called from cleanup_net() work,
	 * and this work is the only process, that may delete
	 * a net from net_namespace_list. So, when the below
	 * is executing, the list may only grow. Thus, we do not
	 * use for_each_net_rcu() or net_rwsem.
	 */
	for_each_net(tmp) { ///< Block Logic: Iterate through all active network namespaces.
		int id;

		spin_lock_bh(&tmp->nsid_lock); ///< Acquire spinlock for nsid.
		id = __peernet2id(tmp, net); ///< Find ID of `net` in `tmp`'s IDR.
		if (id >= 0)
			idr_remove(&tmp->netns_ids, id); ///< Remove ID if found.
		spin_unlock_bh(&tmp->nsid_lock); ///< Release spinlock.
		if (id >= 0)
			rtnl_net_notifyid(tmp, RTM_DELNSID, id, 0, NULL,
					  GFP_KERNEL); ///< Notify userspace of ID deletion.
		if (tmp == last)
			break; ///< Stop if `last` net is reached.
	}
	spin_lock_bh(&net->nsid_lock); ///< Acquire spinlock for `net`'s nsid.
	idr_destroy(&net->netns_ids); ///< Destroy `net`'s IDR tree.
	spin_unlock_bh(&net->nsid_lock); ///< Release spinlock.
}

static LLIST_HEAD(cleanup_list); ///< List for network namespaces awaiting cleanup.

struct task_struct *cleanup_net_task; ///< Pointer to the task performing network namespace cleanup.

/**
 * @brief Cleans up a network namespace when its reference count drops to zero.
 * Functional Utility: This function is executed as a workqueue item. It detaches
 * the network namespace from global lists, removes nsids, undoes pernet operations,
 * and frees associated memory.
 *
 * @param work Pointer to the `work_struct`.
 */
static void cleanup_net(struct work_struct *work)
{
	struct llist_node *net_kill_list;
	struct net *net, *tmp, *last;
	LIST_HEAD(net_exit_list); ///< List to collect network namespaces for exit.

	WRITE_ONCE(cleanup_net_task, current); ///< Functional Utility: Record the task performing cleanup.

	/* Atomically snapshot the list of namespaces to cleanup */
	net_kill_list = llist_del_all(&cleanup_list); ///< Get all network namespaces from cleanup list.

	down_read(&pernet_ops_rwsem); ///< Acquire read lock for pernet operations.

	/* Don't let anyone else find us. */
	down_write(&net_rwsem); ///< Acquire write lock for `net_namespace_list`.
	llist_for_each_entry(net, net_kill_list, cleanup_list)
		list_del_rcu(&net->list); ///< Remove network namespace from global list (RCU-protected).
	/* Cache last net. After we unlock rtnl, no one new net
	 * added to net_namespace_list can assign nsid pointer
	 * to a net from net_kill_list (see peernet2id_alloc()).
	 * So, we skip them in unhash_nsid().
	 *
	 * Note, that unhash_nsid() does not delete nsid links
	 * between net_kill_list's nets, as they've already
	 * deleted from net_namespace_list. But, this would be
	 * useless anyway, as netns_ids are destroyed there.
	 */
	last = list_last_entry(&net_namespace_list, struct net, list); ///< Get last entry in the list.
	up_write(&net_rwsem); ///< Release write lock.

	llist_for_each_entry(net, net_kill_list, cleanup_list) { ///< Block Logic: Process each network namespace marked for killing.
		unhash_nsid(net, last); ///< Unhash nsids associated with this network namespace.
		list_add_tail(&net->exit_list, &net_exit_list); ///< Add to network exit list.
	}

	ops_undo_list(&pernet_list, NULL, &net_exit_list, true); ///< Undo all pernet operations.

	up_read(&pernet_ops_rwsem); ///< Release read lock.

	/* Ensure there are no outstanding rcu callbacks using this
	 * network namespace.
	 */
	rcu_barrier(); ///< Wait for RCU grace period for all pending RCU callbacks to complete.

	net_complete_free(); ///< Complete deferred freeing of network namespaces.

	/* Finally it is safe to free my network namespace structure */
	list_for_each_entry_safe(net, tmp, &net_exit_list, exit_list) { ///< Block Logic: Iterate through network namespaces for final freeing.
		list_del_init(&net->exit_list); ///< Remove from exit list.
		dec_net_namespaces(net->ucounts); ///< Decrement user namespace count.
#ifdef CONFIG_KEYS
		key_remove_domain(net->key_domain); ///< Remove key domain.
#endif
		put_user_ns(net->user_ns); ///< Put user namespace reference.
		net_passive_dec(net); ///< Decrement passive reference count.
	}
	WRITE_ONCE(cleanup_net_task, NULL); ///< Clear cleanup task pointer.
}

/**
 * @brief Waits until concurrent network namespace cleanup work is done.
 * Functional Utility: This barrier function ensures that all network namespace
 * cleanup (`cleanup_net` workqueue) has completed. It's used to prevent modules
 * from being unloaded while they might still be cleaning up network namespaces.
 */
void net_ns_barrier(void)
{
	down_write(&pernet_ops_rwsem); ///< Acquire write lock to serialize with cleanup.
	up_write(&pernet_ops_rwsem); ///< Release write lock.
}
EXPORT_SYMBOL(net_ns_barrier); ///< Export this symbol.

static DECLARE_WORK(net_cleanup_work, cleanup_net); ///< Declare a work item for network namespace cleanup.

/**
 * @brief Internal function to put (decrement reference count) a network namespace.
 * Functional Utility: This is the primary function for decrementing the active
 * reference count of a `net` structure. If the count drops to zero, it schedules
 * the network namespace for deferred cleanup via a workqueue.
 *
 * @param net The network namespace to put.
 */
void __put_net(struct net *net)
{
	ref_tracker_dir_exit(&net->refcnt_tracker); ///< Exit reference count tracker.
	/* Cleanup the network namespace in process context */
	if (llist_add(&net->cleanup_list, &cleanup_list)) ///< Block Logic: Add to cleanup list and queue work if it's the first addition.
		queue_work(netns_wq, &net_cleanup_work); ///< Queue cleanup work.
}
EXPORT_SYMBOL_GPL(__put_net); ///< Export this symbol (GPL-only).

/**
 * @brief Increments the refcount of a network namespace.
 * Functional Utility: Provides a mechanism to acquire an active reference to a
 * network namespace (`net_common` type), ensuring it remains allocated.
 *
 * @param ns Common namespace structure (expected to be a `net` namespace).
 * @return Pointer to the `ns_common` of the network namespace, or `ERR_PTR` if ref is zero.
 */
struct ns_common *get_net_ns(struct ns_common *ns)
{
	struct net *net;

	net = maybe_get_net(container_of(ns, struct net, ns)); ///< Functional Utility: Get a reference to `net` from `ns_common`.
	if (net)
		return &net->ns; ///< Return `ns_common` if valid.
	return ERR_PTR(-EINVAL); ///< Return error if `net` is invalid.
}
EXPORT_SYMBOL_GPL(get_net_ns); ///< Export this symbol (GPL-only).

/**
 * @brief Retrieves a network namespace by file descriptor.
 * Functional Utility: Given an open file descriptor to a `/proc/pid/ns/net`
 * entry, this function retrieves the corresponding `net` structure.
 *
 * @param fd The file descriptor.
 * @return Pointer to the `net` structure, or `ERR_PTR` on failure.
 */
struct net *get_net_ns_by_fd(int fd)
{
	CLASS(fd, f)(fd); ///< Get file structure from file descriptor.

	if (fd_empty(f))
		return ERR_PTR(-EBADF); ///< Return error if file descriptor is invalid.

	if (proc_ns_file(fd_file(f))) { ///< Block Logic: Check if it's a /proc namespace file.
		struct ns_common *ns = get_proc_ns(file_inode(fd_file(f))); ///< Get `ns_common` from inode.
		if (ns->ops == &netns_operations) ///< Block Logic: Check if it's a network namespace.
			return get_net(container_of(ns, struct net, ns)); ///< Get reference to `net`.
	}

	return ERR_PTR(-EINVAL); ///< Return error for invalid file or namespace type.
}
EXPORT_SYMBOL_GPL(get_net_ns_by_fd); ///< Export this symbol (GPL-only).
#endif

/**
 * @brief Retrieves a network namespace by process ID (PID).
 * Functional Utility: Given a process ID, this function finds the task
 * and then retrieves the network namespace it belongs to.
 *
 * @param pid The process ID.
 * @return Pointer to the `net` structure, or `ERR_PTR` on failure.
 */
struct net *get_net_ns_by_pid(pid_t pid)
{
	struct task_struct *tsk;
	struct net *net;

	/* Lookup the network namespace */
	net = ERR_PTR(-ESRCH); ///< Default error if task not found.
	rcu_read_lock(); ///< Acquire RCU read lock.
	tsk = find_task_by_vpid(pid); ///< Find task by virtual PID.
	if (tsk) { ///< Block Logic: If task found.
		struct nsproxy *nsproxy;
		task_lock(tsk); ///< Lock task structure.
		nsproxy = tsk->nsproxy; ///< Get namespace proxy.
		if (nsproxy)
			net = get_net(nsproxy->net_ns); ///< Get reference to network namespace.
		task_unlock(tsk); ///< Unlock task structure.
	}
	rcu_read_unlock(); ///< Release RCU read lock.
	return net;
}
EXPORT_SYMBOL_GPL(get_net_ns_by_pid); ///< Export this symbol (GPL-only).

/**
 * @brief Initializes a network namespace for pernet operations.
 * Functional Utility: Sets the `proc_ns_operations` for the network namespace
 * and assigns its initial inode number. For `init_net`, it uses a predefined
 * inode number; for others, it allocates a new one.
 *
 * @param net The network namespace to initialize.
 * @return 0 on success, or a negative errno on failure.
 */
static __net_init int net_ns_net_init(struct net *net)
{
#ifdef CONFIG_NET_NS
	net->ns.ops = &netns_operations; ///< Assign network namespace operations.
#endif
	if (net == &init_net) { ///< Block Logic: If it's the initial network namespace.
		net->ns.inum = PROC_NET_INIT_INO; ///< Assign predefined inode number.
		return 0;
	}
	return ns_alloc_inum(&net->ns); ///< Allocate new inode number for non-initial namespaces.
}

/**
 * @brief Exits (cleans up) a network namespace for pernet operations.
 * Functional Utility: Frees the inode number associated with a network namespace
 * during its destruction. This is part of the `pernet_operations` cleanup.
 *
 * @param net The network namespace to exit.
 */
static __net_exit void net_ns_net_exit(struct net *net)
{
	/*
	 * Initial network namespace doesn't exit so we don't need any
	 * special checks here.
	 */
	ns_free_inum(&net->ns); ///< Free the inode number.
}

/**
 * @brief `pernet_operations` structure for network namespaces.
 * Functional Utility: Defines the initialization and exit callbacks for the
 * core network namespace functionality, integrating it into the per-net
 * operations framework.
 */
static struct pernet_operations __net_initdata net_ns_ops = {
	.init = net_ns_net_init, ///< Initialization function.
	.exit = net_ns_net_exit, ///< Exit function.
};

/**
 * @brief Netlink attribute policy for `rtnl_net` messages.
 * Functional Utility: Defines the expected type and structure of Netlink
 * attributes when handling `rtnl_net` messages for network namespace IDs.
 */
static const struct nla_policy rtnl_net_policy[NETNSA_MAX + 1] __initconst = {
	[NETNSA_NONE]		= { .type = NLA_UNSPEC },
	[NETNSA_NSID]		= { .type = NLA_S32 },
	[NETNSA_PID]		= { .type = NLA_U32 },
	[NETNSA_FD]		= { .type = NLA_U32 },
	[NETNSA_TARGET_NSID]	= { .type = NLA_S32 },
};

/**
 * @brief Handles `RTM_NEWNSID` Netlink messages to assign new nsids.
 * Functional Utility: Processes Netlink requests from userspace to assign
 * a network namespace ID (`nsid`) to a peer network namespace. It validates
 * the request and either assigns a new ID or confirms an existing one.
 *
 * @param skb The socket buffer containing the Netlink message.
 * @param nlh The Netlink message header.
 * @param extack Extended Netlink error acknowledgment.
 * @return 0 on success, or a negative errno on failure.
 */
static int rtnl_net_newid(struct sk_buff *skb, struct nlmsghdr *nlh,
			  struct netlink_ext_ack *extack)
{
	struct net *net = sock_net(skb->sk); ///< Get the network namespace of the sender.
	struct nlattr *tb[NETNSA_MAX + 1]; ///< Netlink attribute buffer.
	struct nlattr *nla;
	struct net *peer;
	int nsid, err;

	err = nlmsg_parse_deprecated(nlh, sizeof(struct rtgenmsg), tb,
				     NETNSA_MAX, rtnl_net_policy, extack); ///< Parse Netlink attributes.
	if (err < 0)
		return err;
	if (!tb[NETNSA_NSID]) { ///< Block Logic: `nsid` attribute is mandatory.
		NL_SET_ERR_MSG(extack, "nsid is missing");
		return -EINVAL;
	}
	nsid = nla_get_s32(tb[NETNSA_NSID]); ///< Get requested `nsid`.

	if (tb[NETNSA_PID]) { ///< Block Logic: Peer identified by PID.
		peer = get_net_ns_by_pid(nla_get_u32(tb[NETNSA_PID]));
		nla = tb[NETNSA_PID];
	} else if (tb[NETNSA_FD]) { ///< Block Logic: Peer identified by file descriptor.
		peer = get_net_ns_by_fd(nla_get_u32(tb[NETNSA_FD]));
		nla = tb[NETNSA_FD];
	} else { ///< Block Logic: Peer reference is missing.
		NL_SET_ERR_MSG(extack, "Peer netns reference is missing");
		return -EINVAL;
	}
	if (IS_ERR(peer)) { ///< Block Logic: Check if peer lookup failed.
		NL_SET_BAD_ATTR(extack, nla);
		NL_SET_ERR_MSG(extack, "Peer netns reference is invalid");
		return PTR_ERR(peer);
	}

	spin_lock_bh(&net->nsid_lock); ///< Acquire spinlock.
	if (__peernet2id(net, peer) >= 0) { ///< Block Logic: Check if ID already exists.
		spin_unlock_bh(&net->nsid_lock);
		err = -EEXIST;
		NL_SET_BAD_ATTR(extack, nla);
		NL_SET_ERR_MSG(extack,
			       "Peer netns already has a nsid assigned");
		goto out;
	}

	err = alloc_netid(net, peer, nsid); ///< Allocate new ID.
	spin_unlock_bh(&net->nsid_lock); ///< Release spinlock.
	if (err >= 0) { ///< Block Logic: If ID allocation successful.
		rtnl_net_notifyid(net, RTM_NEWNSID, err, NETLINK_CB(skb).portid,
				  nlh, GFP_KERNEL); ///< Notify userspace.
		err = 0;
	} else if (err == -ENOSPC && nsid >= 0) { ///< Block Logic: If specific nsid requested but not available.
		err = -EEXIST;
		NL_SET_BAD_ATTR(extack, tb[NETNSA_NSID]);
		NL_SET_ERR_MSG(extack, "The specified nsid is already used");
	}
out:
	put_net(peer); ///< Release reference to peer network namespace.
	return err;
}

/**
 * @brief Calculates the size needed for a `rtnl_net_getid` Netlink message.
 * Functional Utility: Determines the buffer size required to serialize a
 * Netlink message containing network namespace ID information.
 *
 * @return The calculated message size.
 */
static int rtnl_net_get_size(void)
{
	return NLMSG_ALIGN(sizeof(struct rtgenmsg)) ///< Size of generic Netlink header.
	       + nla_total_size(sizeof(s32)) /* NETNSA_NSID */ ///< Size for NSID attribute.
	       + nla_total_size(sizeof(s32)) /* NETNSA_CURRENT_NSID */ ///< Size for CURRENT_NSID attribute.
	       ;
}

/**
 * @struct net_fill_args
 * @brief Arguments structure for `rtnl_net_fill`.
 * Functional Utility: Stores parameters needed to construct a Netlink message
 * for network namespace ID notifications.
 */
struct net_fill_args {
	u32 portid;
	u32 seq;
	int flags;
	int cmd;
	int nsid;
	bool add_ref;
	int ref_nsid;
};

/**
 * @brief Fills a Netlink message with network namespace ID information.
 * Functional Utility: Constructs a Netlink message containing a `rtgenmsg`
 * header and `NETNSA_NSID` and optionally `NETNSA_CURRENT_NSID` attributes
 * for network namespace ID notifications.
 *
 * @param skb The socket buffer to fill.
 * @param args Pointer to `net_fill_args` containing message parameters.
 * @return 0 on success, or `-EMSGSIZE` if message too large.
 */
static int rtnl_net_fill(struct sk_buff *skb, struct net_fill_args *args)
{
	struct nlmsghdr *nlh;
	struct rtgenmsg *rth;

	nlh = nlmsg_put(skb, args->portid, args->seq, args->cmd, sizeof(*rth),
			args->flags); ///< Put Netlink message header.
	if (!nlh)
		return -EMSGSIZE;

	rth = nlmsg_data(nlh); ///< Get generic Netlink data area.
	rth->rtgen_family = AF_UNSPEC; ///< Set address family to unspecified.

	if (nla_put_s32(skb, NETNSA_NSID, args->nsid)) ///< Put NSID attribute.
		goto nla_put_failure;

	if (args->add_ref && ///< Block Logic: If `add_ref` is true, put `CURRENT_NSID`.
	    nla_put_s32(skb, NETNSA_CURRENT_NSID, args->ref_nsid))
		goto nla_put_failure;

	nlmsg_end(skb, nlh); ///< Finalize Netlink message.
	return 0;

nla_put_failure: ///< Error handling: cancel message if attribute putting fails.
	nlmsg_cancel(skb, nlh);
	return -EMSGSIZE;
}

/**
 * @brief Validates a Netlink `RTM_GETNSID` request.
 * Functional Utility: Parses and validates the attributes of a Netlink
 * `RTM_GETNSID` request, ensuring that only supported attributes are present.
 *
 * @param skb The socket buffer.
 * @param nlh The Netlink message header.
 * @param tb Netlink attribute buffer.
 * @param extack Extended Netlink error acknowledgment.
 * @return 0 on success, or a negative errno on failure.
 */
static int rtnl_net_valid_getid_req(struct sk_buff *skb,
				    const struct nlmsghdr *nlh,
				    struct nlattr **tb,
				    struct netlink_ext_ack *extack)
{
	int i, err;

	if (!netlink_strict_get_check(skb)) ///< Block Logic: If strict checking is not enabled.
		return nlmsg_parse_deprecated(nlh, sizeof(struct rtgenmsg),
					      tb, NETNSA_MAX, rtnl_net_policy,
					      extack); ///< Parse without strictness.

	err = nlmsg_parse_deprecated_strict(nlh, sizeof(struct rtgenmsg), tb,
					    NETNSA_MAX, rtnl_net_policy,
					    extack); ///< Parse with strictness.
	if (err)
		return err;

	for (i = 0; i <= NETNSA_MAX; i++) { ///< Block Logic: Iterate through attributes for validation.
		if (!tb[i])
			continue;

		switch (i) { ///< Block Logic: Check allowed attributes.
		case NETNSA_PID:
		case NETNSA_FD:
		case NETNSA_NSID:
		case NETNSA_TARGET_NSID:
			break;
		default:
			NL_SET_ERR_MSG(extack, "Unsupported attribute in peer netns getid request");
			return -EINVAL; ///< Return error for unsupported attribute.
		}
	}

	return 0;
}

/**
 * @brief Handles `RTM_GETNSID` Netlink messages to retrieve nsids.
 * Functional Utility: Processes Netlink requests from userspace to retrieve
 * the network namespace ID (`nsid`) for a peer network namespace, identified
 * by PID, file descriptor, or an existing nsid. It constructs and sends a
 * Netlink response.
 *
 * @param skb The socket buffer containing the Netlink message.
 * @param nlh The Netlink message header.
 * @param extack Extended Netlink error acknowledgment.
 * @return 0 on success, or a negative errno on failure.
 */
static int rtnl_net_getid(struct sk_buff *skb, struct nlmsghdr *nlh,
			  struct netlink_ext_ack *extack)
{
	struct net *net = sock_net(skb->sk); ///< Get the network namespace of the sender.
	struct nlattr *tb[NETNSA_MAX + 1]; ///< Netlink attribute buffer.
	struct net_fill_args fillargs = { ///< Arguments for filling Netlink message.
		.portid = NETLINK_CB(skb).portid,
		.seq = nlh->nlmsg_seq,
		.cmd = RTM_NEWNSID,
	};
	struct net *peer, *target = net; ///< Peer and target network namespaces.
	struct nlattr *nla;
	struct sk_buff *msg;
	int err;

	err = rtnl_net_valid_getid_req(skb, nlh, tb, extack); ///< Validate the request.
	if (err < 0)
		return err;
	if (tb[NETNSA_PID]) { ///< Block Logic: Peer identified by PID.
		peer = get_net_ns_by_pid(nla_get_u32(tb[NETNSA_PID]));
		nla = tb[NETNSA_PID];
	} else if (tb[NETNSA_FD]) { ///< Block Logic: Peer identified by file descriptor.
		peer = get_net_ns_by_fd(nla_get_u32(tb[NETNSA_FD]));
		nla = tb[NETNSA_FD];
	} else if (tb[NETNSA_NSID]) { ///< Block Logic: Peer identified by existing nsid.
		peer = get_net_ns_by_id(net, nla_get_s32(tb[NETNSA_NSID]));
		if (!peer)
			peer = ERR_PTR(-ENOENT);
		nla = tb[NETNSA_NSID];
	} else { ///< Block Logic: Peer reference is missing.
		NL_SET_ERR_MSG(extack, "Peer netns reference is missing");
		return -EINVAL;
	}

	if (IS_ERR(peer)) { ///< Block Logic: Check if peer lookup failed.
		NL_SET_BAD_ATTR(extack, nla);
		NL_SET_ERR_MSG(extack, "Peer netns reference is invalid");
		return PTR_ERR(peer);
	}

	if (tb[NETNSA_TARGET_NSID]) { ///< Block Logic: If a target network namespace is specified.
		int id = nla_get_s32(tb[NETNSA_TARGET_NSID]);

		target = rtnl_get_net_ns_capable(NETLINK_CB(skb).sk, id); ///< Get target netns.
		if (IS_ERR(target)) {
			NL_SET_BAD_ATTR(extack, tb[NETNSA_TARGET_NSID]);
			NL_SET_ERR_MSG(extack,
				       "Target netns reference is invalid");
			err = PTR_ERR(target);
			goto out;
		}
		fillargs.add_ref = true; ///< Add reference to target.
		fillargs.ref_nsid = peernet2id(net, peer); ///< Get reference nsid.
	}

	msg = nlmsg_new(rtnl_net_get_size(), GFP_KERNEL); ///< Allocate new Netlink message.
	if (!msg) {
		err = -ENOMEM;
		goto out;
	}

	fillargs.nsid = peernet2id(target, peer); ///< Get nsid of peer in target context.
	err = rtnl_net_fill(msg, &fillargs); ///< Fill the message.
	if (err < 0)
		goto err_out;

	err = rtnl_unicast(msg, net, NETLINK_CB(skb).portid); ///< Send unicast response.
	goto out;

err_out: ///< Error handling: free message.
	nlmsg_free(msg);
out:
	if (fillargs.add_ref)
		put_net(target); ///< Put reference to target.
	put_net(peer); ///< Put reference to peer.
	return err;
}

/**
 * @struct rtnl_net_dump_cb
 * @brief Callback structure for dumping network ID information.
 * Functional Utility: Stores context and arguments for the Netlink dump
 * function `rtnl_net_dumpid_one`, allowing it to build a multi-part
 * Netlink message.
 */
struct rtnl_net_dump_cb {
	struct net *tgt_net; ///< Target network namespace.
	struct net *ref_net; ///< Reference network namespace.
	struct sk_buff *skb; ///< Socket buffer for the message.
	struct net_fill_args fillargs; ///< Arguments for filling Netlink message.
	int idx; ///< Current index in iteration.
	int s_idx; ///< Starting index for dump.
};

/* Runs in RCU-critical section. */
/**
 * @brief Callback for `idr_for_each` to dump network ID information.
 * Functional Utility: This function is called for each ID in an IDR tree
 * during a Netlink dump request. It constructs a Netlink message containing
 * the nsid and other relevant information.
 *
 * @param id The current ID being iterated.
 * @param peer The `net` structure associated with the ID.
 * @param data Pointer to `rtnl_net_dump_cb` containing dump context.
 * @return 0 to continue iteration, or a negative errno to stop.
 */
static int rtnl_net_dumpid_one(int id, void *peer, void *data)
{
	struct rtnl_net_dump_cb *net_cb = (struct rtnl_net_dump_cb *)data;
	int ret;

	if (net_cb->idx < net_cb->s_idx) ///< Block Logic: Skip already dumped entries.
		goto cont;

	net_cb->fillargs.nsid = id; ///< Set nsid in fill arguments.
	if (net_cb->fillargs.add_ref) ///< Block Logic: If `add_ref` is true, get `ref_nsid`.
		net_cb->fillargs.ref_nsid = __peernet2id(net_cb->ref_net, peer);
	ret = rtnl_net_fill(net_cb->skb, &net_cb->fillargs); ///< Fill the Netlink message.
	if (ret < 0)
		return ret;

cont:
	net_cb->idx++; ///< Increment index.
	return 0;
}

/**
 * @brief Validates a Netlink `RTM_GETNSID` dump request.
 * Functional Utility: Parses and validates the attributes of a Netlink
 * `RTM_GETNSID` dump request, ensuring only supported attributes are used
 * and setting up the dump callback context.
 *
 * @param nlh The Netlink message header.
 * @param sk The socket.
 * @param net_cb Pointer to `rtnl_net_dump_cb` to fill with context.
 * @param cb Netlink callback structure.
 * @return 0 on success, or a negative errno on failure.
 */
static int rtnl_valid_dump_net_req(const struct nlmsghdr *nlh, struct sock *sk,
				   struct rtnl_net_dump_cb *net_cb,
				   struct netlink_callback *cb)
{
	struct netlink_ext_ack *extack = cb->extack;
	struct nlattr *tb[NETNSA_MAX + 1];
	int err, i;

	err = nlmsg_parse_deprecated_strict(nlh, sizeof(struct rtgenmsg), tb,
					    NETNSA_MAX, rtnl_net_policy,
					    extack); ///< Parse Netlink attributes with strictness.
	if (err < 0)
		return err;

	for (i = 0; i <= NETNSA_MAX; i++) { ///< Block Logic: Iterate through attributes for validation.
		if (!tb[i])
			continue;

		if (i == NETNSA_TARGET_NSID) { ///< Block Logic: If `TARGET_NSID` is present.
			struct net *net;

			net = rtnl_get_net_ns_capable(sk, nla_get_s32(tb[i])); ///< Get target netns.
			if (IS_ERR(net)) {
				NL_SET_BAD_ATTR(extack, tb[i]);
				NL_SET_ERR_MSG(extack,
					       "Invalid target network namespace id");
				return PTR_ERR(net);
			}
			net_cb->fillargs.add_ref = true; ///< Set `add_ref` flag.
			net_cb->ref_net = net_cb->tgt_net; ///< Set reference network namespace.
			net_cb->tgt_net = net; ///< Set target network namespace.
		} else { ///< Block Logic: Unsupported attribute.
			NL_SET_BAD_ATTR(extack, tb[i]);
			NL_SET_ERR_MSG(extack,
				       "Unsupported attribute in dump request");
			return -EINVAL;
		}
	}

	return 0;
}

/**
 * @brief Handles `RTM_GETNSID` Netlink dump requests.
 * Functional Utility: Processes Netlink requests from userspace to dump
 * all currently assigned network namespace IDs. It iterates through the
 * IDR tree and constructs a multi-part Netlink message.
 *
 * @param skb The socket buffer containing the Netlink message.
 * @param cb Netlink callback structure for dump operations.
 * @return 0 on success, or a negative errno on failure.
 */
static int rtnl_net_dumpid(struct sk_buff *skb, struct netlink_callback *cb)
{
	struct rtnl_net_dump_cb net_cb = { ///< Initialize dump callback context.
		.tgt_net = sock_net(skb->sk),
		.skb = skb,
		.fillargs = {
			.portid = NETLINK_CB(cb->skb).portid,
			.seq = cb->nlh->nlmsg_seq,
			.flags = NLM_F_MULTI,
			.cmd = RTM_NEWNSID,
		},
		.idx = 0,
		.s_idx = cb->args[0], ///< Starting index for dump.
	};
	int err = 0;

	if (cb->strict_check) { ///< Block Logic: Perform strict validation if enabled.
		err = rtnl_valid_dump_net_req(cb->nlh, skb->sk, &net_cb, cb);
		if (err < 0)
			goto end;
	}

	rcu_read_lock(); ///< Acquire RCU read lock.
	idr_for_each(&net_cb.tgt_net->netns_ids, rtnl_net_dumpid_one, &net_cb); ///< Iterate through IDs and dump.
	rcu_read_unlock(); ///< Release RCU read lock.

	cb->args[0] = net_cb.idx; ///< Update current position for next dump iteration.
end:
	if (net_cb.fillargs.add_ref)
		put_net(net_cb.tgt_net); ///< Put reference to target.
	return err;
}

/**
 * @brief Notifies Netlink listeners about network ID changes.
 * Functional Utility: Sends a Netlink message (`RTM_NEWNSID` or `RTM_DELNSID`)
 * to interested userspace applications (e.g., `iproute2`) whenever a network
 * namespace ID is assigned or removed.
 *
 * @param net The network namespace generating the notification.
 * @param cmd The Netlink command (e.g., `RTM_NEWNSID`).
 * @param id The network ID being notified.
 * @param portid The Netlink port ID of the sender.
 * @param nlh Original Netlink message header (if a response).
 * @param gfp GFP flags for memory allocation.
 */
static void rtnl_net_notifyid(struct net *net, int cmd, int id, u32 portid,
			      struct nlmsghdr *nlh, gfp_t gfp)
{
	struct net_fill_args fillargs = { ///< Arguments for filling Netlink message.
		.portid = portid,
		.seq = nlh ? nlh->nlmsg_seq : 0,
		.cmd = cmd,
		.nsid = id,
	};
	struct sk_buff *msg;
	int err = -ENOMEM;

	msg = nlmsg_new(rtnl_net_get_size(), gfp); ///< Allocate new Netlink message.
	if (!msg)
		goto out;

	err = rtnl_net_fill(msg, &fillargs); ///< Fill the message.
	if (err < 0)
		goto err_out;

	rtnl_notify(msg, net, portid, RTNLGRP_NSID, nlh, gfp); ///< Send Netlink notification.
	return;

err_out: ///< Error handling: free message.
	nlmsg_free(msg);
out:
	rtnl_set_sk_err(net, RTNLGRP_NSID, err); ///< Set socket error for Netlink group.
}

#ifdef CONFIG_NET_NS ///< Conditional compilation: Only if network namespaces are enabled.
/**
 * @brief Performs cacheline assertions for `netns_ipv4` structure.
 * Functional Utility: Compile-time (or early init) checks to ensure that
 * frequently accessed fields within the `netns_ipv4` structure are correctly
 * placed on cache lines to optimize performance. This helps prevent false sharing.
 */
static void __init netns_ipv4_struct_check(void)
{
	/* TX readonly hotpath cache lines */
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_early_retrans);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_tso_win_divisor);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_tso_rtt_log);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_autocorking);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_min_snd_mss);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_notsent_lowat);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_limit_output_bytes);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_min_rtt_wlen);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_tcp_wmem);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_tx,
				      sysctl_ip_fwd_use_pmtu);
	CACHELINE_ASSERT_GROUP_SIZE(struct netns_ipv4, netns_ipv4_read_tx, 33);

	/* TXRX readonly hotpath cache lines */
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_txrx,
				      sysctl_tcp_moderate_rcvbuf);
	CACHELINE_ASSERT_GROUP_SIZE(struct netns_ipv4, netns_ipv4_read_txrx, 1);

	/* RX readonly hotpath cache line */
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_rx,
				      sysctl_ip_early_demux);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_rx,
				      sysctl_tcp_early_demux);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_rx,
				      sysctl_tcp_l3mdev_accept);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_rx,
				      sysctl_tcp_reordering);
	CACHELINE_ASSERT_GROUP_MEMBER(struct netns_ipv4, netns_ipv4_read_rx,
				      sysctl_tcp_rmem);
	CACHELINE_ASSERT_GROUP_SIZE(struct netns_ipv4, netns_ipv4_read_rx, 22);
}
#endif

/**
 * @brief Netlink message handlers for network namespace ID management.
 * Functional Utility: Defines the set of `rtnl_msg_handler` structures
 * for `RTM_NEWNSID` and `RTM_GETNSID` Netlink messages, specifying which
 * functions (`rtnl_net_newid`, `rtnl_net_getid`) should handle these requests.
 */
static const struct rtnl_msg_handler net_ns_rtnl_msg_handlers[] __initconst = {
	{.msgtype = RTM_NEWNSID, .doit = rtnl_net_newid, ///< Handler for new nsid requests.
	 .flags = RTNL_FLAG_DOIT_UNLOCKED},
	{.msgtype = RTM_GETNSID, .doit = rtnl_net_getid, ///< Handler for get nsid requests.
	 .dumpit = rtnl_net_dumpid, ///< Dump function for nsid.
	 .flags = RTNL_FLAG_DOIT_UNLOCKED | RTNL_FLAG_DUMP_UNLOCKED},
};

/**
 * @brief Initializes the network namespace subsystem.
 * Functional Utility: This is the main initialization function for the network
 * namespace core. It sets up internal caches, workqueues, pre-initializes the
 * `init_net` (initial network namespace), registers pernet operations, and
 * registers Netlink message handlers for nsid management.
 */
void __init net_ns_init(void)
{
	struct net_generic *ng;

#ifdef CONFIG_NET_NS ///< Block Logic: Only if network namespaces are enabled.
	netns_ipv4_struct_check(); ///< Perform cacheline checks for IPv4 struct.
	net_cachep = kmem_cache_create("net_namespace", sizeof(struct net),
					SMP_CACHE_BYTES,
					SLAB_PANIC|SLAB_ACCOUNT, NULL); ///< Create slab cache for `net` structures.

	/* Create workqueue for cleanup */
	netns_wq = create_singlethread_workqueue("netns"); ///< Create workqueue for network namespace cleanup.
	if (!netns_wq)
		panic("Could not create netns workq"); ///< Panic if workqueue creation fails.
#endif

	ng = net_alloc_generic(); ///< Allocate generic per-net data for `init_net`.
	if (!ng)
		panic("Could not allocate generic netns"); ///< Panic if allocation fails.

	rcu_assign_pointer(init_net.gen, ng); ///< Assign generic data to `init_net`, RCU-protected.

#ifdef CONFIG_KEYS ///< Block Logic: If CONFIG_KEYS is enabled, assign key domain.
	init_net.key_domain = &init_net_key_domain;
#endif
	preinit_net(&init_net, &init_user_ns); ///< Pre-initialize the initial network namespace.

	down_write(&pernet_ops_rwsem); ///< Acquire write lock for pernet operations.
	if (setup_net(&init_net)) ///< Set up the initial network namespace.
		panic("Could not setup the initial network namespace"); ///< Panic if setup fails.

	init_net_initialized = true; ///< Mark initial network namespace as initialized.
	up_write(&pernet_ops_rwsem); ///< Release write lock.

	if (register_pernet_subsys(&net_ns_ops)) ///< Register network namespace pernet operations.
		panic("Could not register network namespace subsystems"); ///< Panic if registration fails.

	rtnl_register_many(net_ns_rtnl_msg_handlers); ///< Register Netlink message handlers.
}

#ifdef CONFIG_NET_NS ///< Conditional compilation: Only if network namespaces are enabled.
/**
 * @brief Registers pernet operations, initializing them for existing network namespaces.
 * Functional Utility: Adds a new set of `pernet_operations` to the `pernet_list`
 * and then calls its `init` function for all currently existing network namespaces.
 * This ensures that newly registered subsystems are properly initialized in all active
 * network environments.
 *
 * @param list The list to add the operations to.
 * @param ops Pointer to the `pernet_operations` structure to register.
 * @return 0 on success, or a negative errno on failure.
 */
static int __register_pernet_operations(struct list_head *list,
					struct pernet_operations *ops)
{
	LIST_HEAD(net_exit_list); ///< Temporary list for net exit.
	struct net *net;
	int error;

	list_add_tail(&ops->list, list); ///< Add operation to the list.
	if (ops->init || ops->id) { ///< Block Logic: If init function or ID is provided.
		/* We held write locked pernet_ops_rwsem, and parallel
		 * setup_net() and cleanup_net() are not possible.
		 */
		for_each_net(net) { ///< Block Logic: Iterate through existing network namespaces.
			error = ops_init(ops, net); ///< Initialize operation for current netns.
			if (error)
				goto out_undo; ///< If error, jump to undo.
			list_add_tail(&net->exit_list, &net_exit_list); ///< Add netns to exit list for potential undo.
		}
	}
	return 0;

out_undo: ///< Error handling: undo already initialized operations.
	/* If I have an error cleanup all namespaces I initialized */
	list_del(&ops->list); ///< Remove operation from list.
	ops_undo_single(ops, &net_exit_list); ///< Undo operations for processed netns.
	return error;
}

/**
 * @brief Unregisters pernet operations, exiting them for existing network namespaces.
 * Functional Utility: Removes a set of `pernet_operations` from the `pernet_list`
 * and then calls its `exit` and `free` functions for all currently existing network namespaces.
 * This ensures proper cleanup when a subsystem is unregistered.
 *
 * @param ops Pointer to the `pernet_operations` structure to unregister.
 */
static void __unregister_pernet_operations(struct pernet_operations *ops)
{
	LIST_HEAD(net_exit_list); ///< Temporary list for net exit.
	struct net *net;

	/* See comment in __register_pernet_operations() */
	for_each_net(net) ///< Block Logic: Collect all existing network namespaces.
		list_add_tail(&net->exit_list, &net_exit_list);

	list_del(&ops->list); ///< Remove operation from list.
	ops_undo_single(ops, &net_exit_list); ///< Undo operations for collected netns.
}

#else ///< Conditional compilation: If network namespaces are NOT enabled.

/**
 * @brief Registers pernet operations (stub for non-NS build).
 * Functional Utility: Simplified registration when network namespaces are disabled,
 * only initializing for `init_net` if already initialized.
 *
 * @param list The list to add the operations to.
 * @param ops Pointer to the `pernet_operations` structure.
 * @return 0 on success, or an error from `ops_init`.
 */
static int __register_pernet_operations(struct list_head *list,
					struct pernet_operations *ops)
{
	if (!init_net_initialized) { ///< Block Logic: If `init_net` not yet initialized.
		list_add_tail(&ops->list, list); ///< Just add to list.
		return 0;
	}

	return ops_init(ops, &init_net); ///< Initialize for `init_net`.
}

/**
 * @brief Unregisters pernet operations (stub for non-NS build).
 * Functional Utility: Simplified unregistration when network namespaces are disabled,
 * only undoing operations for `init_net` if already initialized.
 *
 * @param ops Pointer to the `pernet_operations` structure.
 */
static void __unregister_pernet_operations(struct pernet_operations *ops)
{
	if (!init_net_initialized) { ///< Block Logic: If `init_net` not yet initialized.
		list_del(&ops->list); ///< Just remove from list.
	} else { ///< Block Logic: If `init_net` is initialized.
		LIST_HEAD(net_exit_list);

		list_add(&init_net.exit_list, &net_exit_list); ///< Add `init_net` to exit list.
		ops_undo_single(ops, &net_exit_list); ///< Undo operations for `init_net`.
	}
}

#endif /* CONFIG_NET_NS */

static DEFINE_IDA(net_generic_ids); ///< ID allocator for generic per-net IDs.

/**
 * @brief Registers pernet operations.
 * Functional Utility: Centralized registration function for all `pernet_operations`.
 * It allocates a unique ID if needed, updates the maximum generic pointer count,
 * and calls `__register_pernet_operations` to initialize the operation across
 * existing network namespaces.
 *
 * @param list The list to add the operations to.
 * @param ops Pointer to the `pernet_operations` structure to register.
 * @return 0 on success, or a negative errno on failure.
 */
static int register_pernet_operations(struct list_head *list,
				      struct pernet_operations *ops)
{
	int error;

	if (WARN_ON(!!ops->id ^ !!ops->size)) ///< Functional Utility: Ensure `id` and `size` are consistently set or unset.
		return -EINVAL;

	if (ops->id) { ///< Block Logic: If operation requires a generic ID.
		error = ida_alloc_min(&net_generic_ids, MIN_PERNET_OPS_ID,
				GFP_KERNEL); ///< Allocate ID from IDR.
		if (error < 0)
			return error;
		*ops->id = error; ///< Assign allocated ID.
		/* This does not require READ_ONCE as writers already hold
		 * pernet_ops_rwsem. But WRITE_ONCE is needed to protect
		 * net_alloc_generic.
		 */
		WRITE_ONCE(max_gen_ptrs, max(max_gen_ptrs, *ops->id + 1)); ///< Update max generic pointers count.
	}
	error = __register_pernet_operations(list, ops); ///< Call internal registration function.
	if (error) { ///< Block Logic: On error, clean up allocated ID.
		rcu_barrier(); ///< Wait for RCU grace period.
		if (ops->id)
			ida_free(&net_generic_ids, *ops->id); ///< Free allocated ID.
	}

	return error;
}

/**
 * @brief Unregisters pernet operations.
 * Functional Utility: Centralized unregistration function for all `pernet_operations`.
 * It calls `__unregister_pernet_operations` to clean up the operation across existing
 * network namespaces and frees the allocated generic ID.
 *
 * @param ops Pointer to the `pernet_operations` structure to unregister.
 */
static void unregister_pernet_operations(struct pernet_operations *ops)
{
	__unregister_pernet_operations(ops); ///< Call internal unregistration function.
	rcu_barrier(); ///< Wait for RCU grace period.
	if (ops->id)
		ida_free(&net_generic_ids, *ops->id); ///< Free allocated ID.
}

/**
 *      register_pernet_subsys - register a network namespace subsystem
 *	@ops:  pernet operations structure for the subsystem
 *
 *	Register a subsystem which has init and exit functions
 *	that are called when network namespaces are created and
 *	destroyed respectively.
 *
 *	When registered all network namespace init functions are
 *	called for every existing network namespace.  Allowing kernel
 *	modules to have a race free view of the set of network namespaces.
 *
 *	When a new network namespace is created all of the init
 *	methods are called in the order in which they were registered.
 *
 *	When a network namespace is destroyed all of the exit methods
 *	are called in the reverse of the order with which they were
 *	registered.
 */
int register_pernet_subsys(struct pernet_operations *ops)
{
	int error;
	down_write(&pernet_ops_rwsem); ///< Acquire write lock to protect `pernet_list`.
	error =  register_pernet_operations(first_device, ops); ///< Register operations.
	up_write(&pernet_ops_rwsem); ///< Release write lock.
	return error;
}
EXPORT_SYMBOL_GPL(register_pernet_subsys); ///< Export this symbol (GPL-only).

/**
 *      unregister_pernet_subsys - unregister a network namespace subsystem
 *	@ops: pernet operations structure to manipulate
 *
 *	Remove the pernet operations structure from the list to be
 *	used when network namespaces are created or destroyed.  In
 *	addition run the exit method for all existing network
 *	namespaces.
 */
void unregister_pernet_subsys(struct pernet_operations *ops)
{
	down_write(&pernet_ops_rwsem); ///< Acquire write lock.
	unregister_pernet_operations(ops); ///< Unregister operations.
	up_write(&pernet_ops_rwsem); ///< Release write lock.
}
EXPORT_SYMBOL_GPL(unregister_pernet_subsys); ///< Export this symbol (GPL-only).

/**
 *      register_pernet_device - register a network namespace device
 *	@ops:  pernet operations structure for the subsystem
 *
 *	Register a device which has init and exit functions
 *	that are called when network namespaces are created and
 *	destroyed respectively.
 *
 *	When registered all network namespace init functions are
 *	called for every existing network namespace.  Allowing kernel
 *	modules to have a race free view of the set of network namespaces.
 *
 *	When a new network namespace is created all of the init
 *	methods are called in the order in which they were registered.
 *
 *	When a network namespace is destroyed all of the exit methods
 *	are called in the reverse of the order with which they were
 *	registered.
 */
int register_pernet_device(struct pernet_operations *ops)
{
	int error;
	down_write(&pernet_ops_rwsem); ///< Acquire write lock.
	error = register_pernet_operations(&pernet_list, ops); ///< Register operations.
	if (!error && (first_device == &pernet_list)) ///< Block Logic: Update `first_device` if newly registered is first.
		first_device = &ops->list;
	up_write(&pernet_ops_rwsem); ///< Release write lock.
	return error;
}
EXPORT_SYMBOL_GPL(register_pernet_device); ///< Export this symbol (GPL-only).

/**
 *      unregister_pernet_device - unregister a network namespace netdevice
 *	@ops: pernet operations structure to manipulate
 *
 *	Remove the pernet operations structure from the list to be
 *	used when network namespaces are created or destroyed.  In
 *	addition run the exit method for all existing network
 *	namespaces.
 */
void unregister_pernet_device(struct pernet_operations *ops)
{
	down_write(&pernet_ops_rwsem); ///< Acquire write lock.
	if (&ops->list == first_device) ///< Block Logic: Update `first_device` if unregistering the current first.
		first_device = first_device->next;
	unregister_pernet_operations(ops); ///< Unregister operations.
	up_write(&pernet_ops_rwsem); ///< Release write lock.
}
EXPORT_SYMBOL_GPL(unregister_pernet_device); ///< Export this symbol (GPL-only).

#ifdef CONFIG_NET_NS ///< Conditional compilation: Only if network namespaces are enabled.
/**
 * @brief Retrieves the `ns_common` for a task's network namespace.
 * Functional Utility: Callback function for `proc_ns_operations` to get a
 * reference to the network namespace associated with a given `task_struct`.
 *
 * @param task The `task_struct` to query.
 * @return Pointer to the `ns_common` of the network namespace, or `NULL`.
 */
static struct ns_common *netns_get(struct task_struct *task)
{
	struct net *net = NULL;
	struct nsproxy *nsproxy;

	task_lock(task); ///< Lock task structure.
	nsproxy = task->nsproxy; ///< Get namespace proxy.
	if (nsproxy)
		net = get_net(nsproxy->net_ns); ///< Get reference to network namespace.
	task_unlock(task); ///< Unlock task structure.

	return net ? &net->ns : NULL; ///< Return `ns_common` if valid network namespace.
}

/**
 * @brief Converts an `ns_common` pointer to a `net` pointer.
 * Functional Utility: A type-safe way to cast a generic `ns_common` pointer
 * back to its specific `net` structure, relying on the `container_of` macro.
 *
 * @param ns Pointer to the `ns_common` structure.
 * @return Pointer to the `net` structure.
 */
static inline struct net *to_net_ns(struct ns_common *ns)
{
	return container_of(ns, struct net, ns); ///< Use `container_of` for type conversion.
}

/**
 * @brief Puts (decrements reference count) a network namespace.
 * Functional Utility: Callback function for `proc_ns_operations` to release
 * a reference to a network namespace.
 *
 * @param ns Common namespace structure.
 */
static void netns_put(struct ns_common *ns)
{
	put_net(to_net_ns(ns)); ///< Put network namespace reference.
}

/**
 * @brief Installs a network namespace for a process.
 * Functional Utility: Callback function for `proc_ns_operations` to install
 * a specified network namespace into the `nsproxy` of a task (represented by `nsset`).
 * It performs capability checks.
 *
 * @param nsset Pointer to `nsset` containing task's namespace information.
 * @param ns Common namespace structure to install.
 * @return 0 on success, or `-EPERM` for permission errors.
 */
static int netns_install(struct nsset *nsset, struct ns_common *ns)
{
	struct nsproxy *nsproxy = nsset->nsproxy;
	struct net *net = to_net_ns(ns);

	if (!ns_capable(net->user_ns, CAP_SYS_ADMIN) || ///< Functional Utility: Check capabilities in the target user namespace.
	    !ns_capable(nsset->cred->user_ns, CAP_SYS_ADMIN)) ///< Functional Utility: Check capabilities in the current user namespace.
		return -EPERM;

	put_net(nsproxy->net_ns); ///< Put reference to old network namespace.
	nsproxy->net_ns = get_net(net); ///< Get reference to new network namespace and assign.
	return 0;
}

/**
 * @brief Retrieves the owner user namespace of a network namespace.
 * Functional Utility: Callback function for `proc_ns_operations` to get
 * the `user_namespace` that owns a given network namespace.
 *
 * @param ns Common namespace structure.
 * @return Pointer to the `user_namespace` structure.
 */
static struct user_namespace *netns_owner(struct ns_common *ns)
{
	return to_net_ns(ns)->user_ns; ///< Return the owning user namespace.
}

/**
 * @brief `proc_ns_operations` structure for network namespaces.
 * Functional Utility: Defines the set of callbacks for the Netlink subsystem
 * to expose network namespace information and allow manipulation through
 * `/proc/pid/ns/net` entries.
 */
const struct proc_ns_operations netns_operations = {
	.name		= "net", ///< Name of the namespace.
	.type		= CLONE_NEWNET, ///< Type of namespace (clone flag).
	.get		= netns_get, ///< Get operation.
	.put		= netns_put, ///< Put operation.
	.install	= netns_install, ///< Install operation.
	.owner		= netns_owner, ///< Owner operation.
};
#endif