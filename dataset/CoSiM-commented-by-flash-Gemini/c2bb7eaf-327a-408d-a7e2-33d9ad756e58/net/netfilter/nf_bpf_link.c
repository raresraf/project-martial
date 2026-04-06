/**
 * @file nf_bpf_link.c
 * @brief Netfilter BPF Link Implementation for the Linux Kernel.
 *
 * This file implements the core logic for integrating BPF (Berkeley Packet Filter)
 * programs with the Netfilter framework within the Linux kernel. It provides a
 * mechanism for attaching BPF programs to Netfilter hooks, enabling highly flexible
 * and programmable packet filtering, modification, and inspection capabilities.
 *
 * Key functionalities include:
 * - Registering BPF programs as Netfilter hooks to process `sk_buff` (socket buffer)
 *   data at various points in the network stack.
 * - Managing the lifecycle of these BPF Netfilter links, including creation,
 *   attachment, detachment, and resource cleanup.
 * - Handling dynamic loading and management of IP defragmentation hooks,
 *   which can be conditionally enabled for BPF programs operating on fragmented packets.
 * - Providing necessary BPF verifier operations to ensure the safety and
 *   correctness of BPF programs interacting with Netfilter contexts.
 *
 * This integration is crucial for advanced networking use cases, offering a powerful
 * and efficient way to extend Netfilter's capabilities without requiring kernel module
 * modifications for each new filtering or processing logic.
 *
 * @note This code operates within the kernel space and utilizes specific kernel
 *   APIs and concepts such as Netfilter hook infrastructure, BPF program loading
 *   and execution, RCU (Read-Copy Update) for concurrent data access, module
 *   reference counting, and network namespace isolation.
 */
// SPDX-License-Identifier: GPL-2.0
#include <linux/bpf.h>
#include <linux/filter.h>
#include <linux/kmod.h>
#include <linux/module.h>
#include <linux/netfilter.h>

#include <net/netfilter/nf_bpf_link.h>
#include <uapi/linux/netfilter_ipv4.h>

/**
 * @brief Netfilter hook callback function for BPF programs.
 *
 * This function serves as the entry point for BPF programs attached to Netfilter hooks.
 * When a packet triggers a Netfilter hook associated with a BPF program, this function
 * is invoked to execute the BPF program.
 *
 * @param bpf_prog A pointer to the `bpf_prog` structure representing the compiled BPF program.
 * @param skb A pointer to the `sk_buff` (socket buffer) that contains the network packet data.
 * @param s A pointer to the `nf_hook_state` structure, providing context about the Netfilter hook.
 * @return The verdict from the BPF program execution, typically an `NF_ACCEPT`, `NF_DROP`, etc.
 *
 * @details
 * The function first casts the generic `bpf_prog` pointer to `const struct bpf_prog *`.
 * It then populates a `bpf_nf_ctx` context structure with the `nf_hook_state` and `sk_buff`.
 * Finally, it executes the BPF program using `bpf_prog_run`, passing the BPF program and
 * the prepared context. The return value of the BPF program determines the Netfilter verdict.
 */
static unsigned int nf_hook_run_bpf(void *bpf_prog, struct sk_buff *skb,
				    const struct nf_hook_state *s)
{
	const struct bpf_prog *prog = bpf_prog;
	struct bpf_nf_ctx ctx = {
		.state = s,
		.skb = skb,
	};

	return bpf_prog_run(prog, &ctx);
}

/**
 * @brief Represents a Netfilter BPF link.
 *
 * This structure extends the generic `bpf_link` to include Netfilter-specific
 * information and state for a BPF program attached to a Netfilter hook.
 */
struct bpf_nf_link {
	struct bpf_link link; /**< @brief The base BPF link object. Must be the first member. */
	struct nf_hook_ops hook_ops; /**< @brief Netfilter hook operations structure, defining the hook point and callback. */
	netns_tracker ns_tracker; /**< @brief Tracks the network namespace associated with this link. */
	struct net *net; /**< @brief Pointer to the network namespace this link operates within. */
	u32 dead; /**< @brief Flag indicating if the link is marked for detachment/release (atomic operation). */
	const struct nf_defrag_hook *defrag_hook; /**< @brief Optional IP defragmentation hook, if enabled for this link. */
};

#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV4) || IS_ENABLED(CONFIG_NF_DEFRAG_IPV6)
static const struct nf_defrag_hook *
get_proto_defrag_hook(struct bpf_nf_link *link,
		      const struct nf_defrag_hook __rcu **ptr_global_hook,
		      const char *mod)
{
	/**
	 * @brief Dynamically retrieves and manages reference counts for IP defragmentation hooks.
	 *
	 * This function is responsible for obtaining a pointer to a specific IP defragmentation hook
	 * (IPv4 or IPv6) and ensuring its associated kernel module is loaded and has its reference
	 * count incremented. It uses RCU for safe access to global hook pointers and handles
	 * dynamic module loading if the hook is not initially available.
	 *
	 * @param link A pointer to the `bpf_nf_link` structure, used to access the network namespace.
	 * @param ptr_global_hook A pointer to an RCU-protected global pointer to the `nf_defrag_hook` structure.
	 * @param mod The name of the kernel module to request if the hook is not found (e.g., "nf_defrag_ipv4").
	 * @return A pointer to the `nf_defrag_hook` on success, or an `ERR_PTR` containing an errno on failure.
	 *
	 * @details
	 * 1. **RCU Read Lock**: Initiates an RCU read-side critical section to safely dereference
	 *    the global `ptr_global_hook` without holding a lock.
	 * 2. **Hook Retrieval**: Attempts to get the `nf_defrag_hook` pointer.
	 * 3. **Module Request (if needed)**: If the hook is not found, `request_module` is called
	 *    to dynamically load the corresponding defragmentation module. RCU read lock is
	 *    re-acquired after module request.
	 * 4. **Module Reference Count**: If a hook is found, `try_module_get` is used to increment
	 *    the module's reference count. This prevents the module from being unloaded while
	 *    this link is active.
	 * 5. **RCU Pointer Handoff**: If `try_module_get` succeeds, `rcu_pointer_handoff` is used
	 *    to safely transition the RCU-protected pointer to a normal pointer, indicating
	 *    that the module's lifetime is now managed by the reference count.
	 * 6. **Error Handling**: Various error conditions are checked, such as module loading failures
	 *    or a bad hook registration.
	 * 7. **Hook Enablement**: If successful, the hook's `enable` callback is invoked to activate
	 *    defragmentation for the associated network namespace. If this fails, the module reference
	 *    is dropped, and an error is returned.
	 * 8. **RCU Read Unlock**: Releases the RCU read-side critical section.
	 *
	 * @note This function is only compiled if `CONFIG_NF_DEFRAG_IPV4` or `CONFIG_NF_DEFRAG_IPV6`
	 *   is enabled in the kernel configuration.
	 * @see nf_defrag_hook
	 */

	const struct nf_defrag_hook *hook;
	int err;

	/* RCU protects us from races against module unloading */
	rcu_read_lock();
	hook = rcu_dereference(*ptr_global_hook);
	if (!hook) {
		rcu_read_unlock();
		err = request_module("%s", mod);
		if (err)
			return ERR_PTR(err < 0 ? err : -EINVAL);

		rcu_read_lock();
		hook = rcu_dereference(*ptr_global_hook);
	}

	if (hook && try_module_get(hook->owner)) {
		/* Once we have a refcnt on the module, we no longer need RCU */
		hook = rcu_pointer_handoff(hook);
	} else {
		WARN_ONCE(!hook, "%s has bad registration", mod);
		hook = ERR_PTR(-ENOENT);
	}
	rcu_read_unlock();

	if (!IS_ERR(hook)) {
		err = hook->enable(link->net);
		if (err) {
			module_put(hook->owner);
			hook = ERR_PTR(err);
		}
	}

	return hook;
}
#endif

static int bpf_nf_enable_defrag(struct bpf_nf_link *link)
{
	/**
	 * @brief Enables IP defragmentation for a Netfilter BPF link.
	 *
	 * This function attempts to enable IP defragmentation for the network packets
	 * processed by a given `bpf_nf_link`. It dynamically loads and enables the
	 * appropriate IPv4 or IPv6 defragmentation hook based on the protocol family
	 * configured in the link's Netfilter hook operations.
	 *
	 * @param link A pointer to the `bpf_nf_link` for which defragmentation is to be enabled.
	 * @return 0 on success, or a negative errno value on failure.
	 *
	 * @details
	 * The function uses a `switch` statement to determine the protocol family (`pf`)
	 * from `link->hook_ops`.
	 * - For IPv4 (`NFPROTO_IPV4`), it calls `get_proto_defrag_hook` to obtain and enable
	 *   the `nf_defrag_ipv4` module. This path is only compiled if `CONFIG_NF_DEFRAG_IPV4`
	 *   is enabled.
	 * - For IPv6 (`NFPROTO_IPV6`), it calls `get_proto_defrag_hook` to obtain and enable
	 *   the `nf_defrag_ipv6` module. This path is only compiled if `CONFIG_NF_DEFRAG_IPV6`
	 *   is enabled.
	 * - For any other unsupported protocol family, it returns `-EAFNOSUPPORT`.
	 * If `get_proto_defrag_hook` returns an error, it is propagated. On success, the
	 * obtained defragmentation hook is stored in `link->defrag_hook`.
	 *
	 * @see get_proto_defrag_hook
	 * @see bpf_nf_disable_defrag
	 */
	const struct nf_defrag_hook __maybe_unused *hook;

	switch (link->hook_ops.pf) {
#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV4)
	case NFPROTO_IPV4: /* Handles IPv4 protocol family. */
		hook = get_proto_defrag_hook(link, &nf_defrag_v4_hook, "nf_defrag_ipv4");
		if (IS_ERR(hook))
			return PTR_ERR(hook);

		link->defrag_hook = hook;
		return 0;
#endif
#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV6)
	case NFPROTO_IPV6:
		hook = get_proto_defrag_hook(link, &nf_defrag_v6_hook, "nf_defrag_ipv6");
		if (IS_ERR(hook))
			return PTR_ERR(hook);

		link->defrag_hook = hook;
		return 0;
#endif
	default:
		return -EAFNOSUPPORT;
	}
}

static void bpf_nf_disable_defrag(struct bpf_nf_link *link)
{
	/**
	 * @brief Disables IP defragmentation for a Netfilter BPF link.
	 *
	 * This function is responsible for disabling the previously enabled IP defragmentation
	 * hook associated with a `bpf_nf_link` and releasing the corresponding module reference.
	 *
	 * @param link A pointer to the `bpf_nf_link` for which defragmentation is to be disabled.
	 *
	 * @details
	 * The function first checks if a `defrag_hook` is present in the `bpf_nf_link` structure.
	 * If a hook exists, it calls the hook's `disable` callback to deactivate defragmentation
	 * for the associated network namespace. Finally, it decrements the reference count
	 * of the module owning the defragmentation hook using `module_put`, allowing the module
	 * to be unloaded if no other users remain.
	 *
	 * @see bpf_nf_enable_defrag
	 * @see nf_defrag_hook
	 */
	const struct nf_defrag_hook *hook = link->defrag_hook;

	if (!hook)
		return;
	hook->disable(link->net);
	module_put(hook->owner);
}

static void bpf_nf_link_release(struct bpf_link *link)
{
	/**
	 * @brief Releases resources associated with a Netfilter BPF link.
	 *
	 * This function is invoked when a `bpf_nf_link` is being released,
	 * either explicitly detached or as part of link cleanup. It ensures
	 * that all resources held by the link are properly deallocated and
	 * Netfilter hooks are unregistered.
	 *
	 * @param link A pointer to the generic `bpf_link` object being released.
	 *
	 * @details
	 * 1. **Dead Check**: It first checks the `dead` flag of the `bpf_nf_link` structure.
	 *    If the link is already marked as `dead`, it means the resources have been
	 *    released previously (e.g., by `bpf_nf_link_detach`), and thus, it returns
	 *    to prevent a double-release.
	 * 2. **Atomic State Update**: `cmpxchg(&nf_link->dead, 0, 1)` is used to atomically
	 *    set the `dead` flag from 0 to 1. This ensures that the cleanup operations
	 *    are performed only once, even if `bpf_nf_link_release` is called concurrently
	 *    (e.g., from both explicit detach and internal BPF link cleanup).
	 * 3. **Unregister Netfilter Hook**: The Netfilter hook associated with the link
	 *    is unregistered from the network stack using `nf_unregister_net_hook`.
	 * 4. **Disable Defrag Hook**: If an IP defragmentation hook was enabled for this link,
	 *    `bpf_nf_disable_defrag` is called to disable it and release its module reference.
	 * 5. **Release Network Namespace Reference**: The reference to the network namespace
	 *    tracked by `ns_tracker` is released using `put_net_track`.
	 *
	 * @see bpf_nf_link_dealloc
	 * @see bpf_nf_link_detach
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	if (nf_link->dead)
		return;

	/* do not double release in case .detach was already called */
	if (!cmpxchg(&nf_link->dead, 0, 1)) {
		nf_unregister_net_hook(nf_link->net, &nf_link->hook_ops);
		bpf_nf_disable_defrag(nf_link);
		put_net_track(nf_link->net, &nf_link->ns_tracker);
	}
}

static void bpf_nf_link_dealloc(struct bpf_link *link)
{
	/**
	 * @brief Deallocates the memory for a Netfilter BPF link.
	 *
	 * This function is the final step in the destruction of a `bpf_nf_link` object.
	 * It is responsible for freeing the memory allocated for the `bpf_nf_link` structure
	 * itself, after all associated resources (like Netfilter hooks and defrag hooks)
	 * have been released by `bpf_nf_link_release`.
	 *
	 * @param link A pointer to the generic `bpf_link` object whose memory is to be deallocated.
	 *
	 * @details
	 * The function uses `container_of` to get the `bpf_nf_link` instance from the generic
	 * `bpf_link` pointer, and then calls `kfree` to release the memory. This function
	 * assumes that all other resources have already been properly cleaned up.
	 *
	 * @see bpf_nf_link_release
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	kfree(nf_link);
}

static int bpf_nf_link_detach(struct bpf_link *link)
{
	/**
	 * @brief Detaches a Netfilter BPF link.
	 *
	 * This function is part of the `bpf_link_ops` and is called when a user
	 * explicitly detaches a `bpf_nf_link`. It orchestrates the release of
	 * resources associated with the link.
	 *
	 * @param link A pointer to the generic `bpf_link` object to be detached.
	 * @return 0 on successful detachment.
	 *
	 * @details
	 * The function's primary role is to call `bpf_nf_link_release` to perform
	 * the actual cleanup and unregistration of the Netfilter hook and
	 * defragmentation hook (if any). This separation ensures that the complex
	 * resource management logic resides in one place, while `detach` simply
	 * triggers it.
	 *
	 * @see bpf_nf_link_release
	 */
	bpf_nf_link_release(link);
	return 0;
}

static void bpf_nf_link_show_info(const struct bpf_link *link,
				  struct seq_file *seq)
{
	/**
	 * @brief Displays information about a Netfilter BPF link to a `seq_file`.
	 *
	 * This function is used to expose details about an active `bpf_nf_link`
	 * to userspace via the `seq_file` interface, typically accessible through
	 * debugfs or similar mechanisms. It provides a human-readable summary
	 * of the link's configuration.
	 *
	 * @param link A pointer to the generic `bpf_link` object.
	 * @param seq A pointer to the `seq_file` where the information should be printed.
	 *
	 * @details
	 * The function extracts the `bpf_nf_link` instance from the generic `bpf_link`.
	 * It then prints the Netfilter protocol family (`pf`), hook number (`hooknum`),
	 * and priority (`priority`) associated with the link's `nf_hook_ops` to the
	 * provided `seq_file`.
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	seq_printf(seq, "pf:\t%u\thooknum:\t%u\tprio:\t%d\n",
		   nf_link->hook_ops.pf, nf_link->hook_ops.hooknum,
		   nf_link->hook_ops.priority);
}

static int bpf_nf_link_fill_link_info(const struct bpf_link *link,
				      struct bpf_link_info *info)
{
	/**
	 * @brief Fills a `bpf_link_info` structure with Netfilter BPF link details.
	 *
	 * This function is part of the BPF link operations and is used to provide
	 * structured information about a `bpf_nf_link` to userspace. This allows
	 * tools to introspect the state and configuration of active Netfilter BPF links.
	 *
	 * @param link A pointer to the generic `bpf_link` object.
	 * @param info A pointer to the `bpf_link_info` structure to be populated.
	 * @return 0 on success.
	 *
	 * @details
	 * The function extracts the `bpf_nf_link` instance from the generic `bpf_link`.
	 * It then populates the `netfilter` specific fields within the `bpf_link_info` structure:
	 * - `pf`: The Netfilter protocol family (e.g., `NFPROTO_IPV4`, `NFPROTO_IPV6`).
	 * - `hooknum`: The Netfilter hook point (e.g., `NF_INET_PRE_ROUTING`).
	 * - `priority`: The priority of the Netfilter hook.
	 * - `flags`: A bitmask indicating specific features, such as `BPF_F_NETFILTER_IP_DEFRAG`
	 *            if an IP defragmentation hook is active for this link.
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);
	const struct nf_defrag_hook *hook = nf_link->defrag_hook;

	info->netfilter.pf = nf_link->hook_ops.pf;
	info->netfilter.hooknum = nf_link->hook_ops.hooknum;
	info->netfilter.priority = nf_link->hook_ops.priority;
	info->netfilter.flags = hook ? BPF_F_NETFILTER_IP_DEFRAG : 0;

	return 0;
}

static int bpf_nf_link_update(struct bpf_link *link, struct bpf_prog *new_prog,
			      struct bpf_prog *old_prog)
{
	/**
	 * @brief Updates the BPF program associated with a Netfilter BPF link.
	 *
	 * This function is intended to facilitate the atomic replacement of an
	 * existing BPF program (`old_prog`) with a new BPF program (`new_prog`)
	 * on an active Netfilter BPF link.
	 *
	 * @param link A pointer to the `bpf_link` whose associated program is to be updated.
	 * @param new_prog A pointer to the new `bpf_prog` to be attached.
	 * @param old_prog A pointer to the currently attached `bpf_prog`.
	 * @return Always returns `-EOPNOTSUPP` as this functionality is currently not supported.
	 *
	 * @details
	 * At present, the ability to hot-swap BPF programs on a Netfilter BPF link
	 * is not implemented. Therefore, any attempt to call this function will
	 * result in an "Operation not supported" error.
	 */
	return -EOPNOTSUPP;
}

static const struct bpf_link_ops bpf_nf_link_lops = {
	/**
	 * @brief Operations table for the Netfilter BPF link type.
	 *
	 * This structure defines the set of callback functions that implement the
	 * generic BPF link operations (`bpf_link_ops`) specifically for the
	 * Netfilter BPF link type. These operations manage the lifecycle and
	 * introspection of Netfilter BPF links.
	 *
	 * - `.release`: Called to release resources held by the link.
	 * - `.dealloc`: Called to deallocate the memory of the link object.
	 * - `.detach`: Called when the link is explicitly detached.
	 * - `.show_fdinfo`: Used to display link information in `fdinfo` (via `bpf_nf_link_show_info`).
	 * - `.fill_link_info`: Used to populate `bpf_link_info` with link details.
	 * - `.update_prog`: Reserved for future functionality to update the BPF program associated with the link.
	 */
	.release = bpf_nf_link_release,
	.dealloc = bpf_nf_link_dealloc,
	.detach = bpf_nf_link_detach,
	.show_fdinfo = bpf_nf_link_show_info,
	.fill_link_info = bpf_nf_link_fill_link_info,
	.update_prog = bpf_nf_link_update,
};

static int bpf_nf_check_pf_and_hooks(const union bpf_attr *attr)
{
	/**
	 * @brief Validates Netfilter protocol family, hook number, and priority for a BPF link.
	 *
	 * This function performs a series of validation checks on the attributes provided
	 * for creating a new Netfilter BPF link. These checks ensure that the requested
	 * configuration is valid and does not interfere with critical Netfilter operations.
	 *
	 * @param attr A pointer to the `bpf_attr` union, containing the link creation attributes.
	 * @return 0 on successful validation, or a negative errno value on failure.
	 *
	 * @details
	 * 1. **Protocol Family Check**:
	 *    - Verifies that the `pf` (protocol family) is either `NFPROTO_IPV4` or `NFPROTO_IPV6`.
	 *    - For supported families, it checks if the `hooknum` is within the valid range
	 *      (`NF_INET_NUMHOOKS`).
	 *    - Returns `-EAFNOSUPPORT` for unsupported protocol families.
	 * 2. **Flags Check**:
	 *    - Ensures that only the `BPF_F_NETFILTER_IP_DEFRAG` flag is set, returning
	 *      `-EOPNOTSUPP` for any other unsupported flags.
	 * 3. **Priority Checks**:
	 *    - Prevents attachment at `NF_IP_PRI_FIRST` or `NF_IP_PRI_LAST` to avoid
	 *      conflicts with core Netfilter operations like "sabotage_in" or "conntrack confirm".
	 *    - If `BPF_F_NETFILTER_IP_DEFRAG` is requested, it ensures that the BPF program's
	 *      priority (`prio`) is not higher than or equal to `NF_IP_PRI_CONNTRACK_DEFRAG`,
	 *      as the BPF program cannot process defragmented packets if it runs before
	 *      the defragmentation hook.
	 */
	int prio;

	switch (attr->link_create.netfilter.pf) {
	case NFPROTO_IPV4:
	case NFPROTO_IPV6:
		if (attr->link_create.netfilter.hooknum >= NF_INET_NUMHOOKS)
			return -EPROTO;
		break;
	default:
		return -EAFNOSUPPORT;
	}

	if (attr->link_create.netfilter.flags & ~BPF_F_NETFILTER_IP_DEFRAG)
		return -EOPNOTSUPP;

	/* make sure conntrack confirm is always last */
	prio = attr->link_create.netfilter.priority;
	if (prio == NF_IP_PRI_FIRST)
		return -ERANGE;  /* sabotage_in and other warts */
	else if (prio == NF_IP_PRI_LAST)
		return -ERANGE;  /* e.g. conntrack confirm */
	else if ((attr->link_create.netfilter.flags & BPF_F_NETFILTER_IP_DEFRAG) &&
		 prio <= NF_IP_PRI_CONNTRACK_DEFRAG)
		return -ERANGE;  /* cannot use defrag if prog runs before nf_defrag */

	return 0;
}

int bpf_nf_link_attach(const union bpf_attr *attr, struct bpf_prog *prog)
{
	/**
	 * @brief Creates and attaches a new Netfilter BPF link.
	 *
	 * This is the primary function for establishing a connection between a BPF program
	 * and a Netfilter hook. It validates the provided attributes, allocates resources,
	 * initializes the link, and registers the Netfilter hook with the kernel.
	 *
	 * @param attr A pointer to the `bpf_attr` union, containing configuration
	 *   details for the new link (protocol family, hook number, priority, flags).
	 * @param prog A pointer to the `bpf_prog` that will be executed when the
	 *   Netfilter hook is triggered.
	 * @return A file descriptor representing the BPF link on success, or a
	 *   negative errno value on failure.
	 *
	 * @details
	 * 1. **Attribute Validation**: It first checks the `link_create.flags` for
	 *    unsupported flags and then calls `bpf_nf_check_pf_and_hooks` to validate
	 *    the Netfilter-specific attributes (protocol family, hook number, priority).
	 * 2. **Memory Allocation**: A new `bpf_nf_link` structure is allocated using `kzalloc`.
	 * 3. **BPF Link Initialization**: The generic `bpf_link` part of `bpf_nf_link`
	 *    is initialized with the `BPF_LINK_TYPE_NETFILTER` and the `bpf_nf_link_lops`
	 *    operations table. The provided BPF program (`prog`) is also associated.
	 * 4. **Netfilter Hook Operations Configuration**: The `nf_hook_ops` structure
	 *    within `bpf_nf_link` is populated with the BPF hook callback (`nf_hook_run_bpf`),
	 *    the BPF program as `priv` data, protocol family, priority, and hook number
	 *    from the input attributes.
	 * 5. **Network Namespace Tracking**: The current network namespace is stored in `link->net`.
	 * 6. **BPF Link Priming**: `bpf_link_prime` prepares the link for attachment.
	 * 7. **IP Defragmentation Handling**: If the `BPF_F_NETFILTER_IP_DEFRAG` flag is set,
	 *    `bpf_nf_enable_defrag` is called to activate IP defragmentation for the link.
	 * 8. **Netfilter Hook Registration**: The configured `nf_hook_ops` is registered
	 *    with the Netfilter framework using `nf_register_net_hook`.
	 * 9. **Network Namespace Reference**: A reference to the current network namespace
	 *    is taken using `get_net_track` to ensure the namespace remains valid as long as
	 *    the link exists.
	 * 10. **BPF Link Settling**: `bpf_link_settle` finalizes the link attachment and
	 *     returns the link file descriptor.
	 *
	 * @error Handling: Throughout the process, if any step fails (e.g., memory allocation,
	 *        validation, hook enablement, registration), appropriate cleanup is performed
	 *        (e.g., `kfree`, `bpf_link_cleanup`, `bpf_nf_disable_defrag`) before returning
	 *        an error code.
	 *
	 * @see bpf_nf_check_pf_and_hooks
	 * @see bpf_nf_enable_defrag
	 * @see nf_register_net_hook
	 */


const struct bpf_prog_ops netfilter_prog_ops = {
	/**
	 * @brief Defines BPF program operations specific to Netfilter BPF programs.
	 *
	 * This structure provides a set of callback functions that integrate Netfilter-specific
	 * functionalities into the generic BPF program operations framework.
	 *
	 * - `.test_run`: Specifies the function (`bpf_prog_test_run_nf`) to be used
	 *   for testing Netfilter BPF programs. This allows for simulation and
	 *   validation of BPF program behavior in a Netfilter context.
	 */
	.test_run = bpf_prog_test_run_nf,
};

static bool nf_ptr_to_btf_id(struct bpf_insn_access_aux *info, const char *name)
{
	/**
	 * @brief Converts a pointer's type information into a BTF ID for BPF verifier.
	 *
	 * This helper function is used by the BPF verifier to provide type-aware context
	 * for memory accesses within BPF programs. It looks up a structure by its name
	 * in the `vmlinux` BTF (BPF Type Format) and populates the `bpf_insn_access_aux`
	 * structure with the relevant BTF ID and type.
	 *
	 * @param info A pointer to `bpf_insn_access_aux` to be populated with BTF information.
	 * @param name The name of the structure (e.g., "sk_buff", "nf_hook_state") to look up.
	 * @return `true` if the BTF ID is successfully retrieved and `info` is populated,
	 *   `false` otherwise.
	 *
	 * @details
	 * 1. **Get vmlinux BTF**: Retrieves the BTF data for the running kernel (`vmlinux`).
	 *    If retrieval fails or returns NULL, it indicates BTF is not available, and the
	 *    function returns `false`.
	 * 2. **Find Structure by Name**: Uses `btf_find_by_name_kind` to search for a structure
	 *    (`BTF_KIND_STRUCT`) with the given `name` within the `vmlinux` BTF.
	 * 3. **Error Handling**: If the structure is not found or an error occurs during lookup,
	 *    a `WARN_ON_ONCE` is triggered, and the function returns `false`.
	 * 4. **Populate `info`**: On success, the `btf` pointer, `btf_id`, and `reg_type`
	 *    (indicating a trusted pointer to a BTF ID) fields of the `info` structure are populated.
	 */
	struct btf *btf;
	s32 type_id;

	btf = bpf_get_btf_vmlinux();
	if (IS_ERR_OR_NULL(btf))
		return false;

	type_id = btf_find_by_name_kind(btf, name, BTF_KIND_STRUCT);
	if (WARN_ON_ONCE(type_id < 0))
		return false;

	info->btf = btf;
	info->btf_id = type_id;
	info->reg_type = PTR_TO_BTF_ID | PTR_TRUSTED;
	return true;
}

static bool nf_is_valid_access(int off, int size, enum bpf_access_type type,
			       const struct bpf_prog *prog,
			       struct bpf_insn_access_aux *info)
{
	/**
	 * @brief Validates BPF program memory access to the `bpf_nf_ctx` context.
	 *
	 * This function is a crucial part of the BPF verifier's safety checks for
	 * Netfilter BPF programs. It ensures that any memory access (read or write)
	 * performed by a BPF program on the `bpf_nf_ctx` structure is within
	 * bounds and adheres to type constraints.
	 *
	 * @param off The byte offset within the `bpf_nf_ctx` structure.
	 * @param size The size of the memory access in bytes.
	 * @param type The type of access (read or write), defined by `enum bpf_access_type`.
	 * @param prog A pointer to the `bpf_prog` currently being verified.
	 * @param info A pointer to `bpf_insn_access_aux` to be populated with BTF information
	 *   for known pointer types.
	 * @return `true` if the access is valid, `false` otherwise.
	 *
	 * @details
	 * 1. **Bounds Check**: Verifies that the `off` is within the bounds of `sizeof(struct bpf_nf_ctx)`.
	 * 2. **Write Restriction**: Disallows any write access (`BPF_WRITE`) to the `bpf_nf_ctx`,
	 *    as the context is intended to be read-only for BPF programs.
	 * 3. **Field-Specific Checks**:
	 *    - **`skb` field**: If the access targets the `skb` field, it checks for correct size
	 *      and then calls `nf_ptr_to_btf_id` to associate BTF type information for "sk_buff".
	 *    - **`state` field**: Similarly, if the access targets the `state` field, it checks
	 *      size and uses `nf_ptr_to_btf_id` for "nf_hook_state" BTF information.
	 * 4. **Default Invalid**: Any other access (invalid offset, unknown field) defaults to `false`.
	 */
	if (off < 0 || off >= sizeof(struct bpf_nf_ctx))
		return false;

	if (type == BPF_WRITE)
		return false;

	switch (off) {
	case bpf_ctx_range(struct bpf_nf_ctx, skb):
		if (size != sizeof_field(struct bpf_nf_ctx, skb))
			return false;

		return nf_ptr_to_btf_id(info, "sk_buff");
	case bpf_ctx_range(struct bpf_nf_ctx, state):
		if (size != sizeof_field(struct bpf_nf_ctx, state))
			return false;

		return nf_ptr_to_btf_id(info, "nf_hook_state");
	default:
		return false;
	}

	return false;
}

static const struct bpf_func_proto *
bpf_nf_func_proto(enum bpf_func_id func_id, const struct bpf_prog *prog)
{
	/**
	 * @brief Retrieves the function prototype for BPF helper functions.
	 *
	 * This function is part of the BPF verifier's process to determine the
	 * allowed helper functions and their signatures for Netfilter BPF programs.
	 * It ensures that BPF programs call helper functions with correct arguments
	 * and return types.
	 *
	 * @param func_id The ID of the BPF helper function being queried.
	 * @param prog A pointer to the `bpf_prog` for which the helper prototype is requested.
	 * @return A pointer to `bpf_func_proto` structure describing the helper function's
	 *   signature, or `NULL` if the `func_id` is not recognized.
	 *
	 * @details
	 * Currently, Netfilter BPF programs primarily rely on the generic set of
	 * BPF helper functions provided by the kernel. Therefore, this function
	 * delegates the prototype lookup to `bpf_base_func_proto`, which handles
	 * the common BPF helper functions. This implies that no Netfilter-specific
	 * BPF helper functions are currently exposed through this mechanism.
	 */
	return bpf_base_func_proto(func_id, prog);
}

const struct bpf_verifier_ops netfilter_verifier_ops = {
	/**
	 * @brief BPF Verifier operations specific to Netfilter BPF programs.
	 *
	 * This structure provides custom callback functions to the BPF verifier,
	 * allowing it to perform Netfilter-specific validation checks when
	 * analyzing BPF programs intended for Netfilter hooks. This ensures
	 * the safety and security of BPF programs operating within the Netfilter
	 * context.
	 *
	 * - `.is_valid_access`: A callback to validate memory accesses made by the
	 *   BPF program to Netfilter-specific context structures (e.g., `bpf_nf_ctx`).
	 * - `.get_func_proto`: A callback to retrieve function prototypes for BPF
	 *   helper functions that can be called by Netfilter BPF programs.
	 */
	.is_valid_access	= nf_is_valid_access,
	.get_func_proto		= bpf_nf_func_proto,
};
