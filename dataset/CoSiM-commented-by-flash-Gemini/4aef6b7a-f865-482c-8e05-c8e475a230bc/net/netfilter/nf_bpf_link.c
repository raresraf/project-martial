// SPDX-License-Identifier: GPL-2.0
/**
 * @4aef6b7a-f865-482c-8e05-c8e475a230bc/net/netfilter/nf_bpf_link.c
 * @brief Implements the Netfilter BPF link type, enabling BPF programs to be
 *        attached to Netfilter hooks for packet processing and manipulation.
 * Architectural Intent: To integrate BPF's flexible and programmable packet
 *        processing capabilities directly into the Linux Netfilter framework,
 *        allowing for highly efficient and dynamic network policy enforcement
 *        and data plane programmability without requiring kernel module reloads.
 * Key Concepts: BPF programs, Netfilter hooks, network namespaces, IP defragmentation.
 */
#include <linux/bpf.h>
#include <linux/filter.h>
#include <linux/kmod.h>
#include <linux/module.h>
#include <linux/netfilter.h>

#include <net/netfilter/nf_bpf_link.h>
#include <uapi/linux/netfilter_ipv4.h>

static unsigned int nf_hook_run_bpf(void *bpf_prog, struct sk_buff *skb,
				    const struct nf_hook_state *s)
{
	/**
	 * Functional Utility: Invokes a BPF program from a Netfilter hook.
	 * This function acts as the bridge between the Netfilter framework
	 * and an eBPF program, allowing packet processing logic defined in BPF
	 * to be executed at specific Netfilter hook points.
	 *
	 * @param bpf_prog: Pointer to the BPF program to be executed.
	 * @param skb: The socket buffer containing the packet.
	 * @param s: The Netfilter hook state, providing context for the hook.
	 * @return: The verdict from the BPF program, dictating further Netfilter actions.
	 */
	const struct bpf_prog *prog = bpf_prog; // Cast the void pointer to a BPF program pointer.
	struct bpf_nf_ctx ctx = {              // Prepare the context structure for the BPF program.
		.state = s,                    // Pass the Netfilter hook state.
		.skb = skb,                    // Pass the socket buffer.
	};

	// Execute the BPF program with the prepared context.
	return bpf_prog_run_pin_on_cpu(prog, &ctx);
}

struct bpf_nf_link {
	/**
	 * @brief Represents a BPF Netfilter link, extending the generic BPF link
	 *        with Netfilter-specific operational details and state.
	 * @field link: The base BPF link structure, providing common BPF link functionalities.
	 * @field hook_ops: The Netfilter hook operations structure, defining where and how
	 *                  the BPF program is attached to the Netfilter pipeline.
	 * @field ns_tracker: Tracks the network namespace associated with this BPF link.
	 * @field net: Pointer to the `struct net` representing the network namespace.
	 * @field dead: A flag indicating if the link is marked for detachment or is already detached.
	 * @field defrag_hook: Optional pointer to the Netfilter defragmentation hook, used
	 *                     if IP defragmentation is required before BPF processing.
	 */
	struct bpf_link link;
	struct nf_hook_ops hook_ops;
	netns_tracker ns_tracker;
	struct net *net;
	u32 dead;
	const struct nf_defrag_hook *defrag_hook;
};

#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV4) || IS_ENABLED(CONFIG_NF_DEFRAG_IPV6)
static const struct nf_defrag_hook *
get_proto_defrag_hook(struct bpf_nf_link *link,
		      const struct nf_defrag_hook __rcu **ptr_global_hook,
		      const char *mod)
{
	/**
	 * Functional Utility: Retrieves and enables the appropriate IP defragmentation
	 * hook for a given protocol family (IPv4 or IPv6).
	 * This ensures that packets are fully reassembled before being processed by BPF,
	 * which is critical for BPF programs operating on complete packets.
	 *
	 * @param link: Pointer to the bpf_nf_link structure.
	 * @param ptr_global_hook: Pointer to the global RCU-protected defragmentation hook pointer.
	 * @param mod: The name of the module that provides the defragmentation hook (e.g., "nf_defrag_ipv4").
	 * @return: A pointer to the enabled nf_defrag_hook on success, or an ERR_PTR on failure.
	 */
	const struct nf_defrag_hook *hook;
	int err;

	// RCU read lock protects against races with module unloading.
	rcu_read_lock();
	hook = rcu_dereference(*ptr_global_hook); // Dereference the RCU-protected global hook pointer.
	if (!hook) {
		rcu_read_unlock();
		// If the hook is not found, attempt to dynamically load the module.
		err = request_module("%s", mod);
		if (err)
			return ERR_PTR(err < 0 ? err : -EINVAL); // Return error if module loading fails.

		rcu_read_lock(); // Re-acquire RCU lock after module load.
		hook = rcu_dereference(*ptr_global_hook); // Try to dereference again.
	}

	// If a hook is found, try to get a reference to its owner module.
	if (hook && try_module_get(hook->owner)) {
		// Once we have a refcnt on the module, RCU protection is no longer strictly needed.
		hook = rcu_pointer_handoff(hook);
	} else {
		WARN_ONCE(!hook, "%s has bad registration", mod); // Warn if hook is NULL but expected.
		hook = ERR_PTR(-ENOENT); // Return error if hook is invalid.
	}
	rcu_read_unlock(); // Release RCU read lock.

	// If hook retrieval was successful, enable it for the specific network namespace.
	if (!IS_ERR(hook)) {
		err = hook->enable(link->net);
		if (err) {
			module_put(hook->owner); // Release module reference on error.
			hook = ERR_PTR(err);
		}
	}

	return hook; // Return the defragmentation hook or an error pointer.
}
#endif

static int bpf_nf_enable_defrag(struct bpf_nf_link *link)
{
	/**
	 * Functional Utility: Enables IP defragmentation for a Netfilter BPF link
	 * based on the protocol family (IPv4 or IPv6) specified in the hook operations.
	 * This ensures that fragmented packets are reassembled before the BPF program
	 * processes them, which is essential for many packet analysis and filtering tasks.
	 *
	 * @param link: Pointer to the bpf_nf_link structure.
	 * @return: 0 on success, or a negative errno on failure.
	 */
	const struct nf_defrag_hook __maybe_unused *hook;

	switch (link->hook_ops.pf) { // Switch based on the protocol family (e.g., IPv4, IPv6).
#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV4) // Compile-time check for IPv4 defragmentation support.
	case NFPROTO_IPV4:
		// Attempt to get and enable the IPv4 defragmentation hook.
		hook = get_proto_defrag_hook(link, &nf_defrag_v4_hook, "nf_defrag_ipv4");
		if (IS_ERR(hook))
			return PTR_ERR(hook); // Return error if hook could not be enabled.

		link->defrag_hook = hook; // Store the enabled defragmentation hook.
		return 0;                 // Success.
#endif
#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV6) // Compile-time check for IPv6 defragmentation support.
	case NFPROTO_IPV6:
		// Attempt to get and enable the IPv6 defragmentation hook.
		hook = get_proto_defrag_hook(link, &nf_defrag_v6_hook, "nf_defrag_ipv6");
		if (IS_ERR(hook))
			return PTR_ERR(hook); // Return error if hook could not be enabled.

		link->defrag_hook = hook; // Store the enabled defragmentation hook.
		return 0;                 // Success.
#endif
	default:
		// Return EAFNOSUPPORT if the protocol family is not supported for defragmentation.
		return -EAFNOSUPPORT;
	}
}

static void bpf_nf_disable_defrag(struct bpf_nf_link *link)
{
	/**
	 * Functional Utility: Disables IP defragmentation for a Netfilter BPF link
	 * and releases the associated module reference.
	 * This is the counterpart to `bpf_nf_enable_defrag`, ensuring proper cleanup
	 * when the BPF link no longer requires defragmentation.
	 *
	 * @param link: Pointer to the bpf_nf_link structure.
	 */
	const struct nf_defrag_hook *hook = link->defrag_hook;

	if (!hook) // If no defragmentation hook was enabled, do nothing.
		return;
	hook->disable(link->net); // Call the disable function of the defragmentation hook.
	module_put(hook->owner);  // Release the reference to the defragmentation module.
}

static void bpf_nf_link_release(struct bpf_link *link)
{
	/**
	 * Functional Utility: Releases resources associated with a Netfilter BPF link.
	 * This function detaches the BPF program from the Netfilter hook, disables
	 * IP defragmentation (if enabled), and releases network namespace tracking.
	 * It ensures that resources are properly cleaned up when the BPF link is no longer needed.
	 *
	 * @param link: Pointer to the generic bpf_link structure.
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	if (nf_link->dead) // If the link is already marked as dead, return immediately to prevent double release.
		return;

	// Atomically set the 'dead' flag to 1. If it was successfully changed from 0 to 1,
	// proceed with unregistration and cleanup. This prevents double release issues.
	if (!cmpxchg(&nf_link->dead, 0, 1)) {
		nf_unregister_net_hook(nf_link->net, &nf_link->hook_ops); // Unregister the Netfilter hook.
		bpf_nf_disable_defrag(nf_link);                            // Disable IP defragmentation if it was enabled.
		put_net_track(nf_link->net, &nf_link->ns_tracker);         // Release the network namespace tracking.
	}
}

static void bpf_nf_link_dealloc(struct bpf_link *link)
{
	/**
	 * Functional Utility: Deallocates the memory associated with a Netfilter BPF link.
	 * This function is responsible for freeing the `bpf_nf_link` structure itself
	 * after all its resources have been released.
	 *
	 * @param link: Pointer to the generic bpf_link structure.
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	kfree(nf_link); // Free the memory allocated for the bpf_nf_link structure.
}

static int bpf_nf_link_detach(struct bpf_link *link)
{
	/**
	 * Functional Utility: Detaches a Netfilter BPF link, stopping its operation
	 * and releasing associated resources.
	 * This function acts as a wrapper around `bpf_nf_link_release`.
	 *
	 * @param link: Pointer to the generic bpf_link structure.
	 * @return: 0 on successful detachment.
	 */
	bpf_nf_link_release(link); // Call the common release function to clean up resources.
	return 0;                  // Indicate successful detachment.
}

static void bpf_nf_link_show_info(const struct bpf_link *link,
				  struct seq_file *seq)
{
	/**
	 * Functional Utility: Outputs information about a Netfilter BPF link to a `seq_file`.
	 * This function is used for debugging and introspection, providing details
	 * about the Netfilter protocol family, hook number, and priority.
	 *
	 * @param link: Pointer to the generic bpf_link structure.
	 * @param seq: Pointer to the `seq_file` to write the information to.
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	// Output the protocol family, hook number, and priority of the Netfilter hook.
	seq_printf(seq, "pf:\t%u\thooknum:\t%u\tprio:\t%d\n",
		   nf_link->hook_ops.pf, nf_link->hook_ops.hooknum,
		   nf_link->hook_ops.priority);
}

static int bpf_nf_link_fill_link_info(const struct bpf_link *link,
				      struct bpf_link_info *info)
{
	/**
	 * Functional Utility: Fills a `bpf_link_info` structure with Netfilter-specific
	 * details about a BPF link.
	 * This is used to provide userspace with comprehensive information about
	 * the attached BPF program's Netfilter context.
	 *
	 * @param link: Pointer to the generic bpf_link structure.
	 * @param info: Pointer to the `bpf_link_info` structure to populate.
	 * @return: 0 on success.
	 */
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);
	const struct nf_defrag_hook *hook = nf_link->defrag_hook;

	// Populate Netfilter-specific fields in the bpf_link_info structure.
	info->netfilter.pf = nf_link->hook_ops.pf;             // Protocol family.
	info->netfilter.hooknum = nf_link->hook_ops.hooknum;   // Netfilter hook number.
	info->netfilter.priority = nf_link->hook_ops.priority; // Priority of the hook.
	// Set a flag if IP defragmentation is enabled for this link.
	info->netfilter.flags = hook ? BPF_F_NETFILTER_IP_DEFRAG : 0;

	return 0; // Indicate success.
}

static int bpf_nf_link_update(struct bpf_link *link, struct bpf_prog *new_prog,
			      struct bpf_prog *old_prog)
{
	/**
	 * Functional Utility: Attempts to atomically update the BPF program associated
	 * with a Netfilter BPF link.
	 * This operation is currently not supported for Netfilter BPF links,
	 * as indicated by the immediate return of -EOPNOTSUPP.
	 *
	 * @param link: Pointer to the generic bpf_link structure.
	 * @param new_prog: Pointer to the new BPF program to be attached.
	 * @param old_prog: Pointer to the old BPF program currently attached.
	 * @return: -EOPNOTSUPP, indicating that this operation is not supported.
	 */
	return -EOPNOTSUPP;
}

static const struct bpf_link_ops bpf_nf_link_lops = {
	/**
	 * @brief Defines the operations table for Netfilter BPF link types.
	 * This structure provides function pointers for managing the lifecycle
	 * and state of a Netfilter BPF link, integrating it into the generic
	 * BPF link management framework.
	 * @field release: Function to release resources associated with the link.
	 * @field dealloc: Function to deallocate memory for the link structure.
	 * @field detach: Function to detach the BPF program from the Netfilter hook.
	 * @field show_fdinfo: Function to display link information in `fdinfo`.
	 * @field fill_link_info: Function to populate `bpf_link_info` with link-specific details.
	 * @field update_prog: Function to update the BPF program associated with the link (currently unsupported).
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
	 * Functional Utility: Validates the protocol family, hook number, and priority
	 * specified in the BPF link creation attributes for Netfilter.
	 * This ensures that the requested Netfilter hook point is valid and
	 * that any defragmentation requirements are met concerning hook priority.
	 *
	 * @param attr: Pointer to the BPF attributes union containing link creation parameters.
	 * @return: 0 on successful validation, or a negative errno on failure.
	 */
	int prio;

	switch (attr->link_create.netfilter.pf) { // Check the protocol family.
	case NFPROTO_IPV4:
	case NFPROTO_IPV6:
		// Validate the Netfilter hook number for IPv4/IPv6.
		if (attr->link_create.netfilter.hooknum >= NF_INET_NUMHOOKS)
			return -EPROTO; // Invalid hook number.
		break;
	default:
		return -EAFNOSUPPORT; // Unsupported protocol family.
	}

	// Check for unsupported flags in Netfilter link creation attributes.
	if (attr->link_create.netfilter.flags & ~BPF_F_NETFILTER_IP_DEFRAG)
		return -EOPNOTSUPP; // Unsupported flags specified.

	// Functional Utility: Validate hook priority against Netfilter conventions and defragmentation requirements.
	prio = attr->link_create.netfilter.priority;
	if (prio == NF_IP_PRI_FIRST)
		return -ERANGE;  // Priority conflicts with system-reserved first priority.
	else if (prio == NF_IP_PRI_LAST)
		return -ERANGE;  // Priority conflicts with system-reserved last priority (e.g., conntrack confirm).
	// If IP defragmentation is requested, ensure the BPF program runs after defragmentation has occurred.
	else if ((attr->link_create.netfilter.flags & BPF_F_NETFILTER_IP_DEFRAG) &&
		 prio <= NF_IP_PRI_CONNTRACK_DEFRAG)
		return -ERANGE;  // BPF program cannot use defrag if it runs before nf_defrag.

	return 0; // All checks passed.
}

int bpf_nf_link_attach(const union bpf_attr *attr, struct bpf_prog *prog)
{
	/**
	 * Functional Utility: Attaches a BPF program to a Netfilter hook, creating a new
	 * Netfilter BPF link.
	 * This involves validating attributes, allocating and initializing the link,
	 * optionally enabling IP defragmentation, and registering the Netfilter hook.
	 *
	 * @param attr: Pointer to the BPF attributes union containing link creation parameters.
	 * @param prog: Pointer to the BPF program to be attached.
	 * @return: A file descriptor representing the BPF link on success, or a negative errno on failure.
	 */
	struct net *net = current->nsproxy->net_ns; // Get the current network namespace.
	struct bpf_link_primer link_primer;         // Structure for priming the BPF link.
	struct bpf_nf_link *link;
	int err;

	if (attr->link_create.flags)
		return -EINVAL; // Return error for unsupported flags.

	// Validate the protocol family, hook number, and priority.
	err = bpf_nf_check_pf_and_hooks(attr);
	if (err)
		return err;

	// Allocate memory for the bpf_nf_link structure.
	link = kzalloc(sizeof(*link), GFP_USER);
	if (!link)
		return -ENOMEM; // Return error if memory allocation fails.

	// Initialize the generic BPF link part of the bpf_nf_link structure.
	bpf_link_init(&link->link, BPF_LINK_TYPE_NETFILTER, &bpf_nf_link_lops, prog);

	// Initialize the Netfilter hook operations.
	link->hook_ops.hook = nf_hook_run_bpf;           // Set the function to run the BPF program.
	link->hook_ops.hook_ops_type = NF_HOOK_OP_BPF;   // Specify BPF hook operation type.
	link->hook_ops.priv = prog;                      // Store the BPF program as private data.

	// Set Netfilter-specific parameters from the attributes.
	link->hook_ops.pf = attr->link_create.netfilter.pf;
	link->hook_ops.priority = attr->link_create.netfilter.priority;
	link->hook_ops.hooknum = attr->link_create.netfilter.hooknum;

	link->net = net;            // Associate the link with the current network namespace.
	link->dead = false;         // Initialize 'dead' flag to false.
	link->defrag_hook = NULL;   // Initialize defragmentation hook to NULL.

	// Prime the BPF link, preparing it for activation.
	err = bpf_link_prime(&link->link, &link_primer);
	if (err) {
		kfree(link);    // Free allocated memory on error.
		return err;
	}

	// If IP defragmentation flag is set, enable defragmentation.
	if (attr->link_create.netfilter.flags & BPF_F_NETFILTER_IP_DEFRAG) {
		err = bpf_nf_enable_defrag(link);
		if (err) {
			bpf_link_cleanup(&link_primer); // Clean up primed link on error.
			return err;
		}
	}

	// Register the Netfilter hook.
	err = nf_register_net_hook(net, &link->hook_ops);
	if (err) {
		bpf_nf_disable_defrag(link);    // Disable defragmentation if hook registration fails.
		bpf_link_cleanup(&link_primer); // Clean up primed link on error.
		return err;
	}

	// Track the network namespace to ensure proper lifetime management.
	get_net_track(net, &link->ns_tracker, GFP_KERNEL);

	// Settle the BPF link, making it active and returning its file descriptor.
	return bpf_link_settle(&link_primer);
}

const struct bpf_prog_ops netfilter_prog_ops = {
	/**
	 * @brief Defines the BPF program operations specific to Netfilter.
	 * This structure specifies custom operations, such as how to test-run
	 * a BPF program for a Netfilter context.
	 * @field test_run: Function pointer to the test-run implementation for Netfilter BPF programs.
	 */
	.test_run = bpf_prog_test_run_nf,
};

static bool nf_ptr_to_btf_id(struct bpf_insn_access_aux *info, const char *name)
{
	/**
	 * Functional Utility: Populates `bpf_insn_access_aux` with BTF (BPF Type Format)
	 * information for a given structure name.
	 * This helps the BPF verifier understand the type of data being accessed
	 * by a BPF program when dealing with pointers, enhancing safety and correctness.
	 *
	 * @param info: Pointer to the `bpf_insn_access_aux` structure to populate.
	 * @param name: The name of the structure (e.g., "sk_buff", "nf_hook_state").
	 * @return: True if BTF information was successfully retrieved and populated, False otherwise.
	 */
	struct btf *btf;
	s32 type_id;

	btf = bpf_get_btf_vmlinux(); // Get the BTF information for the vmlinux kernel.
	if (IS_ERR_OR_NULL(btf))
		return false; // Return false if BTF information is not available or an error occurred.

	// Find the type ID for the specified structure name within the BTF data.
	type_id = btf_find_by_name_kind(btf, name, BTF_KIND_STRUCT);
	if (WARN_ON_ONCE(type_id < 0)) // Warn if the structure type is not found (shouldn't happen for known types).
		return false;

	info->btf = btf;          // Store the BTF object.
	info->btf_id = type_id;   // Store the type ID of the structure.
	info->reg_type = PTR_TO_BTF_ID | PTR_TRUSTED; // Mark the register as a trusted pointer to a BTF ID.
	return true;              // Successfully populated BTF information.
}

static bool nf_is_valid_access(int off, int size, enum bpf_access_type type,
			       const struct bpf_prog *prog,
			       struct bpf_insn_access_aux *info)
{
	/**
	 * Functional Utility: Implements custom access validation for Netfilter BPF context
	 * (`struct bpf_nf_ctx`).
	 * This function is part of the BPF verifier's mechanism to ensure that BPF programs
	 * only access valid memory regions and data types within their context, enhancing
	 * kernel security and stability.
	 *
	 * @param off: The byte offset within `struct bpf_nf_ctx` that the BPF program is attempting to access.
	 * @param size: The size of the access (e.g., 1 for byte, 4 for int).
	 * @param type: The type of access (BPF_READ or BPF_WRITE).
	 * @param prog: Pointer to the BPF program performing the access.
	 * @param info: Auxiliary information for the BPF verifier, including BTF details.
	 * @return: True if the access is valid, False otherwise.
	 */
	if (off < 0 || off >= sizeof(struct bpf_nf_ctx))
		return false; // Access is out of bounds for `struct bpf_nf_ctx`.

	if (type == BPF_WRITE)
		return false; // Write access to `bpf_nf_ctx` is not allowed.

	switch (off) {
	// Case for accessing the `skb` (socket buffer) member of `struct bpf_nf_ctx`.
	case bpf_ctx_range(struct bpf_nf_ctx, skb):
		if (size != sizeof_field(struct bpf_nf_ctx, skb))
			return false; // Size mismatch.

		// Populate BTF info for `sk_buff` to allow further verification.
		return nf_ptr_to_btf_id(info, "sk_buff");
	// Case for accessing the `state` (nf_hook_state) member of `struct bpf_nf_ctx`.
	case bpf_ctx_range(struct bpf_nf_ctx, state):
		if (size != sizeof_field(struct bpf_nf_ctx, state))
			return false; // Size mismatch.

		// Populate BTF info for `nf_hook_state` to allow further verification.
		return nf_ptr_to_btf_id(info, "nf_hook_state");
	default:
		return false; // Access to other fields of `struct bpf_nf_ctx` is not explicitly allowed.
	}

	return false; // Should not be reached.
}

static const struct bpf_func_proto *
bpf_nf_func_proto(enum bpf_func_id func_id, const struct bpf_prog *prog)
{
	/**
	 * Functional Utility: Provides BPF verifier prototypes for BPF helper functions
	 * relevant to Netfilter.
	 * This function is part of the BPF verifier's mechanism to ensure that BPF programs
	 * use helper functions correctly and safely.
	 *
	 * @param func_id: The ID of the BPF helper function.
	 * @param prog: Pointer to the BPF program making the helper call.
	 * @return: A pointer to the `bpf_func_proto` for the specified helper function.
	 */
	return bpf_base_func_proto(func_id, prog); // Delegate to the base BPF function prototype provider.
}

const struct bpf_verifier_ops netfilter_verifier_ops = {
	/**
	 * @brief Defines BPF verifier operations specific to Netfilter BPF programs.
	 * This structure provides custom callbacks to the BPF verifier for
	 * Netfilter-specific context access validation and helper function prototyping,
	 * ensuring the safety and correctness of Netfilter BPF programs.
	 * @field is_valid_access: Function pointer to custom access validation logic for Netfilter context.
	 * @field get_func_proto: Function pointer to retrieve prototypes for Netfilter-specific BPF helper functions.
	 */
	.is_valid_access	= nf_is_valid_access,
	.get_func_proto		= bpf_nf_func_proto,
};
