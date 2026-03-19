// SPDX-License-Identifier: GPL-2.0
/**
 * @file nf_bpf_link.c
 * @brief Manages the attachment of BPF programs to Netfilter hooks.
 *
 * This file implements the "glue" logic that allows a BPF program to be attached
 * to a standard Netfilter hook point. It handles the creation, lifecycle management,
 * and safe execution of BPF programs in the context of the Netfilter packet
 * processing path. This enables highly programmable and efficient packet filtering
 * and manipulation.
 */
#include <linux/bpf.h>
#include <linux/filter.h>
#include <linux/kmod.h>
#include <linux/module.h>
#include <linux/netfilter.h>

#include <net/netfilter/nf_bpf_link.h>
#include <uapi/linux/netfilter_ipv4.h>

/**
 * nf_hook_run_bpf - The callback function executed by the Netfilter framework.
 * @bpf_prog: A pointer to the compiled BPF program to be executed.
 * @skb: The socket buffer containing the network packet being processed.
 * @s: The Netfilter hook state, providing context about the packet's traversal.
 *
 * This function is registered with a Netfilter hook. For each packet that
 * traverses the hook, this function prepares a bpf_nf_ctx and executes the
 * attached BPF program against it.
 *
 * Return: The verdict from the BPF program (e.g., NF_ACCEPT, NF_DROP).
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
 * struct bpf_nf_link - Represents the link between a BPF program and a Netfilter hook.
 * @link: The generic BPF link object.
 * @hook_ops: The Netfilter hook operations structure that registers nf_hook_run_bpf.
 * @ns_tracker: Tracks the network namespace associated with this link.
 * @net: Pointer to the network namespace.
 * @dead: A flag to prevent double-release on teardown.
 * @defrag_hook: A pointer to the protocol-specific defragmentation hooks, if enabled.
 */
struct bpf_nf_link {
	struct bpf_link link;
	struct nf_hook_ops hook_ops;
	netns_tracker ns_tracker;
	struct net *net;
	u32 dead;
	const struct nf_defrag_hook *defrag_hook;
};

#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV4) || IS_ENABLED(CONFIG_NF_DEFRAG_IPV6)
/**
 * get_proto_defrag_hook - Dynamically loads and enables a defragmentation module.
 * @link: The BPF netfilter link.
 * @ptr_global_hook: A pointer to the global hook for the required protocol (IPv4/v6).
 * @mod: The name of the kernel module to request (e.g., "nf_defrag_ipv4").
 *
 * This function ensures that the necessary IP defragmentation module is loaded
 * and enabled for the link's network namespace. BPF programs often require
 * fully formed packets, making defragmentation a prerequisite. It handles
 * module loading, reference counting, and registration.
 *
 * Return: A pointer to the defrag hook on success, or an ERR_PTR on failure.
 */
static const struct nf_defrag_hook *
get_proto_defrag_hook(struct bpf_nf_link *link,
			      const struct nf_defrag_hook __rcu **ptr_global_hook,
			      const char *mod)
{
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

/**
 * bpf_nf_enable_defrag - Enables IP defragmentation for the link based on protocol family.
 * @link: The BPF netfilter link to enable defragmentation for.
 *
 * Return: 0 on success, or a negative error code.
 */
static int bpf_nf_enable_defrag(struct bpf_nf_link *link)
{
	const struct nf_defrag_hook __maybe_unused *hook;

	switch (link->hook_ops.pf) {
#if IS_ENABLED(CONFIG_NF_DEFRAG_IPV4)
	case NFPROTO_IPV4:
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

/**
 * bpf_nf_disable_defrag - Disables IP defragmentation and releases the module.
 * @link: The BPF netfilter link.
 */
static void bpf_nf_disable_defrag(struct bpf_nf_link *link)
{
	const struct nf_defrag_hook *hook = link->defrag_hook;

	if (!hook)
		return;
	hook->disable(link->net);
	module_put(hook->owner);
}

/**
 * bpf_nf_link_release - Releases resources associated with a BPF Netfilter link.
 * @link: The generic bpf_link to be released.
 *
 * This function is called when the link's reference count drops to zero.
 * It unregisters the Netfilter hook, disables defragmentation, and releases
 * the network namespace tracker.
 */
static void bpf_nf_link_release(struct bpf_link *link)
{
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

/**
 * bpf_nf_link_dealloc - Deallocates the memory for a bpf_nf_link.
 * @link: The generic bpf_link to be deallocated.
 */
static void bpf_nf_link_dealloc(struct bpf_link *link)
{
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	kfree(nf_link);
}

/**
 * bpf_nf_link_detach - Explicitly detaches a BPF Netfilter link.
 * @link: The generic bpf_link to be detached.
 *
 * This is the implementation for the .detach operation, which simply
 * triggers the release logic.
 *
 * Return: 0 on success.
 */
static int bpf_nf_link_detach(struct bpf_link *link)
{
	bpf_nf_link_release(link);
	return 0;
}

/**
 * bpf_nf_link_show_info - Provides debugging information for user space (e.g., bpftool).
 * @link: The link to show information about.
 * @seq: The sequence file to write the information to.
 */
static void bpf_nf_link_show_info(const struct bpf_link *link,
				  struct seq_file *seq)
{
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	seq_printf(seq, "pf:\t%u\thooknum:\t%u\tprio:\t%d\n",
		   nf_link->hook_ops.pf, nf_link->hook_ops.hooknum,
		   nf_link->hook_ops.priority);
}

/**
 * bpf_nf_link_fill_link_info - Fills a bpf_link_info structure for introspection.
 * @link: The link to get information from.
 * @info: The structure to fill.
 *
 * This function provides structured information about the Netfilter link's
 * properties (protocol family, hook number, priority, and flags).
 *
 * Return: 0 on success.
 */
static int bpf_nf_link_fill_link_info(const struct bpf_link *link,
				      struct bpf_link_info *info)
{
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);
	const struct nf_defrag_hook *hook = nf_link->defrag_hook;

	info->netfilter.pf = nf_link->hook_ops.pf;
	info->netfilter.hooknum = nf_link->hook_ops.hooknum;
	info->netfilter.priority = nf_link->hook_ops.priority;
	info->netfilter.flags = hook ? BPF_F_NETFILTER_IP_DEFRAG : 0;

	return 0;
}

/**
 * bpf_nf_link_update - Updates the BPF program associated with the link.
 *
 * This operation is not supported for Netfilter links.
 *
 * Return: -EOPNOTSUPP
 */
static int bpf_nf_link_update(struct bpf_link *link, struct bpf_prog *new_prog,
			      struct bpf_prog *old_prog)
{
	return -EOPNOTSUPP;
}

static const struct bpf_link_ops bpf_nf_link_lops = {
	.release = bpf_nf_link_release,
	.dealloc = bpf_nf_link_dealloc,
	.detach = bpf_nf_link_detach,
	.show_fdinfo = bpf_nf_link_show_info,
	.fill_link_info = bpf_nf_link_fill_link_info,
	.update_prog = bpf_nf_link_update,
};

/**
 * bpf_nf_check_pf_and_hooks - Validates user-provided Netfilter parameters.
 * @attr: The bpf_attr union from the bpf() syscall.
 *
 * This function performs sanity checks on the protocol family, hook number,
 * priority, and flags to ensure they are valid and do not conflict with
 * critical Netfilter priorities (like conntrack).
 *
 * Return: 0 on success, or a negative error code.
 */
static int bpf_nf_check_pf_and_hooks(const union bpf_attr *attr)
{
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

/**
 * bpf_nf_link_attach - Attaches a BPF program to a Netfilter hook.
 * @attr: The bpf_attr union containing link creation parameters.
 * @prog: The BPF program to attach.
 *
 * This is the main entry point for creating a Netfilter-BPF link. It validates
 * the request, allocates and initializes the bpf_nf_link structure, optionally
 * enables defragmentation, and registers the hook with the Netfilter core.
 *
 * Return: A file descriptor for the link on success, or a negative error code.
 */
int bpf_nf_link_attach(const union bpf_attr *attr, struct bpf_prog *prog)
{
	struct net *net = current->nsproxy->net_ns;
	struct bpf_link_primer link_primer;
	struct bpf_nf_link *link;
	int err;

	if (attr->link_create.flags)
		return -EINVAL;

	err = bpf_nf_check_pf_and_hooks(attr);
	if (err)
		return err;

	link = kzalloc(sizeof(*link), GFP_USER);
	if (!link)
		return -ENOMEM;

	bpf_link_init(&link->link, BPF_LINK_TYPE_NETFILTER, &bpf_nf_link_lops, prog);

	link->hook_ops.hook = nf_hook_run_bpf;
	link->hook_ops.hook_ops_type = NF_HOOK_OP_BPF;
	link->hook_ops.priv = prog;

	link->hook_ops.pf = attr->link_create.netfilter.pf;
	link->hook_ops.priority = attr->link_create.netfilter.priority;
	link->hook_ops.hooknum = attr->link_create.netfilter.hooknum;

	link->net = net;
	link->dead = false;
	link->defrag_hook = NULL;

	err = bpf_link_prime(&link->link, &link_primer);
	if (err) {
		kfree(link);
		return err;
	}

	if (attr->link_create.netfilter.flags & BPF_F_NETFILTER_IP_DEFRAG) {
		err = bpf_nf_enable_defrag(link);
		if (err) {
			bpf_link_cleanup(&link_primer);
			return err;
		}
	}

	err = nf_register_net_hook(net, &link->hook_ops);
	if (err) {
		bpf_nf_disable_defrag(link);
		bpf_link_cleanup(&link_primer);
		return err;
	}

	get_net_track(net, &link->ns_tracker, GFP_KERNEL);

	return bpf_link_settle(&link_primer);
}

const struct bpf_prog_ops netfilter_prog_ops = {
	.test_run = bpf_prog_test_run_nf,
};

static bool nf_ptr_to_btf_id(struct bpf_insn_access_aux *info, const char *name)
{
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

/**
 * nf_is_valid_access - Verifier callback to check for valid context access.
 * @off: Offset within the context struct.
 * @size: Size of the access.
 * @type: Type of access (read/write).
 * @prog: The BPF program being verified.
 * @info: BPF verifier internal state.
 *
 * This function is a security-critical part of the BPF verifier. It ensures
 * that a BPF program attached to a Netfilter hook only accesses the fields
 * of `struct bpf_nf_ctx` that it is permitted to. It prevents writing to the
 * context and only allows reading the pointers to `sk_buff` and `nf_hook_state`.
 *
 * Return: True if access is valid, false otherwise.
 */
static bool nf_is_valid_access(int off, int size, enum bpf_access_type type,
			       const struct bpf_prog *prog,
			       struct bpf_insn_access_aux *info)
{
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
	return bpf_base_func_proto(func_id, prog);
}

const struct bpf_verifier_ops netfilter_verifier_ops = {
	.is_valid_access	= nf_is_valid_access,
	.get_func_proto		= bpf_nf_func_proto,
};