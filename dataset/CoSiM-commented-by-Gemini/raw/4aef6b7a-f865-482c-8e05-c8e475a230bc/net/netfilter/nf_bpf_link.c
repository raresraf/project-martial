// SPDX-License-Identifier: GPL-2.0
#include <linux/bpf.h>
#include <linux/filter.h>
#include <linux/kmod.h>
#include <linux/module.h>
#include <linux/netfilter.h>

#include <net/netfilter/nf_bpf_link.h>
#include <uapi/linux/netfilter_ipv4.h>

/**
 * nf_hook_run_bpf - Executes a BPF program attached to a Netfilter hook.
 * @bpf_prog: Pointer to the compiled BPF program.
 * @skb: The socket buffer containing the network packet.
 * @s: The Netfilter hook state.
 *
 * This function serves as the Netfilter hook's callback. It prepares the
 * BPF context (@bpf_nf_ctx) and runs the BPF program. The return value from
 * the BPF program determines the packet's fate (e.g., NF_ACCEPT, NF_DROP).
 *
 * Return: The verdict from the BPF program.
 */
static unsigned int nf_hook_run_bpf(void *bpf_prog, struct sk_buff *skb,
				    const struct nf_hook_state *s)
{
	const struct bpf_prog *prog = bpf_prog;
	struct bpf_nf_ctx ctx = {
		.state = s,
		.skb = skb,
	};

	return bpf_prog_run_pin_on_cpu(prog, &ctx);
}

/**
 * struct bpf_nf_link - Represents a link between a BPF program and a Netfilter hook.
 * @link: The generic BPF link structure.
 * @hook_ops: The Netfilter hook operations structure that registers our BPF
 *            program with the Netfilter framework.
 * @ns_tracker: Tracks the network namespace this link is associated with.
 * @net: Pointer to the network namespace.
 * @dead: Flag to prevent double-release of resources.
 * @defrag_hook: Pointer to the protocol-specific defragmentation hooks, if enabled.
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
 * get_proto_defrag_hook - Retrieves and enables a protocol-specific defrag hook.
 * @link: The bpf_nf_link structure.
 * @ptr_global_hook: A pointer to the global defragmentation hook for the protocol.
 * @mod: The name of the kernel module to request if the hook is not loaded.
 *
 * This function ensures the necessary IP defragmentation module (e.g.,
 * "nf_defrag_ipv4") is loaded and enabled for the link's network namespace.
 * It takes a reference to the module to prevent it from being unloaded while in use.
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
		/* If hook is not present, dynamically load the required module. */
		err = request_module("%s", mod);
		if (err)
			return ERR_PTR(err < 0 ? err : -EINVAL);

		rcu_read_lock();
		hook = rcu_dereference(*ptr_global_hook);
	}

	/* Try to get a reference on the module to ensure it's not unloaded. */
	if (hook && try_module_get(hook->owner)) {
		/* Once we have a refcnt on the module, we no longer need RCU */
		hook = rcu_pointer_handoff(hook);
	} else {
		WARN_ONCE(!hook, "%s has bad registration", mod);
		hook = ERR_PTR(-ENOENT);
	}
	rcu_read_unlock();

	if (!IS_ERR(hook)) {
		/* Enable defragmentation for the specific network namespace. */
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
 * bpf_nf_enable_defrag - Enables IP defragmentation for the link.
 * @link: The bpf_nf_link to enable defragmentation for.
 *
 * Checks the protocol family of the link and calls the appropriate helper
 * to load and enable the IPv4 or IPv6 defragmentation module.
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
 * bpf_nf_disable_defrag - Disables IP defragmentation and releases module reference.
 * @link: The bpf_nf_link to disable defragmentation for.
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
 * bpf_nf_link_release - Callback to release the resources held by the link.
 * @link: The generic BPF link.
 *
 * This function is called when the last reference to the BPF link (e.g., a
 * file descriptor) is dropped. It unregisters the Netfilter hook, disables
 * defragmentation, and releases the network namespace tracker.
 */
static void bpf_nf_link_release(struct bpf_link *link)
{
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	if (nf_link->dead)
		return;

	/* Use cmpxchg to ensure release logic runs only once. */
	if (!cmpxchg(&nf_link->dead, 0, 1)) {
		nf_unregister_net_hook(nf_link->net, &nf_link->hook_ops);
		bpf_nf_disable_defrag(nf_link);
		put_net_track(nf_link->net, &nf_link->ns_tracker);
	}
}

/**
 * bpf_nf_link_dealloc - Callback to deallocate the memory for the link.
 * @link: The generic BPF link.
 */
static void bpf_nf_link_dealloc(struct bpf_link *link)
{
	struct bpf_nf_link *nf_link = container_of(link, struct bpf_nf_link, link);

	kfree(nf_link);
}

/**
 * bpf_nf_link_detach - Explicitly detach the link and release resources.
 * @link: The generic BPF link.
 *
 * Return: 0 on success.
 */
static int bpf_nf_link_detach(struct bpf_link *link)
{
	bpf_nf_link_release(link);
	return 0;
}

/**
 * bpf_nf_link_show_info - Provides information about the link for fdinfo.
 * @link: The generic BPF link.
 * @seq: The sequence file to write the info to.
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
 * bpf_nf_link_fill_link_info - Fills a bpf_link_info struct with link details.
 * @link: The generic BPF link.
 * @info: The info structure to be filled.
 *
 * This is used by tools like bpftool to query link properties.
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
 * bpf_nf_link_update - Updates the BPF program on an existing link.
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

/* Defines the set of operations for a Netfilter BPF link. */
static const struct bpf_link_ops bpf_nf_link_lops = {
	.release = bpf_nf_link_release,
	.dealloc = bpf_nf_link_dealloc,
	.detach = bpf_nf_link_detach,
	.show_fdinfo = bpf_nf_link_show_info,
	.fill_link_info = bpf_nf_link_fill_link_info,
	.update_prog = bpf_nf_link_update,
};

/**
 * bpf_nf_check_pf_and_hooks - Validates user-provided link attributes.
 * @attr: The BPF syscall attributes from userspace.
 *
 * Verifies that the protocol family, hook number, priority, and flags
 * are valid and not conflicting with existing Netfilter priorities.
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

	/* BPF programs should not interfere with core Netfilter functionality
	 * that relies on running at the highest or lowest priorities.
	 */
	prio = attr->link_create.netfilter.priority;
	if (prio == NF_IP_PRI_FIRST)
		return -ERANGE;  /* Reserved for internal use (e.g., sabotage_in) */
	else if (prio == NF_IP_PRI_LAST)
		return -ERANGE;  /* Reserved for internal use (e.g., conntrack confirm) */
	else if ((attr->link_create.netfilter.flags & BPF_F_NETFILTER_IP_DEFRAG) &&
		 prio <= NF_IP_PRI_CONNTRACK_DEFRAG)
		/* A BPF program requesting defrag must run after the defrag hook. */
		return -ERANGE;

	return 0;
}

/**
 * bpf_nf_link_attach - Attaches a BPF program to a Netfilter hook.
 * @attr: The BPF syscall attributes from userspace.
 * @prog: The BPF program to attach.
 *
 * This is the main entry point for creating a Netfilter BPF link. It allocates,
 * initializes, and registers the link and its associated Netfilter hook.
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

/* Defines test run operations for Netfilter BPF programs. */
const struct bpf_prog_ops netfilter_prog_ops = {
	.test_run = bpf_prog_test_run_nf,
};

/**
 * nf_ptr_to_btf_id - Helper to get the BTF type ID for a given structure name.
 * @info: The BPF instruction access info to be filled.
 * @name: The name of the structure to find in vmlinux BTF.
 *
 * This function is used by the verifier to associate a pointer in the BPF
 * context with its actual kernel type, enabling field access verification.
 *
 * Return: True on success, false on failure.
 */
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
 * nf_is_valid_access - Verifier callback to check context access validity.
 * @off: The offset within the context structure being accessed.
 * @size: The size of the access.
 * @type: The type of access (BPF_READ or BPF_WRITE).
 * @prog: The BPF program being verified.
 * @info: The BPF instruction access info.
 *
 * This function is a security critical part of the BPF verifier. It ensures
 * that a BPF program attached to a Netfilter hook only accesses the fields of
* `struct bpf_nf_ctx` in a valid way (e.g., read-only, correct size).
 *
 * Return: True if the access is valid, false otherwise.
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

/**
 * bpf_nf_func_proto - Verifier callback to get function prototypes.
 * @func_id: The ID of the BPF helper function being called.
 * @prog: The BPF program being verified.
 *
 * This allows BPF programs attached to Netfilter hooks to use a standard set
 * of BPF helper functions.
 *
 * Return: A pointer to the function prototype, or NULL if not found.
 */
static const struct bpf_func_proto *
bpf_nf_func_proto(enum bpf_func_id func_id, const struct bpf_prog *prog)
{
	return bpf_base_func_proto(func_id, prog);
}

/* Defines the verifier operations for Netfilter BPF programs. */
const struct bpf_verifier_ops netfilter_verifier_ops = {
	.is_valid_access	= nf_is_valid_access,
	.get_func_proto		= bpf_nf_func_proto,
};
