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
/**
 * @file
 * @brief This file implements the core functionality of x_tables, the backend for
 *        iptables, ip6tables, arptables, and ebtables.
 *
 * It provides a generic framework for packet filtering and manipulation by
 * managing tables, chains, rules, matches, and targets. It handles the
 * registration and unregistration of these components, as well as the
 * translation of rules from userspace to the kernel. This file also
 * includes support for per-CPU counters, compatibility with 32-bit userspace
 * on 64-bit systems, and the /proc interface for inspecting table and
 * component information.
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/socket.h>
#include <linux/net.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/string.h>
#include <linux/vmalloc.h>
#include <linux/mutex.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/audit.h>
#include <linux/user_namespace.h>
#include <net/net_namespace.h>
#include <net/netns/generic.h>

#include <linux/netfilter/x_tables.h>
#include <linux/netfilter_arp.h>
#include <linux/netfilter_ipv4/ip_tables.h>
#include <linux/netfilter_ipv6/ip6_tables.h>
#include <linux/netfilter_arp/arp_tables.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Harald Welte <laforge@netfilter.org>");
MODULE_DESCRIPTION("{ip,ip6,arp,eb}_tables backend module");

#define XT_PCPU_BLOCK_SIZE 4096
#define XT_MAX_TABLE_SIZE	(512 * 1024 * 1024)

struct xt_template {
	struct list_head list;

	/* called when table is needed in the given netns */
	int (*table_init)(struct net *net);

	struct module *me;

	/* A unique name...
 */
	char name[XT_TABLE_MAXNAMELEN];
};

static struct list_head xt_templates[NFPROTO_NUMPROTO];

struct xt_pernet {
	struct list_head tables[NFPROTO_NUMPROTO];
};

struct compat_delta {
	unsigned int offset; /* offset in kernel */
	int delta; /* delta in 32bit user land */
};

struct xt_af {
	struct mutex mutex;
	struct list_head match;
	struct list_head target;
#ifdef CONFIG_NETFILTER_XTABLES_COMPAT
	struct mutex compat_mutex;
	struct compat_delta *compat_tab;
	unsigned int number; /* number of slots in compat_tab[] */
	unsigned int cur; /* number of used slots in compat_tab[] */
#endif
};

static unsigned int xt_pernet_id __read_mostly;
static struct xt_af *xt __read_mostly;

static const char *const xt_prefix[NFPROTO_NUMPROTO] = {
	[NFPROTO_UNSPEC] = "x",
	[NFPROTO_IPV4]   = "ip",
	[NFPROTO_ARP]    = "arp",
	[NFPROTO_BRIDGE] = "eb",
	[NFPROTO_IPV6]   = "ip6",
};

/**
 * @brief Registers a new xtables target.
 * @param target A pointer to the xt_target structure to be registered.
 * @return 0 on success.
 */
int xt_register_target(struct xt_target *target)
{
	u_int8_t af = target->family;

	mutex_lock(&xt[af].mutex);
	list_add(&target->list, &xt[af].target);
	mutex_unlock(&xt[af].mutex);
	return 0;
}
EXPORT_SYMBOL(xt_register_target);

/**
 * @brief Unregisters an xtables target.
 * @param target A pointer to the xt_target structure to be unregistered.
 */
void
xt_unregister_target(struct xt_target *target)
{
	u_int8_t af = target->family;

	mutex_lock(&xt[af].mutex);
	list_del(&target->list);
	mutex_unlock(&xt[af].mutex);
}
EXPORT_SYMBOL(xt_unregister_target);

/**
 * @brief Registers an array of xtables targets.
 * @param target A pointer to the array of xt_target structures.
 * @param n The number of targets in the array.
 * @return 0 on success, or a negative error code on failure.
 */
int
xt_register_targets(struct xt_target *target, unsigned int n)
{
	unsigned int i;
	int err = 0;

	for (i = 0; i < n; i++) {
		err = xt_register_target(&target[i]);
		if (err)
			goto err;
	}
	return err;

err:
	if (i > 0)
		xt_unregister_targets(target, i);
	return err;
}
EXPORT_SYMBOL(xt_register_targets);

/**
 * @brief Unregisters an array of xtables targets.
 * @param target A pointer to the array of xt_target structures.
 * @param n The number of targets in the array.
 */
void
xt_unregister_targets(struct xt_target *target, unsigned int n)
{
	while (n-- > 0)
		xt_unregister_target(&target[n]);
}
EXPORT_SYMBOL(xt_unregister_targets);

/**
 * @brief Registers a new xtables match.
 * @param match A pointer to the xt_match structure to be registered.
 * @return 0 on success.
 */
int xt_register_match(struct xt_match *match)
{
	u_int8_t af = match->family;

	mutex_lock(&xt[af].mutex);
	list_add(&match->list, &xt[af].match);
	mutex_unlock(&xt[af].mutex);
	return 0;
}
EXPORT_SYMBOL(xt_register_match);

/**
 * @brief Unregisters an xtables match.
 * @param match A pointer to the xt_match structure to be unregistered.
 */
void
xt_unregister_match(struct xt_match *match)
{
	u_int8_t af = match->family;

	mutex_lock(&xt[af].mutex);
	list_del(&match->list);
	mutex_unlock(&xt[af].mutex);
}
EXPORT_SYMBOL(xt_unregister_match);

/**
 * @brief Registers an array of xtables matches.
 * @param match A pointer to the array of xt_match structures.
 * @param n The number of matches in the array.
 * @return 0 on success, or a negative error code on failure.
 */
int
xt_register_matches(struct xt_match *match, unsigned int n)
{
	unsigned int i;
	int err = 0;

	for (i = 0; i < n; i++) {
		err = xt_register_match(&match[i]);
		if (err)
			goto err;
	}
	return err;

err:
	if (i > 0)
		xt_unregister_matches(match, i);
	return err;
}
EXPORT_SYMBOL(xt_register_matches);

/**
 * @brief Unregisters an array of xtables matches.
 * @param match A pointer to the array of xt_match structures.
 * @param n The number of matches in the array.
 */
void
xt_unregister_matches(struct xt_match *match, unsigned int n)
{
	while (n-- > 0)
		xt_unregister_match(&match[n]);
}
EXPORT_SYMBOL(xt_unregister_matches);


/*
 * These are weird, but module loading must not be done with mutex
 * held (since they will register), and we have to have a single
 * function to use.
 */

/**
 * @brief Finds an xtables match by name and revision.
 * @param af The address family.
 * @param name The name of the match.
 * @param revision The revision of the match.
 * @return A pointer to the xt_match structure on success, or an error pointer
 *         on failure. A reference to the module is taken on success.
 */
struct xt_match *xt_find_match(u8 af, const char *name, u8 revision)
{
	struct xt_match *m;
	int err = -ENOENT;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL);

	mutex_lock(&xt[af].mutex);
	list_for_each_entry(m, &xt[af].match, list) {
		if (strcmp(m->name, name) == 0) {
			if (m->revision == revision) {
				if (try_module_get(m->me)) {
					mutex_unlock(&xt[af].mutex);
					return m;
				}
			} else
				err = -EPROTOTYPE; /* Found something. */
		}
	}
	mutex_unlock(&xt[af].mutex);

	if (af != NFPROTO_UNSPEC)
		/* Try searching again in the family-independent list */
		return xt_find_match(NFPROTO_UNSPEC, name, revision);

	return ERR_PTR(err);
}
EXPORT_SYMBOL(xt_find_match);

/**
 * @brief Finds an xtables match, requesting the module if not found.
 * @param nfproto The protocol family.
 * @param name The name of the match.
 * @param revision The revision of the match.
 * @return A pointer to the xt_match structure on success, or an error pointer
 *         on failure. A reference to the module is taken on success.
 */
struct xt_match *
xt_request_find_match(uint8_t nfproto, const char *name, uint8_t revision)
{
	struct xt_match *match;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL);

	match = xt_find_match(nfproto, name, revision);
	if (IS_ERR(match)) {
		request_module("%st_%s", xt_prefix[nfproto], name);
		match = xt_find_match(nfproto, name, revision);
	}

	return match;
}
EXPORT_SYMBOL_GPL(xt_request_find_match);

/**
 * @brief Finds an xtables target by name and revision.
 * @param af The address family.
 * @param name The name of the target.
 * @param revision The revision of the target.
 * @return A pointer to the xt_target structure on success, or an error pointer
 *         on failure. A reference to the module is taken on success.
 */
static struct xt_target *xt_find_target(u8 af, const char *name, u8 revision)
{
	struct xt_target *t;
	int err = -ENOENT;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL);

	mutex_lock(&xt[af].mutex);
	list_for_each_entry(t, &xt[af].target, list) {
		if (strcmp(t->name, name) == 0) {
			if (t->revision == revision) {
				if (try_module_get(t->me)) {
					mutex_unlock(&xt[af].mutex);
					return t;
				}
			} else
				err = -EPROTOTYPE; /* Found something. */
		}
	}
	mutex_unlock(&xt[af].mutex);

	if (af != NFPROTO_UNSPEC)
		/* Try searching again in the family-independent list */
		return xt_find_target(NFPROTO_UNSPEC, name, revision);

	return ERR_PTR(err);
}

/**
 * @brief Finds an xtables target, requesting the module if not found.
 * @param af The address family.
 * @param name The name of the target.
 * @param revision The revision of the target.
 * @return A pointer to the xt_target structure on success, or an error pointer
 *         on failure. A reference to the module is taken on success.
 */
struct xt_target *xt_request_find_target(u8 af, const char *name, u8 revision)
{
	struct xt_target *target;

	if (strnlen(name, XT_EXTENSION_MAXNAMELEN) == XT_EXTENSION_MAXNAMELEN)
		return ERR_PTR(-EINVAL);

	target = xt_find_target(af, name, revision);
	if (IS_ERR(target)) {
		request_module("%st_%s", xt_prefix[af], name);
		target = xt_find_target(af, name, revision);
	}

	return target;
}
EXPORT_SYMBOL_GPL(xt_request_find_target);


/**
 * @brief Copies an object's size, name, and revision to userspace.
 * @param psize A user-space pointer to the size.
 * @param size The size of the object.
 * @param pname A user-space pointer to the name.
 * @param name The name of the object.
 * @param prev A user-space pointer to the revision.
 * @param rev The revision of the object.
 * @return 0 on success, or a negative error code on failure.
 */
static int xt_obj_to_user(u16 __user *psize, u16 size,
			  void __user *pname, const char *name,
			  u8 __user *prev, u8 rev)
{
	if (put_user(size, psize))
		return -EFAULT;
	if (copy_to_user(pname, name, strlen(name) + 1))
		return -EFAULT;
	if (put_user(rev, prev))
		return -EFAULT;

	return 0;
}

#define XT_OBJ_TO_USER(U, K, TYPE, C_SIZE)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				* The new string is identical to the old string, so no changes are needed. The original string is returned.```json
{