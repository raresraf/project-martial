/**
 * @file addr_prefs.c
 * @brief AFS Server Address Preference Management
 *
 * Copyright (C) 2023 Red Hat, Inc. All Rights Reserved.
 * Written by David Howells (dhowells@redhat.com)
 *
 * This file implements a system for managing address preferences for connecting to
 * AFS (Andrew File System) servers. It allows an administrator to define a
 * prioritized list of server network addresses, including specific subnets,
 * through a procfs interface (`/proc/fs/afs/addr_prefs`). This is used to
 * influence the selection of server endpoints, for example, to prefer local
 * network paths over remote ones.
 *
 * The core of the implementation is a sorted list of `afs_addr_preference`
 * objects. Updates to this list are handled using a copy-on-write strategy
 * combined with RCU (Read-Copy-Update) to ensure that modifications from
 * userspace do not interfere with concurrent reads by kernel threads attempting
 * to establish connections.
 */

// SPDX-License-Identifier: GPL-2.0-or-later
/* Address preferences management
 *
 * Copyright (C) 2023 Red Hat, Inc. All Rights Reserved.
 * Written by David Howells (dhowells@redhat.com)
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": addr_prefs: " fmt
#include <linux/slab.h>
#include <linux/ctype.h>
#include <linux/inet.h>
#include <linux/seq_file.h>
#include <keys/rxrpc-type.h>
#include "internal.h"

/**
 * afs_seq2net_single - Get the AFS network namespace from a seq_file context.
 * @m: The seq_file instance.
 *
 * This is a helper function to extract the AFS-specific network namespace from
 * the private data of a seq_file, typically used in procfs handlers.
 */
static inline struct afs_net *afs_seq2net_single(struct seq_file *m)
{
	return afs_net(seq_file_single_net(m));
}

/**
 * afs_split_string - Split a string by whitespace for command parsing.
 * @pbuf: A pointer to the current position in the buffer, updated on exit.
 * @strv: An array to be filled with pointers to the start of each word.
 * @maxstrv: The maximum number of elements in the strv array.
 *
 * This function tokenizes a NUL-terminated string, splitting it at whitespace.
 * It modifies the source string in-place by inserting NUL characters to
 * terminate each token. It is primarily used to parse commands read from the
 * procfs interface.
 *
 * Returns:
 * The number of tokens found, or a negative error code on failure (e.g.,
 * -EINVAL if too many elements are found).
 */
static int afs_split_string(char **pbuf, char *strv[], unsigned int maxstrv)
{
	unsigned int count = 0;
	char *p = *pbuf;

	maxstrv--; /* Allow for terminal NULL */
	for (;;) {
		/* Pre-condition: p points to the next character to be processed. */
		/* Invariant: count holds the number of tokens found so far. */
		while (isspace(*p)) {
			if (*p == '\n') {
				p++;
				break;
			}
			p++;
		}
		if (!*p)
			break;

		/* Mark start of word */
		if (count >= maxstrv) {
			pr_warn("Too many elements in string\n");
			return -EINVAL;
		}
		strv[count++] = p;

		/* Skip over word */
		while (!isspace(*p) && *p)
			p++;
		if (!*p)
			break;

		/* Mark end of word */
		if (*p == '\n') {
			*p++ = 0;
			break;
		}
		*p++ = 0;
	}

	*pbuf = p;
	strv[count] = NULL;
	return count;
}

/**
 * afs_parse_address - Parse an IP address with an optional subnet mask.
 * @p: The string containing the address to parse.
 * @pref: The structure to fill with the parsed address information.
 *
 * This function parses a string representation of an IP address, which can be
 * either IPv4 or IPv6. It handles optional subnet masks (e.g., "/24") and
 * bracketed IPv6 addresses (e.g., "[::1]").
 *
 * Returns:
 * 0 on success, or a negative error code on failure.
 */
static int afs_parse_address(char *p, struct afs_addr_preference *pref)
{
	const char *stop;
	unsigned long mask, tmp;
	char *end = p + strlen(p);
	bool bracket = false;

	if (*p == '[') {
		p++;
		bracket = true;
	}

#if 0
	if (*p == '[') {
		p++;
		q = memchr(p, ']', end - p);
		if (!q) {
			pr_warn("Can't find closing ']'\n");
			return -EINVAL;
		}
	} else {
		for (q = p; q < end; q++)
			if (*q == '/')
				break;
	}
#endif

	if (in4_pton(p, end - p, (u8 *)&pref->ipv4_addr, -1, &stop)) {
		pref->family = AF_INET;
		mask = 32;
	} else if (in6_pton(p, end - p, (u8 *)&pref->ipv6_addr, -1, &stop)) {
		pref->family = AF_INET6;
		mask = 128;
	} else {
		pr_warn("Can't determine address family\n");
		return -EINVAL;
	}

	p = (char *)stop;
	if (bracket) {
		if (*p != ']') {
			pr_warn("Can't find closing ']'\n");
			return -EINVAL;
		}
		p++;
	}

	// Block Logic: Parse the optional subnet mask that follows the address.
	if (*p == '/') {
		p++;
		tmp = simple_strtoul(p, &p, 10);
		if (tmp > mask) {
			pr_warn("Subnet mask too large\n");
			return -EINVAL;
		}
		if (tmp == 0) {
			pr_warn("Subnet mask too small\n");
			return -EINVAL;
		}
		mask = tmp;
	}

	if (*p) {
		pr_warn("Invalid address\n");
		return -EINVAL;
	}

	pref->subnet_mask = mask;
	return 0;
}

/**
 * @enum cmp_ret
 * @brief Result of comparing two address preferences.
 */
enum cmp_ret {
	CONTINUE_SEARCH, /**< The target is greater than the current item; keep searching. */
	INSERT_HERE,     /**< The target is less than the current item; insert before it. */
	EXACT_MATCH,     /**< The target and current item are identical addresses and masks. */
	SUBNET_MATCH,    /**< The target is a more specific subnet of the current item. */
};

/**
 * afs_cmp_address_pref - Compare two address preferences for sorting.
 * @a: The first address preference (the one to be inserted or found).
 * @b: The second address preference (from the existing list).
 *
 * This function compares two address preferences to determine their relative
 * order in the sorted preference list. The comparison is based on address family,
 * the network prefix (masked address), and finally the subnet mask length.
 *
 * Returns:
 * An enum cmp_ret value indicating the relationship between the two addresses.
 */
static enum cmp_ret afs_cmp_address_pref(const struct afs_addr_preference *a,
					 const struct afs_addr_preference *b)
{
	int subnet = min(a->subnet_mask, b->subnet_mask);
	const __be32 *pa, *pb;
	u32 mask, na, nb;
	int diff;

	if (a->family != b->family)
		return INSERT_HERE;

	switch (a->family) {
	case AF_INET6:
		pa = a->ipv6_addr.s6_addr32;
		pb = b->ipv6_addr.s6_addr32;
		break;
	case AF_INET:
		pa = &a->ipv4_addr.s_addr;
		pb = &b->ipv4_addr.s_addr;
		break;
	}

	// Block Logic: Compare the address prefixes word by word.
	// Invariant: At the start of each iteration, 'subnet' contains the remaining bits to check.
	while (subnet > 32) {
		diff = ntohl(*pa++) - ntohl(*pb++);
		if (diff < 0)
			return INSERT_HERE; /* a<b */
		if (diff > 0)
			return CONTINUE_SEARCH; /* a>b */
		subnet -= 32;
	}

	if (subnet == 0)
		return EXACT_MATCH;

	// Block Logic: Compare the final, partially-masked word of the address.
	mask = 0xffffffffU << (32 - subnet);
	na = ntohl(*pa);
	nb = ntohl(*pb);
	diff = (na & mask) - (nb & mask);
	//kdebug("diff %08x %08x %08x %d", na, nb, mask, diff);
	if (diff < 0)
		return INSERT_HERE; /* a<b */
	if (diff > 0)
		return CONTINUE_SEARCH; /* a>b */
	if (a->subnet_mask == b->subnet_mask)
		return EXACT_MATCH;
	if (a->subnet_mask > b->subnet_mask)
		return SUBNET_MATCH; /* a binds tighter than b */
	return CONTINUE_SEARCH; /* b binds tighter than a */
}

/**
 * afs_insert_address_pref - Insert a new address preference into the list.
 * @_preflist: A pointer to the current preference list, which may be replaced.
 * @pref: The new preference to insert.
 * @index: The index at which to insert the new preference.
 *
 * This function inserts a preference into the list at a specific index. If the
 * list is full, it reallocates a larger list, copies the existing data, and
 * updates the list pointer. This is part of the copy-on-write mechanism.
 *
 * Returns:
 * 0 on success, or a negative error code (e.g., -ENOMEM) on failure.
 */
static int afs_insert_address_pref(struct afs_addr_preference_list **_preflist,
				   struct afs_addr_preference *pref,
				   int index)
{
	struct afs_addr_preference_list *preflist = *_preflist, *old = preflist;
	size_t size, max_prefs;

	_enter("{%u/%u/%u},%u", preflist->ipv6_off, preflist->nr, preflist->max_prefs, index);

	if (preflist->nr == 255)
		return -ENOSPC;
	
	// Block Logic: Resize the preference list if it is full.
	// A new, larger buffer is allocated, and the old data is copied over.
	if (preflist->nr >= preflist->max_prefs) {
		max_prefs = preflist->max_prefs + 1;
		size = struct_size(preflist, prefs, max_prefs);
		size = roundup_pow_of_two(size);
		max_prefs = min_t(size_t, (size - sizeof(*preflist)) / sizeof(*pref), 255);
		preflist = kmalloc(size, GFP_KERNEL);
		if (!preflist)
			return -ENOMEM;
		*preflist = **_preflist;
		preflist->max_prefs = max_prefs;
		*_preflist = preflist;

		if (index < preflist->nr)
			memcpy(preflist->prefs + index + 1, old->prefs + index,
			       sizeof(*pref) * (preflist->nr - index));
		if (index > 0)
			memcpy(preflist->prefs, old->prefs, sizeof(*pref) * index);
	} else {
		// Block Logic: If there is space, make room for the new element.
		if (index < preflist->nr)
			memmove(preflist->prefs + index + 1, preflist->prefs + index,
			       sizeof(*pref) * (preflist->nr - index));
	}

	preflist->prefs[index] = *pref;
	preflist->nr++;
	if (pref->family == AF_INET)
		preflist->ipv6_off++;
	return 0;
}

/**
 * afs_add_address_pref - Process an "add" command for an address preference.
 * @net: The AFS network namespace.
 * @_preflist: Pointer to the current preference list.
 * @argc: Argument count.
 * @argv: Argument vector.
 *
 * Handles the "add <proto> <IP>[/<mask>] <prio>" command from procfs. It
 * parses the arguments, finds the correct position in the sorted list, and
 * either inserts a new entry or updates the priority of an existing one.
 *
 * Returns:
 * 0 on success, or a negative error code on failure.
 */
static int afs_add_address_pref(struct afs_net *net, struct afs_addr_preference_list **_preflist,
				int argc, char **argv)
{
	struct afs_addr_preference_list *preflist = *_preflist;
	struct afs_addr_preference pref;
	enum cmp_ret cmp;
	int ret, i, stop;

	if (argc != 3) {
		pr_warn("Wrong number of params\n");
		return -EINVAL;
	}

	if (strcmp(argv[0], "udp") != 0) {
		pr_warn("Unsupported protocol\n");
		return -EINVAL;
	}

	ret = afs_parse_address(argv[1], &pref);
	if (ret < 0)
		return ret;

	ret = kstrtou16(argv[2], 10, &pref.prio);
	if (ret < 0) {
		pr_warn("Invalid priority\n");
		return ret;
	}

	if (pref.family == AF_INET) {
		i = 0;
		stop = preflist->ipv6_off;
	} else {
		i = preflist->ipv6_off;
		stop = preflist->nr;
	}

	// Block Logic: Search for the correct insertion point or an exact match.
	// The list is sorted, so we can stop as soon as we find the right place.
	for (; i < stop; i++) {
		cmp = afs_cmp_address_pref(&pref, &preflist->prefs[i]);
		switch (cmp) {
		case CONTINUE_SEARCH:
			continue;
		case INSERT_HERE:
		case SUBNET_MATCH:
			return afs_insert_address_pref(_preflist, &pref, i);
		case EXACT_MATCH:
			// If an exact match is found, just update its priority.
			preflist->prefs[i].prio = pref.prio;
			return 0;
		}
	}

	// If the loop completes, the new entry should be appended at the end of its family block.
	return afs_insert_address_pref(_preflist, &pref, i);
}

/**
 * afs_delete_address_pref - Remove an address preference by its index.
 * @_preflist: A pointer to the current preference list.
 * @index: The index of the preference to remove.
 *
 * This function removes an entry from the preference list, shifting subsequent
 * elements to fill the gap.
 *
 * Returns:
 * 0 on success, or -ENOENT if the list is empty.
 */
static int afs_delete_address_pref(struct afs_addr_preference_list **_preflist,
				   int index)
{
	struct afs_addr_preference_list *preflist = *_preflist;

	_enter("{%u/%u/%u},%u", preflist->ipv6_off, preflist->nr, preflist->max_prefs, index);

	if (preflist->nr == 0)
		return -ENOENT;

	if (index < preflist->nr - 1)
		memmove(preflist->prefs + index, preflist->prefs + index + 1,
			sizeof(preflist->prefs[0]) * (preflist->nr - index - 1));

	if (index < preflist->ipv6_off)
		preflist->ipv6_off--;
	preflist->nr--;
	return 0;
}

/**
 * afs_del_address_pref - Process a "del" command for an address preference.
 * @net: The AFS network namespace.
 * @_preflist: Pointer to the current preference list.
 * @argc: Argument count.
 * @argv: Argument vector.
 *
 * Handles the "del <proto> <IP>[/<mask>]" command from procfs. It parses the
 * address, searches for an exact match in the preference list, and removes it
 * if found.
 *
 * Returns:
 * 0 on success, or a negative error code on failure (-ENOANO if not found).
 */
static int afs_del_address_pref(struct afs_net *net, struct afs_addr_preference_list **_preflist,
				int argc, char **argv)
{
	struct afs_addr_preference_list *preflist = *_preflist;
	struct afs_addr_preference pref;
	enum cmp_ret cmp;
	int ret, i, stop;

	if (argc != 2) {
		pr_warn("Wrong number of params\n");
		return -EINVAL;
	}

	if (strcmp(argv[0], "udp") != 0) {
		pr_warn("Unsupported protocol\n");
		return -EINVAL;
	}

	ret = afs_parse_address(argv[1], &pref);
	if (ret < 0)
		return ret;

	if (pref.family == AF_INET) {
		i = 0;
		stop = preflist->ipv6_off;
	} else {
		i = preflist->ipv6_off;
		stop = preflist->nr;
	}

	// Block Logic: Search for an exact match to delete.
	for (; i < stop; i++) {
		cmp = afs_cmp_address_pref(&pref, &preflist->prefs[i]);
		switch (cmp) {
		case CONTINUE_SEARCH:
			continue;
		case INSERT_HERE:
		case SUBNET_MATCH:
			return 0; // Not found, but not an error.
		case EXACT_MATCH:
			return afs_delete_address_pref(_preflist, i);
		}
	}

	return -ENOANO;
}

/**
 * afs_proc_addr_prefs_write - Write handler for /proc/fs/afs/addr_prefs.
 * @file: The file structure.
 * @buf: The user-provided buffer containing commands.
 * @size: The size of the buffer.
 *
 * This function orchestrates the update of the address preference list. It
 * employs a copy-on-write RCU mechanism for thread safety.
 *
 * 1. A new candidate list is allocated and initialized with the current data.
 * 2. Commands ("add" or "del") are parsed from the user buffer and applied
 *    to this new list.
 * 3. If all commands are successful, the global pointer `net->address_prefs`
 *    is atomically updated to point to the new list via `rcu_assign_pointer`.
 * 4. The old list is scheduled for freeing after a grace period using `kfree_rcu`.
 *
 * This ensures that readers accessing the list concurrently are not affected
 * by the update.
 *
 * Returns:
 * The number of bytes written on success, or a negative error code.
 */
int afs_proc_addr_prefs_write(struct file *file, char *buf, size_t size)
{
	struct afs_addr_preference_list *preflist, *old;
	struct seq_file *m = file->private_data;
	struct afs_net *net = afs_seq2net_single(m);
	size_t psize;
	char *argv[5];
	int ret, argc, max_prefs;

	inode_lock(file_inode(file));

	/* RCU copy-on-write: Allocate a new list and copy the old content. */
	old = rcu_dereference_protected(net->address_prefs,
					lockdep_is_held(&file_inode(file)->i_rwsem));

	if (old)
		max_prefs = old->nr + 1;
	else
		max_prefs = 1;

	psize = struct_size(old, prefs, max_prefs);
	psize = roundup_pow_of_two(psize);
	max_prefs = min_t(size_t, (psize - sizeof(*old)) / sizeof(old->prefs[0]), 255);

	ret = -ENOMEM;
	preflist = kmalloc(struct_size(preflist, prefs, max_prefs), GFP_KERNEL);
	if (!preflist)
		goto done;

	if (old)
		memcpy(preflist, old, struct_size(preflist, prefs, old->nr));
	else
		memset(preflist, 0, sizeof(*preflist));
	preflist->max_prefs = max_prefs;

	// Block Logic: Process all commands in the buffer against the new list.
	do {
		argc = afs_split_string(&buf, argv, ARRAY_SIZE(argv));
		if (argc < 0) {
			ret = argc;
			goto done;
		}
		if (argc < 2)
			goto inval;

		if (strcmp(argv[0], "add") == 0)
			ret = afs_add_address_pref(net, &preflist, argc - 1, argv + 1);
		else if (strcmp(argv[0], "del") == 0)
			ret = afs_del_address_pref(net, &preflist, argc - 1, argv + 1);
		else
			goto inval;
		if (ret < 0)
			goto done;
	} while (*buf);

	/* RCU publish: Atomically update the global list and schedule old for destruction. */
	preflist->version++;
	rcu_assign_pointer(net->address_prefs, preflist);
	/* Store prefs before version to ensure consistency for readers. */
	smp_store_release(&net->address_pref_version, preflist->version);
	kfree_rcu(old, rcu);
	preflist = NULL;
	ret = 0;

done:
	kfree(preflist);
	inode_unlock(file_inode(file));
	_leave(" = %d", ret);
	return ret;

inval:
	pr_warn("Invalid Command\n");
	ret = -EINVAL;
	goto done;
}

/**
 * afs_get_address_preferences_rcu - Update address priorities under RCU lock.
 * @net: The AFS network namespace.
 * @alist: The address list whose priorities are to be updated.
 *
 * This function iterates through a list of server addresses (`alist`) and
 * applies priorities from the global preference list (`net->address_prefs`).
 * It must be called under an RCU read lock to safely access the global list.
 * It checks the version to avoid redundant work if the priorities are already up to date.
 */
void afs_get_address_preferences_rcu(struct afs_net *net, struct afs_addr_list *alist)
{
	const struct afs_addr_preference_list *preflist =
		rcu_dereference(net->address_prefs);
	const struct sockaddr_in6 *sin6;
	const struct sockaddr_in *sin;
	const struct sockaddr *sa;
	struct afs_addr_preference test;
	enum cmp_ret cmp;
	int i, j;

	if (!preflist || !preflist->nr || !alist->nr_addrs ||
	    smp_load_acquire(&alist->addr_pref_version) == preflist->version)
		return;

	test.family = AF_INET;
	test.subnet_mask = 32;
	test.prio = 0;
	// Block Logic: Apply preferences to IPv4 addresses in the list.
	for (i = 0; i < alist->nr_ipv4; i++) {
		sa = rxrpc_kernel_remote_addr(alist->addrs[i].peer);
		sin = (const struct sockaddr_in *)sa;
		test.ipv4_addr = sin->sin_addr;
		for (j = 0; j < preflist->ipv6_off; j++) {
			cmp = afs_cmp_address_pref(&test, &preflist->prefs[j]);
			switch (cmp) {
			case CONTINUE_SEARCH:
				continue;
			case INSERT_HERE:
				break;
			case EXACT_MATCH:
			case SUBNET_MATCH:
				// Atomically update the priority for the matching address.
				WRITE_ONCE(alist->addrs[i].prio, preflist->prefs[j].prio);
				break;
			}
		}
	}

	test.family = AF_INET6;
	test.subnet_mask = 128;
	test.prio = 0;
	// Block Logic: Apply preferences to IPv6 addresses in the list.
	for (; i < alist->nr_addrs; i++) {
		sa = rxrpc_kernel_remote_addr(alist->addrs[i].peer);
		sin6 = (const struct sockaddr_in6 *)sa;
		test.ipv6_addr = sin6->sin6_addr;
		for (j = preflist->ipv6_off; j < preflist->nr; j++) {
			cmp = afs_cmp_address_pref(&test, &preflist->prefs[j]);
			switch (cmp) {
			case CONTINUE_SEARCH:
				continue;
			case INSERT_HERE:
				break;
			case EXACT_MATCH:
			case SUBNET_MATCH:
				// Atomically update the priority for the matching address.
				WRITE_ONCE(alist->addrs[i].prio, preflist->prefs[j].prio);
				break;
			}
		}
	}

	// Mark the list as updated to the current preference version.
	smp_store_release(&alist->addr_pref_version, preflist->version);
}

/**
 * afs_get_address_preferences - Update priorities if the preference table has changed.
 * @net: The AFS network namespace.
 * @alist: The address list to update.
 *
 * This function is a wrapper that checks if the global address preference list
 * has been updated since the last time this address list was checked. If so, it
 * acquires an RCU read lock and calls `afs_get_address_preferences_rcu` to
 * apply the new priorities. This version check serves as a fast path to avoid
 * taking the RCU lock unnecessarily.
 */
void afs_get_address_preferences(struct afs_net *net, struct afs_addr_list *alist)
{
	if (!net->address_prefs ||
	    /* Load version before prefs using an acquire barrier. */
	    smp_load_acquire(&net->address_pref_version) == alist->addr_pref_version)
		return;

	rcu_read_lock();
	afs_get_address_preferences_rcu(net, alist);
	rcu_read_unlock();
}
