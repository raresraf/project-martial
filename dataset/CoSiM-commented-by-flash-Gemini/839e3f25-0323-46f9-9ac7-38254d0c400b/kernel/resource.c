/**
 * @file linux/kernel/resource.c
 * @brief Provides a framework for arbitrary resource management within the Linux kernel.
 *
 * This file implements the core logic for managing system resources such as I/O ports
 * and memory regions. It offers mechanisms for requesting, releasing, inserting,
 * adjusting, and searching for resources, ensuring proper allocation and preventing
 * conflicts between different kernel components and device drivers.
 *
 * Copyright (C) 1999	Linus Torvalds
 * Copyright (C) 1999	Martin Mares <mj@ucw.cz>
 */
// SPDX-License-Identifier: GPL-2.0-only

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/export.h>
#include <linux/errno.h>
#include <linux/ioport.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/pseudo_fs.h>
#include <linux/sched.h>
#include <linux/seq_file.h>
#include <linux/device.h>
#include <linux/pfn.h>
#include <linux/mm.h>
#include <linux/mount.h>
#include <linux/resource_ext.h>
#include <uapi/linux/magic.h>
#include <linux/string.h>
#include <linux/vmalloc.h>
#include <asm/io.h>


/**
 * @brief Global resource structure for managing PCI I/O port space.
 * @details This structure defines the base resource for PCI I/O operations,
 * covering the entire range from 0 to IO_SPACE_LIMIT. Resources
 * within this range are typically allocated to device drivers for
 * accessing hardware registers and other I/O functionalities.
 */
struct resource ioport_resource = {
	.name	= "PCI IO",
	.start	= 0,
	.end	= IO_SPACE_LIMIT,
	.flags	= IORESOURCE_IO,
};
EXPORT_SYMBOL(ioport_resource);

/**
 * @brief Global resource structure for managing PCI memory space.
 * @details This structure defines the base resource for PCI memory-mapped I/O (MMIO)
 * operations, covering the entire addressable memory range. It is used to
 * track and allocate memory regions for device drivers to access hardware
 * devices as if they were memory locations.
 */
struct resource iomem_resource = {
	.name	= "PCI mem",
	.start	= 0,
	.end	= -1,
	.flags	= IORESOURCE_MEM,
};
EXPORT_SYMBOL(iomem_resource);

/**
 * @brief Read-write lock to protect the resource tree.
 * @details This spinlock ensures atomicity and consistency when modifying the
 * global resource tree, preventing race conditions during resource
 * allocation, deallocation, and other management operations.
 * Readers acquire a read lock, writers acquire a write lock.
 */
static DEFINE_RWLOCK(resource_lock);

/**
 * @brief Returns the next node in a pre-order traversal of the resource tree.
 * @param p The current resource node.
 * @param skip_children If true, skip the descendant nodes of @p in traversal.
 * @param subtree_root If @p is a descendant of @subtree_root, only traverse the subtree under @subtree_root.
 * @return The next resource node in pre-order traversal, or NULL if the traversal is complete or restricted by @subtree_root.
 */
static struct resource *next_resource(struct resource *p, bool skip_children,
				      struct resource *subtree_root)
{
	// Pre-condition: If not skipping children and current node has a child, traverse to child.
	// Invariant: The returned node will be the next node in pre-order traversal respecting skip_children and subtree_root.
	if (!skip_children && p->child)
		return p->child;
	// Invariant: Continue traversing up the parent chain if no sibling is found.
	while (!p->sibling && p->parent) {
		// Pre-condition: Current node is not the subtree root.
		// Invariant: If current node is the subtree root, traversal ends.
		p = p->parent;
		if (p == subtree_root)
			return NULL;
	}
	// Invariant: Returns the sibling if available, or NULL if at the end of a branch.
	return p->sibling;
}

/**
 * @brief Macro to traverse the resource subtree in pre-order.
 * @details This macro iterates through the resource nodes in a pre-order fashion,
 * starting from the children of the specified @_root. It excludes the @_root
 * itself from the traversal. It's designed for situations where the traversal
 * needs to begin from the immediate children of a given resource.
 *
 * @param _root The root of the subtree to traverse. The traversal starts from its children.
 * @param _p A pointer to a struct resource that will be updated with the current node in each iteration.
 * @param _skip_children A boolean flag. If true, the descendants of the current node will be skipped.
 *
 * NOTE: '__p' is an internal variable introduced to avoid shadowing '_p'
 * outside the loop, and it is referenced to prevent unused variable warnings.
 */
#define for_each_resource(_root, _p, _skip_children) \
	for (typeof(_root) __root = (_root), __p = _p = __root->child;	\
	     __p && _p; _p = next_resource(_p, _skip_children, __root))

#ifdef CONFIG_PROC_FS

enum { MAX_IORES_LEVEL = 5 };

/**
 * @brief Callback for seq_file to start iterating through resources.
 * @details This function is part of the `seq_operations` for `/proc/iomem` and `/proc/ioports`.
 * It acquires a read lock on `resource_lock` and finds the resource corresponding
 * to the given position in the traversal.
 *
 * @param m The seq_file handle.
 * @param pos The current position in the sequence, updated to reflect the next item.
 * @return A pointer to the resource at the given position, or NULL if no more resources.
 * @acquires resource_lock
 */
static void *r_start(struct seq_file *m, loff_t *pos)
	__acquires(resource_lock)
{
	struct resource *root = pde_data(file_inode(m->file));
	struct resource *p;
	loff_t l = *pos;

	read_lock(&resource_lock);
	// Pre-condition: 'root' points to the starting resource for traversal.
	// Invariant: The loop iterates through resources, decrementing 'l' until it reaches 0 or no more resources.
	for_each_resource(root, p, false) {
		// Pre-condition: 'l' is the remaining count to skip.
		// Invariant: When 'l' becomes 0, 'p' holds the resource at the desired position.
		if (l-- == 0)
			break;
	}

	return p;
}

/**
 * @brief Callback for seq_file to get the next resource in the iteration.
 * @details This function is part of the `seq_operations` for `/proc/iomem` and `/proc/ioports`.
 * It increments the position counter and returns the next resource using `next_resource`.
 *
 * @param m The seq_file handle.
 * @param v The current resource node, previously returned by r_start or r_next.
 * @param pos The current position in the sequence, incremented by this function.
 * @return A pointer to the next resource node, or NULL if no more resources.
 */
static void *r_next(struct seq_file *m, void *v, loff_t *pos)
{
	struct resource *p = v;

	// Invariant: Increment the position for the next iteration.
	(*pos)++;

	// Pre-condition: 'p' is a valid resource.
	// Invariant: Returns the next resource in pre-order traversal.
	return (void *)next_resource(p, false, NULL);
}

/**
 * @brief Callback for seq_file to stop iterating through resources.
 * @details This function is part of the `seq_operations` for `/proc/iomem` and `/proc/ioports`.
 * It releases the read lock on `resource_lock` that was acquired by `r_start`.
 *
 * @param m The seq_file handle.
 * @param v The last resource node processed (unused in this function).
 * @releases resource_lock
 */
static void r_stop(struct seq_file *m, void *v)
	__releases(resource_lock)
{
	// Invariant: Release the read lock, completing the critical section.
	read_unlock(&resource_lock);
}

/**
 * @brief Callback for seq_file to display resource information.
 * @details This function is part of the `seq_operations` for `/proc/iomem` and `/proc/ioports`.
 * It formats and prints details about a given resource, including its depth in the
 * resource tree, start and end addresses, and name. Access to actual addresses
 * is restricted to users with `CAP_SYS_ADMIN` capabilities within the initial
 * user namespace.
 *
 * @param m The seq_file handle.
 * @param v A pointer to the resource node to display.
 * @return 0 on success.
 */
static int r_show(struct seq_file *m, void *v)
{
	struct resource *root = pde_data(file_inode(m->file));
	struct resource *r = v, *p;
	unsigned long long start, end;
	int width = root->end < 0x10000 ? 4 : 8;
	int depth;

	// Pre-condition: 'r' is a valid resource node, 'root' is the tree root.
	// Invariant: 'depth' will represent the nesting level of 'r' under 'root', up to MAX_IORES_LEVEL.
	for (depth = 0, p = r; depth < MAX_IORES_LEVEL; depth++, p = p->parent)
		// Pre-condition: The current parent 'p->parent' is the 'root'.
		// Invariant: If 'p->parent' is the 'root', 'depth' is the direct child depth.
		if (p->parent == root)
			break;

	// Pre-condition: User capabilities determine visibility of resource addresses.
	// Invariant: 'start' and 'end' are either the actual resource boundaries or zeroed out for unprivileged users.
	if (file_ns_capable(m->file, &init_user_ns, CAP_SYS_ADMIN)) {
		start = r->start;
		end = r->end;
	} else {
		start = end = 0;
	}

	// Pre-condition: Formatted strings for depth, start, end, and name are available.
	// Invariant: The resource information is printed to the seq_file in a structured format.
	seq_printf(m, "%*s%0*llx-%0*llx : %s\n",
			depth * 2, "",
			width, start,
			width, end,
			r->name ? r->name : "<BAD>");
	return 0;
}

/**
 * @brief `seq_operations` structure for displaying resource information in `/proc`.
 * @details This structure defines the operations required by the `seq_file` interface
 * to iterate and display resource information, typically used for `/proc/iomem`
 * and `/proc/ioports` entries. It connects the `r_start`, `r_next`, `r_stop`,
 * and `r_show` functions to the kernel's sequence file mechanism.
 */
static const struct seq_operations resource_op = {
	.start	= r_start,
	.next	= r_next,
	.stop	= r_stop,
	.show	= r_show,
};

/**
 * @brief Initializes the `/proc/ioports` and `/proc/iomem` entries.
 * @details This function creates the `/proc/ioports` and `/proc/iomem` files,
 * which allow userspace to inspect the kernel's I/O port and memory
 * resource allocations, respectively. It associates these `/proc` entries
 * with the `resource_op` sequence file operations.
 *
 * @return 0 on successful initialization.
 */
static int __init ioresources_init(void)
{
	// Invariant: Create /proc/ioports entry with read-only permissions,
	// using resource_op for sequence operations and ioport_resource as private data.
	proc_create_seq_data("ioports", 0, NULL, &resource_op,
			&ioport_resource);
	// Invariant: Create /proc/iomem entry with read-only permissions,
	// using resource_op for sequence operations and iomem_resource as private data.
	proc_create_seq_data("iomem", 0, NULL, &resource_op, &iomem_resource);
	return 0;
}
__initcall(ioresources_init);

#endif /* CONFIG_PROC_FS */

/**
 * @brief Frees a resource structure if it was allocated from the slab.
 * @details This function checks if the provided resource was allocated using `kmalloc`
 * (specifically, if it resides on a slab page) and, if so, frees it.
 * Resources allocated by `memblock` early in boot are not freed here to
 * avoid overcomplicating resource handling and potential memory leaks
 * due to partial page returns.
 *
 * @param res A pointer to the resource structure to be freed.
 */
static void free_resource(struct resource *res)
{
	// Pre-condition: 'res' is a valid resource pointer.
	// Invariant: If 'res' is valid and was allocated from the slab (PageSlab returns true),
	// it will be freed using kfree. Otherwise, no action is taken.
	if (res && PageSlab(virt_to_head_page(res)))
		kfree(res);
}

/**
 * @brief Allocates and zeroes out a new resource structure.
 * @details This function allocates memory for a `struct resource` and
 * initializes all its fields to zero. The allocation is performed
 * using `kzalloc`, which takes `gfp_t` flags to specify memory
 * allocation behavior (e.g., `GFP_KERNEL` for normal kernel allocations).
 *
 * @param flags Kernel GFP flags for memory allocation.
 * @return A pointer to the newly allocated and zeroed `struct resource` on success, or NULL on failure.
 */
static struct resource *alloc_resource(gfp_t flags)
{
	// Invariant: kzalloc attempts to allocate memory for a resource structure and clears it.
	return kzalloc(sizeof(struct resource), flags);
}

/**
 * @brief Attempts to request a new resource within a resource tree.
 * @details This function tries to insert a `new` resource into the resource tree
 * rooted at `root`. It checks for various validity conditions, such as
 * `new` resource's start and end addresses being within `root`'s boundaries.
 * If the `new` resource conflicts with an existing resource, it returns
 * the conflicting resource. Otherwise, it inserts the `new` resource and returns NULL.
 *
 * @param root The root resource of the tree where the new resource is to be requested.
 * @param new The new resource to be requested.
 * @return A pointer to the conflicting resource if insertion fails, or NULL if successful.
 */
static struct resource * __request_resource(struct resource *root, struct resource *new)
{
	resource_size_t start = new->start;
	resource_size_t end = new->end;
	struct resource *tmp, **p;

	// Pre-condition: Ensure the new resource's end address is not less than its start address.
	// Invariant: If invalid range, return root as conflict.
	if (end < start)
		return root;
	// Pre-condition: Ensure the new resource starts within the root's bounds.
	// Invariant: If out of bounds, return root as conflict.
	if (start < root->start)
		return root;
	// Pre-condition: Ensure the new resource ends within the root's bounds.
	// Invariant: If out of bounds, return root as conflict.
	if (end > root->end)
		return root;

	p = &root->child;
	// Invariant: Loop indefinitely until a conflict is found or the resource is successfully inserted.
	for (;;) {
		tmp = *p;
		// Pre-condition: No existing resource at current pointer, or existing resource starts after new resource ends.
		// Invariant: If no conflict, insert new resource at this position.
		if (!tmp || tmp->start > end) {
			new->sibling = tmp; // Link new resource to the next one
			*p = new;           // Insert new resource into the list
			new->parent = root; // Set parent of new resource
			return NULL;        // Successfully inserted
		}
		p = &tmp->sibling; // Move to the next sibling pointer
		// Pre-condition: Current existing resource ends before new resource starts.
		// Invariant: No conflict with current resource, continue to next.
		if (tmp->end < start)
			continue;
		// Invariant: Conflict found, return the conflicting resource.
		return tmp;
	}
}

/**
 * @brief Internal function to release a resource from its parent.
 * @details This function removes a specified resource `old` from its parent's child list.
 * It handles two cases for child resources:
 * 1. If `release_child` is true or `old` has no children, `old` is simply removed,
 *    and its siblings are re-linked.
 * 2. If `release_child` is false and `old` has children, `old`'s children are
 *    promoted to become siblings of `old`'s previous siblings, effectively
 *    removing `old` but preserving its subtree.
 *
 * @param old The resource to be released.
 * @param release_child If true, any children of `old` are also conceptually released
 *                      (removed along with `old`). If false, children are promoted.
 * @return 0 on success, or -EINVAL if `old` is not found in its parent's child list.
 */
static int __release_resource(struct resource *old, bool release_child)
{
	struct resource *tmp, **p, *chd;

	p = &old->parent->child;
	// Invariant: Loop through the parent's children to find the 'old' resource.
	for (;;) {
		tmp = *p;
		// Pre-condition: 'tmp' is NULL, meaning 'old' was not found in the list.
		// Invariant: Exit loop, 'old' not found.
		if (!tmp)
			break;
		// Pre-condition: 'tmp' is the 'old' resource to be removed.
		if (tmp == old) {
			// Invariant: If children are to be released or no children exist,
			// simply remove 'tmp' from the list.
			if (release_child || !(tmp->child)) {
				*p = tmp->sibling;
			} else {
				// Invariant: Children are promoted to replace 'tmp'.
				// Iterate through children to update their parent pointers.
				for (chd = tmp->child;; chd = chd->sibling) {
					chd->parent = tmp->parent;
					if (!(chd->sibling))
						break;
				}
				// Link parent to 'tmp's first child.
				*p = tmp->child;
				// Link 'tmp's last child to 'tmp's former sibling.
				chd->sibling = tmp->sibling;
			}
			old->parent = NULL; // Clear parent pointer for the released resource.
			return 0; // Successfully released.
		}
		p = &tmp->sibling; // Move to the next sibling pointer.
	}
	return -EINVAL; // Resource not found, invalid argument.
}

/**
 * @brief Recursively releases all child resources of a given resource.
 * @details This internal function iterates through all direct children of `r`,
 * clears their parent and sibling pointers, and then recursively calls
 * itself to release their children. After releasing the subtree, it
 * restores the size of the released child resource (resetting start/end)
 * while preserving its flags.
 *
 * @param r The parent resource whose children are to be released.
 */
static void __release_child_resources(struct resource *r)
{
	struct resource *tmp, *p;
	resource_size_t size;

	p = r->child;
	r->child = NULL; /* Disconnect all children from the parent 'r' to begin independent processing. */

	/*
	 * Pre-condition: 'p' points to the first child of the original parent 'r' (or NULL if no children).
	 * Invariant: This loop iterates through each direct child of the initial resource 'r',
	 *            detaching them, recursively processing their own children, and resetting
	 *            their start/end fields while preserving flags.
	 */
	while (p) {
		tmp = p; /* Store current child for processing. */
		p = p->sibling; /* Advance 'p' to the next sibling before processing 'tmp' to maintain loop integrity. */

		tmp->parent = NULL; /* Clear parent pointer for the detached child. */
		tmp->sibling = NULL; /* Clear sibling pointer as it's now an independent resource for recursive processing. */
		/*
		 * Pre-condition: 'tmp' is a valid resource node.
		 * Invariant: All descendants of 'tmp' will be recursively processed and detached.
		 */
		__release_child_resources(tmp);

		printk(KERN_DEBUG "release child resource %pR\n", tmp);
		/* Restore original size and preserve flags for the released child resource. */
		size = resource_size(tmp); /* Calculate the original size of the resource based on its previous start and end. */
		tmp->start = 0; /* Reset start address to 0. */
		tmp->end = size - 1; /* Reset end address to reflect its original size from 0. */
	}
}

/**
 * @brief Releases all child resources of a given resource, protected by a write lock.
 * @details This function acquires a write lock on the global `resource_lock` to ensure
 * exclusive access during the modification of the resource tree. It then delegates
 * the recursive release of child resources to `__release_child_resources` and
 * finally releases the write lock.
 *
 * @param r The parent resource whose children are to be released.
 */
void release_child_resources(struct resource *r)
{
	write_lock(&resource_lock);
	__release_child_resources(r);
	write_unlock(&resource_lock);
}

/**
 * @brief Requests and reserves an I/O or memory resource, returning the conflicting resource if any.
 * @details This function attempts to reserve a resource `new` under the `root` resource.
 * If the request causes a conflict with an existing resource, a pointer to the
 * conflicting resource is returned. Otherwise, the new resource is successfully
 * requested, and NULL is returned. The operation is protected by a write lock.
 *
 * @param root The root resource descriptor.
 * @param new The resource descriptor desired by the caller.
 * @return A pointer to the conflicting resource on error, or NULL on success.
 */
struct resource *request_resource_conflict(struct resource *root, struct resource *new)
{
	struct resource *conflict;

	write_lock(&resource_lock);
	conflict = __request_resource(root, new);
	write_unlock(&resource_lock);
	return conflict;
}

/**
 * @brief Requests and reserves an I/O or memory resource.
 * @details This function attempts to reserve a resource `new` under the `root` resource.
 * It's a wrapper around `request_resource_conflict`, returning 0 on success
 * and a negative error code (`-EBUSY`) if a conflict is encountered.
 *
 * @param root The root resource descriptor.
 * @param new The resource descriptor desired by the caller.
 * @return 0 for success, negative error code on error.
 */
int request_resource(struct resource *root, struct resource *new)
{
	struct resource *conflict;

	conflict = request_resource_conflict(root, new);
	return conflict ? -EBUSY : 0;
}

EXPORT_SYMBOL(request_resource);

/**
 * @brief Releases a previously reserved resource.
 * @details This function releases a resource pointed to by `old`. It acquires
 * a write lock to ensure exclusive access to the resource tree during the
 * release operation. The `__release_resource` internal function is used
 * with `true` for `release_child`, meaning any children of `old` are also
 * conceptually released.
 *
 * @param old The resource pointer to release.
 * @return 0 on success, or an error code if the release fails.
 */
int release_resource(struct resource *old)
{
	int retval;

	write_lock(&resource_lock);
	retval = __release_resource(old, true);
	write_unlock(&resource_lock);
	return retval;
}

EXPORT_SYMBOL(release_resource);

/**
 * @brief Checks if a resource's flags and descriptor match the specified criteria.
 * @details This utility function determines if the given resource `p` has all the `flags`
 * set and if its descriptor `p->desc` matches the `desc` parameter. If `desc`
 * is `IORES_DESC_NONE`, the descriptor check is skipped. This is crucial for
 * filtering resources during search operations.
 *
 * @param p A pointer to the `struct resource` to check.
 * @param flags The flags that must be set in `p->flags`.
 * @param desc The descriptor that `p->desc` must match, or `IORES_DESC_NONE` to skip.
 * @return True if the resource matches the type criteria, false otherwise.
 */
static bool is_type_match(struct resource *p, unsigned long flags, unsigned long desc)
{
	return (p->flags & flags) == flags && (desc == IORES_DESC_NONE || desc == p->desc);
}

/**
 * @brief Finds the lowest I/O memory resource that covers part of a specified range.
 * @details This function searches for the lowest I/O memory (`iomem_resource`)
 * that overlaps with the given range (`@start` to `@end`) and matches the
 * specified `flags` and `desc`. If a matching resource is found, it returns
 * 0 and the `res` parameter is overwritten with the intersecting part of that
 * resource. If no resource is found, it returns `-ENODEV`. Invalid parameters
 * result in `-EINVAL`. The operation is protected by a read lock.
 *
 * @param start Start address of the resource searched for.
 * @param end End address of same resource.
 * @param flags Flags which the resource must have.
 * @param desc Descriptor the resource must have (or `IORES_DESC_NONE`).
 * @param res Return pointer, if resource found.
 * @return 0 for success, -ENODEV if none found, -EINVAL for invalid parameters.
 */
static int find_next_iomem_res(resource_size_t start, resource_size_t end,
			       unsigned long flags, unsigned long desc,
			       struct resource *res)
{
	struct resource *p;

	if (!res)
		return -EINVAL;

	if (start >= end)
		return -EINVAL;

	read_lock(&resource_lock);

	for_each_resource(&iomem_resource, p, false) {
		/* If we passed the resource we are looking for, stop */
		if (p->start > end) {
			p = NULL;
			break;
		}

		/* Skip until we find a range that matches what we look for */
		if (p->end < start)
			continue;

		/* Found a match, break */
		if (is_type_match(p, flags, desc))
			break;
	}

	if (p) {
		/* copy data */
		*res = (struct resource) {
			.start = max(start, p->start),
			.end = min(end, p->end),
			.flags = p->flags,
			.desc = p->desc,
			.parent = p->parent,
		};
	}

	read_unlock(&resource_lock);
	return p ? 0 : -ENODEV;
}

/**
 * @brief Internal function to walk through I/O memory resources matching criteria and apply a callback.
 * @details This function iterates over I/O memory resources within a specified range (`start` to `end`)
 * that match the given `flags` and `desc`. For each matching resource, it invokes the
 * provided callback function `func`. The iteration stops if the callback returns a non-zero
 * value or if no more matching resources are found within the range.
 *
 * @param start The starting address of the region to search.
 * @param end The ending address of the region to search.
 * @param flags Resource flags to match.
 * @param desc Resource descriptor to match (or `IORES_DESC_NONE` to ignore).
 * @param arg An opaque argument passed to the callback function.
 * @param func The callback function to execute for each matching resource.
 * @return The return value of the last executed callback function, or -EINVAL if an invalid parameter.
 */
static int __walk_iomem_res_desc(resource_size_t start, resource_size_t end,
				 unsigned long flags, unsigned long desc,
				 void *arg,
				 int (*func)(struct resource *, void *))
{
	struct resource res;
	int ret = -EINVAL;

	// Invariant: Continues as long as 'start' is less than 'end' and a next I/O memory resource is found.
	while (start < end &&
	       !find_next_iomem_res(start, end, flags, desc, &res)) {
		// Pre-condition: A matching resource 'res' has been found.
		// Invariant: Execute the callback function with the found resource.
		ret = (*func)(&res, arg);
		// Pre-condition: Callback returned a non-zero value.
		// Invariant: Stop iteration if the callback indicates completion or an error.
		if (ret)
			break;

		start = res.end + 1; // Move 'start' past the current resource for the next iteration.
	}

	return ret;
}

/**
 * @brief Walks through I/O memory resources and calls func() with matching resource ranges.
 * @details This function is the public interface for `__walk_iomem_res_desc`.
 * It iterates over all memory ranges which overlap `start`, `end`, and also
 * match `flags` and `desc`. For each qualifying resource area, it calls the
 * provided `func` callback.
 *
 * @param desc I/O resource descriptor. Use `IORES_DESC_NONE` to skip `desc` check.
 * @param flags I/O resource flags.
 * @param start Start address.
 * @param end End address.
 * @param arg Function argument for the callback `func`.
 * @param func Callback function that is called for each qualifying resource area.
 */
int walk_iomem_res_desc(unsigned long desc, unsigned long flags, u64 start,
		u64 end, void *arg, int (*func)(struct resource *, void *))
{
	return __walk_iomem_res_desc(start, end, flags, desc, arg, func);
}
EXPORT_SYMBOL_GPL(walk_iomem_res_desc);

/**
 * @brief Walks through System RAM resources and calls a callback function.
 * @details This function iterates over all memory ranges of type `System RAM` that are
 * marked with `IORESOURCE_SYSTEM_RAM` and `IORESOURCE_BUSY`. It calls the
 * provided `func` callback for each matching resource range. This function
 * is specifically designed for System RAM and processes full ranges, not PFNs.
 * It's important to note that if resources are not PFN-aligned, direct PFN
 * handling could lead to truncated ranges, which this function avoids by
 * working with full resource ranges.
 *
 * @param start The starting address for the search.
 * @param end The ending address for the search.
 * @param arg An opaque argument passed to the callback function.
 * @param func The callback function to execute for each qualifying resource area.
 * @return The return value of the underlying `__walk_iomem_res_desc` function.
 */
int walk_system_ram_res(u64 start, u64 end, void *arg,
			int (*func)(struct resource *, void *))
{
	unsigned long flags = IORESOURCE_SYSTEM_RAM | IORESOURCE_BUSY;

	return __walk_iomem_res_desc(start, end, flags, IORES_DESC_NONE, arg,
				     func);
}

/**
 * @brief Walks through System RAM resources in reverse order and calls a callback function.
 * @details This function is a variant of `walk_system_ram_res()`. It calls the `func`
 * callback against all memory ranges of type System RAM which are marked as
 * `IORESOURCE_SYSTEM_RAM` and `IORESOURCE_BUSY`, but processes them in reversed
 * order, i.e., from higher to lower addresses. It collects all matching resources
 * first and then iterates over them in reverse.
 *
 * @param start The starting address for the search.
 * @param end The ending address for the search.
 * @param arg An opaque argument passed to the callback function.
 * @param func The callback function to execute for each qualifying resource area.
 * @return The return value of the last executed callback function, or -ENOMEM if memory allocation fails.
 */
int walk_system_ram_res_rev(u64 start, u64 end, void *arg,
				int (*func)(struct resource *, void *))
{
	struct resource res, *rams;
	int rams_size = 16, i;
	unsigned long flags;
	int ret = -1;

	/* Pre-condition: 'rams' must be allocated to store found resources. */
	/* Invariant: Allocate memory to store resource descriptors for reverse traversal. */
	rams = kvcalloc(rams_size, sizeof(struct resource), GFP_KERNEL);
	if (!rams)
		return ret;

	flags = IORESOURCE_SYSTEM_RAM | IORESOURCE_BUSY;
	i = 0;
	/* Pre-condition: 'start' must be less than 'end' and a new resource must be found. */
	/* Invariant: Collects all matching system RAM resources within the specified range. */
	while ((start < end) &&
		(!find_next_iomem_res(start, end, flags, IORES_DESC_NONE, &res))) {
		/* Pre-condition: Check if the allocated array is large enough. */
		/* Invariant: Reallocate 'rams' if current capacity is exceeded. */
		if (i >= rams_size) {
			/* re-alloc */
			struct resource *rams_new;

			rams_new = kvrealloc(rams, (rams_size + 16) * sizeof(struct resource),
					     GFP_KERNEL);
			if (!rams_new)
				goto out;

			rams = rams_new;
			rams_size += 16;
		}

		rams[i++] = res;
		start = res.end + 1;
	}

	/* Pre-condition: 'rams' contains collected resources. */
	/* Invariant: Iterate through the collected resources in reverse order and apply the callback. */
	for (i--; i >= 0; i--) {
		ret = (*func)(&rams[i], arg);
		if (ret)
			break;
	}

out:
	kvfree(rams);
	return ret;
}

/**
 * @brief Walks through generic memory resources and calls a callback function.
 * @details This function iterates over all memory ranges that are marked with
 * `IORESOURCE_MEM` and `IORESOURCE_BUSY` flags. For each matching resource,
 * it invokes the provided `func` callback. This is a general-purpose function
 * for traversing busy memory regions.
 *
 * @param start The starting address for the search.
 * @param end The ending address for the search.
 * @param arg An opaque argument passed to the callback function.
 * @param func The callback function to execute for each qualifying resource area.
 * @return The return value of the underlying `__walk_iomem_res_desc` function.
 */
int walk_mem_res(u64 start, u64 end, void *arg,
		 int (*func)(struct resource *, void *))
{
	unsigned long flags = IORESOURCE_MEM | IORESOURCE_BUSY;

	return __walk_iomem_res_desc(start, end, flags, IORES_DESC_NONE, arg,
				     func);
}

/**
 * @brief Walks through System RAM page ranges and calls a callback function.
 * @details This function iterates over all memory ranges of type System RAM that are
 * marked as `IORESOURCE_SYSTEM_RAM` and `IORESOURCE_BUSY`. It converts these
 * resource ranges into page frame numbers (PFNs) and calls the provided
 * `func` callback for each valid PFN range. This function is specifically
 * designed for System RAM and operates on page-aligned ranges.
 *
 * @param start_pfn The starting Page Frame Number for the search.
 * @param nr_pages The number of pages to search from `start_pfn`.
 * @param arg An opaque argument passed to the callback function.
 * @param func The callback function to execute for each qualifying PFN range.
 * @return The return value of the last executed callback, or -EINVAL if an error occurs.
 */
int walk_system_ram_range(unsigned long start_pfn, unsigned long nr_pages,
			  void *arg, int (*func)(unsigned long, unsigned long, void *))
{
	resource_size_t start, end;
	unsigned long flags;
	struct resource res;
	unsigned long pfn, end_pfn;
	int ret = -EINVAL;

	/* Pre-condition: Calculate byte addresses from PFNs. */
	/* Invariant: 'start' and 'end' define the byte range corresponding to the given PFNs. */
	start = (u64) start_pfn << PAGE_SHIFT;
	end = ((u64)(start_pfn + nr_pages) << PAGE_SHIFT) - 1;
	flags = IORESOURCE_SYSTEM_RAM | IORESOURCE_BUSY;

	/* Pre-condition: 'start' must be less than 'end' and a matching resource must be found. */
	/* Invariant: Iterate through matching I/O memory resources within the specified byte range. */
	while (start < end &&
	       !find_next_iomem_res(start, end, flags, IORES_DESC_NONE, &res)) {
		/* Pre-condition: 'res' is a valid resource. */
		/* Invariant: Convert resource boundaries to PFNs and call the callback function. */
		pfn = PFN_UP(res.start);
		end_pfn = PFN_DOWN(res.end + 1);
		if (end_pfn > pfn)
			ret = (*func)(pfn, end_pfn - pfn, arg);
		if (ret) /* If callback returns non-zero, stop iteration. */
			break;
		start = res.end + 1; /* Move start past the current resource for the next iteration. */
	}
	return ret;
}

/**
 * @brief Simple callback function to indicate a region is RAM.
 * @details This function serves as a straightforward callback for `walk_system_ram_range`.
 * It always returns 1, effectively marking the queried PFN range as System RAM.
 * It does not perform any complex checks, merely confirms the presence of RAM
 * as determined by the caller of the `walk_system_ram_range` function.
 *
 * @param pfn The starting Page Frame Number.
 * @param nr_pages The number of pages.
 * @param arg An opaque argument (unused in this function).
 * @return Always returns 1 to indicate the region is RAM.
 */
static int __is_ram(unsigned long pfn, unsigned long nr_pages, void *arg)
{
	return 1;
}

/**
 * @brief Checks if a specified page frame number (PFN) corresponds to System RAM.
 * @details This function is a weak alias that queries the `iomem_resource` list
 * to determine if the given PFN falls within a registered System RAM region.
 * It uses `walk_system_ram_range` with the `__is_ram` callback to perform this check.
 *
 * @param pfn The Page Frame Number to check.
 * @return True if the PFN corresponds to System RAM, false otherwise.
 */
int __weak page_is_ram(unsigned long pfn)
{
	return walk_system_ram_range(pfn, 1, NULL, __is_ram) == 1;
}
EXPORT_SYMBOL_GPL(page_is_ram);

/**
 * @brief Internal function to determine how a given region intersects with existing resources.
 * @details This function checks for intersections between a specified memory region (defined by `start` and `size`)
 * and the child resources of a given `parent` resource. It categorizes the intersection into
 * three types: `REGION_DISJOINT`, `REGION_INTERSECTS`, or `REGION_MIXED`, based on whether
 * the region overlaps with resources matching the specified `flags` and `desc`, or with
 * other types of resources. It also handles nested resources to properly account for coverage.
 *
 * @param parent The root resource whose children are to be checked for intersection.
 * @param start The starting address of the region to check.
 * @param size The size of the region to check.
 * @param flags Resource flags to match for specific intersections.
 * @param desc Resource descriptor to match (or `IORES_DESC_NONE` to ignore).
 * @return `REGION_DISJOINT` if no overlap, `REGION_INTERSECTS` if overlap only with matching types,
 *         `REGION_MIXED` if overlap with matching types and other resources.
 */
static int __region_intersects(struct resource *parent, resource_size_t start,
			       size_t size, unsigned long flags,
			       unsigned long desc)
{
	int type = 0; /* Counts intersections with resources matching specified flags/desc. */
	int other = 0; /* Counts intersections with resources not matching specified flags/desc. */
	struct resource *p, *dp;
	struct resource res, o;
	bool covered;

	res = DEFINE_RES(start, size, 0); /* Define a temporary resource representing the input region. */

	/* Invariant: Iterate through all direct children of the parent resource. */
	for (p = parent->child; p ; p = p->sibling) {
		/* Pre-condition: Check if the current child resource 'p' intersects with the region 'res'. */
		if (!resource_intersection(p, &res, &o))
			continue; /* If no intersection, move to the next sibling. */
		/* Pre-condition: Check if the intersecting part 'o' matches the desired type. */
		if (is_type_match(p, flags, desc)) {
			type++; /* Increment count for matching resource type. */
			continue; /* Move to the next sibling. */
		}
		/*
		 * This block handles cases where 'p' itself doesn't match the flags/desc,
		 * but its descendants might. This is crucial for properly assessing regions
		 * that might be covered by nested resources of the desired type, even if
		 * an encompassing parent resource is of a different or generic type.
		 *
		 * For example, a larger "CXL Window" might contain smaller "System RAM" regions.
		 * When searching for "System RAM", an overlap with the "CXL Window" needs
		 * further examination of its children to find the actual "System RAM" within.
		 */
		covered = false; /* Flag to track if the overlapping portion 'o' is fully covered by matching descendants. */
		/* Invariant: Iterate through all descendants of 'p' (including 'p' itself if using for_each_resource). */
		for_each_resource(p, dp, false) {
			/* Pre-condition: Check if the descendant 'dp' overlaps with the region 'res'. */
			if (!resource_overlaps(dp, &res))
				continue; /* If no overlap, move to the next descendant. */
			/* Pre-condition: Check if the overlapping descendant 'dp' matches the desired type. */
			if (is_type_match(dp, flags, desc)) {
				type++; /* Increment count for matching resource type. */
				/*
				 * Pre-condition: 'dp' starts after the current 'o.start'.
				 * Invariant: This means there's a gap in 'o' that isn't covered by 'dp',
				 *            so the search for full coverage within 'o' should break.
				 */
				if (dp->start > o.start)
					break;
				/* Pre-condition: 'dp' covers the remaining part of 'o'. */
				if (dp->end >= o.end) {
					covered = true; /* Mark as fully covered. */
					break; /* Exit descendant loop as 'o' is fully covered. */
				}
				/* Invariant: If 'dp' covers only a part, adjust 'o.start' to continue searching for coverage. */
				o.start = max(o.start, dp->end + 1);
			}
		}
		/* Pre-condition: If 'o' (the intersection with 'p') was not fully covered by matching descendants. */
		if (!covered)
			other++; /* Increment count for other intersecting resources. */
	}

	/* Pre-condition: No intersections with resources matching the specified flags/desc. */
	if (type == 0)
		return REGION_DISJOINT; /* No matching resources found in the region. */

	/* Pre-condition: Intersections only with resources matching the specified flags/desc, no other types. */
	if (other == 0)
		return REGION_INTERSECTS; /* Region intersects exclusively with specified resource types. */

	/* Invariant: Intersections with both matching types and other resource types. */
	return REGION_MIXED; /* Region intersects with a mix of resource types. */
}

/**
 * @brief Determines intersection of a region with known resources.
 * @details Checks if the specified region partially overlaps or fully eclipses a
 * resource identified by `flags` and `desc` (optional with `IORES_DESC_NONE`).
 * Returns `REGION_DISJOINT` if the region does not overlap `flags`/`desc`,
 * returns `REGION_MIXED` if the region overlaps `flags`/`desc` and another
 * resource, and returns `REGION_INTERSECTS` if the region overlaps `flags`/`desc`
 * and no other defined resource. Note that `REGION_INTERSECTS` is also
 * returned in the case when the specified region overlaps RAM and undefined
 * memory holes.
 *
 * `region_intersects()` is used by memory remapping functions to ensure
 * the user is not remapping RAM and is a vast speed up over walking
 * through the resource table page by page.
 *
 * @param start Region start address.
 * @param size Size of region.
 * @param flags Flags of resource (in `iomem_resource`).
 * @param desc Descriptor of resource (in `iomem_resource`) or `IORES_DESC_NONE`.
 * @return `REGION_DISJOINT`, `REGION_INTERSECTS`, or `REGION_MIXED`.
 */
int region_intersects(resource_size_t start, size_t size, unsigned long flags,
		      unsigned long desc)
{
	int ret;

	read_lock(&resource_lock);
	ret = __region_intersects(&iomem_resource, start, size, flags, desc);
	read_unlock(&resource_lock);

	return ret;
}
EXPORT_SYMBOL_GPL(region_intersects);

/**
 * @brief Weak declaration for architecture-specific removal of reservations.
 * @details This function is a weak symbol, meaning it can be overridden by
 * architecture-specific implementations. Its purpose is to perform any
 * necessary architectural clean-up or removal of reservations for a given
 * resource `avail`. By default, it does nothing.
 *
 * @param avail The resource for which to remove reservations.
 */
void __weak arch_remove_reservations(struct resource *avail)
{
}

/**
 * @brief Clips a resource's start and end addresses to within specified bounds.
 * @details This utility function modifies the provided resource `res` such that its
 * `start` address is not less than `min` and its `end` address is not
 * greater than `max`. This ensures that the resource remains within a valid
 * or desired address range.
 *
 * @param res A pointer to the `struct resource` to be clipped.
 * @param min The minimum allowed start address for the resource.
 * @param max The maximum allowed end address for the resource.
 */
static void resource_clip(struct resource *res, resource_size_t min,
			  resource_size_t max)
{
	// Pre-condition: 'res->start' is potentially less than 'min'.
	// Invariant: 'res->start' will be at least 'min' after this operation.
	if (res->start < min)
		res->start = min;
	// Pre-condition: 'res->end' is potentially greater than 'max'.
	// Invariant: 'res->end' will be at most 'max' after this operation.
	if (res->end > max)
		res->end = max;
}

/**
 * @brief Finds an empty space in the resource tree that satisfies given range and alignment constraints.
 * @details This internal function searches for a contiguous block of available resource space
 * within the resource tree rooted at `root`. It takes into account an optional
 * `old` resource (which can be a hole in the search if it's being reallocated),
 * the required `size`, and various `constraint` properties like minimum/maximum
 * bounds and alignment requirements. It attempts to find a suitable region for `new`
 * and updates `new->start` and `new->end` upon success.
 *
 * @param root The root of the resource tree to search within.
 * @param old An optional existing resource that is being reallocated; its space is temporarily considered free.
 * @param new The new resource structure to populate with the found space.
 * @param size The required size of the resource space.
 * @param constraint A structure specifying alignment, minimum, and maximum address constraints.
 * @return 0 on success, -EBUSY if no suitable space is found.
 */
static int __find_resource_space(struct resource *root, struct resource *old,
				 struct resource *new, resource_size_t size,
				 struct resource_constraint *constraint)
{
	struct resource *this = root->child;
	struct resource tmp = *new, avail, alloc;
	resource_alignf alignf = constraint->alignf;

	tmp.start = root->start;
	/*
	 * Pre-condition: 'this' exists and starts at the root's start.
	 * Invariant: Adjust 'tmp.start' to skip the initial allocated resource or proceed from its end.
	 *            This avoids underflow when calculating 'tmp.end' if 'this->start - 1' is used.
	 */
	if (this && this->start == root->start) {
		tmp.start = (this == old) ? old->start : this->end + 1;
		this = this->sibling;
	}
	/* Invariant: Loop to iterate through segments of available space until a suitable one is found or all segments are checked. */
	for(;;) {
		/* Pre-condition: 'this' points to a valid resource or is NULL. */
		/* Invariant: Determine the end boundary of the current potential free space. */
		if (this)
			tmp.end = (this == old) ?  this->end : this->start - 1;
		else
			tmp.end = root->end;

		/* Pre-condition: The calculated temporary end is less than the temporary start. */
		if (tmp.end < tmp.start)
			goto next; /* Skip to the next potential space if the current one is invalid. */

		resource_clip(&tmp, constraint->min, constraint->max);
		arch_remove_reservations(&tmp);

		/* Pre-condition: Check for potential overflow after applying alignment to the start address. */
		/* Invariant: 'avail' defines the aligned and clipped available region. */
		avail.start = ALIGN(tmp.start, constraint->align);
		avail.end = tmp.end;
		avail.flags = new->flags & ~IORESOURCE_UNSET;
		/* Pre-condition: The aligned start address is not less than the original temporary start. */
		if (avail.start >= tmp.start) {
			alloc.flags = avail.flags;
			/* Invariant: If a custom alignment function is provided, use it; otherwise, use the aligned start. */
			if (alignf) {
				alloc.start = alignf(constraint->alignf_data,
						     &avail, size, constraint->align);
			} else {
				alloc.start = avail.start;
			}
			alloc.end = alloc.start + size - 1;
			/* Pre-condition: The allocated region is valid and contained within the available region. */
			if (alloc.start <= alloc.end &&
			    resource_contains(&avail, &alloc)) {
				new->start = alloc.start;
				new->end = alloc.end;
				return 0; /* Successfully found and allocated space. */
			}
		}

next:		/* Pre-condition: No more resources to check or the end of the root resource has been reached. */
		if (!this || this->end == root->end)
			break; /* Exit loop if no more potential spaces. */

		/* Invariant: Move to the next potential free space after the current resource. */
		if (this != old)
			tmp.start = this->end + 1;
		this = this->sibling;
	}
	return -EBUSY; /* No suitable empty space found. */
}

/**
 * @brief Finds empty space in the resource tree given range & alignment.
 * @details Finds an empty space under `root` in the resource tree satisfying range and
 * alignment `constraints`. This function is a public wrapper around
 * `__find_resource_space` for initial resource allocation (where `old` is `NULL`).
 *
 * @param root Root resource descriptor.
 * @param new Resource descriptor awaiting an empty resource space.
 * @param size The minimum size of the empty space.
 * @param constraint The range and alignment constraints to be met.
 * @return 0 if successful, `-EBUSY` if no empty space was found.
 */
int find_resource_space(struct resource *root, struct resource *new,
			resource_size_t size,
			struct resource_constraint *constraint)
{
	return  __find_resource_space(root, NULL, new, size, constraint);
}
EXPORT_SYMBOL_GPL(find_resource_space);

/**
 * @brief Reallocates an existing resource within the resource tree, potentially relocating it.
 * @details This function attempts to resize and potentially relocate an existing resource `old`
 * within the resource tree rooted at `root`. It first tries to find a suitable space
 * for the `newsize` considering the `constraint`. If the new size can be accommodated
 * in the existing location of `old`, it updates `old` in place. Otherwise, if `old`
 * has no children, it releases the `old` resource and requests a new one at the
 * determined location. If `old` has children and cannot be relocated, it returns an error.
 *
 * @param root The root resource of the tree where the reallocation is to occur.
 * @param old The existing resource to be reallocated.
 * @param newsize The desired new size for the resource.
 * @param constraint The range and alignment constraints that must be met.
 * @return 0 on success, -EBUSY if reallocation fails (e.g., no space, or `old` has children and must be relocated).
 */
static int reallocate_resource(struct resource *root, struct resource *old,
			       resource_size_t newsize,
			       struct resource_constraint *constraint)
{
	int err=0;
	struct resource new = *old;
	struct resource *conflict;

	write_lock(&resource_lock); /* Acquire write lock to protect resource tree modifications. */

	/* Pre-condition: Attempt to find a suitable space for the new resource size. */
	/* Invariant: If `__find_resource_space` returns an error, propagation of error occurs. */
	if ((err = __find_resource_space(root, old, &new, newsize, constraint)))
		goto out;

	/* Pre-condition: Check if the newly found space completely contains the old resource's region. */
	/* Invariant: If the new region fully contains the old, update `old` in place and exit. */
	if (resource_contains(&new, old)) {
		old->start = new.start;
		old->end = new.end;
		goto out;
	}

	/* Pre-condition: Check if the old resource has any child resources. */
	/* Invariant: If `old` has children, relocation is not allowed for simplicity. */
	if (old->child) {
		err = -EBUSY;
		goto out;
	}

	/* Pre-condition: Check if the old resource completely contains the newly found space. */
	/* Invariant: If old resource contains new space, update old in place. Otherwise, release old and request new. */
	if (resource_contains(old, &new)) {
		old->start = new.start;
		old->end = new.end;
	} else {
		/* Pre-condition: Old resource does not contain the new space, implying relocation is needed. */
		/* Invariant: Release the old resource, update its descriptor, and request it as a new resource. */
		__release_resource(old, true);
		*old = new;
		conflict = __request_resource(root, old);
		BUG_ON(conflict); /* This should ideally not happen if __find_resource_space worked correctly. */
	}
out:
	write_unlock(&resource_lock); /* Release write lock. */
	return err;
}


/**
 * @brief Allocates an empty slot in the resource tree given range & alignment.
 * @details This function allocates an empty slot in the resource tree, allowing for reallocation
 * of an existing resource if it was already allocated. It uses `__find_resource_space`
 * to locate a suitable region and then `__request_resource` to claim it.
 *
 * @param root Root resource descriptor.
 * @param new Resource descriptor desired by caller.
 * @param size Requested resource region size.
 * @param min Minimum boundary to allocate.
 * @param max Maximum boundary to allocate.
 * @param align Alignment requested, in bytes.
 * @param alignf Alignment function, optional, called if not NULL.
 * @param alignf_data Arbitrary data to pass to the `alignf` function.
 * @return 0 if successful, `-EBUSY` if no empty space was found or a conflict occurred.
 */
int allocate_resource(struct resource *root, struct resource *new,
		      resource_size_t size, resource_size_t min,
		      resource_size_t max, resource_size_t align,
		      resource_alignf alignf,
		      void *alignf_data)
{
	int err;
	struct resource_constraint constraint;

	constraint.min = min;
	constraint.max = max;
	constraint.align = align;
	constraint.alignf = alignf;
	constraint.alignf_data = alignf_data;

	// Pre-condition: Check if the resource is already allocated (has a parent).
	// Invariant: If already allocated, attempt to reallocate it with new constraints.
	if ( new->parent ) {
		/* resource is already allocated, try reallocating with
		   the new constraints */
		return reallocate_resource(root, new, size, &constraint);
	}

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	err = find_resource_space(root, new, size, &constraint); // Find an empty space.
	// Pre-condition: If space was found, attempt to request the resource.
	// Invariant: If a conflict arises during request, set error to -EBUSY.
	if (err >= 0 && __request_resource(root, new))
		err = -EBUSY;
	write_unlock(&resource_lock); // Release write lock.
	return err;
}

EXPORT_SYMBOL(allocate_resource);

/**
 * @brief Looks up an existing resource by its start address.
 * @details This function searches the resource tree rooted at `root` for a resource
 * that has the specified `start` address. It performs a read-locked traversal
 * of the children and their siblings.
 *
 * @param root Root resource descriptor.
 * @param start Resource start address.
 * @return A pointer to the resource if found, NULL otherwise.
 */
struct resource *lookup_resource(struct resource *root, resource_size_t start)
{
	struct resource *res;

	read_lock(&resource_lock); // Acquire read lock for safe tree traversal.
	// Invariant: Iterate through child resources and their siblings.
	for (res = root->child; res; res = res->sibling) {
		// Pre-condition: Check if the current resource's start address matches the target.
		if (res->start == start)
			break; // Found matching resource, exit loop.
	}
	read_unlock(&resource_lock); // Release read lock.

	return res;
}

/**
 * @brief Internal function to insert a resource into the resource tree, handling conflicts by making them children.
 * @details This function attempts to insert a `new` resource under the `parent` resource.
 * If the `new` resource conflicts with existing resources in the tree, and those
 * conflicting resources entirely fit within the range of the `new` resource,
 * then the `new` resource is inserted, and the conflicting resources become
 * its children. This differs from `__request_resource` which simply returns
 * the conflicting resource without modification.
 *
 * @param parent The parent resource under which `new` is to be inserted.
 * @param new The new resource to insert.
 * @return NULL on successful insertion, or a pointer to the conflicting resource
 *         if insertion fails due to an unresolvable overlap.
 */
static struct resource * __insert_resource(struct resource *parent, struct resource *new)
{
	struct resource *first, *next;

	/*
	 * Invariant: This loop iteratively attempts to insert 'new' into 'parent'.
	 * If a conflict ('first') is found, 'parent' is updated to 'first',
	 * effectively trying to insert 'new' as a child of the conflicting resource,
	 * until 'new' is successfully inserted or an unresolvable conflict occurs.
	 */
	for (;; parent = first) {
		/* Pre-condition: Attempt to request the new resource under the current parent. */
		first = __request_resource(parent, new);
		if (!first) /* If no conflict, insertion is successful. */
			return first;

		/* Pre-condition: 'first' is the same as 'parent', indicating a fundamental conflict with the root. */
		if (first == parent)
			return first;
		/* Pre-condition: Detects and warns about duplicated insertion attempts. */
		if (WARN_ON(first == new))	/* duplicated insertion */
			return first;

		/*
		 * Pre-condition: If the conflicting resource 'first' does not fully encompass
		 * the 'new' resource, or vice-versa, it implies a partial overlap which
		 * cannot be resolved by making 'first' a child of 'new'.
		 */
		if ((first->start > new->start) || (first->end < new->end))
			break;
		if ((first->start == new->start) && (first->end == new->end))
			break;
	}

	/*
	 * Invariant: After the first loop, 'first' is the first of a chain of
	 * resources that fully overlap with 'new'. This loop finds the last
	 * resource in this chain that 'new' will encompass.
	 */
	for (next = first; ; next = next->sibling) {
		/* Pre-condition: Checks for partial overlap; if 'next' is not fully contained within 'new', it's an unfixable conflict. */
		if (next->start < new->start || next->end > new->end)
			return next;
		if (!next->sibling) /* If 'next' is the last sibling, break. */
			break;
		if (next->sibling->start > new->end) /* If the next sibling is beyond 'new''s end, 'next' is the last encompassing sibling. */
			break;
	}

	/* Restructure the resource tree: 'new' becomes the parent of 'first' and its encompassing siblings. */
	new->parent = parent;
	new->sibling = next->sibling;
	new->child = first;

	next->sibling = NULL; /* Terminate the sibling list of the newly formed child subtree. */
	/* Invariant: Update the parent pointers for all resources that become children of 'new'. */
	for (next = first; next; next = next->sibling)
		next->parent = new;

	/* Invariant: Update the parent's child pointer to point to the newly inserted 'new' resource. */
	if (parent->child == first) {
		parent->child = new;
	} else {
		next = parent->child;
		/* Pre-condition: Traverse the parent's children list to find the sibling preceding 'first'. */
		while (next->sibling != first)
			next = next->sibling;
		next->sibling = new; /* Link 'new' into the parent's child list. */
	}
	return NULL; /* Successful insertion. */
}

/**
 * @brief Inserts resource in the resource tree, returning a conflict resource if any.
 * @details This function is equivalent to `request_resource_conflict` when no conflict
 * happens. If a conflict happens, and the conflicting resources
 * entirely fit within the range of the new resource, then the new
 * resource is inserted and the conflicting resources become children of
 * the new resource. This function is intended for producers of resources,
 * such as FW modules and bus drivers.
 *
 * @param parent Parent of the new resource.
 * @param new New resource to insert.
 * @return 0 on success, conflict resource if the resource can't be inserted.
 */
struct resource *insert_resource_conflict(struct resource *parent, struct resource *new)
{
	struct resource *conflict;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	conflict = __insert_resource(parent, new); // Attempt to insert the resource.
	write_unlock(&resource_lock); // Release write lock.
	return conflict;
}

/**
 * @brief Inserts a resource in the resource tree.
 * @details This function is intended for producers of resources, such as FW modules
 * and bus drivers. It's a wrapper around `insert_resource_conflict`, returning
 * 0 on success and `-EBUSY` if the resource can't be inserted due to an
 * unresolvable conflict.
 *
 * @param parent Parent of the new resource.
 * @param new New resource to insert.
 * @return 0 on success, `-EBUSY` if the resource can't be inserted.
 */
int insert_resource(struct resource *parent, struct resource *new)
{
	struct resource *conflict;

	conflict = insert_resource_conflict(parent, new);
	return conflict ? -EBUSY : 0;
}
EXPORT_SYMBOL_GPL(insert_resource);

/**
 * @brief Inserts a resource into the resource tree, possibly expanding it to encompass conflicts.
 * @details This function attempts to insert a `new` resource into the resource tree
 * rooted at `root`. If conflicts arise, it iteratively expands the `new` resource's
 * range to cover the conflicting resources and retries the insertion. This ensures
 * that the `new` resource eventually encompasses any overlapping existing resources.
 * This interface is typically used by early boot memory map parsing, PCI resource
 * discovery, and late discovery of CXL resources.
 *
 * @param root Root resource descriptor.
 * @param new New resource to insert.
 */
void insert_resource_expand_to_fit(struct resource *root, struct resource *new)
{
	// Pre-condition: If the resource already has a parent, it's already in the tree; no action needed.
	if (new->parent)
		return;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	// Invariant: Continuously attempt to insert or expand the new resource until successfully inserted or a fundamental conflict occurs.
	for (;;) {
		struct resource *conflict;

		conflict = __insert_resource(root, new); // Attempt to insert the resource.
		if (!conflict) // If no conflict, insertion is successful.
			break;
		if (conflict == root) // If conflict is with the root itself, cannot expand further.
			break;

		/* Ok, expand resource to cover the conflict, then try again .. */
		// Pre-condition: Conflict's start is before new's start.
		// Invariant: Adjust new's start to encompass the conflict.
		if (conflict->start < new->start)
			new->start = conflict->start;
		// Pre-condition: Conflict's end is after new's end.
		// Invariant: Adjust new's end to encompass the conflict.
		if (conflict->end > new->end)
			new->end = conflict->end;

		pr_info("Expanded resource %s due to conflict with %s\n", new->name, conflict->name);
	}
	write_unlock(&resource_lock); // Release write lock.
}
/*
 * Not for general consumption, only early boot memory map parsing, PCI
 * resource discovery, and late discovery of CXL resources are expected
 * to use this interface. The former are built-in and only the latter,
 * CXL, is a module.
 */
EXPORT_SYMBOL_NS_GPL(insert_resource_expand_to_fit, "CXL");

/**
 * @brief Removes a resource from the resource tree.
 * @details This function removes a resource previously inserted by `insert_resource()`
 * or `insert_resource_conflict()`. If the removed resource had children,
 * those children are moved up to where they were before the parent was inserted.
 * This function is intended for producers of resources, such as FW modules and bus drivers.
 *
 * @param old Resource to remove.
 * @return 0 on success, `-EINVAL` if the resource is not valid.
 */
int remove_resource(struct resource *old)
{
	int retval;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	retval = __release_resource(old, false); // Release the resource, promoting children if any.
	write_unlock(&resource_lock); // Release write lock.
	return retval;
}
EXPORT_SYMBOL_GPL(remove_resource);

/**
 * @brief Internal function to modify a resource's start address and size, ensuring it fits within constraints.
 * @details This function attempts to adjust the `start` address and `size` of an existing
 * resource `res`. It performs several checks to ensure that the new region:
 * 1. Has a valid parent.
 * 2. Remains within its parent's boundaries.
 * 3. Does not overlap with its siblings.
 * 4. Contains all its existing children.
 * If all checks pass, the resource's `start` and `end` are updated.
 *
 * @param res The resource to modify.
 * @param start The new desired start address for the resource.
 * @param size The new desired size for the resource.
 * @return 0 on success, -EBUSY if the adjustment is not possible due to conflicts or boundary violations.
 */
static int __adjust_resource(struct resource *res, resource_size_t start,
				resource_size_t size)
{
	struct resource *tmp, *parent = res->parent;
	resource_size_t end = start + size - 1;
	int result = -EBUSY;

	/* Pre-condition: If the resource has no parent, some checks are skipped. */
	if (!parent)
		goto skip;

	/* Pre-condition: The new resource range must be fully contained within its parent's range. */
	if ((start < parent->start) || (end > parent->end))
		goto out;

	/* Pre-condition: The new resource range must not overlap with its next sibling. */
	if (res->sibling && (res->sibling->start <= end))
		goto out;

	tmp = parent->child;
	/* Pre-condition: If 'res' is not the first child, find its preceding sibling. */
	if (tmp != res) {
		/* Invariant: Traverse siblings until 'res' is found. */
		while (tmp->sibling != res)
			tmp = tmp->sibling;
		/* Pre-condition: The new resource range must not overlap with its previous sibling. */
		if (start <= tmp->end)
			goto out;
	}

skip:
	/* Invariant: Iterate through all children of 'res' to ensure they are contained within the new adjusted range. */
	for (tmp = res->child; tmp; tmp = tmp->sibling)
		/* Pre-condition: If any child falls outside the new range, the adjustment is invalid. */
		if ((tmp->start < start) || (tmp->end > end))
			goto out;

	/* Invariant: If all checks pass, update the resource's start and end addresses. */
	res->start = start;
	res->end = end;
	result = 0;

 out:
	return result;
}

/**
 * @brief Modifies a resource's start and size.
 * @details Given an existing resource, this function changes its start address and size
 * to match the arguments. It acquires a write lock to ensure thread safety
 * during the modification. Existing children of the resource are assumed to be immutable.
 *
 * @param res Resource to modify.
 * @param start New start value.
 * @param size New size.
 * @return 0 on success, -EBUSY if it can't fit.
 */
int adjust_resource(struct resource *res, resource_size_t start,
		    resource_size_t size)
{
	int result;

	write_lock(&resource_lock); // Acquire write lock for resource modification.
	result = __adjust_resource(res, start, size); // Attempt to adjust the resource.
	write_unlock(&resource_lock); // Release write lock.
	return result;
}
EXPORT_SYMBOL(adjust_resource);

/**
 * @brief Reserves a region in the resource tree, splitting it if conflicts arise.
 * @details This internal function attempts to reserve a specified memory region (`start` to `end`)
 * within the resource tree rooted at `root`. If the requested region conflicts
 * with an already existing resource, it attempts to split the requested region
 * around the conflict and reserve the non-conflicting parts. This allows for
 * reserving fragmented regions in the presence of existing allocations.
 *
 * @param root The root resource where the region is to be reserved.
 * @param start The starting address of the region to reserve.
 * @param end The ending address of the region to reserve.
 * @param name The name to assign to the reserved resource(s).
 */
static void __init
__reserve_region_with_split(struct resource *root, resource_size_t start,
			    resource_size_t end, const char *name)
{
	struct resource *parent = root;
	struct resource *conflict;
	struct resource *res = alloc_resource(GFP_ATOMIC);
	struct resource *next_res = NULL;
	int type = resource_type(root);

	/* Pre-condition: Ensure resource allocation is successful. */
	if (!res)
		return;

	res->name = name;
	res->start = start;
	res->end = end;
	res->flags = type | IORESOURCE_BUSY;
	res->desc = IORES_DESC_NONE;

	/* Invariant: Loop continues until the entire requested region is reserved or all conflicts are handled. */
	while (1) {
		/* Pre-condition: Attempt to reserve the current part of the region. */
		conflict = __request_resource(parent, res);
		/* Invariant: If no conflict, the current 'res' part is successfully reserved. */
		if (!conflict) {
			if (!next_res) /* If no `next_res` is pending, all parts are reserved. */
				break;
			res = next_res; /* Process the next pending part. */
			next_res = NULL;
			continue;
		}

		/* Pre-condition: The conflicting resource completely covers the current requested 'res'. */
		/* Invariant: If conflict covers the whole area, the requested part cannot be reserved. */
		if (conflict->start <= res->start &&
				conflict->end >= res->end) {
			free_resource(res); /* Free the current 'res' as it's fully conflicted. */
			WARN_ON(next_res); /* Should not have `next_res` if current one is fully consumed. */
			break;
		}

		/* Invariant: If a conflict exists but doesn't cover the whole area, split 'res' and try again. */
		if (conflict->start > res->start) {
			/* Pre-condition: The conflict starts after 'res.start', meaning a free region exists before the conflict. */
			end = res->end; /* Store original end of 'res'. */
			res->end = conflict->start - 1; /* Adjust 'res.end' to be just before the conflict. */
			if (conflict->end < end) {
				/* Pre-condition: There's a free region after the conflict as well. */
				next_res = alloc_resource(GFP_ATOMIC); /* Allocate for the part after the conflict. */
				if (!next_res) {
					free_resource(res);
					break;
				}
				next_res->name = name;
				next_res->start = conflict->end + 1;
				next_res->end = end;
				next_res->flags = type | IORESOURCE_BUSY;
				next_res->desc = IORES_DESC_NONE;
			}
		} else {
			/* Pre-condition: The conflict starts at or before 'res.start', so adjust 'res.start' to be after the conflict. */
			res->start = conflict->end + 1;
		}
	}
}
/**
 * @brief Reserves a region in the resource tree, splitting it if conflicts arise.
 * @details This function is a wrapper around `__reserve_region_with_split`, adding
 * error handling and debug messages. It ensures that the requested region is
 * within the bounds of the `root` resource and logs warnings or errors if
 * the region is invalid or adjusted. The entire operation is protected by a write lock.
 *
 * @param root The root resource where the region is to be reserved.
 * @param start The starting address of the region to reserve.
 * @param end The ending address of the region to reserve.
 * @param name The name to assign to the reserved resource(s).
 */
void __init
reserve_region_with_split(struct resource *root, resource_size_t start,
			  resource_size_t end, const char *name)
{
	int abort = 0;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	// Pre-condition: Check if the requested range is outside the root's bounds.
	if (root->start > start || root->end < end) {
		pr_err("requested range [0x%llx-0x%llx] not in root %pr\n",
		       (unsigned long long)start, (unsigned long long)end,
		       root);
		// Invariant: If the requested range is completely disjoint or partially outside, adjust or abort.
		if (start > root->end || end < root->start)
			abort = 1;
		else {
			// Invariant: Clip the requested range to fit within the root's bounds.
			if (end > root->end)
				end = root->end;
			if (start < root->start)
				start = root->start;
			pr_err("fixing request to [0x%llx-0x%llx]\n",
			       (unsigned long long)start,
			       (unsigned long long)end);
		}
		dump_stack(); // Log a stack trace for debugging.
	}
	// Pre-condition: If not aborted, proceed with the reservation.
	if (!abort)
		__reserve_region_with_split(root, start, end, name); // Perform the actual reservation.
	write_unlock(&resource_lock); // Release write lock.
}

/**
 * @brief Calculates a resource's alignment.
 * @details This function determines the alignment requirement for a given resource
 * based on its `IORESOURCE_SIZEALIGN` or `IORESOURCE_STARTALIGN` flags.
 *
 * @param res Resource pointer.
 * @return Alignment on success, 0 (invalid alignment) on failure.
 */
resource_size_t resource_alignment(struct resource *res)
{
	// Pre-condition: Check the flags for alignment type.
	switch (res->flags & (IORESOURCE_SIZEALIGN | IORESOURCE_STARTALIGN)) {
	// Invariant: If IORESOURCE_SIZEALIGN is set, the resource size is the alignment.
	case IORESOURCE_SIZEALIGN:
		return resource_size(res);
	// Invariant: If IORESOURCE_STARTALIGN is set, the resource start address is the alignment.
	case IORESOURCE_STARTALIGN:
		return res->start;
	// Invariant: If neither flag is set, return 0 for invalid alignment.
	default:
		return 0;
	}
}

/**
 * @brief Inserts resource in the resource tree, returning a conflict resource if any.
 * @details This function is equivalent to `request_resource_conflict` when no conflict
 * happens. If a conflict happens, and the conflicting resources
 * entirely fit within the range of the new resource, then the new
 * resource is inserted and the conflicting resources become children of
 * the new resource. This function is intended for producers of resources,
 * such as FW modules and bus drivers.
 *
 * @param parent Parent of the new resource.
 * @param new New resource to insert.
 * @return 0 on success, conflict resource if the resource can't be inserted.
 */
struct resource *insert_resource_conflict(struct resource *parent, struct resource *new)
{
	struct resource *conflict;

	// Invariant: Acquire write lock for resource tree modification.
	write_lock(&resource_lock);
	// Pre-condition: Attempt to insert the resource with potential conflicts.
	// Invariant: 'conflict' will hold a pointer to the conflicting resource if any, otherwise NULL.
	conflict = __insert_resource(parent, new);
	// Invariant: Release write lock.
	write_unlock(&resource_lock);
	return conflict;
}

/**
 * @brief Inserts a resource in the resource tree.
 * @details This function is intended for producers of resources, such as FW modules
 * and bus drivers. It's a wrapper around `insert_resource_conflict`, returning
 * 0 on success and `-EBUSY` if the resource can't be inserted due to an
 * unresolvable conflict.
 *
 * @param parent Parent of the new resource.
 * @param new New resource to insert.
 * @return 0 on success, `-EBUSY` if the resource can't be inserted.
 */
int insert_resource(struct resource *parent, struct resource *new)
{
	struct resource *conflict;

	// Pre-condition: Attempt to insert the resource, which may result in a conflict.
	// Invariant: 'conflict' will point to a conflicting resource if insertion fails, otherwise NULL.
	conflict = insert_resource_conflict(parent, new);
	return conflict ? -EBUSY : 0;
}
EXPORT_SYMBOL_GPL(insert_resource);

/**
 * insert_resource_expand_to_fit - Insert a resource into the resource tree
 * @root: root resource descriptor
 * @new: new resource to insert
 *
 * Insert a resource into the resource tree, possibly expanding it in order
 * to make it encompass any conflicting resources.
 */
void insert_resource_expand_to_fit(struct resource *root, struct resource *new)
{
	// Pre-condition: If the resource already has a parent, it's already in the tree; no action needed.
	if (new->parent)
		return;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	// Invariant: Continuously attempt to insert or expand the new resource until successfully inserted or a fundamental conflict occurs.
	for (;;) {
		struct resource *conflict;

		// Pre-condition: Attempt to insert the resource.
		// Invariant: 'conflict' will be non-NULL if an overlap is found.
		conflict = __insert_resource(root, new);
		if (!conflict) // If no conflict, insertion is successful.
			break;
		if (conflict == root) // If conflict is with the root itself, cannot expand further.
			break;

		/* Ok, expand resource to cover the conflict, then try again .. */
		// Pre-condition: Conflict's start is before new's start.
		// Invariant: Adjust new's start to encompass the conflict.
		if (conflict->start < new->start)
			new->start = conflict->start;
		// Pre-condition: Conflict's end is after new's end.
		// Invariant: Adjust new's end to encompass the conflict.
		if (conflict->end > new->end)
			new->end = conflict->end;

		pr_info("Expanded resource %s due to conflict with %s\n", new->name, conflict->name);
	}
	write_unlock(&resource_lock); // Release write lock.
}
/*
 * Not for general consumption, only early boot memory map parsing, PCI
 * resource discovery, and late discovery of CXL resources are expected
 * to use this interface. The former are built-in and only the latter,
 * CXL, is a module.
 */
EXPORT_SYMBOL_NS_GPL(insert_resource_expand_to_fit, "CXL");

/**
 * remove_resource - Remove a resource in the resource tree
 * @old: resource to remove
 *
 * Returns 0 on success, -EINVAL if the resource is not valid.
 *
 * This function removes a resource previously inserted by insert_resource()
 * or insert_resource_conflict(), and moves the children (if any) up to
 * where they were before.  insert_resource() and insert_resource_conflict()
 * insert a new resource, and move any conflicting resources down to the
 * children of the new resource.
 *
 * insert_resource(), insert_resource_conflict() and remove_resource() are
 * intended for producers of resources, such as FW modules and bus drivers.
 */
int remove_resource(struct resource *old)
{
	int retval;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	// Pre-condition: 'old' is the resource to be removed. 'false' indicates that children, if any, should be promoted.
	// Invariant: 'retval' will store the result of the resource release operation (0 on success, -EINVAL on failure).
	retval = __release_resource(old, false); // Release the resource, promoting children if any.
	write_unlock(&resource_lock); // Release write lock.
	return retval;
}
EXPORT_SYMBOL_GPL(remove_resource);

/**
 * @brief Internal function to modify a resource's start address and size, ensuring it fits within constraints.
 *
 * This function attempts to adjust the `start` address and `size` of an existing
 * resource `res`. It performs several checks to ensure that the new region:
 * 1. Has a valid parent.
 * 2. Remains within its parent's boundaries.
 * 3. Does not overlap with its siblings.
 * 4. Contains all its existing children.
 * If all checks pass, the resource's `start` and `end` are updated.
 *
 * @param res The resource to modify.
 * @param start The new desired start address for the resource.
 * @param size The new desired size for the resource.
 * @return 0 on success, -EBUSY if the adjustment is not possible due to conflicts or boundary violations.
 */
static int __adjust_resource(struct resource *res, resource_size_t start,
				resource_size_t size)
{
	struct resource *tmp, *parent = res->parent;
	resource_size_t end = start + size - 1;
	int result = -EBUSY;

	/* Pre-condition: If the resource has no parent, some checks are skipped. */
	if (!parent)
		goto skip;

	/* Pre-condition: The new resource range must be fully contained within its parent's range. */
	if ((start < parent->start) || (end > parent->end))
		goto out;

	/* Pre-condition: The new resource range must not overlap with its next sibling. */
	if (res->sibling && (res->sibling->start <= end))
		goto out;

	tmp = parent->child;
	/* Pre-condition: If 'res' is not the first child, find its preceding sibling. */
	if (tmp != res) {
		/* Invariant: Traverse siblings until 'res' is found. */
		while (tmp->sibling != res)
			tmp = tmp->sibling;
		/* Pre-condition: The new resource range must not overlap with its previous sibling. */
		if (start <= tmp->end)
			goto out;
	}

skip:
	/* Invariant: Iterate through all children of 'res' to ensure they are contained within the new adjusted range. */
	for (tmp = res->child; tmp; tmp = tmp->sibling)
		/* Pre-condition: If any child falls outside the new range, the adjustment is invalid. */
		if ((tmp->start < start) || (tmp->end > end))
			goto out;

	/* Invariant: If all checks pass, update the resource's start and end addresses. */
	res->start = start;
	res->end = end;
	result = 0;

 out:
	return result;
}

/**
 * adjust_resource - modify a resource's start and size
 * @res: resource to modify
 * @start: new start value
 * @size: new size
 *
 * Given an existing resource, change its start and size to match the
 * arguments.  Returns 0 on success, -EBUSY if it can't fit.
 * Existing children of the resource are assumed to be immutable.
 */
int adjust_resource(struct resource *res, resource_size_t start,
		    resource_size_t size)
{
	int result;

	write_lock(&resource_lock);
	result = __adjust_resource(res, start, size);
	write_unlock(&resource_lock);
	return result;
}
EXPORT_SYMBOL(adjust_resource);

/**
 * @brief Reserves a region in the resource tree, splitting it if conflicts arise.
 *
 * This internal function attempts to reserve a specified memory region (`start` to `end`)
 * within the resource tree rooted at `root`. If the requested region conflicts
 * with an already existing resource, it attempts to split the requested region
 * around the conflict and reserve the non-conflicting parts. This allows for
 * reserving fragmented regions in the presence of existing allocations.
 *
 * @param root The root resource where the region is to be reserved.
 * @param start The starting address of the region to reserve.
 * @param end The ending address of the region to reserve.
 * @param name The name to assign to the reserved resource(s).
 */
static void __init
__reserve_region_with_split(struct resource *root, resource_size_t start,
			    resource_size_t end, const char *name)
{
	struct resource *parent = root;
	struct resource *conflict;
	struct resource *res = alloc_resource(GFP_ATOMIC);
	struct resource *next_res = NULL;
	int type = resource_type(root);

	/* Pre-condition: Ensure resource allocation is successful. */
	if (!res)
		return;

	res->name = name;
	res->start = start;
	res->end = end;
	res->flags = type | IORESOURCE_BUSY;
	res->desc = IORES_DESC_NONE;

	/* Invariant: Loop continues until the entire requested region is reserved or all conflicts are handled. */
	while (1) {
		/* Pre-condition: Attempt to reserve the current part of the region. */
		conflict = __request_resource(parent, res);
		/* Invariant: If no conflict, the current 'res' part is successfully reserved. */
		if (!conflict) {
			if (!next_res) /* If no `next_res` is pending, all parts are reserved. */
				break;
			res = next_res; /* Process the next pending part. */
			next_res = NULL;
			continue;
		}

		/* Pre-condition: The conflicting resource completely covers the current requested 'res'. */
		/* Invariant: If conflict covers the whole area, the requested part cannot be reserved. */
		if (conflict->start <= res->start &&
				conflict->end >= res->end) {
			free_resource(res); /* Free the current 'res' as it's fully conflicted. */
			WARN_ON(next_res); /* Should not have `next_res` if current one is fully consumed. */
			break;
		}

		/* Invariant: If a conflict exists but doesn't cover the whole area, split 'res' and try again. */
		if (conflict->start > res->start) {
			/* Pre-condition: The conflict starts after 'res.start', meaning a free region exists before the conflict. */
			end = res->end; /* Store original end of 'res'. */
			res->end = conflict->start - 1; /* Adjust 'res.end' to be just before the conflict. */
			if (conflict->end < end) {
				/* Pre-condition: There's a free region after the conflict as well. */
				next_res = alloc_resource(GFP_ATOMIC); /* Allocate for the part after the conflict. */
				if (!next_res) {
					free_resource(res);
					break;
				}
				next_res->name = name;
				next_res->start = conflict->end + 1;
				next_res->end = end;
				next_res->flags = type | IORESOURCE_BUSY;
				next_res->desc = IORES_DESC_NONE;
			}
		} else {
			/* Pre-condition: The conflict starts at or before 'res.start', so adjust 'res.start' to be after the conflict. */
			res->start = conflict->end + 1;
		}
	}
}
void __init
reserve_region_with_split(struct resource *root, resource_size_t start,
			  resource_size_t end, const char *name)
{
	int abort = 0;

	write_lock(&resource_lock);
	if (root->start > start || root->end < end) {
		pr_err("requested range [0x%llx-0x%llx] not in root %pr\n",
		       (unsigned long long)start, (unsigned long long)end,
		       root);
		if (start > root->end || end < root->start)
			abort = 1;
		else {
			if (end > root->end)
				end = root->end;
			if (start < root->start)
				start = root->start;
			pr_err("fixing request to [0x%llx-0x%llx]\n",
			       (unsigned long long)start,
			       (unsigned long long)end);
		}
		dump_stack();
	}
	if (!abort)
		__reserve_region_with_split(root, start, end, name);
	write_unlock(&resource_lock);
}

/**
 * @brief Calculates a resource's alignment.
 * @details This function determines the alignment requirement for a given resource
 * based on its `IORESOURCE_SIZEALIGN` or `IORESOURCE_STARTALIGN` flags.
 *
 * @param res Resource pointer.
 * @return Alignment on success, 0 (invalid alignment) on failure.
 */
resource_size_t resource_alignment(struct resource *res)
{
	// Pre-condition: Check the flags for alignment type.
	switch (res->flags & (IORESOURCE_SIZEALIGN | IORESOURCE_STARTALIGN)) {
	// Invariant: If IORESOURCE_SIZEALIGN is set, the resource size is the alignment.
	case IORESOURCE_SIZEALIGN:
		return resource_size(res);
	// Invariant: If IORESOURCE_STARTALIGN is set, the resource start address is the alignment.
	case IORESOURCE_STARTALIGN:
		return res->start;
	// Invariant: If neither flag is set, return 0 for invalid alignment.
	default:
		return 0;
	}
}

/*
 * @brief Compatibility layer for I/O resources management.
 * @details This section provides functions for managing I/O resources,
 * specifically handling busy regions and their allocation/deallocation.
 * Unlike the more generic resource management functions, these are aware
 * of the "busy" flag and other I/O specific flag meanings.
 * `request_region` creates a new busy region, and `release_region` releases it.
 */

/**
 * @brief Wait queue head for muxed resources.
 * @details This wait queue is used to synchronize access to multiplexed
 * I/O resources. Processes attempting to acquire a muxed resource that
 * is currently held will be added to this queue and woken up when the
 * resource becomes available.
 */
/**
 * @brief Wait queue head for muxed resources.
 * @details This wait queue is used to synchronize access to multiplexed
 * I/O resources. Processes attempting to acquire a muxed resource that
 * is currently held will be added to this queue and woken up when the
 * resource becomes available.
 */
static DECLARE_WAIT_QUEUE_HEAD(muxed_resource_wait);

/**
 * @brief Inode for the /dev/mem mapping.
 * @details This static inode is used to manage file operations for /dev/mem,
 * particularly for revoking user-space memory mappings when a kernel driver
 * claims a physical memory region.
 */
static struct inode *iomem_inode;

#ifdef CONFIG_IO_STRICT_DEVMEM
/**
 * @brief Revokes I/O memory mappings for a given resource.
 * @details This function is conditionally compiled based on `CONFIG_IO_STRICT_DEVMEM`.
 * It attempts to unmap any existing user-space mappings for the specified
 * resource `res` to prevent unauthorized access, especially when a driver
 * claims a physical memory region. It checks if `iomem=relaxed` kernel parameter is set,
 * which can override this strict behavior.
 *
 * @param res The resource for which to revoke I/O memory mappings.
 */
static void revoke_iomem(struct resource *res)
{
	/* pairs with smp_store_release() in iomem_init_inode() */
	struct inode *inode = smp_load_acquire(&iomem_inode);

	/*
	 * Pre-condition: Check that the `iomem_inode` has been initialized.
	 * Invariant: If not initialized, drivers might claim resources before
	 * fs_initcall level, potentially preventing user-space mappings.
	 */
	if (!inode)
		return;

	/*
	 * Pre-condition: The driver should have marked the resource busy.
	 * Invariant: `devmem_is_allowed()` should return false for the region.
	 * Performance optimization: does not iterate entire resource range.
	 */
	if (devmem_is_allowed(PHYS_PFN(res->start)) &&
	    devmem_is_allowed(PHYS_PFN(res->end))) {
		/*
		 * *cringe* iomem=relaxed says "go ahead, what's the
		 * worst that can happen?"
		 */
		return; /* `iomem=relaxed` allows mapping even if it should be strict. */
	}

	// Invariant: Unmap the user-space mapping range for the given resource.
	unmap_mapping_range(inode->i_mapping, res->start, resource_size(res), 1);
}
#else
static void revoke_iomem(struct resource *res) {}
#endif

/**
 * @brief Retrieves the address space mapping for I/O memory.
 * @details This function provides access to the `address_space` structure associated
 * with I/O memory, which is used for managing memory-mapped I/O operations.
 * It ensures that the `iomem_inode` has been initialized before returning
 * its `i_mapping`. This function is typically called from file open paths,
 * guaranteeing `fs_initcalls` completion.
 *
 * @return A pointer to the `address_space` for I/O memory.
 */
struct address_space *iomem_get_mapping(void)
{
	/*
	 * This function is only called from file open paths, hence guaranteed
	 * that fs_initcalls have completed and no need to check for NULL. But
	 * since revoke_iomem can be called before the initcall we still need
	 * the barrier to appease checkers.
	 */
	// Invariant: The `iomem_inode` is guaranteed to be initialized before `iomem_get_mapping` is called, ensuring a valid `i_mapping` is returned.
	return smp_load_acquire(&iomem_inode)->i_mapping;
}

/**
 * @brief Internal function to request an I/O resource region, with waiting for muxed resources.
 * @details This function attempts to request a resource region specified by `start` and `n`
 * under a `parent` resource. If a conflict occurs, and the conflicting resource
 * has the `IORESOURCE_MUXED` flag set, the function will put the current process
 * to sleep and wait for the resource to become available. This allows for contention
 * resolution for multiplexed I/O resources.
 *
 * @param res The resource structure to populate with the requested region details.
 * @param parent The parent resource descriptor.
 * @param start Resource start address.
 * @param n Resource region size.
 * @param name Reserving caller's ID string.
 * @param flags I/O resource flags.
 * @return 0 on success, or `-EBUSY` if the resource cannot be acquired.
 */
static int __request_region_locked(struct resource *res, struct resource *parent,
				   resource_size_t start, resource_size_t n,
				   const char *name, int flags)
{
	DECLARE_WAITQUEUE(wait, current);

	res->name = name;
	res->start = start;
	res->end = start + n - 1;

	// Invariant: Loop indefinitely until the resource is successfully requested or an unresolvable conflict occurs.
	for (;;) {
		struct resource *conflict;

		res->flags = resource_type(parent) | resource_ext_type(parent);
		res->flags |= IORESOURCE_BUSY | flags;
		res->desc = parent->desc;

		// Pre-condition: Attempt to request the resource under the current parent.
		// Invariant: 'conflict' will hold the conflicting resource if any, otherwise NULL.
		conflict = __request_resource(parent, res);
		// Invariant: If no conflict, the current 'res' part is successfully reserved.
		// Invariant: If no conflict, the current 'res' part is successfully reserved.
		if (!conflict)
			break;
		/*
		 * mm/hmm.c reserves physical addresses which then
		 * become unavailable to other users.  Conflicts are
		 * not expected.  Warn to aid debugging if encountered.
		 */
		// Pre-condition: If the parent is `iomem_resource` and the conflict is with device private memory.
		if (parent == &iomem_resource &&
		    conflict->desc == IORES_DESC_DEVICE_PRIVATE_MEMORY) {
			// Invariant: Log a warning for unexpected conflicts with unaddressable device memory.
			pr_warn("Unaddressable device %s %pR conflicts with %pR\n",
				conflict->name, conflict, res);
		}
		// Pre-condition: Conflict is not with the parent itself.
		// Invariant: If conflict is with a child resource that is not busy, try to request under that child.
		if (conflict != parent) {
			if (!(conflict->flags & IORESOURCE_BUSY)) {
				parent = conflict;
				continue;
			}
		}
		// Pre-condition: Both conflicting resource and requested flags have IORESOURCE_MUXED set.
		// Invariant: If a muxed resource conflict occurs, wait for it to become available.
		if (conflict->flags & flags & IORESOURCE_MUXED) {
			add_wait_queue(&muxed_resource_wait, &wait); // Add current process to wait queue.
			write_unlock(&resource_lock); // Release write lock before sleeping.
			set_current_state(TASK_UNINTERRUPTIBLE); // Set task state to uninterruptible.
			schedule(); // Schedule out the current task.
			remove_wait_queue(&muxed_resource_wait, &wait); // Remove from wait queue after waking up.
			write_lock(&resource_lock); // Reacquire write lock.
			continue; // Retry the request.
		}
		/* Uhhuh, that didn't work out.. */
		return -EBUSY; // Unresolvable conflict.
	}

	return 0; // Success.
}

/**
 * @brief Creates a new busy resource region.
 * @details This function allocates a new `struct resource`, populates it with the
 * requested details (`start`, `n`, `name`, `flags`), and attempts to
 * acquire the region using `__request_region_locked`. If successful,
 * and the parent is `iomem_resource`, it calls `revoke_iomem` to
 * potentially unmap user-space mappings.
 *
 * @param parent Parent resource descriptor.
 * @param start Resource start address.
 * @param n Resource region size.
 * @param name Reserving caller's ID string.
 * @param flags IO resource flags.
 * @return A pointer to the newly allocated and requested resource on success, NULL on failure.
 */
struct resource *__request_region(struct resource *parent,
				  resource_size_t start, resource_size_t n,
				  const char *name, int flags)
{
	// Pre-condition: Attempt to allocate memory for the new resource.
	struct resource *res = alloc_resource(GFP_KERNEL);
	if (!res)
		return NULL; // Failed to allocate resource.

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	// Pre-condition: Attempt to request the region using the internal locked function.
	// Invariant: 'ret' will be 0 on success, or an error code on failure.
	int ret = __request_region_locked(res, parent, start, n, name, flags); // Attempt to request the region.
	write_unlock(&resource_lock); // Release write lock.

	if (ret) {
		free_resource(res); // Free allocated resource on failure.
		return NULL;
	}

	// Pre-condition: If the parent is the I/O memory resource, revoke its user-space mappings.
	if (parent == &iomem_resource)
		revoke_iomem(res);

	return res; // Return the successfully requested resource.
}
EXPORT_SYMBOL(__request_region);

/**
 * @brief Releases a previously reserved resource region.
 * @details This function releases a resource region previously reserved using
 * `__request_region`. It searches the children of the `parent` resource
 * for a matching busy region and removes it from the tree.
 *
 * @param parent Parent resource descriptor.
 * @param start Resource start address.
 * @param n Resource region size.
 */
void __release_region(struct resource *parent, resource_size_t start,
		      resource_size_t n)
{
	struct resource **p;
	resource_size_t end;

	p = &parent->child;
	end = start + n - 1;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.

	// Invariant: Iterate through the parent's children to find the resource to be released.
	for (;;) {
		struct resource *res = *p;

		if (!res) // No more resources to check.
			break;
		// Pre-condition: Check if the current resource 'res' completely encompasses the target region.
		if (res->start <= start && res->end >= end) {
			// Pre-condition: If the resource is not busy, it means it's an ancestor of the target.
			// Invariant: Traverse into the children of the current resource.
			if (!(res->flags & IORESOURCE_BUSY)) {
				p = &res->child;
				continue;
			}
			// Pre-condition: If the resource exactly matches the target, release it.
			if (res->start != start || res->end != end)
				break; // Partial match, but not exact, implies error or wrong logic.
			*p = res->sibling; // Remove the resource from the list.
			write_unlock(&resource_lock); // Release write lock before waking up.
			// Pre-condition: If the resource was muxed, wake up waiting processes.
			if (res->flags & IORESOURCE_MUXED)
				wake_up(&muxed_resource_wait);
			free_resource(res); // Free the resource.
			return; // Successfully released.
		}
		p = &res->sibling; // Move to the next sibling.
	}

	write_unlock(&resource_lock); // Release write lock.

	pr_warn("Trying to free nonexistent resource <%pa-%pa>\n", &start, &end); // Log warning for nonexistent resource.
}
EXPORT_SYMBOL(__release_region);

#ifdef CONFIG_MEMORY_HOTREMOVE
/**
 * @brief Releases a previously reserved memory region, with support for adjustment.
 * @details This interface is intended for memory hot-delete. The requested region
 * is released from a currently busy memory resource. The requested region
 * must either match exactly or fit into a single busy resource entry. In
 * the latter case, the remaining resource is adjusted accordingly.
 * Existing children of the busy memory resource must be immutable in the
 * request.
 *
 * @param start Resource start address.
 * @param size Resource region size.
 */
void release_mem_region_adjustable(resource_size_t start, resource_size_t size)
{
	struct resource *parent = &iomem_resource;
	struct resource *new_res = NULL;
	bool alloc_nofail = false;
	struct resource **p;
	struct resource *res;
	resource_size_t end;

	end = start + size - 1;
	// Pre-condition: Warn if the requested region is outside the parent's bounds.
	if (WARN_ON_ONCE((start < parent->start) || (end > parent->end)))
		return;

	/*
	 * We free up quite a lot of memory on memory hotunplug (esp., memap),
	 * just before releasing the region. This is highly unlikely to
	 * fail - let's play save and make it never fail as the caller cannot
	 * perform any error handling (e.g., trying to re-add memory will fail
	 * similarly).
	 */
retry:
	new_res = alloc_resource(GFP_KERNEL | (alloc_nofail ? __GFP_NOFAIL : 0));

	p = &parent->child;
	write_lock(&resource_lock); // Acquire write lock.

	// Invariant: Iterate through the parent's children to find and adjust the target resource.
	while ((res = *p)) {
		// Pre-condition: Current resource starts after the target end, so no overlap.
		if (res->start >= end)
			break;

		/* look for the next resource if it does not fit into */
		// Pre-condition: Current resource's range does not fully encompass the target.
		if (res->start > start || res->end < end) {
			p = &res->sibling; // Move to the next sibling.
			continue;
		}

		// Pre-condition: If not an IORESOURCE_MEM type, break.
		if (!(res->flags & IORESOURCE_MEM))
			break;

		// Pre-condition: If not busy, it's an ancestor, so traverse children.
		if (!(res->flags & IORESOURCE_BUSY)) {
			p = &res->child;
			continue;
		}

		/* found the target resource; let's adjust accordingly */
		// Pre-condition: Exact match for the entire resource.
		if (res->start == start && res->end == end) {
			/* free the whole entry */
			*p = res->sibling; // Remove from list.
			free_resource(res); // Free the resource.
		} else if (res->start == start && res->end != end) {
			/* adjust the start */
			// Pre-condition: The resource starts at the target start but extends beyond target end.
			// Invariant: Adjust the resource to start after the released portion.
			WARN_ON_ONCE(__adjust_resource(res, end + 1,
						       res->end - end));
		} else if (res->start != start && res->end == end) {
			/* adjust the end */
			// Pre-condition: The resource ends at the target end but starts before target start.
			// Invariant: Adjust the resource to end before the released portion.
			WARN_ON_ONCE(__adjust_resource(res, res->start,
						       start - res->start));
		} else {
			/* split into two entries - we need a new resource */
			// Pre-condition: The target region is in the middle of a larger resource.
			// Invariant: Split the resource into two parts around the released region.
			if (!new_res) {
				new_res = alloc_resource(GFP_ATOMIC); // Allocate for the second part.
				if (!new_res) {
					alloc_nofail = true;
					write_unlock(&resource_lock); // Release lock before retrying.
					goto retry;
				}
			}
			// Invariant: Populate the new resource with the details of the second part.
			new_res->name = res->name;
			new_res->start = end + 1;
			new_res->end = res->end;
			new_res->flags = res->flags;
			new_res->desc = res->desc;
			new_res->parent = res->parent;
			new_res->sibling = res->sibling;
			new_res->child = NULL;

			// Invariant: Adjust the original resource to cover the first part.
			if (WARN_ON_ONCE(__adjust_resource(res, res->start,
							   start - res->start)))
				break; // Break if adjustment fails.
			res->sibling = new_res; // Link the two parts.
			new_res = NULL; // Clear new_res for subsequent allocations.
		}

		break; // Break after processing the target resource.
	}

	write_unlock(&resource_lock); // Release write lock.
	free_resource(new_res); // Free any leftover new_res if not used.
}
#endif	/* CONFIG_MEMORY_HOTREMOVE */

#ifdef CONFIG_MEMORY_HOTPLUG
/**
 * @brief Checks if two System RAM resources are mergeable.
 * @details This utility function determines if two adjacent System RAM resources (`r1` and `r2`)
 * can be merged into a single resource. The conditions for merging are:
 * 1. Both resources have the same flags.
 * 2. `r1` ends immediately before `r2` starts (`r1->end + 1 == r2->start`).
 * 3. Both resources have the same name.
 * 4. Both resources have the same descriptor.
 * 5. Neither resource has any children.
 *
 * @param r1 The first resource.
 * @param r2 The second resource.
 * @return True if the resources are mergeable, false otherwise.
 */
static bool system_ram_resources_mergeable(struct resource *r1,
					   struct resource *r2)
{
	/* We assume either r1 or r2 is IORESOURCE_SYSRAM_MERGEABLE. */
	return r1->flags == r2->flags && r1->end + 1 == r2->start &&
	       r1->name == r2->name && r1->desc == r2->desc &&
	       !r1->child && !r2->child;
}

/**
 * @brief Marks a System RAM resource mergeable and attempts to merge it with adjacent resources.
 * @details This function is intended for memory hotplug scenarios. It marks the given
 * System RAM resource `res` as `IORESOURCE_SYSRAM_MERGEABLE` and then attempts
 * to merge it with any immediately adjacent, mergeable System RAM resources in
 * both forward and backward directions within the resource tree. The caller must
 * ensure that no stale pointers to potentially freed resources are used after this call.
 *
 * @param res Resource descriptor to mark mergeable and attempt to merge.
 */
void merge_system_ram_resource(struct resource *res)
{
	const unsigned long flags = IORESOURCE_SYSTEM_RAM | IORESOURCE_BUSY;
	struct resource *cur;

	// Pre-condition: Warn if the resource does not have the expected System RAM and BUSY flags.
	if (WARN_ON_ONCE((res->flags & flags) != flags))
		return;

	write_lock(&resource_lock); // Acquire write lock for resource tree modification.
	res->flags |= IORESOURCE_SYSRAM_MERGEABLE; // Mark the resource as mergeable.

	/* Try to merge with next item in the list. */
	cur = res->sibling;
	// Pre-condition: Check if there's a next sibling and if it's mergeable with the current resource.
	// Invariant: If mergeable, extend the current resource's end and re-link siblings, then free the merged resource.
	if (cur && system_ram_resources_mergeable(res, cur)) {
		res->end = cur->end;
		res->sibling = cur->sibling;
		free_resource(cur);
	}

	/* Try to merge with previous item in the list. */
	cur = res->parent->child;
	// Invariant: Traverse the parent's children to find the sibling immediately preceding 'res'.
	while (cur && cur->sibling != res)
		cur = cur->sibling;
	// Pre-condition: Check if there's a previous sibling and if it's mergeable with the current resource.
	// Invariant: If mergeable, extend the previous resource's end and re-link siblings, then free the current resource.
	if (cur && system_ram_resources_mergeable(cur, res)) {
		cur->end = res->end;
		cur->sibling = res->sibling;
		free_resource(res);
	}
	write_unlock(&resource_lock); // Release write lock.
}
#endif	/* CONFIG_MEMORY_HOTPLUG */

/**
 * @defgroup ManagedRegionResources Managed Region Resources
 * @brief This section provides functions for managing device-managed resources,
 * ensuring proper allocation, tracking, and automatic release of resources
 * when the associated device is unbound from its driver. These functions
 * simplify resource management for device drivers.
 */

/**
 * @brief Device-managed release callback for a resource.
 * @details This function is registered with the device resource manager (`devres`)
 * to automatically release a `struct resource` when the associated device
 * is unbound from its driver. It calls `release_resource` on the pointer
 * to the resource stored in `ptr`.
 *
 * @param dev The device associated with the resource.
 * @param ptr A pointer to a `struct resource *` that needs to be released.
 */
static void devm_resource_release(struct device *dev, void *ptr)
{
	struct resource **r = ptr;

	release_resource(*r);
}

/**
 * devm_request_resource() - request and reserve an I/O or memory resource
 * @dev: device for which to request the resource
 * @root: root of the resource tree from which to request the resource
 * @new: descriptor of the resource to request
 *
 * This is a device-managed version of request_resource(). There is usually
 * no need to release resources requested by this function explicitly since
 * that will be taken care of when the device is unbound from its driver.
 * If for some reason the resource needs to be released explicitly, because
 * of ordering issues for example, drivers must call devm_release_resource()
 * rather than the regular release_resource().
 *
 * When a conflict is detected between any existing resources and the newly
 * requested resource, an error message will be printed.
 *
 * Returns 0 on success or a negative error code on failure.
 */
int devm_request_resource(struct device *dev, struct resource *root,
			  struct resource *new)
{
	struct resource *conflict, **ptr;

	ptr = devres_alloc(devm_resource_release, sizeof(*ptr), GFP_KERNEL);
	if (!ptr)
		return -ENOMEM;

	*ptr = new;

	conflict = request_resource_conflict(root, new);
	if (conflict) {
		dev_err(dev, "resource collision: %pR conflicts with %s %pR\n",
			new, conflict->name, conflict);
		devres_free(ptr);
		return -EBUSY;
	}

	devres_add(dev, ptr);
	return 0;
}
EXPORT_SYMBOL(devm_request_resource);

/**
 * @brief Device-managed resource matching callback.
 * @details This function is a callback used by the device resource manager
 * (`devres`) to match a stored resource (`res`) with a given data pointer
 * (`data`). It is typically used during resource release to find the correct
 * resource entry to free.
 *
 * @param dev The device associated with the resource.
 * @param res A pointer to the stored resource entry (which is a `struct resource **`).
 * @param data A pointer to the resource being searched for (`struct resource *`).
 * @return 1 if the resources match, 0 otherwise.
 */
static int devm_resource_match(struct device *dev, void *res, void *data)
{
	struct resource **ptr = res;

	return *ptr == data;
}

/**
 * devm_release_resource() - release a previously requested resource
 * @dev: device for which to release the resource
 * @new: descriptor of the resource to release
 *
 * Releases a resource previously requested using devm_request_resource().
 */
void devm_release_resource(struct device *dev, struct resource *new)
{
	WARN_ON(devres_release(dev, devm_resource_release, devm_resource_match,
			       new));
}
EXPORT_SYMBOL(devm_release_resource);

/**
 * @brief Structure to store information for device-managed region resources.
 * @details This structure holds the necessary details to release a region
 * that was requested using device-managed functions. It captures the
 * parent resource, the starting address, and the size of the region.
 */
struct region_devres {
	struct resource *parent; /**< @brief The parent resource of the region. */
	resource_size_t start;   /**< @brief The starting address of the region. */
	resource_size_t n;       /**< @brief The size of the region. */
};

/**
 * @brief Device-managed release callback for a region resource.
 * @details This function is registered with the device resource manager (`devres`)
 * to automatically release a region resource when the associated device
 * is unbound from its driver. It extracts the parent, start, and size
 * information from the `region_devres` structure and calls `__release_region`.
 *
 * @param dev The device associated with the region.
 * @param res A pointer to the `region_devres` structure containing details of the region to be released.
 */
static void devm_region_release(struct device *dev, void *res)
{
	struct region_devres *this = res;

	__release_region(this->parent, this->start, this->n);
}

/**
 * @brief Device-managed region matching callback.
 * @details This function is a callback used by the device resource manager
 * (`devres`) to match a stored `region_devres` entry with a provided
 * `match_data`. It compares the parent resource, start address, and size
 * to determine if they represent the same region.
 *
 * @param dev The device associated with the region.
 * @param res A pointer to the stored `region_devres` entry.
 * @param match_data A pointer to the `region_devres` structure to match against.
 * @return 1 if the regions match, 0 otherwise.
 */
static int devm_region_match(struct device *dev, void *res, void *match_data)
{
	struct region_devres *this = res, *match = match_data;

	return this->parent == match->parent &&
		this->start == match->start && this->n == match->n;
}

/**
 * @brief Device-managed request for a resource region.
 * @details This function provides a device-managed way to request a resource
 * region. It allocates a `region_devres` structure to track the requested
 * region and ensures its automatic release when the device is unbound.
 * It uses `__request_region` to perform the actual resource allocation.
 *
 * @param dev The device for which the resource is being requested.
 * @param parent The parent resource under which the new region should be allocated.
 * @param start The starting address of the requested region.
 * @param n The size of the requested region.
 * @param name The name of the requested region.
 * @return A pointer to the allocated `struct resource` on success, or NULL on failure.
 */
struct resource *
__devm_request_region(struct device *dev, struct resource *parent,
		      resource_size_t start, resource_size_t n, const char *name)
{
	struct region_devres *dr = NULL;
	struct resource *res;

	dr = devres_alloc(devm_region_release, sizeof(struct region_devres),
			  GFP_KERNEL);
	if (!dr)
		return NULL;

	dr->parent = parent;
	dr->start = start;
	dr->n = n;

	res = __request_region(parent, start, n, name, 0);
	if (res)
		devres_add(dev, dr);
	else
		devres_free(dr);

	return res;
}
EXPORT_SYMBOL(__devm_request_region);

/**
 * @brief Device-managed release of a resource region.
 * @details This function provides a device-managed way to explicitly release
 * a resource region that was previously requested using `__devm_request_region`.
 * It uses the `devres_release` mechanism to find and free the associated
 * `region_devres` entry.
 *
 * @param dev The device associated with the region.
 * @param parent The parent resource of the region being released.
 * @param start The starting address of the region being released.
 * @param n The size of the region being released.
 */
void __devm_release_region(struct device *dev, struct resource *parent,
			   resource_size_t start, resource_size_t n)
{
	struct region_devres match_data = { parent, start, n };

	WARN_ON(devres_release(dev, devm_region_release, devm_region_match,
			       &match_data));
}
EXPORT_SYMBOL(__devm_release_region);

/**
 * @brief Reserves I/O ports or memory regions based on kernel command-line parameters.
 * @details This function is an early initialization routine (`__init`) that parses
 * the "reserve=" kernel command-line parameter. It interprets specified start
 * addresses and sizes to reserve I/O port or memory regions. Regions below
 * 0x10000 are treated as I/O ports, and others as memory.
 * It uses a static array `reserve` to track up to `MAXRESERVE` reservations.
 *
 * @param str The string containing the "reserve=" kernel parameter arguments.
 * @return 1 on successful parsing and reservation.
 */
#define MAXRESERVE 4
static int __init reserve_setup(char *str)
{
	static int reserved;
	static struct resource reserve[MAXRESERVE];

	// Invariant: Loop indefinitely to parse multiple "reserve=" entries until no more options are found.
	for (;;) {
		unsigned int io_start, io_num;
		int x = reserved;
		struct resource *parent;

		// Pre-condition: Attempt to retrieve the start address and number of units from the command line string.
		// Invariant: If parsing fails, break the loop as there are no more valid options.
		if (get_option(&str, &io_start) != 2)
			break;
		// Pre-condition: Attempt to retrieve the number of units from the command line string.
		// Invariant: If parsing fails, break the loop.
		if (get_option(&str, &io_num) == 0)
			break;
		// Pre-condition: Check if there is space left in the `reserve` array.
		// Invariant: If there's space, a new resource is prepared and an attempt is made to reserve it.
		if (x < MAXRESERVE) {
			struct resource *res = reserve + x;

			/*
			 * Pre-condition: Check if the I/O region starts below 0x10000.
			 * Invariant: If true, it's treated as I/O port space; otherwise, as memory.
			 */
			if (io_start < 0x10000) {
				*res = DEFINE_RES_IO_NAMED(io_start, io_num, "reserved");
				parent = &ioport_resource;
			} else {
				*res = DEFINE_RES_MEM_NAMED(io_start, io_num, "reserved");
				parent = &iomem_resource;
			}
			res->flags |= IORESOURCE_BUSY; // Mark the resource as busy.
			// Pre-condition: Attempt to request the resource.
			// Invariant: If successful, increment the count of reserved resources.
			if (request_resource(parent, res) == 0)
				reserved = x+1;
		}
	}
	return 1; // Indicate successful processing of the setup string.
}
__setup("reserve=", reserve_setup);

/**
 * @brief Performs a sanity check on a requested I/O memory mapping.
 * @details This function checks if a requested I/O memory region (`addr` to `addr + size - 1`)
 * spans across multiple existing, non-busy resources within the `iomem_resource` tree.
 * It's used to prevent erroneous mappings that might indicate a driver attempting
 * to map a region larger than or spanning multiple disjoint hardware resources.
 *
 * @param addr The starting address of the requested I/O memory region.
 * @param size The size of the requested I/O memory region.
 * @return 0 on success (no sanity check violation), -1 if a sanity check fails.
 */
int iomem_map_sanity_check(resource_size_t addr, unsigned long size)
{
	resource_size_t end = addr + size - 1;
	struct resource *p;
	int err = 0;

	read_lock(&resource_lock);
	// Invariant: Iterate through each resource in the iomem_resource tree.
	for_each_resource(&iomem_resource, p, false) {
		// Pre-condition: Check if the current resource 'p' starts after the requested end.
		// Invariant: If true, no further resources in this branch can overlap, so continue to the next.
		if (p->start > end)
			continue;
		// Pre-condition: Check if the current resource 'p' ends before the requested start.
		// Invariant: If true, no overlap with the requested region, so continue to the next.
		if (p->end < addr)
			continue;
		// Pre-condition: Check if the requested region is fully contained within the current resource 'p'
		// on a PFN (Page Frame Number) basis.
		// Invariant: If true, the current resource 'p' completely covers the requested area, which is fine.
		if (PFN_DOWN(p->start) <= PFN_DOWN(addr) &&
		    PFN_DOWN(p->end) >= PFN_DOWN(end))
			continue;
		/*
		 * Pre-condition: Check if the resource is marked as "BUSY".
		 * Invariant: If a resource is "BUSY", it typically represents a driver's existing mapping,
		 *            not a raw hardware resource. Warnings are not issued for these as partial
		 *            mappings are legitimate in such cases (e.g., vesafb).
		 */
		if (p->flags & IORESOURCE_BUSY)
			continue;

		// Invariant: If none of the above conditions are met, a sanity check violation is detected.
		// This indicates that the requested region spans more than one non-busy hardware resource.
		pr_warn("resource sanity check: requesting [mem %pa-%pa], which spans more than %s %pR\n",
			&addr, &end, p->name, p);
		err = -1;
		break; // Exit the loop as a violation has been found.
	}
	read_unlock(&resource_lock);

	return err;
}

/**
 * @brief Global flag to control strictness of I/O memory access checks.
 * @details This static variable controls whether strict checks are performed
 * for I/O memory access, particularly for mappings to user space via /dev/mem.
 * It can be configured via the "iomem=" kernel parameter.
 * If `CONFIG_STRICT_DEVMEM` is enabled, it defaults to 1 (strict); otherwise, 0 (relaxed).
 */
#ifdef CONFIG_STRICT_DEVMEM
static int strict_iomem_checks = 1;
#else
static int strict_iomem_checks;
#endif

/**
 * @brief Checks if a given memory address range is exclusive to the kernel.
 * @details This function determines if a specified memory region (`addr` to `addr + size`)
 * should not be mapped to user space (e.g., via `/dev/mem`). This is critical
 * for security and stability, preventing user-space access to sensitive kernel-only
 * memory regions. The check considers `IORESOURCE_EXCLUSIVE` flags,
 * `IORESOURCE_SYSTEM_RAM` resources, and the `strict_iomem_checks` setting.
 *
 * @param root The root of the resource tree to search within (e.g., `iomem_resource`).
 * @param addr The starting address of the memory region to check.
 * @param size The size of the memory region to check.
 * @return True if the region is exclusive to the kernel and should not be mapped to user space, false otherwise.
 */
bool resource_is_exclusive(struct resource *root, u64 addr, resource_size_t size)
{
	const unsigned int exclusive_system_ram = IORESOURCE_SYSTEM_RAM |
						  IORESOURCE_EXCLUSIVE;
	bool skip_children = false, err = false;
	struct resource *p;

	read_lock(&resource_lock);
	// Invariant: Iterate through the resources in the tree to find any overlaps with the target region.
	for_each_resource(root, p, skip_children) {
		// Pre-condition: If the current resource starts after the target region ends.
		// Invariant: No further resources in this branch can overlap, so break.
		if (p->start >= addr + size)
			break;
		// Pre-condition: If the current resource ends before the target region starts.
		// Invariant: No overlap, skip children and continue to the next sibling.
		if (p->end < addr) {
			skip_children = true;
			continue;
		}
		// Invariant: Reset skip_children as an overlap or potential overlap is found.
		skip_children = false;

		/*
		 * Pre-condition: Check if the resource is IORESOURCE_SYSTEM_RAM and IORESOURCE_EXCLUSIVE.
		 * Invariant: Such resources are always exclusive, regardless of other settings.
		 */
		if ((p->flags & exclusive_system_ram) == exclusive_system_ram) {
			err = true;
			break;
		}

		/*
		 * Pre-condition: Check if strict iomem checks are enabled AND the resource is not busy.
		 * Invariant: If not busy and strict checks apply, it might not be exclusive.
		 *
		 * Pre-condition: Check if CONFIG_IO_STRICT_DEVMEM is enabled OR the resource has IORESOURCE_EXCLUSIVE.
		 * Invariant: If either is true, and the resource is busy (checked above), it's considered exclusive.
		 */
		if (!strict_iomem_checks || !(p->flags & IORESOURCE_BUSY))
			continue;
		if (IS_ENABLED(CONFIG_IO_STRICT_DEVMEM)
				|| p->flags & IORESOURCE_EXCLUSIVE) {
			err = true;
			break;
		}
	}
	read_unlock(&resource_lock);

	return err;
}

/**
 * @brief Checks if a specific I/O memory address is exclusive to the kernel.
 * @details This is a convenience wrapper around `resource_is_exclusive`
 * tailored for I/O memory (`iomem_resource`). It checks if a given page-aligned
 * address within I/O memory should be exclusive to the kernel and thus
 * not accessible to user space.
 *
 * @param addr The I/O memory address to check. It will be page-aligned internally.
 * @return True if the I/O memory page is exclusive to the kernel, false otherwise.
 */
bool iomem_is_exclusive(u64 addr)
{
	return resource_is_exclusive(&iomem_resource, addr & PAGE_MASK,
				     PAGE_SIZE);
}

/**
 * @brief Creates a new `resource_entry` and optionally initializes its resource.
 * @details This function allocates memory for a `struct resource_entry` and
 * an optional `extra_size` to store additional data. It initializes the
 * list head and sets up the internal `struct resource` pointer. If `res`
 * is provided, it points to that resource; otherwise, it points to an
 * embedded resource within the `resource_entry`.
 *
 * @param res An optional pointer to an existing `struct resource` to associate with the entry.
 *            If NULL, an embedded resource within the entry is used.
 * @param extra_size Additional memory to allocate after the `resource_entry` for custom data.
 * @return A pointer to the newly created and initialized `struct resource_entry` on success, or NULL on failure.
 */
struct resource_entry *resource_list_create_entry(struct resource *res,
						  size_t extra_size)
{
	struct resource_entry *entry;

	entry = kzalloc(sizeof(*entry) + extra_size, GFP_KERNEL);
	if (entry) {
		INIT_LIST_HEAD(&entry->node);
		entry->res = res ? res : &entry->__res;
	}

	return entry;
}
EXPORT_SYMBOL(resource_list_create_entry);

/**
 * @brief Frees all `resource_entry` elements in a linked list.
 * @details This function iterates safely through a list of `resource_entry`
 * structures and frees each entry using `resource_list_destroy_entry`.
 * It is typically used to clean up a list of resources.
 *
 * @param head A pointer to the head of the `list_head` containing `resource_entry` elements.
 */
void resource_list_free(struct list_head *head)
{
	struct resource_entry *entry, *tmp;

	list_for_each_entry_safe(entry, tmp, head, node)
		resource_list_destroy_entry(entry);
}
EXPORT_SYMBOL(resource_list_free);

#ifdef CONFIG_GET_FREE_REGION
/**
 * @brief Flag for `get_free_mem_region` to search in descending order.
 * @details When this flag is set, `get_free_mem_region` will search for a free
 * memory region starting from higher addresses and moving downwards.
 */
#define GFR_DESCENDING		(1UL << 0)
/**
 * @brief Flag for `get_free_mem_region` to request a region.
 * @details When this flag is set, `get_free_mem_region` will use `__request_region_locked`
 * to request and mark the found free memory region as busy.
 */
#define GFR_REQUEST_REGION	(1UL << 1)
#ifdef PA_SECTION_SHIFT
/**
 * @brief Default alignment for free memory regions based on `PA_SECTION_SHIFT`.
 * @details This macro defines the default alignment for memory regions,
 * typically used by `get_free_mem_region`. It leverages `PA_SECTION_SHIFT`
 * if defined, which often relates to architecture-specific page or section sizes.
 */
#define GFR_DEFAULT_ALIGN	(1UL << PA_SECTION_SHIFT)
#else
/**
 * @brief Default alignment for free memory regions, falling back to `PAGE_SIZE`.
 * @details If `PA_SECTION_SHIFT` is not defined, this macro provides a default
 * alignment based on `PAGE_SIZE`, ensuring memory regions are aligned to page boundaries.
 */
#define GFR_DEFAULT_ALIGN	PAGE_SIZE
#endif

static resource_size_t gfr_start(struct resource *base, resource_size_t size,
				 resource_size_t align, unsigned long flags)
{
	if (flags & GFR_DESCENDING) {
		resource_size_t end;

		end = min_t(resource_size_t, base->end, DIRECT_MAP_PHYSMEM_END);
		return end - size + 1;
	}

	return ALIGN(max(base->start, align), align);
}

static bool gfr_continue(struct resource *base, resource_size_t addr,
			 resource_size_t size, unsigned long flags)
{
	if (flags & GFR_DESCENDING)
		return addr > size && addr >= base->start;
	/*
	 * In the ascend case be careful that the last increment by
	 * @size did not wrap 0.
	 */
	return addr > addr - size &&
	       addr <= min_t(resource_size_t, base->end, DIRECT_MAP_PHYSMEM_END);
}

static resource_size_t gfr_next(resource_size_t addr, resource_size_t size,
				unsigned long flags)
{
	if (flags & GFR_DESCENDING)
		return addr - size;
	return addr + size;
}

static void remove_free_mem_region(void *_res)
{
	struct resource *res = _res;

	if (res->parent)
		remove_resource(res);
	free_resource(res);
}

static struct resource *
get_free_mem_region(struct device *dev, struct resource *base,
		    resource_size_t size, const unsigned long align,
		    const char *name, const unsigned long desc,
		    const unsigned long flags)
{
	resource_size_t addr;
	struct resource *res;
	struct region_devres *dr = NULL;

	size = ALIGN(size, align);

	res = alloc_resource(GFP_KERNEL);
	if (!res)
		return ERR_PTR(-ENOMEM);

	if (dev && (flags & GFR_REQUEST_REGION)) {
		dr = devres_alloc(devm_region_release,
				sizeof(struct region_devres), GFP_KERNEL);
		if (!dr) {
			free_resource(res);
			return ERR_PTR(-ENOMEM);
		}
	} else if (dev) {
		if (devm_add_action_or_reset(dev, remove_free_mem_region, res))
			return ERR_PTR(-ENOMEM);
	}

	write_lock(&resource_lock);
	for (addr = gfr_start(base, size, align, flags);
	     gfr_continue(base, addr, align, flags);
	     addr = gfr_next(addr, align, flags)) {
		if (__region_intersects(base, addr, size, 0, IORES_DESC_NONE) !=
		    REGION_DISJOINT)
			continue;

		if (flags & GFR_REQUEST_REGION) {
			if (__request_region_locked(res, &iomem_resource, addr,
						    size, name, 0))
				break;

			if (dev) {
				dr->parent = &iomem_resource;
				dr->start = addr;
				dr->n = size;
				devres_add(dev, dr);
			}

			res->desc = desc;
			write_unlock(&resource_lock);


			/*
			 * A driver is claiming this region so revoke any
			 * mappings.
			 */
			revoke_iomem(res);
		} else {
			*res = DEFINE_RES_NAMED_DESC(addr, size, name, IORESOURCE_MEM, desc);

			/*
			 * Only succeed if the resource hosts an exclusive
			 * range after the insert
			 */
			if (__insert_resource(base, res) || res->child)
				break;

			write_unlock(&resource_lock);
		}

		return res;
	}
	write_unlock(&resource_lock);

	if (flags & GFR_REQUEST_REGION) {
		free_resource(res);
		devres_free(dr);
	} else if (dev)
		devm_release_action(dev, remove_free_mem_region, res);

	return ERR_PTR(-ERANGE);
}

/**
 * devm_request_free_mem_region - find free region for device private memory
 *
 * @dev: device struct to bind the resource to
 * @size: size in bytes of the device memory to add
 * @base: resource tree to look in
 *
 * This function tries to find an empty range of physical address big enough to
 * contain the new resource, so that it can later be hotplugged as ZONE_DEVICE
 * memory, which in turn allocates struct pages.
 */
struct resource *devm_request_free_mem_region(struct device *dev,
		struct resource *base, unsigned long size)
{
	unsigned long flags = GFR_DESCENDING | GFR_REQUEST_REGION;

	return get_free_mem_region(dev, base, size, GFR_DEFAULT_ALIGN,
				   dev_name(dev),
				   IORES_DESC_DEVICE_PRIVATE_MEMORY, flags);
}
EXPORT_SYMBOL_GPL(devm_request_free_mem_region);

struct resource *request_free_mem_region(struct resource *base,
		unsigned long size, const char *name)
{
	unsigned long flags = GFR_DESCENDING | GFR_REQUEST_REGION;

	return get_free_mem_region(NULL, base, size, GFR_DEFAULT_ALIGN, name,
				   IORES_DESC_DEVICE_PRIVATE_MEMORY, flags);
}
EXPORT_SYMBOL_GPL(request_free_mem_region);

/**
 * alloc_free_mem_region - find a free region relative to @base
 * @base: resource that will parent the new resource
 * @size: size in bytes of memory to allocate from @base
 * @align: alignment requirements for the allocation
 * @name: resource name
 *
 * Buses like CXL, that can dynamically instantiate new memory regions,
 * need a method to allocate physical address space for those regions.
 * Allocate and insert a new resource to cover a free, unclaimed by a
 * descendant of @base, range in the span of @base.
 */
struct resource *alloc_free_mem_region(struct resource *base,
				       unsigned long size, unsigned long align,
				       const char *name)
{
	/* Default of ascending direction and insert resource */
	unsigned long flags = 0;

	return get_free_mem_region(NULL, base, size, align, name,
				   IORES_DESC_NONE, flags);
}
EXPORT_SYMBOL_GPL(alloc_free_mem_region);
#endif /* CONFIG_GET_FREE_REGION */

static int __init strict_iomem(char *str)
{
	if (strstr(str, "relaxed"))
		strict_iomem_checks = 0;
	if (strstr(str, "strict"))
		strict_iomem_checks = 1;
	return 1;
}

static int iomem_fs_init_fs_context(struct fs_context *fc)
{
	return init_pseudo(fc, DEVMEM_MAGIC) ? 0 : -ENOMEM;
}

static struct file_system_type iomem_fs_type = {
	.name		= "iomem",
	.owner		= THIS_MODULE,
	.init_fs_context = iomem_fs_init_fs_context,
	.kill_sb	= kill_anon_super,
};

static int __init iomem_init_inode(void)
{
	static struct vfsmount *iomem_vfs_mount;
	static int iomem_fs_cnt;
	struct inode *inode;
	int rc;

	rc = simple_pin_fs(&iomem_fs_type, &iomem_vfs_mount, &iomem_fs_cnt);
	if (rc < 0) {
		pr_err("Cannot mount iomem pseudo filesystem: %d\n", rc);
		return rc;
	}

	inode = alloc_anon_inode(iomem_vfs_mount->mnt_sb);
	if (IS_ERR(inode)) {
		rc = PTR_ERR(inode);
		pr_err("Cannot allocate inode for iomem: %d\n", rc);
		simple_release_fs(&iomem_vfs_mount, &iomem_fs_cnt);
		return rc;
	}

	/*
	 * Publish iomem revocation inode initialized.
	 * Pairs with smp_load_acquire() in revoke_iomem().
	 */
	smp_store_release(&iomem_inode, inode);

	return 0;
}

fs_initcall(iomem_init_inode);

__setup("iomem=", strict_iomem);
