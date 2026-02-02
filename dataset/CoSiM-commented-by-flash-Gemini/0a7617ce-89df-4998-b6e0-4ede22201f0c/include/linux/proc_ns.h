/**
 * @file proc_ns.h
 * @brief Defines the interface for Netfilter namespace operations within the /proc filesystem.
 *
 * This header file provides definitions and declarations for how various kernel
 * namespaces (PID, UTS, IPC, MNT, USER, CGROUP, TIME, NET) are exposed and managed
 * through the /proc filesystem interface. It includes the `proc_ns_operations`
 * structure which specifies callbacks for getting, putting, installing, and
 * querying owner information for different namespace types.
 *
 * Functional Utility: Facilitates the introspection and manipulation of kernel
 * namespaces by userspace tools through the standardized /proc filesystem. It
 * is crucial for container technologies and process isolation.
 *
 * Architecture:
 * - `proc_ns_operations`: A table of function pointers that provides a generic
 *   interface for VFS operations related to namespace files in /proc (e.g.,
 *   `/proc/self/ns/pid`).
 * - Initial Inode Numbers: Defines specific inode numbers for well-known
 *   initial namespace instances (e.g., `PROC_PID_INIT_INO`).
 * - Inum Allocation/Management: Provides functions for allocating and freeing
 *   inode numbers (`inum`) for namespace objects.
 * - Namespace Retrieval: Functions to get and put references to namespace
 *   objects, and to retrieve path information.
 */
/* SPDX-License-Identifier: GPL-2.0 */
/*
 * procfs namespace bits
 */
#ifndef _LINUX_PROC_NS_H ///< Include guard to prevent multiple inclusions of this header file.
#define _LINUX_PROC_NS_H

#include <linux/ns_common.h> ///< Common namespace definitions.
#include <uapi/linux/nsfs.h> ///< User-space API definitions for namespace filesystems.

struct pid_namespace; ///< Forward declaration of `pid_namespace` structure.
struct nsset; ///< Forward declaration of `nsset` structure.
struct path; ///< Forward declaration of `path` structure.
struct task_struct; ///< Forward declaration of `task_struct` structure.
struct inode; ///< Forward declaration of `inode` structure.

/**
 * @struct proc_ns_operations
 * @brief Table of operations for handling a specific type of namespace in /proc.
 * Functional Utility: Provides a generic interface for the Virtual File System (VFS)
 * to interact with different types of kernel namespaces exposed via `/proc/pid/ns/`
 * entries. Each field is a function pointer to a specific operation relevant to
 * the namespace type.
 */
struct proc_ns_operations {
	const char *name; ///< The name of the namespace type (e.g., "pid", "mnt").
	const char *real_ns_name; ///< The real name of the namespace, if different from `name`.
	int type; ///< The type of namespace (e.g., `CLONE_NEWPID`).
	struct ns_common *(*get)(struct task_struct *task); ///< Function to get a reference to the namespace associated with a task.
	void (*put)(struct ns_common *ns); ///< Function to put (release) a reference to the namespace.
	int (*install)(struct nsset *nsset, struct ns_common *ns); ///< Function to install a namespace for a task.
	struct user_namespace *(*owner)(struct ns_common *ns); ///< Function to get the owning user namespace of a namespace.
	struct ns_common *(*get_parent)(struct ns_common *ns); ///< Function to get the parent namespace.
} __randomize_layout; ///< Attribute to randomize the layout of this structure for security.

// External declarations for global `proc_ns_operations` instances, one for each namespace type.
extern const struct proc_ns_operations netns_operations; ///< Operations for network namespaces.
extern const struct proc_ns_operations utsns_operations; ///< Operations for UTS namespaces.
extern const struct proc_ns_operations ipcns_operations; ///< Operations for IPC namespaces.
extern const struct proc_ns_operations pidns_operations; ///< Operations for PID namespaces.
extern const struct proc_ns_operations pidns_for_children_operations; ///< Operations for PID namespaces (children).
extern const struct proc_ns_operations userns_operations; ///< Operations for user namespaces.
extern const struct proc_ns_operations mntns_operations; ///< Operations for mount namespaces.
extern const struct proc_ns_operations cgroupns_operations; ///< Operations for cgroup namespaces.
extern const struct proc_ns_operations timens_operations; ///< Operations for time namespaces.
extern const struct proc_ns_operations timens_for_children_operations; ///< Operations for time namespaces (children).

/**
 * @brief Anonymous enumeration defining initial inode numbers for various namespace types.
 * Functional Utility: These constants provide well-known inode numbers for the initial
 * instances of each namespace type, which are commonly found under `/proc/self/ns/`.
 * These values are usually defined in `uapi/linux/nsfs.h`.
 */
enum {
	PROC_IPC_INIT_INO	= IPC_NS_INIT_INO,
	PROC_UTS_INIT_INO	= UTS_NS_INIT_INO,
	PROC_USER_INIT_INO	= USER_NS_INIT_INO,
	PROC_PID_INIT_INO	= PID_NS_INIT_INO,
	PROC_CGROUP_INIT_INO	= CGROUP_NS_INIT_INO,
	PROC_TIME_INIT_INO	= TIME_NS_INIT_INO,
	PROC_NET_INIT_INO	= NET_NS_INIT_INO,
	PROC_MNT_INIT_INO	= MNT_NS_INIT_INO,
};

#ifdef CONFIG_PROC_FS ///< Conditional compilation: Only if /proc filesystem is enabled.

/**
 * @brief Allocates a new inode number for a /proc namespace entry.
 * Functional Utility: Provides a unique inode number for a namespace object
 * that will be exposed in the /proc filesystem.
 *
 * @param pino Pointer to an unsigned int where the allocated inode number will be stored.
 * @return 0 on success, or a negative errno on failure.
 */
extern int proc_alloc_inum(unsigned int *pino);
/**
 * @brief Frees a previously allocated inode number for a /proc namespace entry.
 * Functional Utility: Releases an inode number when a namespace object is no
 * longer exposed in /proc.
 *
 * @param inum The inode number to free.
 */
extern void proc_free_inum(unsigned int inum);

#else /* CONFIG_PROC_FS */

/**
 * @brief Inline stub for `proc_alloc_inum` when `CONFIG_PROC_FS` is not enabled.
 * Functional Utility: Provides a no-op implementation or a minimal placeholder
 * when the /proc filesystem is not configured, ensuring compilation.
 *
 * @param inum Pointer to an unsigned int.
 * @return Always 0.
 */
static inline int proc_alloc_inum(unsigned int *inum)
{
	*inum = 1;
	return 0;
}
/**
 * @brief Inline stub for `proc_free_inum` when `CONFIG_PROC_FS` is not enabled.
 * Functional Utility: Provides a no-op implementation when the /proc filesystem
 * is not configured.
 *
 * @param inum The inode number to free.
 */
static inline void proc_free_inum(unsigned int inum) {}

#endif /* CONFIG_PROC_FS */

/**
 * @brief Allocates an inode number for a `ns_common` structure.
 * Functional Utility: Wraps `proc_alloc_inum` to assign an inode number directly
 * to a `ns_common` structure, marking it for exposure in /proc.
 *
 * @param ns Pointer to the `ns_common` structure.
 * @return 0 on success, or a negative errno on failure.
 */
static inline int ns_alloc_inum(struct ns_common *ns)
{
	WRITE_ONCE(ns->stashed, NULL); ///< Functional Utility: Ensure no stashed data is present.
	return proc_alloc_inum(&ns->inum); ///< Allocate inode number and assign to `ns->inum`.
}

/**
 * @brief Frees the inode number associated with a `ns_common` structure.
 * Functional Utility: Releases the inode number held by a `ns_common` structure.
 *
 * @param ns Pointer to the `ns_common` structure.
 */
#define ns_free_inum(ns) proc_free_inum((ns)->inum)

/**
 * @brief Gets the `ns_common` structure from an inode's private data.
 * Functional Utility: Provides a convenient way to retrieve the namespace
 * object associated with an inode in the /proc filesystem.
 *
 * @param inode Pointer to the `inode` structure.
 * @return Pointer to the `ns_common` structure.
 */
#define get_proc_ns(inode) ((struct ns_common *)(inode)->i_private)
/**
 * @brief Retrieves the path to a namespace file for a given task.
 * Functional Utility: Constructs a `path` structure that points to the
 * /proc entry representing a specific namespace type for a given task.
 *
 * @param path Pointer to the `path` structure to fill.
 * @param task Pointer to the `task_struct`.
 * @param ns_ops Pointer to `proc_ns_operations` for the namespace type.
 * @return 0 on success, or a negative errno on failure.
 */
extern int ns_get_path(struct path *path, struct task_struct *task,
			const struct proc_ns_operations *ns_ops);
/**
 * @brief Helper type definition for `ns_get_path_cb`.
 * Functional Utility: Defines the signature for a callback function used to
 * obtain a `ns_common` pointer within `ns_get_path_cb`.
 */
typedef struct ns_common *ns_get_path_helper_t(void *);
/**
 * @brief Retrieves the path to a namespace file using a callback helper.
 * Functional Utility: Provides a flexible way to retrieve a namespace path
 * where the actual namespace object is obtained via a provided callback function.
 *
 * @param path Pointer to the `path` structure to fill.
 * @param ns_get_cb Callback function to get the `ns_common` object.
 * @param private_data Private data to pass to the callback.
 * @return 0 on success, or a negative errno on failure.
 */
extern int ns_get_path_cb(struct path *path, ns_get_path_helper_t ns_get_cb,
			    void *private_data);

/**
 * @brief Checks if a given namespace matches specific device and inode numbers.
 * Functional Utility: Determines if a namespace object corresponds to a given
 * `dev_t` and `ino_t`, useful for identifying namespace objects by their filesystem
 * metadata.
 *
 * @param ns Pointer to the `ns_common` structure.
 * @param dev The device number.
 * @param ino The inode number.
 * @return `true` if the namespace matches, `false` otherwise.
 */
extern bool ns_match(const struct ns_common *ns, dev_t dev, ino_t ino);

/**
 * @brief Gets the name of a namespace for a given task.
 * Functional Utility: Fills a buffer with the string name of a namespace
 * associated with a task, suitable for display or logging.
 *
 * @param buf Buffer to store the name.
 * @param size Size of the buffer.
 * @param task Pointer to the `task_struct`.
 * @param ns_ops Pointer to `proc_ns_operations` for the namespace type.
 * @return The length of the name string, or a negative errno on failure.
 */
extern int ns_get_name(char *buf, size_t size, struct task_struct *task,
			const struct proc_ns_operations *ns_ops);
/**
 * @brief Initializes the `nsfs` filesystem.
 * Functional Utility: Performs setup for the `nsfs` (namespace filesystem),
 * which is the underlying filesystem type used to expose kernel namespaces
 * through `/proc/pid/ns/` entries.
 */
extern void nsfs_init(void);

#endif /* _LINUX_PROC_NS_H */