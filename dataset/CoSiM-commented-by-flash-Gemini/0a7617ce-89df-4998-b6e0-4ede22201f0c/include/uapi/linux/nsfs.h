/**
 * @file nsfs.h
 * @brief User-space API (UAPI) definitions for the `nsfs` filesystem.
 *
 * This header file provides definitions for `ioctl` commands and data structures
 * that allow userspace applications to query and interact with kernel namespaces
 * exposed through the `nsfs` (namespace filesystem) typically mounted at
 * `/proc/pid/ns/`. This includes functionality to retrieve information about
 * namespace ownership, hierarchy, and type, as well as translate PID IDs
 * across different PID namespaces.
 *
 * Functional Utility: Crucial for tools that manage and inspect containerized
 * environments, allowing them to understand the isolation boundaries and
 * relationships between different namespaces.
 *
 * Architecture:
 * - `ioctl` commands: A set of `_IO` and `_IOR` commands under the `NSIO` magic
 *   number, enabling queries for user, parent, and specific namespace types
 *   (PID, MNT).
 * - PID Translation: Specific ioctls are provided for translating PIDs between
 *   the caller's PID namespace and a target PID namespace, a key feature for
 *   cross-namespace communication and monitoring.
 * - Mount Namespace Info: Structures and ioctls to retrieve metadata about
 *   mount namespaces, including their ID and number of mounts.
 * - Initial Inode Numbers: Defines well-known inode numbers for the initial
 *   instances of various namespace types.
 */
/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef __LINUX_NSFS_H ///< Include guard to prevent multiple inclusions of this header file.
#define __LINUX_NSFS_H

#include <linux/ioctl.h> ///< Defines `_IO`, `_IOR` macros for `ioctl` commands.
#include <linux/types.h> ///< Defines standard kernel types like `__u32`, `__u64`.

#define NSIO	0xb7 ///< Magic number for namespace `ioctl` commands.

/* Returns a file descriptor that refers to an owning user namespace */
#define NS_GET_USERNS		_IO(NSIO, 0x1) ///< IOCTL: Get a file descriptor to the owning user namespace.
/* Returns a file descriptor that refers to a parent namespace */
#define NS_GET_PARENT		_IO(NSIO, 0x2) ///< IOCTL: Get a file descriptor to the parent namespace.
/* Returns the type of namespace (CLONE_NEW* value) referred to by
   file descriptor */
#define NS_GET_NSTYPE		_IO(NSIO, 0x3) ///< IOCTL: Get the type of the namespace (e.g., `CLONE_NEWPID`).
/* Get owner UID (in the caller's user namespace) for a user namespace */
#define NS_GET_OWNER_UID	_IO(NSIO, 0x4) ///< IOCTL: Get the owner UID of a user namespace in the caller's user namespace.
/* Get the id for a mount namespace */
#define NS_GET_MNTNS_ID		_IOR(NSIO, 0x5, __u64) ///< IOCTL: Get the unique ID for a mount namespace.
/* Translate pid from target pid namespace into the caller's pid namespace. */
#define NS_GET_PID_FROM_PIDNS	_IOR(NSIO, 0x6, int) ///< IOCTL: Translate a PID from a target PID namespace to the caller's PID namespace.
/* Return thread-group leader id of pid in the callers pid namespace. */
#define NS_GET_TGID_FROM_PIDNS	_IOR(NSIO, 0x7, int) ///< IOCTL: Get the thread-group leader's PID from a target PID namespace to the caller's.
/* Translate pid from caller's pid namespace into a target pid namespace. */
#define NS_GET_PID_IN_PIDNS	_IOR(NSIO, 0x8, int) ///< IOCTL: Translate a PID from the caller's PID namespace to a target PID namespace.
/* Return thread-group leader id of pid in the target pid namespace. */
#define NS_GET_TGID_IN_PIDNS	_IOR(NSIO, 0x9, int) ///< IOCTL: Get the thread-group leader's PID from the caller's PID namespace to a target PID namespace.

/**
 * @struct mnt_ns_info
 * @brief Information structure for a mount namespace.
 * Functional Utility: Used with `NS_MNT_GET_INFO`, `NS_MNT_GET_NEXT`, `NS_MNT_GET_PREV`
 * ioctls to retrieve metadata about a mount namespace.
 */
struct mnt_ns_info {
	__u32 size; ///< Size of this structure, for ABI compatibility.
	__u32 nr_mounts; ///< Number of mounts in this mount namespace.
	__u64 mnt_ns_id; ///< Unique ID of this mount namespace.
};

#define MNT_NS_INFO_SIZE_VER0 16 /* size of first published struct */ ///< Size of the initial version of `mnt_ns_info`.

/* Get information about namespace. */
#define NS_MNT_GET_INFO		_IOR(NSIO, 10, struct mnt_ns_info) ///< IOCTL: Get information about a mount namespace.
/* Get next namespace. */
#define NS_MNT_GET_NEXT		_IOR(NSIO, 11, struct mnt_ns_info) ///< IOCTL: Get information about the next mount namespace (by ID).
/* Get previous namespace. */
#define NS_MNT_GET_PREV		_IOR(NSIO, 12, struct mnt_ns_info) ///< IOCTL: Get information about the previous mount namespace (by ID).

/**
 * @enum init_ns_ino
 * @brief Defines well-known inode numbers for initial instances of various kernel namespaces.
 * Functional Utility: These constants provide stable and recognizable inode numbers
 * for the initial (root) instances of each kernel namespace type, which are commonly
 * exposed in the /proc filesystem (e.g., `/proc/self/ns/pid`).
 */
enum init_ns_ino {
	IPC_NS_INIT_INO		= 0xEFFFFFFFU, ///< Initial IPC namespace inode number.
	UTS_NS_INIT_INO		= 0xEFFFFFFEU, ///< Initial UTS namespace inode number.
	USER_NS_INIT_INO	= 0xEFFFFFFDU, ///< Initial user namespace inode number.
	PID_NS_INIT_INO		= 0xEFFFFFFCU, ///< Initial PID namespace inode number.
	CGROUP_NS_INIT_INO	= 0xEFFFFFFBU, ///< Initial cgroup namespace inode number.
	TIME_NS_INIT_INO	= 0xEFFFFFFAU, ///< Initial time namespace inode number.
	NET_NS_INIT_INO		= 0xEFFFFFF9U, ///< Initial network namespace inode number.
	MNT_NS_INIT_INO		= 0xEFFFFFF8U, ///< Initial mount namespace inode number.
};

#endif /* __LINUX_NSFS_H */