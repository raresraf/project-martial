/**
 * @file root.c
 * @brief Linux kernel /proc filesystem root directory handling functions.
 *
 * This file implements the core logic for managing the root directory of the
 * /proc filesystem in the Linux kernel. It is responsible for initializing
 * the /proc filesystem, parsing and applying mount options (like `hidepid` and `gid`),
 * and defining the operations for the /proc root inode and directory.
 *
 * Functional Utility: Provides the foundational structure and behavior for the
 * /proc filesystem, which is a virtual filesystem presenting information about
 * processes and other system state. It allows administrators to control access
 * to process information and customize the visibility of PIDs.
 *
 * Algorithm:
 * - Mount Option Parsing: Uses `fs_context` and `fs_parameter_spec` to parse
 *   mount options provided by userspace (e.g., `gid`, `hidepid`, `subset`).
 * - Superblock Filling: `proc_fill_super` initializes the superblock, sets up
 *   inode operations, and creates the root dentry for the /proc filesystem.
 * - Dynamic Directory Listing: The `proc_root_readdir` function dynamically
 *   lists directories for PIDs (e.g., `/proc/1234`) in addition to static
 *   /proc entries.
 * - Access Control: The `hidepid` option controls the visibility of PIDs and
 *   access to their information, enforcing security policies.
 *
 * Kernel Architecture Details:
 * - VFS Layer: Integrates with the Virtual File System (VFS) layer, providing
 *   `file_system_type`, `fs_context_operations`, and `inode_operations` implementations.
 * - PID Namespaces: Integrates with PID namespaces, ensuring that a /proc
 *   mount reflects the processes visible within its specific PID namespace.
 * - User Namespaces: Considers user namespaces for `gid` mapping and capabilities.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 *  linux/fs/proc/root.c
 *
 *  Copyright (C) 1991, 1992 Linus Torvalds
 *
 *  proc root directory handling functions
 */
#include <linux/errno.h> ///< Standard error codes.
#include <linux/time.h> ///< Time-related definitions.
#include <linux/proc_fs.h> ///< Proc filesystem specific definitions.
#include <linux/stat.h> ///< File status and mode definitions.
#include <linux/init.h>	/* init_rootfs */ ///< Kernel initialization macros.
#include <linux/sched.h> ///< Scheduler definitions, task_struct.
#include <linux/sched/stat.h> ///< Task status definitions.
#include <linux/module.h> ///< Kernel module definitions.
#include <linux/bitops.h> ///< Bit manipulation functions.
#include <linux/user_namespace.h> ///< User namespace definitions.
#include <linux/fs_context.h> ///< Filesystem context API for mount options.
#include <linux/mount.h> ///< Mount definitions.
#include <linux/pid_namespace.h> ///< PID namespace definitions.
#include <linux/fs_parser.h> ///< Filesystem option parsing utilities.
#include <linux/cred.h> ///< Credentials management (UID, GID).
#include <linux/magic.h> ///< Magic numbers for filesystems.
#include <linux/slab.h> ///< Slab allocator for kernel memory.

#include "internal.h" ///< Internal proc filesystem definitions.

/**
 * @struct proc_fs_context
 * @brief Context data structure for parsing /proc filesystem mount options.
 * Functional Utility: Stores parsed parameters from `mount -o` for the /proc
 * filesystem, such as PID namespace, access mask, `hidepid` setting, GID,
 * and `pidonly` mode.
 */
struct proc_fs_context {
	struct pid_namespace	*pid_ns; ///< PID namespace associated with this /proc mount.
	unsigned int		mask; ///< Bitmask of options set by the user.
	enum proc_hidepid	hidepid; ///< Setting for hiding PIDs.
	int			gid; ///< Group ID for `hidepid` policy.
	enum proc_pidonly	pidonly; ///< Flag for PID-only mode.
};

/**
 * @enum proc_param
 * @brief Enumeration for /proc filesystem mount option parameters.
 * Functional Utility: Provides symbolic names for the different mount options
 * that can be specified for the /proc filesystem, used in parsing.
 */
enum proc_param {
	Opt_gid, ///< Option for specifying the group ID.
	Opt_hidepid, ///< Option for specifying PID visibility.
	Opt_subset, ///< Option for specifying a subset of /proc entries (e.g., pid only).
};

/**
 * @brief Array of filesystem parameter specifications for the /proc filesystem.
 * Functional Utility: Defines the expected format and type of mount options
 * for /proc, enabling the generic `fs_parser` to interpret userspace inputs.
 */
static const struct fs_parameter_spec proc_fs_parameters[] = {
	fsparam_u32("gid",	Opt_gid), ///< Parameter for `gid` (unsigned 32-bit integer).
	fsparam_string("hidepid",	Opt_hidepid), ///< Parameter for `hidepid` (string).
	fsparam_string("subset",	Opt_subset), ///< Parameter for `subset` (string).
	{} // Sentinel value to mark the end of the array.
};

/**
 * @brief Checks if a given `hidepid` value is valid.
 * Functional Utility: Ensures that the numerical value provided for the `hidepid`
 * mount option corresponds to one of the defined valid states.
 *
 * @param value The `hidepid` value to check.
 * @return `true` if the value is valid, `false` otherwise.
 */
static inline int valid_hidepid(unsigned int value)
{
	return (value == HIDEPID_OFF ||
		value == HIDEPID_NO_ACCESS ||
		value == HIDEPID_INVISIBLE ||
		value == HIDEPID_NOT_PTRACEABLE);
}

/**
 * @brief Parses the `hidepid` mount option for the /proc filesystem.
 * Functional Utility: Interprets the `hidepid` value, which can be either a
 * numerical string or a symbolic string (e.g., "off", "noaccess"), and
 * converts it into the internal `enum proc_hidepid` representation.
 *
 * @param fc Filesystem context.
 * @param param Filesystem parameter containing the `hidepid` value.
 * @return 0 on success, or a negative errno on failure (`-EINVAL`) for invalid values.
 */
static int proc_parse_hidepid_param(struct fs_context *fc, struct fs_parameter *param)
{
	struct proc_fs_context *ctx = fc->fs_private; ///< Get the /proc filesystem context.
	struct fs_parameter_spec hidepid_u32_spec = fsparam_u32("hidepid", Opt_hidepid); ///< Temporary spec for numerical parsing.
	struct fs_parse_result result;
	int base = (unsigned long)hidepid_u32_spec.data;

	if (param->type != fs_value_is_string)
		return invalf(fc, "proc: unexpected type of hidepid value\n"); ///< Must be a string.

	// Block Logic: Attempt to parse as a numerical value.
	if (!kstrtouint(param->string, base, &result.uint_32)) {
		if (!valid_hidepid(result.uint_32))
			return invalf(fc, "proc: unknown value of hidepid - %s\n", param->string); ///< Check validity of numerical value.
		ctx->hidepid = result.uint_32; ///< Assign numerical hidepid.
		return 0;
	}

	// Block Logic: Parse as a symbolic string.
	if (!strcmp(param->string, "off"))
		ctx->hidepid = HIDEPID_OFF;
	else if (!strcmp(param->string, "noaccess"))
		ctx->hidepid = HIDEPID_NO_ACCESS;
	else if (!strcmp(param->string, "invisible"))
		ctx->hidepid = HIDEPID_INVISIBLE;
	else if (!strcmp(param->string, "ptraceable"))
		ctx->hidepid = HIDEPID_NOT_PTRACEABLE;
	else
		return invalf(fc, "proc: unknown value of hidepid - %s\n", param->string); ///< Invalid symbolic value.

	return 0;
}

/**
 * @brief Parses the `subset` mount option for the /proc filesystem.
 * Functional Utility: Interprets the `subset` value to restrict the contents
 * of the /proc filesystem to a specific subset, currently supporting "pid" only.
 * This is designed to reduce the attack surface or simplify the view of /proc.
 *
 * @param fc Filesystem context.
 * @param value The string value of the `subset` option (e.g., "pid").
 * @return 0 on success, or a negative errno on failure (`-EINVAL`) for unsupported options.
 */
static int proc_parse_subset_param(struct fs_context *fc, char *value)
{
	struct proc_fs_context *ctx = fc->fs_private; ///< Get the /proc filesystem context.

	while (value) { ///< Block Logic: Iterate through comma-separated sub-options.
		char *ptr = strchr(value, ','); // Find next comma.

		if (ptr != NULL)
			*ptr++ = '\0'; // Null-terminate current sub-option and advance pointer.

		if (*value != '\0') {
			if (!strcmp(value, "pid")) {
				ctx->pidonly = PROC_PIDONLY_ON; ///< Set PID-only mode.
			} else {
				return invalf(fc, "proc: unsupported subset option - %s\n", value); ///< Unsupported subset option.
			}
		}
		value = ptr; // Move to the next sub-option.
	}

	return 0;
}

/**
 * @brief Generic parameter parser for /proc filesystem mount options.
 * Functional Utility: Acts as a dispatcher for parsing various mount options
 * (`gid`, `hidepid`, `subset`) using `fs_parse` and then calling specific
 * parsing functions based on the option type.
 *
 * @param fc Filesystem context.
 * @param param Filesystem parameter to parse.
 * @return 0 on success, or a negative errno on failure (`-EINVAL`).
 */
static int proc_parse_param(struct fs_context *fc, struct fs_parameter *param)
{
	struct proc_fs_context *ctx = fc->fs_private; ///< Get the /proc filesystem context.
	struct fs_parse_result result;
	int opt;

	opt = fs_parse(fc, proc_fs_parameters, param, &result); ///< Parse parameter using generic parser.
	if (opt < 0)
		return opt; ///< Return error if generic parsing failed.

	switch (opt) { ///< Block Logic: Handle specific options based on parsed type.
	case Opt_gid:
		ctx->gid = result.uint_32; ///< Assign parsed GID.
		break;

	case Opt_hidepid:
		if (proc_parse_hidepid_param(fc, param)) ///< Call specific parser for `hidepid`.
			return -EINVAL;
		break;

	case Opt_subset:
		if (proc_parse_subset_param(fc, param->string) < 0) ///< Call specific parser for `subset`.
			return -EINVAL;
		break;

	default:
		return -EINVAL; ///< Unknown option.
	}

	ctx->mask |= 1 << opt; ///< Set bit in mask to indicate option was processed.
	return 0;
}

/**
 * @brief Applies parsed mount options to the /proc filesystem information structure.
 * Functional Utility: Transfers the `gid`, `hidepid`, and `pidonly` settings from
 * the `fs_context` (after parsing) into the `proc_fs_info` structure associated
 * with the superblock, making them active for the mount.
 *
 * @param fs_info Pointer to the `proc_fs_info` structure of the superblock.
 * @param fc Filesystem context containing parsed options.
 * @param user_ns The user namespace in which the mount is performed.
 */
static void proc_apply_options(struct proc_fs_info *fs_info,
			       struct fs_context *fc,
			       struct user_namespace *user_ns)
{
	struct proc_fs_context *ctx = fc->fs_private; ///< Get the /proc filesystem context.

	if (ctx->mask & (1 << Opt_gid))
		fs_info->pid_gid = make_kgid(user_ns, ctx->gid); ///< Apply GID option.
	if (ctx->mask & (1 << Opt_hidepid))
		fs_info->hide_pid = ctx->hidepid; ///< Apply hidepid option.
	if (ctx->mask & (1 << Opt_subset))
		fs_info->pidonly = ctx->pidonly; ///< Apply subset option.
}

/**
 * @brief Fills the superblock for a /proc filesystem instance.
 * Functional Utility: Initializes the superblock with appropriate flags, operations,
 * and internal data (`proc_fs_info`) for a new /proc mount. It sets up the root
 * inode and dentry, and applies the parsed mount options.
 *
 * @param s Pointer to the `super_block` to fill.
 * @param fc Filesystem context containing mount options.
 * @return 0 on success, or a negative errno on failure.
 */
static int proc_fill_super(struct super_block *s, struct fs_context *fc)
{
	struct proc_fs_context *ctx = fc->fs_private; ///< Get the /proc filesystem context.
	struct inode *root_inode;
	struct proc_fs_info *fs_info;
	int ret;

	fs_info = kzalloc(sizeof(*fs_info), GFP_KERNEL); ///< Allocate memory for proc-specific superblock info.
	if (!fs_info)
		return -ENOMEM;

	fs_info->pid_ns = get_pid_ns(ctx->pid_ns); ///< Get a reference to the PID namespace.
	proc_apply_options(fs_info, fc, current_user_ns()); ///< Apply mount options.

	// Functional Utility: Set superblock flags and info.
	// User space would break if executables or devices appear on proc.
	s->s_iflags |= SB_I_USERNS_VISIBLE | SB_I_NOEXEC | SB_I_NODEV;
	s->s_flags |= SB_NODIRATIME | SB_NOSUID | SB_NOEXEC;
	s->s_blocksize = 1024;
	s->s_blocksize_bits = 10;
	s->s_magic = PROC_SUPER_MAGIC;
	s->s_op = &proc_sops; // Superblock operations.
	s->s_time_gran = 1;
	s->s_fs_info = fs_info; // Store proc-specific info.

	/*
	 * procfs isn't actually a stacking filesystem; however, there is
	 * too much magic going on inside it to permit stacking things on
	 * top of it
	 */
	s->s_stack_depth = FILESYSTEM_MAX_STACK_DEPTH; // Prevent stacking on /proc.

	/* procfs dentries and inodes don't require IO to create */
	s->s_shrink->seeks = 0; // Optimize shrinker behavior.

	pde_get(&proc_root); // Get a reference to the global proc_root entry.
	root_inode = proc_get_inode(s, &proc_root); ///< Get the inode for the /proc root.
	if (!root_inode) {
		pr_err("proc_fill_super: get root inode failed\n");
		return -ENOMEM;
	}

	s->s_root = d_make_root(root_inode); ///< Create the root dentry.
	if (!s->s_root) {
		pr_err("proc_fill_super: allocate dentry failed\n");
		return -ENOMEM;
	}

	ret = proc_setup_self(s); ///< Set up /proc/self.
	if (ret) {
		return ret;
	}
	return proc_setup_thread_self(s); ///< Set up /proc/thread-self.
}

/**
 * @brief Reconfigures an existing /proc filesystem mount.
 * Functional Utility: Applies new mount options to an already mounted /proc
 * filesystem. This involves syncing the filesystem and then reapplying `gid`,
 * `hidepid`, and `pidonly` settings.
 *
 * @param fc Filesystem context containing new mount options.
 * @return 0 on success.
 */
static int proc_reconfigure(struct fs_context *fc)
{
	struct super_block *sb = fc->root->d_sb; ///< Get the superblock.
	struct proc_fs_info *fs_info = proc_sb_info(sb); ///< Get proc-specific superblock info.

	sync_filesystem(sb); ///< Synchronize filesystem before applying changes.

	proc_apply_options(fs_info, fc, current_user_ns()); ///< Apply new mount options.
	return 0;
}

/**
 * @brief Gets the filesystem tree for /proc.
 * Functional Utility: Serves as the callback for the `get_tree` operation in
 * `fs_context_operations`. It uses `get_tree_nodev` to create a new filesystem
 * instance without backing device, calling `proc_fill_super` to populate it.
 *
 * @param fc Filesystem context.
 * @return 0 on success, or a negative errno on failure.
 */
static int proc_get_tree(struct fs_context *fc)
{
	return get_tree_nodev(fc, proc_fill_super); ///< Create a new tree with `proc_fill_super`.
}

/**
 * @brief Frees the /proc filesystem context.
 * Functional Utility: Releases resources associated with the `proc_fs_context`,
 * specifically the reference to the PID namespace and the context itself.
 *
 * @param fc Filesystem context to free.
 */
static void proc_fs_context_free(struct fs_context *fc)
{
	struct proc_fs_context *ctx = fc->fs_private; ///< Get the /proc filesystem context.

	put_pid_ns(ctx->pid_ns); ///< Release reference to PID namespace.
	kfree(ctx); ///< Free the context memory.
}

/**
 * @brief Defines the filesystem context operations for the /proc filesystem.
 * Functional Utility: Provides the set of callbacks that the VFS uses to
 * manage the lifecycle of a /proc filesystem instance, including parsing
 * parameters, getting the tree, and reconfiguring.
 */
static const struct fs_context_operations proc_fs_context_ops = {
	.free		= proc_fs_context_free, ///< Callback to free the context.
	.parse_param	= proc_parse_param, ///< Callback to parse mount parameters.
	.get_tree	= proc_get_tree, ///< Callback to get the filesystem tree.
	.reconfigure	= proc_reconfigure, ///< Callback to reconfigure an existing mount.
};

/**
 * @brief Initializes the filesystem context for a new /proc mount.
 * Functional Utility: Allocates and initializes a `proc_fs_context` structure,
 * setting up the initial PID namespace and assigning the context operations.
 * This is the first step in preparing a /proc filesystem mount.
 *
 * @param fc Filesystem context to initialize.
 * @return 0 on success, or `-ENOMEM` on memory allocation failure.
 */
static int proc_init_fs_context(struct fs_context *fc)
{
	struct proc_fs_context *ctx;

	ctx = kzalloc(sizeof(struct proc_fs_context), GFP_KERNEL); ///< Allocate memory for the context.
	if (!ctx)
		return -ENOMEM;

	ctx->pid_ns = get_pid_ns(task_active_pid_ns(current)); ///< Get PID namespace of current task.
	put_user_ns(fc->user_ns); ///< Release reference to the initial user namespace.
	fc->user_ns = get_user_ns(ctx->pid_ns->user_ns); ///< Set user namespace to that of the PID namespace.
	fc->fs_private = ctx; ///< Store context in `fs_context`.
	fc->ops = &proc_fs_context_ops; ///< Assign context operations.
	return 0;
}

/**
 * @brief Kills a /proc superblock.
 * Functional Utility: Cleans up resources associated with a /proc superblock,
 * including freeing `proc_fs_info` and putting references to PID namespaces
 * and `/proc/self` entries.
 *
 * @param sb Pointer to the `super_block` to kill.
 */
static void proc_kill_sb(struct super_block *sb)
{
	struct proc_fs_info *fs_info = proc_sb_info(sb); ///< Get proc-specific info.

	if (!fs_info) {
		kill_anon_super(sb); ///< Fallback to generic kill if no specific info.
		return;
	}

	dput(fs_info->proc_self); ///< Release dentry for /proc/self.
	dput(fs_info->proc_thread_self); ///< Release dentry for /proc/thread-self.

	kill_anon_super(sb); ///< Call generic anonymous superblock killer.
	put_pid_ns(fs_info->pid_ns); ///< Release reference to PID namespace.
	kfree_rcu(fs_info, rcu); ///< Free proc-specific info after RCU grace period.
}

/**
 * @brief Defines the `file_system_type` for the /proc filesystem.
 * Functional Utility: Provides the VFS with the necessary information to
 * recognize and manage /proc filesystem mounts, including its name,
 * initialization function, parameter specifications, and superblock killer.
 */
static struct file_system_type proc_fs_type = {
	.name			= "proc", ///< Filesystem name.
	.init_fs_context	= proc_init_fs_context, ///< Context initialization function.
	.parameters		= proc_fs_parameters, ///< Mount option parameters.
	.kill_sb		= proc_kill_sb, ///< Superblock kill function.
	.fs_flags		= FS_USERNS_MOUNT | FS_DISALLOW_NOTIFY_PERM, ///< Filesystem flags.
};

/**
 * @brief Initializes the /proc filesystem root.
 * Functional Utility: Sets up the initial structure of the /proc filesystem
 * during kernel boot. This includes initializing internal caches, creating
 * static directories (e.g., `fs`, `driver`, `bus`), setting up symbolic links
 * (`mounts -> self/mounts`), and registering the /proc filesystem type with the VFS.
 */
void __init proc_root_init(void)
{
	proc_init_kmemcache(); ///< Initialize kernel memory cache for /proc objects.
	set_proc_pid_nlink(); ///< Set initial nlink for PID directories.
	proc_self_init(); ///< Initialize /proc/self.
	proc_thread_self_init(); ///< Initialize /proc/thread-self.
	proc_symlink("mounts", NULL, "self/mounts"); ///< Create symlink for mounts.

	proc_net_init(); ///< Initialize /proc/net.
	proc_mkdir("fs", NULL); ///< Create /proc/fs directory.
	proc_mkdir("driver", NULL); ///< Create /proc/driver directory.
	proc_create_mount_point("fs/nfsd"); /* somewhere for the nfsd filesystem to be mounted */ ///< Create mount point for nfsd.
#if defined(CONFIG_SUN_OPENPROMFS) || defined(CONFIG_SUN_OPENPROMFS_MODULE)
	/* just give it a mountpoint */
	proc_create_mount_point("openprom"); ///< Create mount point for openprom.
#endif
	proc_tty_init(); ///< Initialize /proc/tty.
	proc_mkdir("bus", NULL); ///< Create /proc/bus directory.
	proc_sys_init(); ///< Initialize /proc/sys.

	/*
	 * Last things last. It is not like userspace processes eager
	 * to open /proc files exist at this point but register last
	 * anyway.
	 */
	register_filesystem(&proc_fs_type); ///< Register the /proc filesystem type with the VFS.
}

/**
 * @brief Gets attributes for the /proc root directory.
 * Functional Utility: Populates a `kstat` structure with attributes (mode, nlink, etc.)
 * for the /proc root directory. It dynamically calculates the `nlink` to include
 * directories for all active processes.
 *
 * @param idmap The mount ID mapping (ignored for /proc root).
 * @param path The path to the /proc root directory.
 * @param stat Pointer to the `kstat` structure to fill.
 * @param request_mask Mask of requested attributes.
 * @param query_flags Query flags.
 * @return 0 on success.
 */
static int proc_root_getattr(struct mnt_idmap *idmap,
			     const struct path *path, struct kstat *stat,
			     u32 request_mask, unsigned int query_flags)
{
	generic_fillattr(&nop_mnt_idmap, request_mask, d_inode(path->dentry),
			 stat); ///< Fill generic attributes.
	// Functional Utility: Dynamically calculate nlink to include entries for all processes.
	stat->nlink = proc_root.nlink + nr_processes();
	return 0;
}

/**
 * @brief Looks up an entry within the /proc root directory.
 * Functional Utility: This is a specialized lookup function for the /proc root.
 * It first attempts to look up a PID directory. If that fails, it falls back
 * to the generic /proc lookup for static entries.
 *
 * @param dir The inode of the /proc root directory.
 * @param dentry The dentry to lookup.
 * @param flags Lookup flags.
 * @return Pointer to the dentry if found, or `NULL`.
 */
static struct dentry *proc_root_lookup(struct inode * dir, struct dentry * dentry, unsigned int flags)
{
	if (!proc_pid_lookup(dentry, flags)) ///< Block Logic: Attempt to lookup as a PID directory.
		return NULL;

	return proc_lookup(dir, dentry, flags); ///< Fallback to generic /proc lookup.
}

/**
 * @brief Reads directory entries from the /proc root directory.
 * Functional Utility: Provides a combined directory listing for the /proc root,
 * first enumerating static entries (e.g., `fs`, `sys`) and then dynamically
 * generating entries for all active processes (PIDs).
 *
 * @param file Pointer to the `file` structure for the directory.
 * @param ctx Pointer to the `dir_context` for directory iteration.
 * @return 0 on success, or a negative errno on failure.
 */
static int proc_root_readdir(struct file *file, struct dir_context *ctx)
{
	if (ctx->pos < FIRST_PROCESS_ENTRY) { ///< Block Logic: First, read static entries.
		int error = proc_readdir(file, ctx); ///< Read generic /proc entries.
		if (unlikely(error <= 0))
			return error; ///< Return if error or end of static entries.
		ctx->pos = FIRST_PROCESS_ENTRY; ///< Set position to start of process entries.
	}

	return proc_pid_readdir(file, ctx); ///< Block Logic: Then, read PID directories.
}

/**
 * @brief File operations for the /proc root directory.
 * Functional Utility: Defines the set of operations that can be performed on
 * the /proc root directory, including reading and iterating its contents.
 * This structure customizes the VFS behavior for /proc.
 */
static const struct file_operations proc_root_operations = {
	.read		 = generic_read_dir, ///< Generic read function for directories.
	.iterate_shared	 = proc_root_readdir, ///< Custom iterator to handle both static and PID entries.
	.llseek		= generic_file_llseek, ///< Generic lseek for files.
};

/**
 * @brief Inode operations for the /proc root directory.
 * Functional Utility: Defines the set of operations that can be performed on
 * the inode representing the /proc root directory, including lookup of entries
 * and getting attributes.
 */
static const struct inode_operations proc_root_inode_operations = {
	.lookup		= proc_root_lookup, ///< Custom lookup function.
	.getattr	= proc_root_getattr, ///< Custom getattr function.
};

/**
 * @brief The root `proc_dir_entry` for the /proc filesystem.
 * Functional Utility: This global structure represents the top-level
 * directory of the /proc filesystem. It holds metadata like inode number,
 * name, mode, link count, and pointers to its file and inode operations.
 * It serves as the entry point for accessing all /proc information.
 */
struct proc_dir_entry proc_root = {
	.low_ino	= PROCFS_ROOT_INO, ///< Inode number for the /proc root.
	.namelen	= 5, ///< Length of the name "/proc".
	.mode		= S_IFDIR | S_IRUGO | S_IXUGO, ///< Directory mode: directory, readable/executable by all.
	.nlink		= 2, ///< Initial link count (self and parent).
	.refcnt		= REFCOUNT_INIT(1), ///< Initial reference count.
	.proc_iops	= &proc_root_inode_operations, ///< Inode operations for /proc root.
	.proc_dir_ops	= &proc_root_operations, ///< File operations for /proc root.
	.parent		= &proc_root, ///< Self-reference as its own parent for some internal logic.
	.subdir		= RB_ROOT, ///< Red-black tree for static subdirectories.
	.name		= "/proc", ///< Name of the root directory.
};