/**
 * @file fs.h
 * @brief User-space API (UAPI) definitions for Linux filesystem structures and ioctls.
 *
 * This header file provides fundamental definitions for interacting with the
 * Linux kernel's Virtual File System (VFS) from userspace. It includes constants,
 * structures, and ioctl commands related to file operations, mount attributes,
 * filesystem properties, and process memory mapping.
 *
 * Functional Utility: Serves as the crucial interface between userspace applications
 * and the kernel's filesystem layer, enabling programs to perform advanced file
 * manipulation, query filesystem state, and manage process memory regions.
 *
 * Architecture:
 * - `ioctl` commands: A large collection of `_IO`, `_IOR`, `_IOW`, `_IOWR` macros
 *   define various operations on block devices and filesystems, such as setting
 *   read-only status, re-reading partition tables, trimming, cloning, and querying
 *   extended attributes.
 * - File and Inode Flags: Bitmasks (`FS_XFLAG_`, `FS_FL_`) define specific behaviors
 *   or properties of files and inodes (e.g., immutability, append-only, noatime).
 * - Process Memory Mapping: Structures (`pm_scan_arg`, `procmap_query`) and
 *   associated `ioctl` commands (`PAGEMAP_SCAN`, `PROCMAP_QUERY`) allow userspace
 *   to query details about a process's virtual memory areas.
 * - Cross-cutting Concerns: Includes definitions for `SEEK_` operations, `RENAME_`
 *   flags, `RWF_` flags for `preadv2`/`pwritev2`, and integrity metadata flags.
 *
 * This file is part of the UAPI (User API), meaning its definitions are stable
 * and intended for direct use by userspace programs.
 */
/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef _UAPI_LINUX_FS_H ///< Include guard to prevent multiple inclusions of this header file.
#define _UAPI_LINUX_FS_H

/*
 * This file has definitions for some important file table structures
 * and constants and structures used by various generic file system
 * ioctl's.  Please do not make any changes in this file before
 * sending patches for review to linux-fsdevel@vger.kernel.org and
 * linux-api@vger.kernel.org.
 */

#include <linux/limits.h> ///< Defines limits for various system parameters, e.g., `PATH_MAX`.
#include <linux/ioctl.h> ///< Defines `_IO`, `_IOR`, `_IOW`, `_IOWR` macros for `ioctl` commands.
#include <linux/types.h> ///< Defines standard kernel types like `__u64`, `__s64`, `__u32`, `__s32`.
#ifndef __KERNEL__ ///< Conditional compilation: This block is included only when compiled for userspace.
#include <linux/fscrypt.h> ///< Filesystem encryption definitions (userspace specific part).
#endif

/* Use of MS_* flags within the kernel is restricted to core mount(2) code. */
#if !defined(__KERNEL__) ///< Conditional compilation: This block is included only when compiled for userspace.
#include <linux/mount.h> ///< Mount-related definitions for userspace (e.g., MS_* flags).
#endif

/*
 * It's silly to have NR_OPEN bigger than NR_FILE, but you can change
 * the file limit at runtime and only root can increase the per-process
 * nr_file rlimit, so it's safe to set up a ridiculously high absolute
 * upper limit on files-per-process.
 *
 * Some programs (notably those using select()) may have to be
 * recompiled to take full advantage of the new limits..
 */

/* Fixed constants first: */
#undef NR_OPEN ///< Undefine `NR_OPEN` to ensure the new definition is used.
#define INR_OPEN_CUR 1024	/* Initial setting for nfile rlimits */ ///< Initial soft limit for open files per process.
#define INR_OPEN_MAX 4096	/* Hard limit for nfile rlimits */ ///< Hard limit for open files per process.

#define BLOCK_SIZE_BITS 10 ///< Number of bits for the block size (2^10 = 1024).
#define BLOCK_SIZE (1<<BLOCK_SIZE_BITS) ///< Standard block size (1024 bytes).

/* flags for integrity meta */
#define IO_INTEGRITY_CHK_GUARD		(1U << 0) /* enforce guard check */ ///< Flag to enforce integrity guard checks.
#define IO_INTEGRITY_CHK_REFTAG		(1U << 1) /* enforce ref check */ ///< Flag to enforce integrity reference tag checks.
#define IO_INTEGRITY_CHK_APPTAG		(1U << 2) /* enforce app check */ ///< Flag to enforce integrity application tag checks.

#define IO_INTEGRITY_VALID_FLAGS (IO_INTEGRITY_CHK_GUARD | \
				  IO_INTEGRITY_CHK_REFTAG | \
				  IO_INTEGRITY_CHK_APPTAG) ///< Mask for all valid integrity check flags.

#define SEEK_SET	0	/* seek relative to beginning of file */ ///< Seek from the beginning of the file.
#define SEEK_CUR	1	/* seek relative to current file position */ ///< Seek from the current file position.
#define SEEK_END	2	/* seek relative to end of file */ ///< Seek from the end of the file.
#define SEEK_DATA	3	/* seek to the next data */ ///< Seek to the next data segment (for sparse files).
#define SEEK_HOLE	4	/* seek to the next hole */ ///< Seek to the next hole (for sparse files).
#define SEEK_MAX	SEEK_HOLE ///< Maximum valid `SEEK_` command.

#define RENAME_NOREPLACE	(1 << 0)	/* Don't overwrite target */ ///< Flag for `renameat2`: do not overwrite existing target.
#define RENAME_EXCHANGE		(1 << 1)	/* Exchange source and dest */ ///< Flag for `renameat2`: atomically exchange source and destination.
#define RENAME_WHITEOUT		(1 << 2)	/* Whiteout source */ ///< Flag for `renameat2`: create a whiteout entry at the source.

/*
 * The root inode of procfs is guaranteed to always have the same inode number.
 * For programs that make heavy use of procfs, verifying that the root is a
 * real procfs root and using openat2(RESOLVE_{NO_{XDEV,MAGICLINKS},BENEATH})
 * will allow you to make sure you are never tricked into operating on the
 * wrong procfs file.
 */
/**
 * @enum procfs_ino
 * @brief Defines well-known inode numbers for the /proc filesystem.
 * Functional Utility: Provides a stable inode number for the /proc root,
 * allowing userspace programs to reliably identify it.
 */
enum procfs_ino {
	PROCFS_ROOT_INO = 1, ///< Inode number for the root of the /proc filesystem.
};

/**
 * @struct file_clone_range
 * @brief Structure for the `FICLONERANGE` ioctl, used for cloning file data.
 * Functional Utility: Specifies a range of data to be cloned from one file
 * descriptor to another within the same file or across different files on
 * supporting filesystems.
 */
struct file_clone_range {
	__s64 src_fd; ///< Source file descriptor (or -1 if same as target).
	__u64 src_offset; ///< Offset in the source file to start cloning.
	__u64 src_length; ///< Length of the data to clone.
	__u64 dest_offset; ///< Offset in the destination file to write cloned data.
};

/**
 * @struct fstrim_range
 * @brief Structure for the `FITRIM` ioctl, used for trimming (discarding) blocks.
 * Functional Utility: Specifies a range of blocks on a filesystem to be
 * discarded, typically used with SSDs or thin-provisioned storage to reclaim
 * unused blocks.
 */
struct fstrim_range {
	__u64 start; ///< Starting offset within the filesystem.
	__u64 len; ///< Length of the range to trim.
	__u64 minlen; ///< Minimum amount of contiguous freed space to report.
};

/*
 * We include a length field because some filesystems (vfat) have an identifier
 * that we do want to expose as a UUID, but doesn't have the standard length.
 *
 * We use a fixed size buffer beacuse this interface will, by fiat, never
 * support "UUIDs" longer than 16 bytes; we don't want to force all downstream
 * users to have to deal with that.
 */
/**
 * @struct fsuuid2
 * @brief Represents a filesystem UUID with a length field.
 * Functional Utility: Provides a flexible way to represent filesystem
 * identifiers, accommodating UUIDs of varying lengths up to 16 bytes.
 */
struct fsuuid2 {
	__u8	len; ///< Length of the UUID data.
	__u8	uuid[16]; ///< Buffer for the UUID data.
};

/**
 * @struct fs_sysfs_path
 * @brief Represents a path component under /sys/fs/ for a filesystem.
 * Functional Utility: Used to retrieve the path by which a filesystem
 * is exposed in the `/sys/fs` hierarchy, or `/sys/kernel/debug` for debugfs exports.
 */
struct fs_sysfs_path {
	__u8			len; ///< Length of the name.
	__u8			name[128]; ///< Buffer for the path name.
};

/* extent-same (dedupe) ioctls; these MUST match the btrfs ioctl definitions */
#define FILE_DEDUPE_RANGE_SAME		0 ///< Status code: deduplication succeeded.
#define FILE_DEDUPE_RANGE_DIFFERS	1 ///< Status code: data differs, deduplication not possible.

/* from struct btrfs_ioctl_file_extent_same_info */
/**
 * @struct file_dedupe_range_info
 * @brief Information about a deduplication attempt for a specific range.
 * Functional Utility: Provides detailed status and byte count for a deduplication
 * operation between a source and a destination file range.
 */
struct file_dedupe_range_info {
	__s64 dest_fd;		/* in - destination file */ ///< Destination file descriptor.
	__u64 dest_offset;	/* in - start of extent in destination */ ///< Start offset in destination.
	__u64 bytes_deduped;	/* out - total # of bytes we were able
				 * to dedupe from this file. */ ///< Total bytes successfully deduplicated.
	/* status of this dedupe operation:
	 * < 0 for error
	 * == FILE_DEDUPE_RANGE_SAME if dedupe succeeds
	 * == FILE_DEDUPE_RANGE_DIFFERS if data differs
	 */
	__s32 status;		/* out - see above description */ ///< Status of the dedupe operation.
	__u32 reserved;		/* must be zero */ ///< Reserved for future use.
};

/* from struct btrfs_ioctl_file_extent_same_args */
/**
 * @struct file_dedupe_range
 * @brief Arguments for the `FIDEDUPERANGE` ioctl, used for deduplicating file data.
 * Functional Utility: Specifies one or more ranges in a source file that should
 * be deduplicated against ranges in a destination file.
 */
struct file_dedupe_range {
	__u64 src_offset;	/* in - start of extent in source */ ///< Start offset in source file.
	__u64 src_length;	/* in - length of extent */ ///< Length of the extent to dedupe.
	__u16 dest_count;	/* in - total elements in info array */ ///< Number of destination ranges in `info` array.
	__u16 reserved1;	/* must be zero */ ///< Reserved.
	__u32 reserved2;	/* must be zero */ ///< Reserved.
	struct file_dedupe_range_info info[]; ///< Array of destination range information.
};

/* And dynamically-tunable limits and defaults: */
/**
 * @struct files_stat_struct
 * @brief Statistics and tunable limits for the system's open files.
 * Functional Utility: Provides information about the current usage and
 * maximum allowed number of open file descriptors system-wide.
 */
struct files_stat_struct {
	unsigned long nr_files;		/* read only */ ///< Current number of open files.
	unsigned long nr_free_files;	/* read only */ ///< Number of free file structs.
	unsigned long max_files;		/* tunable */ ///< Tunable maximum number of open files.
};

/**
 * @struct inodes_stat_t
 * @brief Statistics for the system's inodes.
 * Functional Utility: Provides information about the current usage and
 * availability of inodes system-wide.
 */
struct inodes_stat_t {
	long nr_inodes; ///< Current number of allocated inodes.
	long nr_unused; ///< Number of unused inodes.
	long dummy[5];		/* padding for sysctl ABI compatibility */ ///< Padding for ABI compatibility.
};


#define NR_FILE  8192	/* this can well be larger on a larger system */ ///< Default maximum number of open files.

/*
 * Structure for FS_IOC_FSGETXATTR[A] and FS_IOC_FSSETXATTR.
 */
/**
 * @struct fsxattr
 * @brief Extended attributes structure for `FS_IOC_FSGETXATTR` and `FS_IOC_FSSETXATTR`.
 * Functional Utility: Allows userspace to query and set various extended
 * filesystem attributes and flags on files, such as `xflags`, `extsize`,
 * `projid`, and `cowextsize`.
 */
struct fsxattr {
	__u32		fsx_xflags;	/* xflags field value (get/set) */ ///< Extended flags.
	__u32		fsx_extsize;	/* extsize field value (get/set)*/ ///< Extent size hint.
	__u32		fsx_nextents;	/* nextents field value (get)	*/ ///< Number of extents (get only).
	__u32		fsx_projid;	/* project identifier (get/set) */ ///< Project ID.
	__u32		fsx_cowextsize;	/* CoW extsize field value (get/set)*/ ///< Copy-on-Write extent size hint.
	unsigned char	fsx_pad[8]; ///< Padding for alignment and future use.
};

/*
 * Flags for the fsx_xflags field
 */
#define FS_XFLAG_REALTIME	0x00000001	/* data in realtime volume */ ///< File data resides on a realtime volume.
#define FS_XFLAG_PREALLOC	0x00000002	/* preallocated file extents */ ///< File extents are preallocated.
#define FS_XFLAG_IMMUTABLE	0x00000008	/* file cannot be modified */ ///< File cannot be modified.
#define FS_XFLAG_APPEND		0x00000010	/* all writes append */ ///< All writes to the file are append-only.
#define FS_XFLAG_SYNC		0x00000020	/* all writes synchronous */ ///< All writes to the file are synchronous.
#define FS_XFLAG_NOATIME	0x00000040	/* do not update access time */ ///< Do not update access time for this file.
#define FS_XFLAG_NODUMP		0x00000080	/* do not include in backups */ ///< Do not include this file in backups.
#define FS_XFLAG_RTINHERIT	0x00000100	/* create with rt bit set */ ///< New files inherit the realtime bit.
#define FS_XFLAG_PROJINHERIT	0x00000200	/* create with parents projid */ ///< New files inherit the parent's project ID.
#define FS_XFLAG_NOSYMLINKS	0x00000400	/* disallow symlink creation */ ///< Disallow symlink creation in this directory.
#define FS_XFLAG_EXTSIZE	0x00000800	/* extent size allocator hint */ ///< Hint for extent size allocator.
#define FS_XFLAG_EXTSZINHERIT	0x00001000	/* inherit inode extent size */ ///< Inherit parent's inode extent size.
#define FS_XFLAG_NODEFRAG	0x00002000	/* do not defragment */ ///< Do not defragment this file.
#define FS_XFLAG_FILESTREAM	0x00004000	/* use filestream allocator */ ///< Use filestream allocator.
#define FS_XFLAG_DAX		0x00008000	/* use DAX for IO */ ///< Use Direct Access (DAX) for IO.
#define FS_XFLAG_COWEXTSIZE	0x00010000	/* CoW extent size allocator hint */ ///< Copy-on-Write extent size allocator hint.
#define FS_XFLAG_HASATTR	0x80000000	/* no DIFLAG for this	*/ ///< Internal flag indicating attributes are present.

/* the read-only stuff doesn't really belong here, but any other place is
   probably as bad and I don't want to create yet another include file. */

// Block device IOCTLs.
#define BLKROSET   _IO(0x12,93)	/* set device read-only (0 = read-write) */ ///< Set block device read-only status.
#define BLKROGET   _IO(0x12,94)	/* get read-only status (0 = read_write) */ ///< Get block device read-only status.
#define BLKRRPART  _IO(0x12,95)	/* re-read partition table */ ///< Re-read partition table.
#define BLKGETSIZE _IO(0x12,96)	/* return device size /512 (long *arg) */ ///< Get device size in 512-byte sectors.
#define BLKFLSBUF  _IO(0x12,97)	/* flush buffer cache */ ///< Flush the buffer cache for the device.
#define BLKRASET   _IO(0x12,98)	/* set read ahead for block device */ ///< Set read-ahead for block device.
#define BLKRAGET   _IO(0x12,99)	/* get current read ahead setting */ ///< Get read-ahead for block device.
#define BLKFRASET  _IO(0x12,100)/* set filesystem (mm/filemap.c) read-ahead */ ///< Set filesystem read-ahead.
#define BLKFRAGET  _IO(0x12,101)/* get filesystem (mm/filemap.c) read-ahead */ ///< Get filesystem read-ahead.
#define BLKSECTSET _IO(0x12,102)/* set max sectors per request (ll_rw_blk.c) */ ///< Set max sectors per request.
#define BLKSECTGET _IO(0x12,103)/* get max sectors per request (ll_rw_blk.c) */ ///< Get max sectors per request.
#define BLKSSZGET  _IO(0x12,104)/* get block device sector size */ ///< Get block device sector size.
#if 0
#define BLKPG      _IO(0x12,105)/* See blkpg.h */

/* Some people are morons.  Do not use sizeof! */

#define BLKELVGET  _IOR(0x12,106,size_t)/* elevator get */
#define BLKELVSET  _IOW(0x12,107,size_t)/* elevator set */
/* This was here just to show that the number is taken -
   probably all these _IO(0x12,*) ioctls should be moved to blkpg.h. */
#endif
/* A jump here: 108-111 have been used for various private purposes. */
#define BLKBSZGET  _IOR(0x12,112,size_t) ///< Get block device block size.
#define BLKBSZSET  _IOW(0x12,113,size_t) ///< Set block device block size.
#define BLKGETSIZE64 _IOR(0x12,114,size_t)	/* return device size in bytes (u64 *arg) */ ///< Get device size in bytes (64-bit).
#define BLKTRACESETUP _IOWR(0x12,115,struct blk_user_trace_setup) ///< Setup block trace.
#define BLKTRACESTART _IO(0x12,116) ///< Start block trace.
#define BLKTRACESTOP _IO(0x12,117) ///< Stop block trace.
#define BLKTRACETEARDOWN _IO(0x12,118) ///< Teardown block trace.
#define BLKDISCARD _IO(0x12,119) ///< Discard (TRIM) blocks.
#define BLKIOMIN _IO(0x12,120) ///< Get minimum I/O size.
#define BLKIOOPT _IO(0x12,121) ///< Get optimal I/O size.
#define BLKALIGNOFF _IO(0x12,122) ///< Get alignment offset.
#define BLKPBSZGET _IO(0x12,123) ///< Get physical block size.
#define BLKDISCARDZEROES _IO(0x12,124) ///< Discard and zero blocks.
#define BLKSECDISCARD _IO(0x12,125) ///< Secure discard blocks.
#define BLKROTATIONAL _IO(0x12,126) ///< Check if device is rotational.
#define BLKZEROOUT _IO(0x12,127) ///< Zero out a range of blocks.
#define BLKGETDISKSEQ _IOR(0x12,128,__u64) ///< Get disk sequence number.
/* 130-136 are used by zoned block device ioctls (uapi/linux/blkzoned.h) */
/* 137-141 are used by blk-crypto ioctls (uapi/linux/blk-crypto.h) */

// Filesystem IOCTLs.
#define BMAP_IOCTL 1		/* obsolete - kept for compatibility */ ///< Obsolete BMAP ioctl.
#define FIBMAP	   _IO(0x00,1)	/* bmap access */ ///< Get physical block number for a logical block.
#define FIGETBSZ   _IO(0x00,2)	/* get the block size used for bmap */ ///< Get block size used for BMAP.
#define FIFREEZE	_IOWR('X', 119, int)	/* Freeze */ ///< Freeze a filesystem (e.g., for snapshots).
#define FITHAW		_IOWR('X', 120, int)	/* Thaw */ ///< Thaw a frozen filesystem.
#define FITRIM		_IOWR('X', 121, struct fstrim_range)	/* Trim */ ///< Trim blocks on a filesystem.
#define FICLONE		_IOW(0x94, 9, int) ///< Clone a file (copy-on-write).
#define FICLONERANGE	_IOW(0x94, 13, struct file_clone_range) ///< Clone a range of a file.
#define FIDEDUPERANGE	_IOWR(0x94, 54, struct file_dedupe_range) ///< Deduplicate a range of file data.

#define FSLABEL_MAX 256	/* Max chars for the interface; each fs may differ */ ///< Maximum characters for a filesystem label.

#define	FS_IOC_GETFLAGS			_IOR('f', 1, long) ///< Get inode flags.
#define	FS_IOC_SETFLAGS			_IOW('f', 2, long) ///< Set inode flags.
#define	FS_IOC_GETVERSION		_IOR('v', 1, long) ///< Get filesystem version.
#define	FS_IOC_SETVERSION		_IOW('v', 2, long) ///< Set filesystem version.
#define FS_IOC_FIEMAP			_IOWR('f', 11, struct fiemap) ///< Get file extent mapping.
#define FS_IOC32_GETFLAGS		_IOR('f', 1, int) ///< Get inode flags (32-bit compat).
#define FS_IOC32_SETFLAGS		_IOW('f', 2, int) ///< Set inode flags (32-bit compat).
#define FS_IOC32_GETVERSION		_IOR('v', 1, int) ///< Get filesystem version (32-bit compat).
#define FS_IOC32_SETVERSION		_IOW('v', 2, int) ///< Set filesystem version (32-bit compat).
#define FS_IOC_FSGETXATTR		_IOR('X', 31, struct fsxattr) ///< Get extended filesystem attributes.
#define FS_IOC_FSSETXATTR		_IOW('X', 32, struct fsxattr) ///< Set extended filesystem attributes.
#define FS_IOC_GETFSLABEL		_IOR(0x94, 49, char[FSLABEL_MAX]) ///< Get filesystem label.
#define FS_IOC_SETFSLABEL		_IOW(0x94, 50, char[FSLABEL_MAX]) ///< Set filesystem label.
/* Returns the external filesystem UUID, the same one blkid returns */
#define FS_IOC_GETFSUUID		_IOR(0x15, 0, struct fsuuid2) ///< Get external filesystem UUID.
/*
 * Returns the path component under /sys/fs/ that refers to this filesystem;
 * also /sys/kernel/debug/ for filesystems with debugfs exports
 */
#define FS_IOC_GETFSSYSFSPATH		_IOR(0x15, 1, struct fs_sysfs_path) ///< Get /sys/fs/ path for filesystem.

/*
 * Inode flags (FS_IOC_GETFLAGS / FS_IOC_SETFLAGS)
 *
 * Note: for historical reasons, these flags were originally used and
 * defined for use by ext2/ext3, and then other file systems started
 * using these flags so they wouldn't need to write their own version
 * of chattr/lsattr (which was shipped as part of e2fsprogs).  You
 * should think twice before trying to use these flags in new
 * contexts, or trying to assign these flags, since they are used both
 * as the UAPI and the on-disk encoding for ext2/3/4.  Also, we are
 * almost out of 32-bit flags.  :-)
 *
 * We have recently hoisted FS_IOC_FSGETXATTR / FS_IOC_FSSETXATTR from
 * XFS to the generic FS level interface.  This uses a structure that
 * has padding and hence has more room to grow, so it may be more
 * appropriate for many new use cases.
 *
 * Please do not change these flags or interfaces before checking with
 * linux-fsdevel@vger.kernel.org and linux-api@vger.kernel.org.
 */
#define	FS_SECRM_FL			0x00000001 /* Secure deletion */ ///< Secure deletion (undeletable).
#define	FS_UNRM_FL			0x00000002 /* Undelete */ ///< Undelete (recoverable).
#define	FS_COMPR_FL			0x00000004 /* Compress file */ ///< File is compressed.
#define FS_SYNC_FL			0x00000008 /* Synchronous updates */ ///< All changes are written synchronously.
#define FS_IMMUTABLE_FL			0x00000010 /* Immutable file */ ///< File cannot be modified.
#define FS_APPEND_FL			0x00000020 /* writes to file may only append */ ///< Writes to file are append-only.
#define FS_NODUMP_FL			0x00000040 /* do not dump file */ ///< Do not include file in backups.
#define FS_NOATIME_FL			0x00000080 /* do not update access time */ ///< Do not update access time.
/* Reserved for compression usage... */
#define FS_DIRTY_FL			0x00000100 ///< Internal flag: file is dirty.
#define FS_COMPRBLK_FL			0x00000200 /* One or more compressed clusters */ ///< One or more compressed clusters exist.
#define FS_NOCOMP_FL			0x00000400 /* Don't compress */ ///< Do not compress this file.
/* End compression flags --- maybe not all used */
#define FS_ENCRYPT_FL			0x00000800 /* Encrypted file */ ///< File is encrypted.
#define FS_BTREE_FL			0x00001000 /* btree format dir */ ///< Directory is in B-tree format.
#define FS_INDEX_FL			0x00001000 /* hash-indexed directory */ ///< Directory is hash-indexed.
#define FS_IMAGIC_FL			0x00002000 /* AFS directory */ ///< AFS directory.
#define FS_JOURNAL_DATA_FL		0x00004000 /* Reserved for ext3 */ ///< Journal data (ext3).
#define FS_NOTAIL_FL			0x00008000 /* file tail should not be merged */ ///< File tail should not be merged.
#define FS_DIRSYNC_FL			0x00010000 /* dirsync behaviour (directories only) */ ///< Directory synchronization behavior.
#define FS_TOPDIR_FL			0x00020000 /* Top of directory hierarchies*/ ///< Top of directory hierarchy.
#define FS_HUGE_FILE_FL			0x00040000 /* Reserved for ext4 */ ///< Huge file (ext4).
#define FS_EXTENT_FL			0x00080000 /* Extents */ ///< File uses extents.
#define FS_VERITY_FL			0x00100000 /* Verity protected inode */ ///< Verity protected inode.
#define FS_EA_INODE_FL			0x00200000 /* Inode used for large EA */ ///< Inode used for large extended attributes.
#define FS_EOFBLOCKS_FL			0x00400000 /* Reserved for ext4 */ ///< EOF blocks (ext4).
#define FS_NOCOW_FL			0x00800000 /* Do not cow file */ ///< Do not Copy-on-Write this file.
#define FS_DAX_FL			0x02000000 /* Inode is DAX */ ///< Inode uses DAX (Direct Access) functionality.
#define FS_INLINE_DATA_FL		0x10000000 /* Reserved for ext4 */ ///< Inline data (ext4).
#define FS_PROJINHERIT_FL		0x20000000 /* Create with parents projid */ ///< Inherit project ID from parent.
#define FS_CASEFOLD_FL			0x40000000 /* Folder is case insensitive */ ///< Folder is case-insensitive.
#define FS_RESERVED_FL			0x80000000 /* reserved for ext2 lib */ ///< Reserved for ext2 library.

#define FS_FL_USER_VISIBLE		0x0003DFFF /* User visible flags */ ///< Mask of all user-visible inode flags.
#define FS_FL_USER_MODIFIABLE		0x000380FF /* User modifiable flags */ ///< Mask of user-modifiable inode flags.


#define SYNC_FILE_RANGE_WAIT_BEFORE	1 ///< Flag for `sync_file_range`: wait for dirty pages to be written.
#define SYNC_FILE_RANGE_WRITE		2 ///< Flag for `sync_file_range`: write dirty pages to disk.
#define SYNC_FILE_RANGE_WAIT_AFTER	4 ///< Flag for `sync_file_range`: wait for writes to complete.
#define SYNC_FILE_RANGE_WRITE_AND_WAIT	(SYNC_FILE_RANGE_WRITE | \
					 SYNC_FILE_RANGE_WAIT_BEFORE | \
					 SYNC_FILE_RANGE_WAIT_AFTER) ///< Combination of write and wait flags.

/*
 * Flags for preadv2/pwritev2:
 */

typedef int __bitwise __kernel_rwf_t; ///< Type for kernel read/write flags.

/* high priority request, poll if possible */
#define RWF_HIPRI	((__force __kernel_rwf_t)0x00000001) ///< High priority read/write request.

/* per-IO O_DSYNC */
#define RWF_DSYNC	((__force __kernel_rwf_t)0x00000002) ///< Per-IO data synchronous write.

/* per-IO O_SYNC */
#define RWF_SYNC	((__force __kernel_rwf_t)0x00000004) ///< Per-IO synchronous write.

/* per-IO, return -EAGAIN if operation would block */
#define RWF_NOWAIT	((__force __kernel_rwf_t)0x00000008) ///< Non-blocking I/O, return if operation would block.

/* per-IO O_APPEND */
#define RWF_APPEND	((__force __kernel_rwf_t)0x00000010) ///< Per-IO append mode.

/* per-IO negation of O_APPEND */
#define RWF_NOAPPEND	((__force __kernel_rwf_t)0x00000020) ///< Per-IO disable append mode.

/* Atomic Write */
#define RWF_ATOMIC	((__force __kernel_rwf_t)0x00000040) ///< Per-IO atomic write.

/* buffered IO that drops the cache after reading or writing data */
#define RWF_DONTCACHE	((__force __kernel_rwf_t)0x00000080) ///< Buffered I/O that drops cache.

/* mask of flags supported by the kernel */
#define RWF_SUPPORTED	(RWF_HIPRI | RWF_DSYNC | RWF_SYNC | RWF_NOWAIT |\
			 RWF_APPEND | RWF_NOAPPEND | RWF_ATOMIC |\
			 RWF_DONTCACHE) ///< Mask of all supported `RWF_` flags.

#define PROCFS_IOCTL_MAGIC 'f' ///< Magic number for /proc filesystem ioctls.

/* Pagemap ioctl */
#define PAGEMAP_SCAN	_IOWR(PROCFS_IOCTL_MAGIC, 16, struct pm_scan_arg) ///< IOCTL for scanning process pagemaps.

/* Bitmasks provided in pm_scan_args masks and reported in page_region.categories. */
#define PAGE_IS_WPALLOWED	(1 << 0) ///< Page is write-protected but allowed.
#define PAGE_IS_WRITTEN		(1 << 1) ///< Page has been written to.
#define PAGE_IS_FILE		(1 << 2) ///< Page is file-backed.
#define PAGE_IS_PRESENT		(1 << 3) ///< Page is present in RAM.
#define PAGE_IS_SWAPPED		(1 << 4) ///< Page is swapped out.
#define PAGE_IS_PFNZERO		(1 << 5) ///< Page has zero PFN (e.g., zero page).
#define PAGE_IS_HUGE		(1 << 6) ///< Page is a huge page.
#define PAGE_IS_SOFT_DIRTY	(1 << 7) ///< Page is soft-dirty (tracked for checkpointing).
#define PAGE_IS_GUARD		(1 << 8) ///< Page is a guard page.

/*
 * struct page_region - Page region with flags
 * @start:	Start of the region
 * @end:	End of the region (exclusive)
 * @categories:	PAGE_IS_* category bitmask for the region
 */
/**
 * @struct page_region
 * @brief Describes a contiguous region of virtual memory with associated properties.
 * Functional Utility: Used by the `PAGEMAP_SCAN` ioctl to report properties
 * of virtual memory pages within a specified range, categorized by bitmasks.
 */
struct page_region {
	__u64 start; ///< Starting address of the region (inclusive).
	__u64 end; ///< Ending address of the region (exclusive).
	__u64 categories; ///< Bitmask of `PAGE_IS_*` categories for the region.
};

/* Flags for PAGEMAP_SCAN ioctl */
#define PM_SCAN_WP_MATCHING	(1 << 0)	/* Write protect the pages matched. */ ///< Flag to write-protect matched pages.
#define PM_SCAN_CHECK_WPASYNC	(1 << 1)	/* Abort the scan when a non-WP-enabled page is found. */ ///< Flag to abort scan if non-WP page found.

/*
 * struct pm_scan_arg - Pagemap ioctl argument
 * @size:		Size of the structure
 * @flags:		Flags for the IOCTL
 * @start:		Starting address of the region
 * @end:		Ending address of the region
 * @walk_end		Address where the scan stopped (written by kernel).
 *			walk_end == end (address tags cleared) informs that the scan completed on entire range.
 * @vec:		Address of page_region struct array for output
 * @vec_len:		Length of the page_region struct array
 * @max_pages:		Optional limit for number of returned pages (0 = disabled)
 * @category_inverted:	PAGE_IS_* categories which values match if 0 instead of 1
 * @category_mask:	Skip pages for which any category doesn't match
 * @category_anyof_mask: Skip pages for which no category matches
 * @return_mask:	PAGE_IS_* categories that are to be reported in `page_region`s returned
 */
/**
 * @struct pm_scan_arg
 * @brief Arguments structure for the `PAGEMAP_SCAN` ioctl.
 * Functional Utility: Configures the behavior of the pagemap scanner, specifying
 * the memory range, filtering criteria, and how results should be returned.
 */
struct pm_scan_arg {
	__u64 size; ///< Size of this structure.
	__u64 flags; ///< Flags for the ioctl, e.g., `PM_SCAN_WP_MATCHING`.
	__u64 start; ///< Starting virtual address of the region to scan.
	__u64 end; ///< Ending virtual address of the region to scan (exclusive).
	__u64 walk_end; ///< (Output) Address where the scan stopped.
	__u64 vec; ///< (Input) Address of a userspace array of `page_region` structs for output.
	__u64 vec_len; ///< (Input) Length of the `page_region` struct array.
	__u64 max_pages; ///< Optional limit for number of returned pages (0 = disabled).
	__u64 category_inverted; ///< `PAGE_IS_*` categories to match if their value is 0 instead of 1.
	__u64 category_mask; ///< Pages are skipped if any category in this mask doesn't match.
	__u64 category_anyof_mask; ///< Pages are skipped if no category in this mask matches.
	__u64 return_mask; ///< `PAGE_IS_*` categories that are to be reported in returned `page_region`s.
};

/* /proc/<pid>/maps ioctl */
#define PROCMAP_QUERY	_IOWR(PROCFS_IOCTL_MAGIC, 17, struct procmap_query) ///< IOCTL for querying process memory maps.

/**
 * @enum procmap_query_flags
 * @brief Flags for the `PROCMAP_QUERY` ioctl to filter and modify query behavior.
 * Functional Utility: Used to specify permissions for VMAs, query modifiers
 * (e.g., next VMA, file-backed), and control how the `PROCMAP_QUERY` ioctl
 * searches for Virtual Memory Areas.
 */
enum procmap_query_flags {
	/*
	 * VMA permission flags.
	 *
	 * Can be used as part of procmap_query.query_flags field to look up
	 * only VMAs satisfying specified subset of permissions. E.g., specifying
	 * PROCMAP_QUERY_VMA_READABLE only will return both readable and read/write VMAs,
	 * while having PROCMAP_QUERY_VMA_READABLE | PROCMAP_QUERY_VMA_WRITABLE will only
	 * return read/write VMAs, though both executable/non-executable and
	 * private/shared will be ignored.
	 *
	 * PROCMAP_QUERY_VMA_* flags are also returned in procmap_query.vma_flags
	 * field to specify actual VMA permissions.
	 */
	PROCMAP_QUERY_VMA_READABLE		= 0x01, ///< VMA is readable.
	PROCMAP_QUERY_VMA_WRITABLE		= 0x02, ///< VMA is writable.
	PROCMAP_QUERY_VMA_EXECUTABLE		= 0x04, ///< VMA is executable.
	PROCMAP_QUERY_VMA_SHARED		= 0x08, ///< VMA is shared (not private).
	/*
	 * Query modifier flags.
	 *
	 * By default VMA that covers provided address is returned, or -ENOENT
	 * is returned. With PROCMAP_QUERY_COVERING_OR_NEXT_VMA flag set, closest
	 * VMA with vma_start > addr will be returned if no covering VMA is
	 * found.
	 *
	 * PROCMAP_QUERY_FILE_BACKED_VMA instructs query to consider only VMAs that
	 * have file backing. Can be combined with PROCMAP_QUERY_COVERING_OR_NEXT_VMA
	 * to iterate all VMAs with file backing.
	 */
	PROCMAP_QUERY_COVERING_OR_NEXT_VMA	= 0x10, ///< Return covering VMA or the next one.
	PROCMAP_QUERY_FILE_BACKED_VMA		= 0x20, ///< Consider only file-backed VMAs.
};

/*
 * Input/output argument structured passed into ioctl() call. It can be used
 * to query a set of VMAs (Virtual Memory Areas) of a process.
 *
 * Each field can be one of three kinds, marked in a short comment to the
 * right of the field:
 *   - "in", input argument, user has to provide this value, kernel doesn't modify it;
 *   - "out", output argument, kernel sets this field with VMA data;
 *   - "in/out", input and output argument; user provides initial value (used
 *     to specify maximum allowable buffer size), and kernel sets it to actual
 *     amount of data written (or zero, if there is no data).
 *
 * If matching VMA is found (according to criterias specified by
 * query_addr/query_flags, all the out fields are filled out, and ioctl()
 * returns 0. If there is no matching VMA, -ENOENT will be returned.
 * In case of any other error, negative error code other than -ENOENT is
 * returned.
 *
 * Most of the data is similar to the one returned as text in /proc/<pid>/maps
 * file, but procmap_query provides more querying flexibility. There are no
 * consistency guarantees between subsequent ioctl() calls, but data returned
 * for matched VMA is self-consistent.
 */
/**
 * @struct procmap_query
 * @brief Arguments structure for the `PROCMAP_QUERY` ioctl.
 * Functional Utility: Enables userspace to query detailed information about
 * a process's Virtual Memory Areas (VMAs), including their addresses,
 * permissions, backing files, and associated metadata. It supports various
 * filtering and lookup options.
 */
struct procmap_query {
	/* Query struct size, for backwards/forward compatibility */
	__u64 size; ///< (Input) Size of this structure, for ABI compatibility.
	/*
	 * Query flags, a combination of enum procmap_query_flags values.
	 * Defines query filtering and behavior, see enum procmap_query_flags.
	 *
	 * Input argument, provided by user. Kernel doesn't modify it.
	 */
	__u64 query_flags;		/* in */ ///< (Input) Flags for filtering and lookup behavior.
	/*
	 * Query address. By default, VMA that covers this address will
	 * be looked up. PROCMAP_QUERY_* flags above modify this default
	 * behavior further.
	 *
	 * Input argument, provided by user. Kernel doesn't modify it.
	 */
	__u64 query_addr;		/* in */ ///< (Input) Address to query for a VMA.
	/* VMA starting (inclusive) and ending (exclusive) address, if VMA is found. */
	__u64 vma_start;		/* out */ ///< (Output) Starting address of the found VMA.
	__u64 vma_end;			/* out */ ///< (Output) Ending address of the found VMA.
	/* VMA permissions flags. A combination of PROCMAP_QUERY_VMA_* flags. */
	__u64 vma_flags;		/* out */ ///< (Output) Permissions flags of the VMA.
	/* VMA backing page size granularity. */
	__u64 vma_page_size;		/* out */ ///< (Output) Page size granularity of the VMA.
	/*
	 * VMA file offset. If VMA has file backing, this specifies offset
	 * within the file that VMA's start address corresponds to.
	 * Is set to zero if VMA has no backing file.
	 */
	__u64 vma_offset;		/* out */ ///< (Output) File offset if file-backed.
	/* Backing file's inode number, or zero, if VMA has no backing file. */
	__u64 inode;			/* out */ ///< (Output) Inode number of backing file (if any).
	/* Backing file's device major/minor number, or zero, if VMA has no backing file. */
	__u32 dev_major;		/* out */ ///< (Output) Major device number of backing file (if any).
	__u32 dev_minor;		/* out */ ///< (Output) Minor device number of backing file (if any).
	/*
	 * If set to non-zero value, signals the request to return VMA name
	 * (i.e., VMA's backing file's absolute path, with " (deleted)" suffix
	 * appended, if file was unlinked from FS) for matched VMA. VMA name
	 * can also be some special name (e.g., "[heap]", "[stack]") or could
	 * be even user-supplied with prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME).
	 *
	 * Kernel will set this field to zero, if VMA has no associated name.
	 * Otherwise kernel will return actual amount of bytes filled in
	 * user-supplied buffer (see vma_name_addr field below), including the
	 * terminating zero.
	 *
	 * If VMA name is longer that user-supplied maximum buffer size,
	 * -E2BIG error is returned.
	 *
	 * If this field is set to non-zero value, vma_name_addr should point
	 * to valid user space memory buffer of at least vma_name_size bytes.
	 * If set to zero, vma_name_addr should be set to zero as well
	 */
	__u32 vma_name_size;		/* in/out */ ///< (Input/Output) Size of VMA name buffer.
	/*
	 * If set to non-zero value, signals the request to extract and return
	 * VMA's backing file's build ID, if the backing file is an ELF file
	 * and it contains embedded build ID.
	 *
	 * Kernel will set this field to zero, if VMA has no backing file,
	 * backing file is not an ELF file, or ELF file has no build ID
	 * embedded.
	 *
	 * Build ID is a binary value (not a string). Kernel will set
	 * build_id_size field to exact number of bytes used for build ID.
	 * If build ID is requested and present, but needs more bytes than
	 * user-supplied maximum buffer size (see build_id_addr field below),
	 * -E2BIG error will be returned.
	 *
	 * If this field is set to non-zero value, build_id_addr should point
	 * to valid user space memory buffer of at least build_id_size bytes.
	 * If set to zero, build_id_addr should be set to zero as well
	 */
	__u32 build_id_size;		/* in/out */ ///< (Input/Output) Size of build ID buffer.
	/*
	 * User-supplied address of a buffer of at least vma_name_size bytes
	 * for kernel to fill with matched VMA's name (see vma_name_size field
	 * description above for details).
	 *
	 * Should be set to zero if VMA name should not be returned.
	 */
	__u64 vma_name_addr;		/* in */ ///< (Input) Userspace address for VMA name buffer.
	/*
	 * User-supplied address of a buffer of at least build_id_size bytes
	 * for kernel to fill with matched VMA's ELF build ID, if available
	 * (see build_id_size field description above for details).
	 *
	 * Should be set to zero if build ID should not be returned.
	 */
	__u64 build_id_addr;		/* in */ ///< (Input) Userspace address for build ID buffer.
};

#endif /* _UAPI_LINUX_FS_H */