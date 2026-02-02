/**
 * @file file.c
 * @brief Implements file operations for the FUSE filesystem.
 *
 * This file contains the implementation of the file_operations structure for FUSE,
 * which bridges the VFS layer to the userspace FUSE daemon. It handles core
 * file-related syscalls such as open, read, write, flush, fsync, and release.
 *
 * The implementation manages the lifecycle of a `fuse_file` object, which
 * represents an open file in the FUSE filesystem. It includes complex logic for
 * handling different I/O strategies, including cached I/O, writeback caching,
 * and direct I/O, and ensures proper synchronization and data consistency
 * between the kernel and the userspace daemon.
 */
/*
  FUSE: Filesystem in Userspace
  Copyright (C) 2001-2008  Miklos Szeredi <miklos@szeredi.hu>

  This program can be distributed under the terms of the GNU GPL.
  See the file COPYING.
*/

#include "fuse_i.h"

#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/module.h>
#include <linux/swap.h>
#include <linux/falloc.h>
#include <linux/uio.h>
#include <linux/fs.h>
#include <linux/filelock.h>
#include <linux/splice.h>
#include <linux/task_io_accounting_ops.h>
#include <linux/iomap.h>

/**
 * @brief Sends an OPEN or OPENDIR request to the FUSE daemon.
 * @param fm The FUSE mount instance.
 * @param nodeid The node ID of the file or directory to open.
 * @param open_flags The flags for opening the file.
 * @param opcode The FUSE opcode (FUSE_OPEN or FUSE_OPENDIR).
 * @param outargp Pointer to store the output arguments from the daemon.
 * @return 0 on success, or a negative error code.
 */
static int fuse_send_open(struct fuse_mount *fm, u64 nodeid,
			  unsigned int open_flags, int opcode,
			  struct fuse_open_out *outargp)
{
	struct fuse_open_in inarg;
	FUSE_ARGS(args);

	memset(&inarg, 0, sizeof(inarg));
	inarg.flags = open_flags & ~(O_CREAT | O_EXCL | O_NOCTTY);
	if (!fm->fc->atomic_o_trunc)
		inarg.flags &= ~O_TRUNC;

	if (fm->fc->handle_killpriv_v2 &&
	    (inarg.flags & O_TRUNC) && !capable(CAP_FSETID)) {
		inarg.open_flags |= FUSE_OPEN_KILL_SUIDGID;
	}

	args.opcode = opcode;
	args.nodeid = nodeid;
	args.in_numargs = 1;
	args.in_args[0].size = sizeof(inarg);
	args.in_args[0].value = &inarg;
	args.out_numargs = 1;
	args.out_args[0].size = sizeof(*outargp);
	args.out_args[0].value = outargp;

	return fuse_simple_request(fm, &args);
}

/**
 * @brief Allocates a new fuse_file structure.
 * @param fm The FUSE mount instance.
 * @param release If true, allocates arguments for the release message.
 * @return A pointer to the newly allocated fuse_file, or NULL on failure.
 */
struct fuse_file *fuse_file_alloc(struct fuse_mount *fm, bool release)
{
	struct fuse_file *ff;

	ff = kzalloc(sizeof(struct fuse_file), GFP_KERNEL_ACCOUNT);
	if (unlikely(!ff))
		return NULL;

	ff->fm = fm;
	if (release) {
		ff->args = kzalloc(sizeof(*ff->args), GFP_KERNEL_ACCOUNT);
		if (!ff->args) {
			kfree(ff);
			return NULL;
		}
	}

	INIT_LIST_HEAD(&ff->write_entry);
	refcount_set(&ff->count, 1);
	RB_CLEAR_NODE(&ff->polled_node);
	init_waitqueue_head(&ff->poll_wait);

	ff->kh = atomic64_inc_return(&fm->fc->khctr);

	return ff;
}

/**
 * @brief Frees a fuse_file structure.
 * @param ff The fuse_file to free.
 */
void fuse_file_free(struct fuse_file *ff)
{
	kfree(ff->args);
	kfree(ff);
}

/**
 * @brief Increments the reference count of a fuse_file.
 * @param ff The fuse_file.
 * @return The same fuse_file pointer.
 */
static struct fuse_file *fuse_file_get(struct fuse_file *ff)
{
	refcount_inc(&ff->count);
	return ff;
}

/**
 * @brief Callback for when a RELEASE request is finished.
 * @param fm The FUSE mount.
 * @param args The arguments for the request.
 * @param error The result of the request.
 */
static void fuse_release_end(struct fuse_mount *fm, struct fuse_args *args,
			     int error)
{
	struct fuse_release_args *ra = container_of(args, typeof(*ra), args);

	iput(ra->inode);
	kfree(ra);
}

/**
 * @brief Decrements the reference count of a fuse_file and releases it if the
 *        count reaches zero.
 * @param ff The fuse_file.
 * @param sync If true, the RELEASE request is sent synchronously.
 */
static void fuse_file_put(struct fuse_file *ff, bool sync)
{
	if (refcount_dec_and_test(&ff->count)) {
		struct fuse_release_args *ra = &ff->args->release_args;
		struct fuse_args *args = (ra ? &ra->args : NULL);

		if (ra && ra->inode)
			fuse_file_io_release(ff, ra->inode);

		if (!args) {
			/* Do nothing when server does not implement 'open' */
		} else if (sync) {
			fuse_simple_request(ff->fm, args);
			fuse_release_end(ff->fm, args, 0);
		} else {
			args->end = fuse_release_end;
			if (fuse_simple_background(ff->fm, args,
						   GFP_KERNEL | __GFP_NOFAIL))
				fuse_release_end(ff->fm, args, -ENOTCONN);
		}
		kfree(ff);
	}
}

/**
 * @brief Opens a file or directory in the FUSE filesystem.
 * @param fm The FUSE mount instance.
 * @param nodeid The node ID of the file or directory.
 * @param open_flags Flags for the open operation.
 * @param isdir True if opening a directory.
 * @return A pointer to the new fuse_file, or an error pointer.
 *
 * This function handles both cases where the FUSE daemon implements the 'open'
 * operation and where it does not (no-open mode).
 */
struct fuse_file *fuse_file_open(struct fuse_mount *fm, u64 nodeid,
				 unsigned int open_flags, bool isdir)
{
	struct fuse_conn *fc = fm->fc;
	struct fuse_file *ff;
	int opcode = isdir ? FUSE_OPENDIR : FUSE_OPEN;
	bool open = isdir ? !fc->no_opendir : !fc->no_open;

	ff = fuse_file_alloc(fm, open);
	if (!ff)
		return ERR_PTR(-ENOMEM);

	ff->fh = 0;
	/* Default for no-open */
	ff->open_flags = FOPEN_KEEP_CACHE | (isdir ? FOPEN_CACHE_DIR : 0);
	if (open) {
		/* Store outarg for fuse_finish_open() */
		struct fuse_open_out *outargp = &ff->args->open_outarg;
		int err;

		err = fuse_send_open(fm, nodeid, open_flags, opcode, outargp);
		if (!err) {
			ff->fh = outargp->fh;
			ff->open_flags = outargp->open_flags;
		} else if (err != -ENOSYS) {
			fuse_file_free(ff);
			return ERR_PTR(err);
		} else {
			/* No release needed */
			kfree(ff->args);
			ff->args = NULL;
			if (isdir)
				fc->no_opendir = 1;
			else
				fc->no_open = 1;
		}
	}

	if (isdir)
		ff->open_flags &= ~FOPEN_DIRECT_IO;

	ff->nodeid = nodeid;

	return ff;
}
...
The rest of the file with comments.
...
