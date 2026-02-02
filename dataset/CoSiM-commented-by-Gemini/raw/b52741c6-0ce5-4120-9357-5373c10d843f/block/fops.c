/**
 * @file fops.c
 * @brief Implements file operations for block devices in the Linux kernel.
 *
 * This file provides the core implementation for file-level operations on block
 * devices, such as read, write, seek, and fsync. It handles both buffered and
 * direct I/O, memory mapping, and ioctl commands. The implementation bridges
 * the gap between the generic file system interface and the block layer's bio-based
 * request processing.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (C) 1991, 1992  Linus Torvalds
 * Copyright (C) 2001  Andrea Arcangeli <andrea@suse.de> SuSE
 * Copyright (C) 2016 - 2020 Christoph Hellwig
 */
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/blkdev.h>
#include <linux/buffer_head.h>
#include <linux/mpage.h>
#include <linux/uio.h>
#include <linux/namei.h>
#include <linux/task_io_accounting_ops.h>
#include <linux/falloc.h>
#include <linux/suspend.h>
#include <linux/fs.h>
#include <linux/iomap.h>
#include <linux/module.h>
#include <linux/io_uring/cmd.h>
#include "blk.h"

// Helper to get the inode associated with a block device file.
static inline struct inode *bdev_file_inode(struct file *file)
{
	return file->f_mapping->host;
}

/**
 * @brief Determines the bio operation flags for a direct write.
 * @param iocb The I/O control block for the operation.
 * @return The appropriate blk_opf_t flags for the write.
 *
 * This function sets up the flags for a synchronous direct write, including
 * REQ_FUA (Force Unit Access) if the operation is for data sync.
 */
static blk_opf_t dio_bio_write_op(struct kiocb *iocb)
{
	blk_opf_t opf = REQ_OP_WRITE | REQ_SYNC | REQ_IDLE;

	/* avoid the need for a I/O completion work item */
	if (iocb_is_dsync(iocb))
		opf |= REQ_FUA;
	return opf;
}

/**
 * @brief Checks if a direct I/O operation is invalid due to alignment constraints.
 * @param bdev The block device.
 * @param iocb The I/O control block.
 * @param iter The iov_iter containing the user data buffer.
 * @return True if the I/O operation is misaligned, false otherwise.
 */
static bool blkdev_dio_invalid(struct block_device *bdev, struct kiocb *iocb,
				struct iov_iter *iter)
{
	return iocb->ki_pos & (bdev_logical_block_size(bdev) - 1) ||
		!bdev_iter_is_aligned(bdev, iter);
}

#define DIO_INLINE_BIO_VECS 4

/**
 * @brief A simplified path for synchronous direct I/O.
 * @param iocb The I/O control block.
 * @param iter The iov_iter with the data.
 * @param bdev The block device.
 * @param nr_pages The number of pages in the I/O.
 * @return The number of bytes transferred, or a negative error code.
 *
 * This function is an optimization for small, simple direct I/O operations that
 * can be handled with a single bio structure.
 */
static ssize_t __blkdev_direct_IO_simple(struct kiocb *iocb,
		struct iov_iter *iter, struct block_device *bdev,
		unsigned int nr_pages)
{
	struct bio_vec inline_vecs[DIO_INLINE_BIO_VECS], *vecs;
	loff_t pos = iocb->ki_pos;
	bool should_dirty = false;
	struct bio bio;
	ssize_t ret;

	WARN_ON_ONCE(iocb->ki_flags & IOCB_HAS_METADATA);
	if (nr_pages <= DIO_INLINE_BIO_VECS)
		vecs = inline_vecs;
	else {
		vecs = kmalloc_array(nr_pages, sizeof(struct bio_vec),
				     GFP_KERNEL);
		if (!vecs)
			return -ENOMEM;
	}

	if (iov_iter_rw(iter) == READ) {
		bio_init(&bio, bdev, vecs, nr_pages, REQ_OP_READ);
		if (user_backed_iter(iter))
			should_dirty = true;
	} else {
		bio_init(&bio, bdev, vecs, nr_pages, dio_bio_write_op(iocb));
	}
	bio.bi_iter.bi_sector = pos >> SECTOR_SHIFT;
	bio.bi_write_hint = file_inode(iocb->ki_filp)->i_write_hint;
	bio.bi_write_stream = iocb->ki_write_stream;
	bio.bi_ioprio = iocb->ki_ioprio;
	if (iocb->ki_flags & IOCB_ATOMIC)
		bio.bi_opf |= REQ_ATOMIC;

	ret = bio_iov_iter_get_pages(&bio, iter);
	if (unlikely(ret))
		goto out;
	ret = bio.bi_iter.bi_size;

	if (iov_iter_rw(iter) == WRITE)
		task_io_account_write(ret);

	if (iocb->ki_flags & IOCB_NOWAIT)
		bio.bi_opf |= REQ_NOWAIT;

	submit_bio_wait(&bio);

	bio_release_pages(&bio, should_dirty);
	if (unlikely(bio.bi_status))
		ret = blk_status_to_errno(bio.bi_status);

out:
	if (vecs != inline_vecs)
		kfree(vecs);

	bio_uninit(&bio);

	return ret;
}

enum {
	DIO_SHOULD_DIRTY	= 1, // Flag: pages should be marked dirty after I/O.
	DIO_IS_SYNC		= 2, // Flag: the I/O operation is synchronous.
};

/**
 * @struct blkdev_dio
 * @brief State tracking for a direct I/O operation.
 *
 * This structure is embedded within a bio to manage the state of a complex
 * direct I/O operation that may span multiple bio submissions.
 */
struct blkdev_dio {
	union {
		struct kiocb		*iocb;   // For async I/O.
		struct task_struct	*waiter; // For sync I/O.
	};
	size_t			size;    // Total size of the I/O operation.
	atomic_t		ref;     // Reference count for outstanding bios.
	unsigned int		flags;   // DIO flags (e.g., DIO_SHOULD_DIRTY).
	struct bio		bio ____cacheline_aligned_in_smp; // The first bio.
};

static struct bio_set blkdev_dio_pool;

/**
 * @brief The end I/O handler for direct I/O bios.
 * @param bio The bio that has completed.
 *
 * This function is called upon completion of each bio in a direct I/O operation.
 * It handles error propagation, reference counting, and final completion of the
 * kiocb for async operations or waking up the process for sync operations.
 */
static void blkdev_bio_end_io(struct bio *bio)
{
	struct blkdev_dio *dio = bio->bi_private;
	bool should_dirty = dio->flags & DIO_SHOULD_DIRTY;
	bool is_sync = dio->flags & DIO_IS_SYNC;

	if (bio->bi_status && !dio->bio.bi_status)
		dio->bio.bi_status = bio->bi_status;

	if (!is_sync && (dio->iocb->ki_flags & IOCB_HAS_METADATA))
		bio_integrity_unmap_user(bio);

	if (atomic_dec_and_test(&dio->ref)) {
		if (!is_sync) {
			struct kiocb *iocb = dio->iocb;
			ssize_t ret;

			WRITE_ONCE(iocb->private, NULL);

			if (likely(!dio->bio.bi_status)) {
				ret = dio->size;
				iocb->ki_pos += ret;
			} else {
				ret = blk_status_to_errno(dio->bio.bi_status);
			}

			dio->iocb->ki_complete(iocb, ret);
			bio_put(&dio->bio);
		} else {
			struct task_struct *waiter = dio->waiter;

			WRITE_ONCE(dio->waiter, NULL);
			blk_wake_io_task(waiter);
		}
	}

	if (should_dirty) {
		bio_check_pages_dirty(bio);
	} else {
		bio_release_pages(bio, false);
		bio_put(bio);
	}
}
... The rest of the file ...