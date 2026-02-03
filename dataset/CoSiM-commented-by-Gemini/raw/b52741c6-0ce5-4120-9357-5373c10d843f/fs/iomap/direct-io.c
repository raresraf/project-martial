/**
 * @file direct-io.c
 * @brief Generic implementation of direct I/O using the iomap interface.
 * @copyright Copyright (C) 2010 Red Hat, Inc.
 * @copyright Copyright (c) 2016-2025 Christoph Hellwig.
 *
 * This file provides a generic direct I/O (DIO) implementation for filesystems
 * that use the iomap interface. It handles both synchronous and asynchronous
 * DIO, and is responsible for building and submitting bios to the block layer,
 * managing I/O completion, and handling potential page faults and errors.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2010 Red Hat, Inc.
 * Copyright (c) 2016-2025 Christoph Hellwig.
 */
#include <linux/fscrypt.h>
#include <linux/pagemap.h>
#include <linux/iomap.h>
#include <linux/task_io_accounting_ops.h>
#include "internal.h"
#include "trace.h"

#include "../internal.h"

/*
 * Private flags for iomap_dio, must not overlap with the public ones in
 * iomap.h:
 */
#define IOMAP_DIO_NO_INVALIDATE	(1U << 25)
#define IOMAP_DIO_CALLER_COMP	(1U << 26)
#define IOMAP_DIO_INLINE_COMP	(1U << 27)
#define IOMAP_DIO_WRITE_THROUGH	(1U << 28)
#define IOMAP_DIO_NEED_SYNC	(1U << 29)
#define IOMAP_DIO_WRITE		(1U << 30)
#define IOMAP_DIO_DIRTY		(1U << 31)

/*
 * Used for sub block zeroing in iomap_dio_zero()
 */
#define IOMAP_ZERO_PAGE_SIZE (SZ_64K)
#define IOMAP_ZERO_PAGE_ORDER (get_order(IOMAP_ZERO_PAGE_SIZE))
static struct page *zero_page;

/**
 * @struct iomap_dio
 * @brief State for an in-flight direct I/O operation.
 *
 * This structure tracks all the necessary information for a direct I/O
 * request, including the iocb, the total size of the I/O, error status,
 * and synchronization primitives for completion handling.
 */
struct iomap_dio {
	struct kiocb		*iocb;
	const struct iomap_dio_ops *dops;
	loff_t			i_size;
	loff_t			size;
	atomic_t		ref;
	unsigned		flags;
	int			error;
	size_t			done_before;
	bool			wait_for_completion;

	union {
		/* used during submission and for synchronous completion: */
		struct {
			struct iov_iter		*iter;
			struct task_struct	*waiter;
		} submit;

		/* used for aio completion: */
		struct {
			struct work_struct	work;
		} aio;
	};
};

/**
 * @brief Allocates a bio for a direct I/O operation.
 * @param iter The iomap_iter for the current extent.
 * @param dio The direct I/O state.
 * @param nr_vecs The number of bio_vecs to allocate.
 * @param opf The operation flags for the bio.
 * @return A pointer to the allocated bio, or NULL on failure.
 *
 * This function allocates a bio from a bioset if one is provided by the
 * filesystem, which can improve performance by reducing contention on global
 * locks.
 */
static struct bio *iomap_dio_alloc_bio(const struct iomap_iter *iter,
		struct iomap_dio *dio, unsigned short nr_vecs, blk_opf_t opf)
{
	// Block Logic: If the filesystem provides a private bioset, use it for
	// allocation to improve performance and scalability. Otherwise, fall
	// back to the generic bio allocator.
	if (dio->dops && dio->dops->bio_set)
		return bio_alloc_bioset(iter->iomap.bdev, nr_vecs, opf,
					GFP_KERNEL, dio->dops->bio_set);
	return bio_alloc(iter->iomap.bdev, nr_vecs, opf, GFP_KERNEL);
}

/**
 * @brief Submits a bio for a direct I/O operation.
 * @param iter The iomap_iter for the current extent.
 * @param dio The direct I/O state.
 * @param bio The bio to be submitted.
 * @param pos The file offset for this I/O.
 *
 * This function submits a bio and, if the filesystem has provided an accounting
 * function, calls it to update I/O statistics.
 */
static void iomap_dio_submit_bio(const struct iomap_iter *iter,
		struct iomap_dio *dio, struct bio *bio, loff_t pos)
{
	// Pre-condition: If the filesystem has a custom accounting function,
	// invoke it before submitting the bio.
	if (dio->dops && dio->dops->submit_io)
		dio->dops->submit_io(dio->iocb, iter, bio);

	submit_bio(bio);
}

/**
 * @brief Gets a reference to the iomap_dio structure.
 * @param dio The direct I/O state.
 */
static void iomap_dio_get(struct iomap_dio *dio)
{
	atomic_inc(&dio->ref);
}

/**
 * @brief The bio completion handler for direct I/O.
 * @param bio The completed bio.
 *
 * This function is called when a bio completes. It updates the error status,
 * releases the bio, and decrements the reference count of the iomap_dio
 * structure, triggering the final completion handling when the count reaches
 * zero.
 */
static void iomap_dio_bio_end_io(struct bio *bio)
{
	struct iomap_dio *dio = bio->bi_private;

	// Invariant: If any part of the I/O fails, the error is recorded.
	if (bio->bi_status)
		dio->error = blk_status_to_errno(bio->bi_status);

	bio_free(bio);
	iomap_dio_put(dio);
}

/**
 * @brief Zeros a range of a block device.
 * @param iter The iomap_iter for the current extent.
 * @param dio The direct I/O state.
 * @param pos The starting offset for zeroing.
 * @param len The length of the range to zero.
 * @return 0 on success, or a negative error code.
 *
 * This function handles zeroing of a device range, using either a single large
 * zero page for efficiency or falling back to issuing a write of zeros if the
 * former is not possible.
 */
static int iomap_dio_zero(const struct iomap_iter *iter, struct iomap_dio *dio,
		loff_t pos, u64 len)
{
	const struct iomap *iomap = &iter->iomap;
	struct bio *bio;
	int ret = 0;

	// Invariant: The loop continues until the entire range has been zeroed.
	while (len) {
		unsigned short nr_vecs = 0;
		u64 blen = len;
		int i = 0;

		if (dio->error)
			break;

		// Block Logic: Attempts to use a large, pre-allocated zero page to
		// efficiently zero out large sections of the device. If the mapping is
		// unwritten, this is a fast path that avoids reading old data.
		if (iomap->flags & IOMAP_F_UNWRITTEN) {
			blen = min_t(u64, blen,
				     (u64)IOMAP_ZERO_PAGE_SIZE << i);
			while ((u64)IOMAP_ZERO_PAGE_SIZE << i < blen &&
			       i < IOMAP_ZERO_PAGE_ORDER)
				i++;
		}
		blen = min_t(u64, blen, BIO_MAX_SIZE);

		bio = iomap_dio_alloc_bio(iter, dio, nr_vecs,
				(dio->flags & IOMAP_DIO_WRITE_THROUGH) ?
				REQ_OP_WRITE | REQ_SYNC : REQ_OP_WRITE);
		if (!bio) {
			ret = -ENOMEM;
			break;
		}

		bio->bi_iter.bi_sector = iomap_sector(iomap, pos);
		bio->bi_private = dio;
		bio->bi_end_io = iomap_dio_bio_end_io;
		iomap_dio_get(dio);

		if (iomap->flags & IOMAP_F_UNWRITTEN) {
			while (blen > 0) {
				u64 pwen = (u64)IOMAP_ZERO_PAGE_SIZE << i;

				while (pwen > blen)
					pwen = (u64)IOMAP_ZERO_PAGE_SIZE << --i;

				ret = bio_add_page(bio, zero_page, pwen, 0);
				if (ret != pwen) {
					ret = -EIO;
					goto out_put_bio;
				}
				blen -= pwen;
			}
		}

		iomap_dio_submit_bio(iter, dio, bio, pos);
		pos += bio->bi_iter.bi_size;
		len -= bio->bi_iter.bi_size;
	}
	return ret;

out_put_bio:
	bio_free(bio);
	return ret;
}

/**
 * @brief Processes a single extent for direct I/O.
 * @param iter The iomap_iter for the current extent.
 * @param dio The direct I/O state.
 * @param pos The starting offset for the I/O.
 * @param len The length of the I/O.
 * @return The number of bytes processed, or a negative error code.
 *
 * This is the core function for handling direct I/O. It pins user pages,
 * builds and submits bios, and manages I/O completion.
 */
static loff_t iomap_dio_rw(const struct iomap_iter *iter,
		struct iomap_dio *dio, loff_t pos, u64 len)
{
	const struct iomap *iomap = &iter->iomap;
	loff_t processed = 0;
	blk_opf_t opf;

	if (dio->flags & IOMAP_DIO_WRITE) {
		opf = REQ_OP_WRITE;
		if (dio->flags & IOMAP_DIO_WRITE_THROUGH)
			opf |= REQ_SYNC;
	} else {
		opf = REQ_OP_READ;
	}

	// Invariant: The loop continues as long as there is data to process and no
	// errors have occurred.
	while (len) {
		struct bio *bio;
		u64 plen;
		int ret;

		if (dio->error)
			break;

		// Block Logic: For hole or inline extents, this function will zero out
		// the corresponding range in the user buffer for reads or on the
		// device for writes, ensuring correct semantics for sparse files.
		if (iomap->type == IOMAP_HOLE || iomap->type == IOMAP_INLINE) {
			if (dio->flags & IOMAP_DIO_WRITE) {
				ret = iomap_dio_zero(iter, dio, pos, len);
				if (ret) {
					dio->error = ret;
					break;
				}
				processed += len;
			} else {
				ret = iov_iter_zero(len, dio->submit.iter);
				if (ret < 0) {
					dio->error = ret;
					break;
				}
				processed += len;
			}
			break;
		}

		bio = iomap_dio_alloc_bio(iter, dio, BIO_MAX_PAGES, opf);
		if (!bio) {
			dio->error = -ENOMEM;
			break;
		}

		plen = len;
		ret = iov_iter_get_pages_alloc(dio->submit.iter, &bio->bi_io_vec,
				plen, &bio->bi_vcnt);
		if (ret < 0) {
			bio_free(bio);
			dio->error = ret;
			break;
		}

		bio->bi_iter.bi_size = ret;
		bio->bi_iter.bi_sector = iomap_sector(iomap, pos);
		bio->bi_private = dio;
		bio->bi_end_io = iomap_dio_bio_end_io;
		iomap_dio_get(dio);

		if (dio->flags & IOMAP_DIO_WRITE)
			task_io_account_write(ret);

		iomap_dio_submit_bio(iter, dio, bio, pos);

		processed += ret;
		pos += ret;
		len -= ret;
	}

	return processed;
}

/**
 * @brief Waits for all in-flight direct I/O to complete.
 * @param dio The direct I/O state.
 *
 * This function puts the current task to sleep until all pending bios for a
 * given direct I/O operation have completed.
 */
static void iomap_dio_wait(struct iomap_dio *dio)
{
	// Invariant: The loop continues until the reference count indicates all I/O
	// has completed.
	for (;;) {
		set_current_state(TASK_UNINTERRUPTIBLE);
		if (atomic_read(&dio->ref) == 1)
			break;
		schedule();
	}
	__set_current_state(TASK_RUNNING);
}

/**
 * @brief The main entry point for performing direct I/O.
 * @param iocb The I/O control block.
 * @param iter The iov_iter specifying the user buffer.
 * @param dops The direct I/O operations for the filesystem.
 * @param flags The flags for the direct I/O operation.
 * @return The number of bytes transferred, or a negative error code.
 *
 * This function orchestrates the entire direct I/O process. It iterates over
 * the file's extents, invalidates the page cache where necessary, and calls
 * `iomap_dio_rw` to handle the actual I/O.
 */
ssize_t
iomap_dio_rw(struct kiocb *iocb, struct iov_iter *iter,
		const struct iomap_dio_ops *dops, unsigned int flags)
{
	struct inode *inode = file_inode(iocb->ki_filp);
	const struct iomap_ops *ops = iomap_inode_ops(inode);
	loff_t pos = iocb->ki_pos;
	ssize_t ret = 0;
	size_t done_before = 0;
	struct iomap_dio *dio;
	struct iomap_iter iter_data = {
		.inode = inode,
		.pos = pos,
		.len = iov_iter_count(iter),
		.flags = IOMAP_NOWAIT,
	};

	dio = kzalloc(sizeof(*dio), GFP_KERNEL);
	if (!dio)
		return -ENOMEM;
	dio->iocb = iocb;
	dio->dops = dops;
	dio->i_size = i_size_read(inode);
	dio->size = iter_data.len;
	atomic_set(&dio->ref, 1);
	dio->flags = flags;
	dio->submit.iter = iter;

	// Invariant: The loop iterates over each extent in the requested I/O range.
	while (iter_data.len > 0) {
		loff_t offset, len;
		ssize_t count;
		int ret2;

		ret2 = iomap_iter(&iter_data, ops);
		if (ret2 <= 0) {
			if (!ret2)
				ret2 = -ENXIO;
			ret = ret2;
			break;
		}

		offset = iter_data.iomap.offset;
		len = iter_data.iomap.length;

		// Block Logic: If the I/O is not aligned to the filesystem's block
		// size, this section handles the partial block at the beginning.
		if (pos < offset) {
			loff_t gap = offset - pos;

			if (iov_iter_count(iter) < gap)
				gap = iov_iter_count(iter);

			count = iomap_dio_rw(&iter_data, dio, pos, gap);
			if (count < 0) {
				ret = count;
				break;
			}

			pos += count;
			iov_iter_advance(iter, count);
			ret += count;
			if (count != gap)
				break;
		}

		count = iomap_dio_rw(&iter_data, dio, offset,
				     min(len, iov_iter_count(iter)));
		if (count < 0) {
			ret = count;
			break;
		}

		pos += count;
		iov_iter_advance(iter, count);
		ret += count;
		if (count != min(len, iov_iter_count(iter)))
			break;
	}

	// Post-condition: If any asynchronous I/O was submitted, this block waits for
	// it to complete.
	if (!is_sync_kiocb(iocb)) {
		dio->wait_for_completion = true;
	} else {
		iomap_dio_wait(dio);
		if (!ret)
			ret = dio->error;
	}

	if (iov_iter_count(iter) > 0 && !ret)
		ret = -EFAULT;
	if (dio->flags & IOMAP_DIO_WRITE && (dio->flags & IOMAP_DIO_DIRTY))
		file_inc_version(iocb->ki_filp);

	iomap_dio_put(dio);
	return ret;
}
