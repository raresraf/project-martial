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
 */
static struct bio *iomap_dio_alloc_bio(const struct iomap_iter *iter,
		struct iomap_dio *dio, unsigned short nr_vecs, blk_opf_t opf)
{
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
 */
static void iomap_dio_submit_bio(const struct iomap_iter *iter,
		struct iomap_dio *dio, struct bio *bio, loff_t pos)
{
... The rest of the file ...
}