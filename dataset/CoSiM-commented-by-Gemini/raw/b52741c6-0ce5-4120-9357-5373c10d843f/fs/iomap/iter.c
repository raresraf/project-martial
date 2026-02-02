/**
 * @file iter.c
 * @brief Iomap iterator implementation.
 * @copyright Copyright (C) 2010 Red Hat, Inc.
 * @copyright Copyright (c) 2016-2021 Christoph Hellwig.
 *
 * This file provides the core logic for iterating over the block mappings of a
 * file using the iomap interface. The `iomap_iter` function is the main entry
 * point, which filesystems use to report their extent mappings to the iomap
 * core.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2010 Red Hat, Inc.
 * Copyright (c) 2016-2021 Christoph Hellwig.
 */
#include <linux/iomap.h>
#include "trace.h"

// Resets the iomap and srcmap members of the iterator.
static inline void iomap_iter_reset_iomap(struct iomap_iter *iter)
{
	iter->status = 0;
	memset(&iter->iomap, 0, sizeof(iter->iomap));
	memset(&iter->srcmap, 0, sizeof(iter->srcmap));
}

/**
 * @brief Advances the iterator's position and updates the remaining length.
 * @param iter The iomap iterator.
 * @param count Pointer to the number of bytes to advance. On return, it will
 *              contain the number of bytes remaining in the current mapping.
 * @return 0 on success, or a negative error code if the advance is invalid.
 */
int iomap_iter_advance(struct iomap_iter *iter, u64 *count)
{
	if (WARN_ON_ONCE(*count > iomap_length(iter)))
		return -EIO;
	iter->pos += *count;
	iter->len -= *count;
	*count = iomap_length(iter);
	return 0;
}

/**
 * @brief Finalizes the state of the iterator after a new mapping is found.
 * @param iter The iomap iterator.
 *
 * This function performs some sanity checks and records the starting position
 * for the next iteration. It also traces the new mapping information.
 */
static inline void iomap_iter_done(struct iomap_iter *iter)
{
	WARN_ON_ONCE(iter->iomap.offset > iter->pos);
	WARN_ON_ONCE(iter->iomap.length == 0);
	WARN_ON_ONCE(iter->iomap.offset + iter->iomap.length <= iter->pos);
	WARN_ON_ONCE(iter->iomap.flags & IOMAP_F_STALE);

	iter->iter_start_pos = iter->pos;

	trace_iomap_iter_dstmap(iter->inode, &iter->iomap);
	if (iter->srcmap.type != IOMAP_HOLE)
		trace_iomap_iter_srcmap(iter->inode, &iter->srcmap);
}

/**
 * @brief Iterates over the block mappings for a given range in a file.
 * @param iter The iomap iterator, initialized with the file, position, and length.
 * @param ops The iomap operations vector provided by the filesystem.
 * @return A positive value to continue the iteration, 0 to stop, or a negative
 *         error code.
 *
 * This is the core iterator function. It calls the filesystem's `iomap_begin`
 * operation to get a new block mapping. The caller is expected to process the
 * returned mapping and then call `iomap_iter` again in a loop until it returns
 * a non-positive value.
 */
int iomap_iter(struct iomap_iter *iter, const struct iomap_ops *ops)
{
...
The rest of the file with comments.
...
}