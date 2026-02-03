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

/**
 * @brief Resets the iomap and srcmap members of the iterator.
 *
 * This function is called at the beginning of each iteration to clear the
 * previous mapping information.
 *
 * @param iter The iomap iterator.
 */
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
 *
 * This function is used by iomap consumers to move the iterator forward within
 * the current mapping. It ensures that the advance does not go beyond the
 * bounds of the current mapping.
 */
int iomap_iter_advance(struct iomap_iter *iter, u64 *count)
{
	// Pre-condition: Checks for an invalid request to advance beyond the current
	// mapping's length.
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
 * for the next iteration. It also traces the new mapping information for
 * debugging and performance analysis.
 */
static inline void iomap_iter_done(struct iomap_iter *iter)
{
	// Invariant: The returned mapping must be valid and within the requested range.
	WARN_ON_ONCE(iter->iomap.offset > iter->pos);
	WARN_ON_ONCE(iter->iomap.length == 0);
	WARN_ON_ONCE(iter->iomap.offset + iter->iomap.length <= iter->pos);
	// Invariant: Stale mappings should not be returned directly by the iterator.
	WARN_ON_ONCE(iter->iomap.flags & IOMAP_F_STALE);

	iter->iter_start_pos = iter->pos;

	trace_iomap_iter_dstmap(iter->inode, &iter->iomap);
	// Pre-condition: Only trace the source map if it's not a hole.
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
	// Pre-condition: The loop continues as long as there is remaining length
	// to be mapped.
	while (iter->len > 0) {
		int ret;

		iomap_iter_reset_iomap(iter);
		ret = ops->iomap_begin(iter->inode, iter->pos, iter->len,
				iter->flags, &iter->iomap, &iter->srcmap);

		// Post-condition: If iomap_begin returns 0, it signifies the end of
		// the iteration. Any other value indicates a valid mapping or an error.
		if (ret) {
			iter->status = ret;
			return ret;
		}

		// Pre-condition: A valid mapping must be returned if the status is 0.
		if (WARN_ON_ONCE(iter->iomap.type == 0)) {
			iter->status = -EIO;
			return -EIO;
		}
		// Invariant: The returned mapping's offset must match the requested position.
		if (WARN_ON_ONCE(iter->iomap.offset != iter->pos)) {
			iter->status = -EIO;
			return -EIO;
		}

		iomap_iter_done(iter);
		return 1;
	}
	return 0;
}
