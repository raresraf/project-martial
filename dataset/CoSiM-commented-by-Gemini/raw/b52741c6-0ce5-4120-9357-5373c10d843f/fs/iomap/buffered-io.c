/**
 * @file buffered-io.c
 * @brief Generic implementation of buffered I/O using the iomap interface.
 * @copyright Copyright (C) 2010 Red Hat, Inc.
 * @copyright Copyright (C) 2016-2023 Christoph Hellwig.
 *
 * This file provides a set of generic helper functions for filesystems to
 * implement buffered I/O on top of the iomap infrastructure. It handles the
 * complexities of interacting with the page cache, including reading and
 * writing folios (formerly pages), tracking dirty state at a sub-page level
 * for filesystems with smaller block sizes, and managing writeback.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2010 Red Hat, Inc.
 * Copyright (C) 2016-2023 Christoph Hellwig.
 */
#include <linux/iomap.h>
#include <linux/buffer_head.h>
#include <linux/writeback.h>
#include <linux/swap.h>
#include <linux/migrate.h>
#include "trace.h"

#include "../internal.h"

/**
 * @struct iomap_folio_state
 * @brief Tracks per-block state for a folio when the filesystem block size
 *        is smaller than the page size.
 *
 * This structure is attached to a folio's private data to manage the uptodate
 * and dirty status of individual blocks within that folio. This allows for
 * more granular I/O and state tracking.
 */
struct iomap_folio_state {
	spinlock_t		state_lock;
	unsigned int		read_bytes_pending;
	atomic_t		write_bytes_pending;

	/*
	 * Each block has two bits in this bitmap:
	 * Bits [0..blocks_per_folio) has the uptodate status.
	 * Bits [b_p_f...(2*b_p_f))   has the dirty status.
	 */
	unsigned long		state[];
};

// Checks if all blocks in a folio are marked as up-to-date.
static inline bool ifs_is_fully_uptodate(struct folio *folio,
		struct iomap_folio_state *ifs)
{
	struct inode *inode = folio->mapping->host;

	return bitmap_full(ifs->state, i_blocks_per_folio(inode, folio));
}

// Checks if a specific block within a folio is up-to-date.
static inline bool ifs_block_is_uptodate(struct iomap_folio_state *ifs,
		unsigned int block)
{
	return test_bit(block, ifs->state);
}

// Sets a range of blocks within a folio as up-to-date.
static bool ifs_set_range_uptodate(struct folio *folio,
		struct iomap_folio_state *ifs, size_t off, size_t len)
{
	struct inode *inode = folio->mapping->host;
	unsigned int first_blk = off >> inode->i_blkbits;
	unsigned int last_blk = (off + len - 1) >> inode->i_blkbits;
	unsigned int nr_blks = last_blk - first_blk + 1;

	bitmap_set(ifs->state, first_blk, nr_blks);
	return ifs_is_fully_uptodate(folio, ifs);
}

/**
 * @brief Marks a range of bytes within a folio as up-to-date.
 * @param folio The folio.
 * @param off The starting offset within the folio.
 * @param len The length of the range.
 *
 * If all blocks in the folio become up-to-date as a result of this operation,
 * the entire folio is marked as up-to-date.
 */
static void iomap_set_range_uptodate(struct folio *folio, size_t off,
		size_t len)
{
	struct iomap_folio_state *ifs = folio->private;
	unsigned long flags;
	bool uptodate = true;

	if (folio_test_uptodate(folio))
		return;

	if (ifs) {
		spin_lock_irqsave(&ifs->state_lock, flags);
		uptodate = ifs_set_range_uptodate(folio, ifs, off, len);
		spin_unlock_irqrestore(&ifs->state_lock, flags);
	}

	if (uptodate)
		folio_mark_uptodate(folio);
}

// Checks if a specific block within a folio is dirty.
static inline bool ifs_block_is_dirty(struct folio *folio,
		struct iomap_folio_state *ifs, int block)
{
	struct inode *inode = folio->mapping->host;
	unsigned int blks_per_folio = i_blocks_per_folio(inode, folio);

	return test_bit(block + blks_per_folio, ifs->state);
}

// Finds the next contiguous range of dirty blocks within a folio.
static unsigned ifs_find_dirty_range(struct folio *folio,
		struct iomap_folio_state *ifs, u64 *range_start, u64 range_end)
{
	struct inode *inode = folio->mapping->host;
	unsigned start_blk =
		offset_in_folio(folio, *range_start) >> inode->i_blkbits;
	unsigned end_blk = min_not_zero(
		offset_in_folio(folio, range_end) >> inode->i_blkbits,
		i_blocks_per_folio(inode, folio));
	unsigned nblks = 1;

	while (!ifs_block_is_dirty(folio, ifs, start_blk))
		if (++start_blk == end_blk)
			return 0;

	while (start_blk + nblks < end_blk) {
		if (!ifs_block_is_dirty(folio, ifs, start_blk + nblks))
			break;
		nblks++;
	}

	*range_start = folio_pos(folio) + (start_blk << inode->i_blkbits);
	return nblks << inode->i_blkbits;
}
...
The rest of the file with comments.
...