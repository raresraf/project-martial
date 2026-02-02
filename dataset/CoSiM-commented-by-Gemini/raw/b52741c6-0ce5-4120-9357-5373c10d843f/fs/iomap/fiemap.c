/**
 * @file fiemap.c
 * @brief Implements fiemap and bmap operations using the iomap interface.
 * @copyright Copyright (c) 2016-2021 Christoph Hellwig.
 *
 * This file provides generic implementations for the FIEMAP (file extent mapping)
 * ioctl and the legacy bmap operation. It leverages the iomap interface to allow
 * filesystems to easily provide this functionality by implementing the iomap_ops
 * callbacks.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2016-2021 Christoph Hellwig.
 */
#include <linux/iomap.h>
#include <linux/fiemap.h>
#include <linux/pagemap.h>

/**
 * @brief Converts an iomap structure to a fiemap extent.
 * @param fi The fiemap extent info structure to fill.
 * @param iomap The source iomap structure.
 * @param flags Additional flags to be set on the fiemap extent.
 * @return 0 if the extent was skipped (hole), 1 if the fiemap extent array is
 * full, or a negative error code.
 *
 * This helper function translates the iomap extent types (hole, delalloc, etc.)
 * and flags into their corresponding fiemap equivalents.
 */
static int iomap_to_fiemap(struct fiemap_extent_info *fi,
		const struct iomap *iomap, u32 flags)
{
	switch (iomap->type) {
	case IOMAP_HOLE:
		/* skip holes */
		return 0;
	case IOMAP_DELALLOC:
		flags |= FIEMAP_EXTENT_DELALLOC | FIEMAP_EXTENT_UNKNOWN;
		break;
	case IOMAP_MAPPED:
		break;
	case IOMAP_UNWRITTEN:
		flags |= FIEMAP_EXTENT_UNWRITTEN;
		break;
	case IOMAP_INLINE:
		flags |= FIEMAP_EXTENT_DATA_INLINE;
		break;
	}

	if (iomap->flags & IOMAP_F_MERGED)
		flags |= FIEMAP_EXTENT_MERGED;
	if (iomap->flags & IOMAP_F_SHARED)
		flags |= FIEMAP_EXTENT_SHARED;

	return fiemap_fill_next_extent(fi, iomap->offset,
			iomap->addr != IOMAP_NULL_ADDR ? iomap->addr : 0,
			iomap->length, flags);
}

/**
 * @brief Processes a single iomap extent during a fiemap iteration.
 * @param iter The iomap iterator.
 * @param fi The fiemap extent info structure.
 * @param prev Pointer to the previously processed iomap extent.
 * @return 0 to continue, 1 to stop (array full), or a negative error code.
 */
static int iomap_fiemap_iter(struct iomap_iter *iter,
		struct fiemap_extent_info *fi, struct iomap *prev)
{
	int ret;

	if (iter->iomap.type == IOMAP_HOLE)
		goto advance;

	ret = iomap_to_fiemap(fi, prev, 0);
	*prev = iter->iomap;
	if (ret < 0)
		return ret;
	if (ret == 1)	/* extent array full */
		return 0;

advance:
	return iomap_iter_advance_full(iter);
}

/**
 * @brief Generic implementation of the FIEMAP ioctl using iomap.
 * @param inode The inode to map.
 * @param fi The fiemap extent info structure provided by the user.
 * @param start The starting offset for the mapping.
 * @param len The length of the range to map.
 * @param ops The iomap operations for the filesystem.
 * @return 0 on success, or a negative error code.
 */
int iomap_fiemap(struct inode *inode, struct fiemap_extent_info *fi,
		u64 start, u64 len, const struct iomap_ops *ops)
{
	struct iomap_iter iter = {
		.inode		= inode,
		.pos		= start,
		.len		= len,
		.flags		= IOMAP_REPORT,
	};
	struct iomap prev = {
		.type		= IOMAP_HOLE,
	};
	int ret;

	ret = fiemap_prep(inode, fi, start, &iter.len, 0);
	if (ret)
		return ret;

	while ((ret = iomap_iter(&iter, ops)) > 0)
		iter.status = iomap_fiemap_iter(&iter, fi, &prev);

	if (prev.type != IOMAP_HOLE) {
		ret = iomap_to_fiemap(fi, &prev, FIEMAP_EXTENT_LAST);
		if (ret < 0)
			return ret;
	}

	/* inode with no (attribute) mapping will give ENOENT */
	if (ret < 0 && ret != -ENOENT)
		return ret;
	return 0;
}
EXPORT_SYMBOL_GPL(iomap_fiemap);

/**
 * @brief Generic implementation of the legacy ->bmap address space operation.
 * @param mapping The address space of the file.
 * @param bno The logical block number to map.
 * @param ops The iomap operations for the filesystem.
 * @return The physical block number, or 0 on error or if the block is not mapped.
 */
sector_t
iomap_bmap(struct address_space *mapping, sector_t bno,
		const struct iomap_ops *ops)
{
	struct iomap_iter iter = {
		.inode	= mapping->host,
		.pos	= (loff_t)bno << mapping->host->i_blkbits,
		.len	= i_blocksize(mapping->host),
		.flags	= IOMAP_REPORT,
	};
	const unsigned int blkshift = mapping->host->i_blkbits - SECTOR_SHIFT;
	int ret;

	if (filemap_write_and_wait(mapping))
		return 0;

	bno = 0;
	while ((ret = iomap_iter(&iter, ops)) > 0) {
		if (iter.iomap.type == IOMAP_MAPPED)
			bno = iomap_sector(&iter.iomap, iter.pos) >> blkshift;
		/* leave iter.status unset to abort loop */
	}
	if (ret)
		return 0;

	return bno;
}
EXPORT_SYMBOL_GPL(iomap_bmap);