/**
 * @file seek.c
 * @brief Implements SEEK_HOLE and SEEK_DATA for iomap-based filesystems.
 * @copyright Copyright (C) 2017 Red Hat, Inc.
 * @copyright Copyright (c) 2018-2021 Christoph Hellwig.
 *
 * This file provides generic implementations for the SEEK_HOLE and SEEK_DATA
 * llseek operations. Filesystems that use the iomap interface can use these
 * helpers to efficiently find the next hole or data segment in a file.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2017 Red Hat, Inc.
 * Copyright (c) 2018-2021 Christoph Hellwig.
 */
#include <linux/iomap.h>
#include <linux/pagemap.h>

/**
 * @brief Processes a single iomap extent during a SEEK_HOLE operation.
 * @param iter The iomap iterator.
 * @param hole_pos Pointer to store the position of the found hole.
 * @return 0 to stop, or a positive value to continue iteration.
 *
 * This helper function checks the type of the current iomap extent. If it's
 * a hole, it records the position and stops. If it's an unwritten extent, it
 * consults the page cache to find a hole within it. Otherwise, it advances
 * to the next extent.
 */
static int iomap_seek_hole_iter(struct iomap_iter *iter,
		loff_t *hole_pos)
{
	loff_t length = iomap_length(iter);

	switch (iter->iomap.type) {
	case IOMAP_UNWRITTEN:
		*hole_pos = mapping_seek_hole_data(iter->inode->i_mapping,
				iter->pos, iter->pos + length, SEEK_HOLE);
		if (*hole_pos == iter->pos + length)
			return iomap_iter_advance(iter, &length);
		return 0;
	case IOMAP_HOLE:
		*hole_pos = iter->pos;
		return 0;
	default:
		return iomap_iter_advance(iter, &length);
	}
}

/**
 * @brief Generic implementation of SEEK_HOLE using the iomap interface.
 * @param inode The inode to search in.
 * @param pos The starting offset for the search.
 * @param ops The iomap operations for the filesystem.
 * @return The starting offset of the next hole, or -ENXIO if no hole is found.
 */
loff_t
iomap_seek_hole(struct inode *inode, loff_t pos, const struct iomap_ops *ops)
{
	loff_t size = i_size_read(inode);
	struct iomap_iter iter = {
		.inode	= inode,
		.pos	= pos,
		.flags	= IOMAP_REPORT,
	};
	int ret;

	/* Nothing to be found before or beyond the end of the file. */
	if (pos < 0 || pos >= size)
		return -ENXIO;

	iter.len = size - pos;
	while ((ret = iomap_iter(&iter, ops)) > 0)
		iter.status = iomap_seek_hole_iter(&iter, &pos);
	if (ret < 0)
		return ret;
	if (iter.len) /* found hole before EOF */
		return pos;
	return size;
}
EXPORT_SYMBOL_GPL(iomap_seek_hole);

/**
 * @brief Processes a single iomap extent during a SEEK_DATA operation.
 * @param iter The iomap iterator.
 * @param data_pos Pointer to store the position of the found data.
 * @return 0 to stop, or a positive value to continue iteration.
 *
 * This helper function checks the type of the current iomap extent. If it's
 * mapped or inline data, it records the position and stops. If it's an
 * unwritten extent, it consults the page cache to find a data segment within
 * it. Otherwise, it advances to the next extent.
 */
static int iomap_seek_data_iter(struct iomap_iter *iter,
		loff_t *data_pos)
{
	loff_t length = iomap_length(iter);

	switch (iter->iomap.type) {
	case IOMAP_HOLE:
		return iomap_iter_advance(iter, &length);
	case IOMAP_UNWRITTEN:
		*data_pos = mapping_seek_hole_data(iter->inode->i_mapping,
				iter->pos, iter->pos + length, SEEK_DATA);
		if (*data_pos < 0)
			return iomap_iter_advance(iter, &length);
		return 0;
	default:
		*data_pos = iter->pos;
		return 0;
	}
}

/**
 * @brief Generic implementation of SEEK_DATA using the iomap interface.
 * @param inode The inode to search in.
 * @param pos The starting offset for the search.
 * @param ops The iomap operations for the filesystem.
 * @return The starting offset of the next data segment, or -ENXIO if no data
 * is found.
 */
loff_t
iomap_seek_data(struct inode *inode, loff_t pos, const struct iomap_ops *ops)
{
	loff_t size = i_size_read(inode);
	struct iomap_iter iter = {
		.inode	= inode,
		.pos	= pos,
		.flags	= IOMAP_REPORT,
	};
	int ret;

	/* Nothing to be found before or beyond the end of the file. */
	if (pos < 0 || pos >= size)
		return -ENXIO;

	iter.len = size - pos;
	while ((ret = iomap_iter(&iter, ops)) > 0)
		iter.status = iomap_seek_data_iter(&iter, &pos);
	if (ret < 0)
		return ret;
	if (iter.len) /* found data before EOF */
		return pos;
	/* We've reached the end of the file without finding data */
	return -ENXIO;
}
EXPORT_SYMBOL_GPL(iomap_seek_data);