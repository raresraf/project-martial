/**
 * @file bmap.h
 * @brief Header file for GFS2 block mapping and allocation.
 * @copyright Copyright (C) Sistina Software, Inc. 1997-2003 All rights reserved.
 * @copyright Copyright (C) 2004-2006 Red Hat, Inc. All rights reserved.
 *
 * This file contains the function prototypes, inline functions, and data
 * structure definitions for GFS2's block mapping functionality. It defines
 * the interface for operations such as allocating blocks, unstuffing inodes,
 * and handling iomap operations, which are implemented in bmap.c.
 */
/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Copyright (C) Sistina Software, Inc.  1997-2003 All rights reserved.
 * Copyright (C) 2004-2006 Red Hat, Inc.  All rights reserved.
 */

#ifndef __BMAP_DOT_H__
#define __BMAP_DOT_H__

#include <linux/iomap.h>

#include "inode.h"

struct inode;
struct gfs2_inode;
struct page;


/**
 * @brief Calculates the number of metadata and data blocks required for a write.
 * @param ip The GFS2 inode for the file.
 * @param len The number of bytes to be written.
 * @param data_blocks Pointer to store the required number of data blocks.
 * @param ind_blocks Pointer to store the required number of indirect (metadata) blocks.
 *
 * This function provides an estimate for the number of blocks needed for a write
 * operation, which is crucial for transaction and resource reservation.
 */
static inline void gfs2_write_calc_reserv(const struct gfs2_inode *ip,
					  unsigned int len,
					  unsigned int *data_blocks,
					  unsigned int *ind_blocks)
{
	const struct gfs2_sbd *sdp = GFS2_SB(&ip->i_inode);
	unsigned int tmp;

	BUG_ON(gfs2_is_dir(ip));
	*data_blocks = (len >> sdp->sd_sb.sb_bsize_shift) + 3;
	*ind_blocks = 3 * (sdp->sd_max_height - 1);

	for (tmp = *data_blocks; tmp > sdp->sd_diptrs;) {
		tmp = DIV_ROUND_UP(tmp, sdp->sd_inptrs);
		*ind_blocks += tmp;
	}
}

// iomap operations for GFS2 files.
extern const struct iomap_ops gfs2_iomap_ops;
extern const struct iomap_write_ops gfs2_iomap_write_ops;
extern const struct iomap_writeback_ops gfs2_writeback_ops;

/**
 * @brief Unstuffs an inode, moving its data from the dinode to external blocks.
 * @param ip The GFS2 inode to unstuff.
 * @return 0 on success, or a negative error code.
 */
int gfs2_unstuff_dinode(struct gfs2_inode *ip);

/**
 * @brief Maps a logical block of a file to a physical block on disk.
 * @param inode The inode of the file.
 * @param lblock The logical block number.
 * @param bh The buffer_head to be filled with the mapping information.
 * @param create If true, allocate new blocks if necessary.
 * @return 0 on success, or a negative error code.
 */
int gfs2_block_map(struct inode *inode, sector_t lblock,
		   struct buffer_head *bh, int create);

/**
 * @brief Retrieves the iomap for a range of a file.
 * @param inode The inode of the file.
 * @param pos The starting byte offset.
 * @param length The length of the range.
 * @param iomap The iomap structure to be filled.
 * @return 0 on success, or a negative error code.
 */
int gfs2_iomap_get(struct inode *inode, loff_t pos, loff_t length,
		   struct iomap *iomap);

/**
 * @brief Allocates blocks for a range of a file and returns the iomap.
 * @param inode The inode of the file.
 * @param pos The starting byte offset.
 * @param length The length of the range.
 * @param iomap The iomap structure to be filled.
 * @return 0 on success, or a negative error code.
 */
int gfs2_iomap_alloc(struct inode *inode, loff_t pos, loff_t length,
		     struct iomap *iomap);

/**
 * @brief Gets an extent (a contiguous range of blocks) for a file.
 * @param inode The inode.
 * @param lblock The starting logical block number.
 * @param dblock Pointer to store the starting physical block number.
 * @param extlen Pointer to store the length of the extent.
 * @return 0 on success, or a negative error code.
 */
int gfs2_get_extent(struct inode *inode, u64 lblock, u64 *dblock,
		    unsigned int *extlen);

/**
 * @brief Allocates an extent for a file.
 * @param inode The inode.
 * @param lblock The starting logical block number.
 * @param dblock Pointer to store the starting physical block number.
 * @param extlen Pointer to store the length of the extent.
 * @param new Pointer to a boolean that will be true if the extent was newly allocated.
 * @return 0 on success, or a negative error code.
 */
int gfs2_alloc_extent(struct inode *inode, u64 lblock, u64 *dblock,
		      unsigned *extlen, bool *new);

/**
 * @brief Sets the size of an inode, handling truncation or extension.
 * @param inode The inode.
 * @param size The new size.
 * @return 0 on success, or a negative error code.
 */
int gfs2_setattr_size(struct inode *inode, u64 size);

/**
 * @brief Resumes a truncate operation that was interrupted.
 * @param ip The GFS2 inode.
 * @return 0 on success, or a negative error code.
 */
int gfs2_truncatei_resume(struct gfs2_inode *ip);

/**
 * @brief Deallocates all blocks of a file, effectively truncating it to zero.
 * @param ip The GFS2 inode.
 * @return 0 on success, or a negative error code.
 */
int gfs2_file_dealloc(struct gfs2_inode *ip);

/**
 * @brief Checks if a write operation requires block allocation.
 * @param ip The GFS2 inode.
 * @param offset The starting offset of the write.
 * @param len The length of the write.
 * @return 1 if allocation is needed, 0 otherwise, or a negative error code.
 */
int gfs2_write_alloc_required(struct gfs2_inode *ip, u64 offset,
			      unsigned int len);

/**
 * @brief Maps the extents of a journal file.
 * @param sdp The GFS2 superblock.
 * @param jd The journal descriptor.
 * @return 0 on success, or a negative error code.
 */
int gfs2_map_journal_extents(struct gfs2_sbd *sdp, struct gfs2_jdesc *jd);

/**
 * @brief Frees the memory used to store a journal's extents.
 * @param jd The journal descriptor.
 */
void gfs2_free_journal_extents(struct gfs2_jdesc *jd);

/**
 * @brief Punches a hole in a file.
 * @param file The file.
 * @param offset The start offset of the hole.
 * @param length The length of the hole.
 * @return 0 on success, or a negative error code.
 */
int __gfs2_punch_hole(struct file *file, loff_t offset, loff_t length);

#endif /* __BMAP_DOT_H__ */