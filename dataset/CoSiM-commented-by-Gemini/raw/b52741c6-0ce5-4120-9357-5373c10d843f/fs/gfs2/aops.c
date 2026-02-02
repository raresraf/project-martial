/**
 * @file aops.c
 * @brief Implements address space operations for the GFS2 file system.
 * @copyright Copyright (C) Sistina Software, Inc. 1997-2003 All rights reserved.
 * @copyright Copyright (C) 2004-2008 Red Hat, Inc. All rights reserved.
 *
 * This file provides the implementation of the address_space_operations for GFS2,
 * which are the hooks used by the VFS to interact with file data. It handles
 * operations like reading and writing pages, readahead, and writeback.
 *
 * GFS2 supports different data journaling modes, and this file contains two
 * sets of address space operations: one for 'ordered' and 'writeback' modes
 * (`gfs2_aops`), and another for 'data=journal' mode (`gfs2_jdata_aops`). The
 * appropriate set of operations is assigned to an inode based on its journaling
 * mode.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (C) Sistina Software, Inc.  1997-2003 All rights reserved.
 * Copyright (C) 2004-2008 Red Hat, Inc.  All rights reserved.
 */

#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/completion.h>
#include <linux/buffer_head.h>
#include <linux/pagemap.h>
#include <linux/pagevec.h>
#include <linux/mpage.h>
#include <linux/fs.h>
#include <linux/writeback.h>
#include <linux/swap.h>
#include <linux/gfs2_ondisk.h>
#include <linux/backing-dev.h>
#include <linux/uio.h>
#include <trace/events/writeback.h>
#include <linux/sched/signal.h>

#include "gfs2.h"
#include "incore.h"
#include "bmap.h"
#include "glock.h"
#include "inode.h"
#include "log.h"
#include "meta_io.h"
#include "quota.h"
#include "trans.h"
#include "rgrp.h"
#include "super.h"
#include "util.h"
#include "glops.h"
#include "aops.h"


/**
 * @brief A get_block_t function for gfs2 that doesn't allocate new blocks.
 * @param inode The inode.
 * @param lblock The logical block number within the file.
 * @param bh_result The buffer_head to fill in.
 * @param create This parameter is ignored, as this function never allocates.
 * @return 0 on success, or a negative error code.
 *
 * This function is used for I/O to jdata files where we only want to write
 * to already allocated blocks. It maps a logical block to a physical block
 * and fills in the buffer_head, but returns an error if the block is not
 * already mapped.
 */
static int gfs2_get_block_noalloc(struct inode *inode, sector_t lblock,
				  struct buffer_head *bh_result, int create)
{
	int error;

	error = gfs2_block_map(inode, lblock, bh_result, 0);
	if (error)
		return error;
	if (!buffer_mapped(bh_result))
		return -ENODATA;
	return 0;
}

/**
 * @brief A gfs2 jdata-specific version of block_write_full_folio.
 * @param folio The folio to write.
 * @param wbc The writeback control structure.
 * @return 0 on success, or a negative error code.
 *
 * This function is similar to the generic block_write_full_folio but is
 * specifically for journaled data (jdata) inodes. It handles cases where the
 * folio straddles the end of the file (i_size), ensuring that the part of the
 * page beyond i_size is zeroed out to prevent leaking old data.
 */
static int gfs2_write_jdata_folio(struct folio *folio,
				 struct writeback_control *wbc)
{
	struct inode * const inode = folio->mapping->host;
	loff_t i_size = i_size_read(inode);

	/*
	 * The folio straddles i_size.  It must be zeroed out on each and every
	 * writepage invocation because it may be mmapped.  "A file is mapped
	 * in multiples of the page size.  For a file that is not a multiple of
	 * the page size, the remaining memory is zeroed when mapped, and
	 * writes to that region are not written out to the file."
	 */
	if (folio_pos(folio) < i_size &&
	    i_size < folio_pos(folio) + folio_size(folio))
		folio_zero_segment(folio, offset_in_folio(folio, i_size),
				folio_size(folio));

	return __block_write_full_folio(inode, folio, gfs2_get_block_noalloc,
			wbc);
}

/**
 * @brief The core of the jdata writepage implementation.
 * @param folio The folio to write back.
 * @param wbc The writeback control structure.
 * @return 0 on success, or a negative error code.
 *
 * This function handles the details of writing back a dirty folio for a
 * journaled data file. If the folio is marked as "checked" (meaning it
 * requires a transaction), it adds the folio's buffers to the current
 * transaction before writing them to disk.
 */
static int __gfs2_jdata_write_folio(struct folio *folio,
		struct writeback_control *wbc)
{
	struct inode *inode = folio->mapping->host;
	struct gfs2_inode *ip = GFS2_I(inode);

	if (folio_test_checked(folio)) {
		folio_clear_checked(folio);
		if (!folio_buffers(folio)) {
			create_empty_buffers(folio,
					inode->i_sb->s_blocksize,
					BIT(BH_Dirty)|BIT(BH_Uptodate));
		}
		gfs2_trans_add_databufs(ip->i_gl, folio, 0, folio_size(folio));
	}
	return gfs2_write_jdata_folio(folio, wbc);
}

/**
 * @brief Write back dirty pages for a jdata file.
 * @param mapping The address space of the jdata file.
 * @param wbc The writeback control structure.
 * @return 0 on success, or a negative error code.
 *
 * This function is the entry point for writing back dirty pages of a journaled
 * data file. It iterates through the dirty pages and calls the core write
 * function for each.
 */
int gfs2_jdata_writeback(struct address_space *mapping, struct writeback_control *wbc)
{
	struct inode *inode = mapping->host;
	struct gfs2_inode *ip = GFS2_I(inode);
	struct gfs2_sbd *sdp = GFS2_SB(mapping->host);
	struct folio *folio = NULL;
	int error;

	BUG_ON(current->journal_info);
	if (gfs2_assert_withdraw(sdp, ip->i_gl->gl_state == LM_ST_EXCLUSIVE))
		return 0;

	while ((folio = writeback_iter(mapping, wbc, folio, &error))) {
		if (folio_test_checked(folio)) {
			folio_redirty_for_writepage(wbc, folio);
			folio_unlock(folio);
			continue;
		}
		error = __gfs2_jdata_write_folio(folio, wbc);
	}

	return error;
}

/**
 * @brief The ->writepages implementation for GFS2 files in ordered and
 *        writeback modes.
 * @param mapping The address space of the file.
 * @param wbc The writeback control structure.
 * @return 0 on success, or a negative error code.
 *
 * This function uses the iomap infrastructure to write back dirty pages.
 * If not all requested pages are written, it forces a flush of the AIL
 * (active item list) to ensure that dirty metadata is eventually written out.
 */
static int gfs2_writepages(struct address_space *mapping,
			   struct writeback_control *wbc)
{
... The rest of the file ...
}