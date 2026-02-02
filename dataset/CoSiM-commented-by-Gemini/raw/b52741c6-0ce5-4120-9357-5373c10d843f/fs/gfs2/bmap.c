/**
 * @file bmap.c
 * @brief Block mapping and allocation for GFS2.
 * @copyright Copyright (C) Sistina Software, Inc. 1997-2003 All rights reserved.
 * @copyright Copyright (C) 2004-2006 Red Hat, Inc. All rights reserved.
 *
 * This file implements the block mapping logic for the GFS2 file system. This
 * involves translating a logical block offset within a file to a physical block
 * number on the storage device. It handles the complexities of GFS2's metadata
 * structure, which uses a B-tree-like hierarchy of indirect blocks to store
 * block pointers.
 *
 * The implementation includes functions for:
 * - Walking the metadata tree (`find_metapath`, `lookup_metapath`).
 * - Allocating new blocks for a file (`__gfs2_iomap_alloc`).
 * - Unstuffing inodes (converting a file that stores its data directly in the
 *   inode to one that uses external data blocks).
 * - Punching holes in files (deallocating blocks).
 * - Providing the `bmap` and `iomap` interfaces for the VFS.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * Copyright (C) Sistina Software, Inc.  1997-2003 All rights reserved.
 * Copyright (C) 2004-2006 Red Hat, Inc.  All rights reserved.
 */

#include <linux/spinlock.h>
#include <linux/completion.h>
#include <linux/buffer_head.h>
#include <linux/blkdev.h>
#include <linux/gfs2_ondisk.h>
#include <linux/crc32.h>
#include <linux/iomap.h>
#include <linux/ktime.h>

#include "gfs2.h"
#include "incore.h"
#include "bmap.h"
#include "glock.h"
#include "inode.h"
#include "meta_io.h"
#include "quota.h"
#include "rgrp.h"
#include "log.h"
#include "super.h"
#include "trans.h"
#include "dir.h"
#include "util.h"
#include "aops.h"
#include "trace_gfs2.h"

/**
 * @struct metapath
 * @brief Represents a path through the metadata B-tree of an inode.
 *
 * This structure stores the buffer heads of the indirect blocks that form
 * a path from the inode's dinode to a specific data block. It also stores
 * the index at each level of the tree.
 */
struct metapath {
	struct buffer_head *mp_bh[GFS2_MAX_META_HEIGHT];
	__u16 mp_list[GFS2_MAX_META_HEIGHT];
	int mp_fheight; /* find_metapath height */
	int mp_aheight; /* actual height (lookup height) */
};

static int punch_hole(struct gfs2_inode *ip, u64 offset, u64 length);

/**
 * @brief Unstuffs a file's data from its dinode into a newly allocated block.
 * @param ip The GFS2 inode.
 * @param dibh The buffer head containing the on-disk inode.
 * @param block The newly allocated block number to store the data.
 * @param folio The folio representing the first page of the file.
 * @return 0 on success, or a negative error code.
 *
 * This function copies the data from the dinode to the first page of the file,
 * which is then written out to the new block.
 */
static int gfs2_unstuffer_folio(struct gfs2_inode *ip, struct buffer_head *dibh,
			       u64 block, struct folio *folio)
{
... The rest of the file ...
}