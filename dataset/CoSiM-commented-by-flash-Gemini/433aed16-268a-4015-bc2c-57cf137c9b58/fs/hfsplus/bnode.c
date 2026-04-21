// SPDX-License-Identifier: GPL-2.0
/*
 *  linux/fs/hfsplus/bnode.c
 *
 * Copyright (C) 2001
 * Brad Boyer (flar@allandria.com)
 * (C) 2003 Ardis Technologies <roman@ardistech.com>
 *
 * Handle basic btree node operations
 */

/**
 * @file bnode.c
 * @brief Low-level B-tree node management for the HFS+ filesystem.
 * 
 * Functional Intent: Provides core primitives for reading, writing, moving, 
 * and caching B-tree nodes. Manages the mapping between logical B-tree 
 * structures and the underlying page cache, ensuring data consistency 
 * through atomic reference counting and hash-based node tracking.
 * 
 * Domain: Production Systems, Kernel File Systems, Storage Management.
 */

#include <linux/string.h>
#include <linux/slab.h>
#include <linux/pagemap.h>
#include <linux/fs.h>
#include <linux/swap.h>

#include "hfsplus_fs.h"
#include "hfsplus_raw.h"

/**
 * hfs_bnode_read - Copy a specified range of bytes from the raw data of a node.
 * 
 * Algorithm: Page-boundary aware sequential copy.
 * Logic: Spans multiple memory pages if the requested range crosses a 4KB boundary.
 */
void hfs_bnode_read(struct hfs_bnode *node, void *buf, int off, int len)
{
	struct page **pagep;
	int l;

	off += node->page_offset;
	pagep = node->page + (off >> PAGE_SHIFT);
	off &= ~PAGE_MASK;

	l = min_t(int, len, PAGE_SIZE - off);
	memcpy_from_page(buf, *pagep, off, l);

	/**
	 * Block Logic: Multi-page retrieval loop.
	 * Invariant: Successfully transfers 'len' bytes by iterating through the node's page array.
	 */
	while ((len -= l) != 0) {
		buf += l;
		l = min_t(int, len, PAGE_SIZE);
		memcpy_from_page(buf, *++pagep, 0, l);
	}
}

/**
 * hfs_bnode_read_u16 - Reads a big-endian 16-bit integer and converts to CPU endianness.
 */
u16 hfs_bnode_read_u16(struct hfs_bnode *node, int off)
{
	__be16 data;
	hfs_bnode_read(node, &data, off, 2);
	return be16_to_cpu(data);
}

u8 hfs_bnode_read_u8(struct hfs_bnode *node, int off)
{
	u8 data;
	hfs_bnode_read(node, &data, off, 1);
	return data;
}

/**
 * hfs_bnode_read_key - Extracts a B-tree key from a node, handling variable length formats.
 * 
 * Logic: Resolves key length based on node type (Leaf vs Index) and tree attributes.
 */
void hfs_bnode_read_key(struct hfs_bnode *node, void *key, int off)
{
	struct hfs_btree *tree;
	int key_len;

	tree = node->tree;
	if (node->type == HFS_NODE_LEAF ||
	    tree->attributes & HFS_TREE_VARIDXKEYS ||
	    node->tree->cnid == HFSPLUS_ATTR_CNID)
		key_len = hfs_bnode_read_u16(node, off) + 2;
	else
		key_len = tree->max_key_len + 2;

	// Pre-condition: Key length must be within architectural limits.
	if (key_len > sizeof(hfsplus_btree_key) || key_len < 1) {
		memset(key, 0, sizeof(hfsplus_btree_key));
		pr_err("hfsplus: Invalid key length: %d\n", key_len);
		return;
	}

	hfs_bnode_read(node, key, off, key_len);
}

/**
 * hfs_bnode_write - Commits data to a B-tree node and marks pages as dirty.
 */
void hfs_bnode_write(struct hfs_bnode *node, void *buf, int off, int len)
{
	struct page **pagep;
	int l;

	off += node->page_offset;
	pagep = node->page + (off >> PAGE_SHIFT);
	off &= ~PAGE_MASK;

	l = min_t(int, len, PAGE_SIZE - off);
	memcpy_to_page(*pagep, off, buf, l);
	set_page_dirty(*pagep);

	while ((len -= l) != 0) {
		buf += l;
		l = min_t(int, len, PAGE_SIZE);
		memcpy_to_page(*++pagep, 0, buf, l);
		set_page_dirty(*++pagep);
	}
}

void hfs_bnode_write_u16(struct hfs_bnode *node, int off, u16 data)
{
	__be16 v = cpu_to_be16(data);
	hfs_bnode_write(node, &v, off, 2);
}

/**
 * hfs_bnode_clear - Zeroes out a range of bytes in a node.
 */
void hfs_bnode_clear(struct hfs_bnode *node, int off, int len)
{
	struct page **pagep;
	int l;

	off += node->page_offset;
	pagep = node->page + (off >> PAGE_SHIFT);
	off &= ~PAGE_MASK;

	l = min_t(int, len, PAGE_SIZE - off);
	memzero_page(*pagep, off, l);
	set_page_dirty(*pagep);

	while ((len -= l) != 0) {
		l = min_t(int, len, PAGE_SIZE);
		memzero_page(*++pagep, 0, l);
		set_page_dirty(*++pagep);
	}
}

/**
 * hfs_bnode_copy - Efficiently copies data between two B-tree nodes.
 * 
 * Logic: Optimizes for the 'same page offset' case using page-level memcpy, 
 * falling back to local mapping for misaligned offsets.
 */
void hfs_bnode_copy(struct hfs_bnode *dst_node, int dst,
		    struct hfs_bnode *src_node, int src, int len)
{
	struct page **src_page, **dst_page;
	int l;

	hfs_dbg(BNODE_MOD, "copybytes: %u,%u,%u\n", dst, src, len);
	if (!len)
		return;
	src += src_node->page_offset;
	dst += dst_node->page_offset;
	src_page = src_node->page + (src >> PAGE_SHIFT);
	src &= ~PAGE_MASK;
	dst_page = dst_node->page + (dst >> PAGE_SHIFT);
	dst &= ~PAGE_MASK;

	if (src == dst) {
		// Optimization: Aligned copy allows direct page-to-page transfer.
		l = min_t(int, len, PAGE_SIZE - src);
		memcpy_page(*dst_page, src, *src_page, src, l);
		set_page_dirty(*dst_page);

		while ((len -= l) != 0) {
			l = min_t(int, len, PAGE_SIZE);
			memcpy_page(*++dst_page, 0, *++src_page, 0, l);
			set_page_dirty(*dst_page);
		}
	} else {
		void *src_ptr, *dst_ptr;

		/**
		 * Block Logic: Misaligned copy loop.
		 * Logic: Maps pages into kernel address space temporarily to perform 
		 * partial transfers across mismatched page boundaries.
		 */
		do {
			dst_ptr = kmap_local_page(*dst_page) + dst;
			src_ptr = kmap_local_page(*src_page) + src;
			if (PAGE_SIZE - src < PAGE_SIZE - dst) {
				l = PAGE_SIZE - src;
				src = 0;
				dst += l;
			} else {
				l = PAGE_SIZE - dst;
				src += l;
				dst = 0;
			}
			l = min(len, l);
			memcpy(dst_ptr, src_ptr, l);
			kunmap_local(src_ptr);
			set_page_dirty(*dst_page);
			kunmap_local(dst_ptr);
			if (!dst)
				dst_page++;
			else
				src_page++;
		} while ((len -= l));
	}
}

/**
 * hfs_bnode_move - In-place data movement within a single B-tree node.
 * 
 * Algorithm: Overlap-safe memory movement (memmove equivalent for page cache).
 * Logic: Accounts for forward vs backward movement to prevent source data 
 * corruption when source and destination ranges overlap.
 */
void hfs_bnode_move(struct hfs_bnode *node, int dst, int src, int len)
{
	struct page **src_page, **dst_page;
	void *src_ptr, *dst_ptr;
	int l;

	hfs_dbg(BNODE_MOD, "movebytes: %u,%u,%u\n", dst, src, len);
	if (!len)
		return;
	src += node->page_offset;
	dst += node->page_offset;
	
	if (dst > src) {
		// Logic: Backward move (tail-to-head) to handle overlapping regions safely.
		src += len - 1;
		src_page = node->page + (src >> PAGE_SHIFT);
		src = (src & ~PAGE_MASK) + 1;
		dst += len - 1;
		dst_page = node->page + (dst >> PAGE_SHIFT);
		dst = (dst & ~PAGE_MASK) + 1;

		if (src == dst) {
			while (src < len) {
				dst_ptr = kmap_local_page(*dst_page);
				src_ptr = kmap_local_page(*src_page);
				memmove(dst_ptr, src_ptr, src);
				kunmap_local(src_ptr);
				set_page_dirty(*dst_page);
				kunmap_local(dst_ptr);
				len -= src;
				src = PAGE_SIZE;
				src_page--;
				dst_page--;
			}
			src -= len;
			dst_ptr = kmap_local_page(*dst_page);
			src_ptr = kmap_local_page(*src_page);
			memmove(dst_ptr + src, src_ptr + src, len);
			kunmap_local(src_ptr);
			set_page_dirty(*dst_page);
			kunmap_local(dst_ptr);
		} else {
			do {
				dst_ptr = kmap_local_page(*dst_page) + dst;
				src_ptr = kmap_local_page(*src_page) + src;
				if (src < dst) {
					l = src;
					src = PAGE_SIZE;
					dst -= l;
				} else {
					l = dst;
					src -= l;
					dst = PAGE_SIZE;
				}
				l = min(len, l);
				memmove(dst_ptr - l, src_ptr - l, l);
				kunmap_local(src_ptr);
				set_page_dirty(*dst_page);
				kunmap_local(dst_ptr);
				if (dst == PAGE_SIZE)
					dst_page--;
				else
					src_page--;
			} while ((len -= l));
		}
	} else {
		// Logic: Forward move (head-to-tail).
		src_page = node->page + (src >> PAGE_SHIFT);
		src &= ~PAGE_MASK;
		dst_page = node->page + (dst >> PAGE_SHIFT);
		dst &= ~PAGE_MASK;

		if (src == dst) {
			l = min_t(int, len, PAGE_SIZE - src);

			dst_ptr = kmap_local_page(*dst_page) + src;
			src_ptr = kmap_local_page(*src_page) + src;
			memmove(dst_ptr, src_ptr, l);
			kunmap_local(src_ptr);
			set_page_dirty(*dst_page);
			kunmap_local(dst_ptr);

			while ((len -= l) != 0) {
				l = min_t(int, len, PAGE_SIZE);
				dst_ptr = kmap_local_page(*++dst_page);
				src_ptr = kmap_local_page(*++src_page);
				memmove(dst_ptr, src_ptr, l);
				kunmap_local(src_ptr);
				set_page_dirty(*dst_page);
				kunmap_local(dst_ptr);
			}
		} else {
			do {
				dst_ptr = kmap_local_page(*dst_page) + dst;
				src_ptr = kmap_local_page(*src_page) + src;
				if (PAGE_SIZE - src <
						PAGE_SIZE - dst) {
					l = PAGE_SIZE - src;
					src = 0;
					dst += l;
				} else {
					l = PAGE_SIZE - dst;
					src += l;
					dst = 0;
				}
				l = min(len, l);
				memmove(dst_ptr, src_ptr, l);
				kunmap_local(src_ptr);
				set_page_dirty(*dst_page);
				kunmap_local(dst_ptr);
				if (!dst)
					dst_page++;
				else
					src_page++;
			} while ((len -= l));
		}
	}
}

/**
 * hfs_bnode_unlink - Removes a node from its B-tree level sequence.
 * 
 * Logic: Adjusts 'next' and 'prev' pointers of adjacent nodes to bypass the current node.
 */
void hfs_bnode_unlink(struct hfs_bnode *node)
{
	struct hfs_btree *tree;
	struct hfs_bnode *tmp;
	__be32 cnid;

	tree = node->tree;
	if (node->prev) {
		tmp = hfs_bnode_find(tree, node->prev);
		if (IS_ERR(tmp))
			return;
		tmp->next = node->next;
		cnid = cpu_to_be32(tmp->next);
		hfs_bnode_write(tmp, &cnid,
			offsetof(struct hfs_bnode_desc, next), 4);
		hfs_bnode_put(tmp);
	} else if (node->type == HFS_NODE_LEAF)
		tree->leaf_head = node->next;

	if (node->next) {
		tmp = hfs_bnode_find(tree, node->next);
		if (IS_ERR(tmp))
			return;
		tmp->prev = node->prev;
		cnid = cpu_to_be32(tmp->prev);
		hfs_bnode_write(tmp, &cnid,
			offsetof(struct hfs_bnode_desc, prev), 4);
		hfs_bnode_put(tmp);
	} else if (node->type == HFS_NODE_LEAF)
		tree->leaf_tail = node->prev;

	if (!node->prev && !node->next)
		hfs_dbg(BNODE_MOD, "hfs_btree_del_level\n");
	if (!node->parent) {
		tree->root = 0;
		tree->depth = 0;
	}
	set_bit(HFS_BNODE_DELETED, &node->flags);
}

/**
 * hfs_bnode_hash - Simple dispersion hash for node CNIDs.
 */
static inline int hfs_bnode_hash(u32 num)
{
	num = (num >> 16) + num;
	num += num >> 8;
	return num & (NODE_HASH_SIZE - 1);
}

/**
 * hfs_bnode_findhash - Fast lookup for nodes already cached in memory.
 */
struct hfs_bnode *hfs_bnode_findhash(struct hfs_btree *tree, u32 cnid)
{
	struct hfs_bnode *node;

	if (cnid >= tree->node_count) {
		pr_err("request for non-existent node %d in B*Tree\n",
		       cnid);
		return NULL;
	}

	for (node = tree->node_hash[hfs_bnode_hash(cnid)];
			node; node = node->next_hash)
		if (node->this == cnid)
			return node;
	return NULL;
}

/**
 * __hfs_bnode_create - Internal allocator for B-tree nodes.
 * 
 * Logic: Initializes node metadata, inserts into hash for caching, 
 * and triggers page cache reads for data residency.
 */
static struct hfs_bnode *__hfs_bnode_create(struct hfs_btree *tree, u32 cnid)
{
	struct hfs_bnode *node, *node2;
	struct address_space *mapping;
	struct page *page;
	int size, block, i, hash;
	loff_t off;

	if (cnid >= tree->node_count) {
		pr_err("request for non-existent node %d in B*Tree\n",
		       cnid);
		return NULL;
	}

	size = sizeof(struct hfs_bnode) + tree->pages_per_bnode *
		sizeof(struct page *);
	node = kzalloc(size, GFP_KERNEL);
	if (!node)
		return NULL;
	node->tree = tree;
	node->this = cnid;
	set_bit(HFS_BNODE_NEW, &node->flags);
	atomic_set(&node->refcnt, 1);
	
	init_waitqueue_head(&node->lock_wq);
	spin_lock(&tree->hash_lock);
	node2 = hfs_bnode_findhash(tree, cnid);
	if (!node2) {
		hash = hfs_bnode_hash(cnid);
		node->next_hash = tree->node_hash[hash];
		tree->node_hash[hash] = node;
		tree->node_hash_cnt++;
	} else {
		spin_unlock(&tree->hash_lock);
		kfree(node);
		// Logic: Wait for concurrently created node to finish initialization.
		wait_event(node2->lock_wq,
			!test_bit(HFS_BNODE_NEW, &node2->flags));
		return node2;
	}
	spin_unlock(&tree->hash_lock);

	mapping = tree->inode->i_mapping;
	off = (loff_t)cnid << tree->node_size_shift;
	block = off >> PAGE_SHIFT;
	node->page_offset = off & ~PAGE_MASK;
	
	// Block Logic: Page residency fulfillment.
	for (i = 0; i < tree->pages_per_bnode; block++, i++) {
		page = read_mapping_page(mapping, block, NULL);
		if (IS_ERR(page))
			goto fail;
		node->page[i] = page;
	}

	return node;
fail:
	set_bit(HFS_BNODE_ERROR, &node->flags);
	return node;
}

/**
 * hfs_bnode_find - Comprehensive node resolution (cache or disk).
 * 
 * Algorithm: Check hash cache -> Create/Read from disk -> Validate structure.
 */
struct hfs_bnode *hfs_bnode_find(struct hfs_btree *tree, u32 num)
{
	struct hfs_bnode *node;
	struct hfs_bnode_desc *desc;
	int i, rec_off, off, next_off;
	int entry_size, key_size;

	spin_lock(&tree->hash_lock);
	node = hfs_bnode_findhash(tree, num);
	if (node) {
		hfs_bnode_get(node);
		spin_unlock(&tree->hash_lock);
		wait_event(node->lock_wq,
			!test_bit(HFS_BNODE_NEW, &node->flags));
		if (test_bit(HFS_BNODE_ERROR, &node->flags))
			goto node_error;
		return node;
	}
	spin_unlock(&tree->hash_lock);
	
	node = __hfs_bnode_create(tree, num);
	if (!node)
		return ERR_PTR(-ENOMEM);
	if (test_bit(HFS_BNODE_ERROR, &node->flags))
		goto node_error;
	if (!test_bit(HFS_BNODE_NEW, &node->flags))
		return node;

	// Block Logic: On-disk structure validation.
	desc = (struct hfs_bnode_desc *)(kmap_local_page(node->page[0]) +
							 node->page_offset);
	node->prev = be32_to_cpu(desc->prev);
	node->next = be32_to_cpu(desc->next);
	node->num_recs = be16_to_cpu(desc->num_recs);
	node->type = desc->type;
	node->height = desc->height;
	kunmap_local(desc);

	// Block Logic: Consistency checks for node height vs type.
	switch (node->type) {
	case HFS_NODE_HEADER:
	case HFS_NODE_MAP:
		if (node->height != 0)
			goto node_error;
		break;
	case HFS_NODE_LEAF:
		if (node->height != 1)
			goto node_error;
		break;
	case HFS_NODE_INDEX:
		if (node->height <= 1 || node->height > tree->depth)
			goto node_error;
		break;
	default:
		goto node_error;
	}

	// Block Logic: Record index validation.
	rec_off = tree->node_size - 2;
	off = hfs_bnode_read_u16(node, rec_off);
	if (off != sizeof(struct hfs_bnode_desc))
		goto node_error;
	for (i = 1; i <= node->num_recs; off = next_off, i++) {
		rec_off -= 2;
		next_off = hfs_bnode_read_u16(node, rec_off);
		if (next_off <= off ||
		    next_off > tree->node_size ||
		    next_off & 1)
			goto node_error;
		entry_size = next_off - off;
		if (node->type != HFS_NODE_INDEX &&
		    node->type != HFS_NODE_LEAF)
			continue;
		key_size = hfs_bnode_read_u16(node, off) + 2;
		if (key_size >= entry_size || key_size & 1)
			goto node_error;
	}
	clear_bit(HFS_BNODE_NEW, &node->flags);
	wake_up(&node->lock_wq);
	return node;

node_error:
	set_bit(HFS_BNODE_ERROR, &node->flags);
	clear_bit(HFS_BNODE_NEW, &node->flags);
	wake_up(&node->lock_wq);
	hfs_bnode_put(node);
	return ERR_PTR(-EIO);
}

/**
 * hfs_bnode_put - Decrements node reference count and performs cleanup if necessary.
 * 
 * Logic: Accesses dirty/access flags on pages and triggers deletion workflow 
 * if the node was previously unlinked.
 */
void hfs_bnode_put(struct hfs_bnode *node)
{
	if (node) {
		struct hfs_btree *tree = node->tree;
		int i;

		BUG_ON(!atomic_read(&node->refcnt));
		if (!atomic_dec_and_lock(&node->refcnt, &tree->hash_lock))
			return;
		for (i = 0; i < tree->pages_per_bnode; i++) {
			if (!node->page[i])
				continue;
			mark_page_accessed(node->page[i]);
		}

		if (test_bit(HFS_BNODE_DELETED, &node->flags)) {
			hfs_bnode_unhash(node);
			spin_unlock(&tree->hash_lock);
			if (hfs_bnode_need_zeroout(tree))
				hfs_bnode_clear(node, 0, tree->node_size);
			hfs_bmap_free(node);
			hfs_bnode_free(node);
			return;
		}
		spin_unlock(&tree->hash_lock);
	}
}

/**
 * hfs_bnode_need_zeroout - Policy check for security-conscious node clearing.
 * 
 * Functional Intent: Ensures deleted catalog nodes are zeroed if the volume fix flag is set.
 */
bool hfs_bnode_need_zeroout(struct hfs_btree *tree)
{
	struct super_block *sb = tree->inode->i_sb;
	struct hfsplus_sb_info *sbi = HFSPLUS_SB(sb);
	const u32 volume_attr = be32_to_cpu(sbi->s_vhdr->attributes);

	return tree->cnid == HFSPLUS_CAT_CNID &&
		volume_attr & HFSPLUS_VOL_UNUSED_NODE_FIX;
}
