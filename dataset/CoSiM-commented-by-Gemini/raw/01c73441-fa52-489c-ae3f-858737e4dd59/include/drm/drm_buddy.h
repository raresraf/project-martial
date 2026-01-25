/**
 * @file drm_buddy.h
 * @brief DRM Buddy Allocator
 *
 * This header file defines the interface for a generic binary buddy system
 * allocator, designed for managing memory within the DRM (Direct Rendering
 * Manager) subsystem. The buddy system is a memory allocation algorithm that
 * allocates and deallocates blocks of memory of sizes that are powers of two.
 *
 * This implementation is used by drivers like `amdgpu` to manage VRAM and GTT
 * (Graphics Translation Table) address space, providing an efficient way to
 * handle allocations of various sizes while minimizing fragmentation.
 */
/* SPDX-License-Identifier: MIT */
/*
 * Copyright © 2021 Intel Corporation
 */

#ifndef __DRM_BUDDY_H__
#define __DRM_BUDDY_H__

#include <linux/bitops.h>
#include <linux/list.h>
#include <linux/slab.h>
#include <linux/sched.h>

#include <drm/drm_print.h>

#define range_overflows(start, size, max) ({ \
	typedef typeof(start) start__ = (start); \
	typedef typeof(size) size__ = (size); \
	typedef typeof(max) max__ = (max); \
	(void)(&start__ == &size__); \
	(void)(&start__ == &max__); \
	start__ >= max__ || size__ > max__ - start__; \
})

#define DRM_BUDDY_RANGE_ALLOCATION	BIT(0)
#define DRM_BUDDY_TOPDOWN_ALLOCATION	BIT(1)
#define DRM_BUDDY_CONTIGUOUS_ALLOCATION	BIT(2)
#define DRM_BUDDY_CLEAR_ALLOCATION	BIT(3)
#define DRM_BUDDY_CLEARED		BIT(4)
#define DRM_BUDDY_TRIM_DISABLE		BIT(5)

/**
 * @struct drm_buddy_block
 * @brief A block of memory managed by the buddy allocator.
 *
 * Each `drm_buddy_block` represents a node in the binary tree of memory
 * blocks. It contains metadata about the block's state (free, allocated, or
 * split), its order (size as a power of two), and its position within the
 * larger memory space.
 *
 * @header: A 64-bit field containing the block's offset, state, and order.
 * @left: Pointer to the left child block.
 * @right: Pointer to the right child block.
 * @parent: Pointer to the parent block.
 * @private: A private pointer for the user of the allocator.
 * @link: A list head for linking the block into free lists or user lists.
 * @tmp_link: A temporary list head for internal use during allocation.
 */
struct drm_buddy_block {
#define DRM_BUDDY_HEADER_OFFSET GENMASK_ULL(63, 12)
#define DRM_BUDDY_HEADER_STATE  GENMASK_ULL(11, 10)
#define   DRM_BUDDY_ALLOCATED	   (1 << 10)
#define   DRM_BUDDY_FREE	   (2 << 10)
#define   DRM_BUDDY_SPLIT	   (3 << 10)
#define DRM_BUDDY_HEADER_CLEAR  GENMASK_ULL(9, 9)
/* Free to be used, if needed in the future */
#define DRM_BUDDY_HEADER_UNUSED GENMASK_ULL(8, 6)
#define DRM_BUDDY_HEADER_ORDER  GENMASK_ULL(5, 0)
	u64 header;

	struct drm_buddy_block *left;
	struct drm_buddy_block *right;
	struct drm_buddy_block *parent;

	void *private; /* owned by creator */

	/*
	 * While the block is allocated by the user through drm_buddy_alloc*,
	 * the user has ownership of the link, for example to maintain within
	 * a list, if so desired. As soon as the block is freed with
	 * drm_buddy_free* ownership is given back to the mm.
	 */
	struct list_head link;
	struct list_head tmp_link;
};

/* Order-zero must be at least SZ_4K */
#define DRM_BUDDY_MAX_ORDER (63 - 12)

/**
 * @struct drm_buddy
 * @brief The main buddy allocator structure.
 *
 * This structure represents an instance of the buddy allocator, managing a
 * specific range of memory.
 *
 * @free_list: An array of list heads, one for each order, for tracking free blocks.
 * @roots: An array of root blocks for the binary trees that represent the memory space.
 * @n_roots: The number of root blocks.
 * @max_order: The maximum order of a block that can be allocated.
 * @chunk_size: The size of the smallest block (order 0).
 * @size: The total size of the memory being managed.
 * @avail: The total amount of available memory.
 * @clear_avail: The amount of available memory that is known to be zeroed.
 */
struct drm_buddy {
	/* Maintain a free list for each order. */
	struct list_head *free_list;

	/*
	 * Maintain explicit binary tree(s) to track the allocation of the
	 * address space. This gives us a simple way of finding a buddy block
	 * and performing the potentially recursive merge step when freeing a
	 * block.  Nodes are either allocated or free, in which case they will
	 * also exist on the respective free list.
	 */
	struct drm_buddy_block **roots;

	/*
	 * Anything from here is public, and remains static for the lifetime of
	 * the mm. Everything above is considered do-not-touch.
	 */
	unsigned int n_roots;
	unsigned int max_order;

	/* Must be at least SZ_4K */
	u64 chunk_size;
	u64 size;
	u64 avail;
	u64 clear_avail;
};

static inline u64
drm_buddy_block_offset(struct drm_buddy_block *block)
{
	return block->header & DRM_BUDDY_HEADER_OFFSET;
}

static inline unsigned int
drm_buddy_block_order(struct drm_buddy_block *block)
{
	return block->header & DRM_BUDDY_HEADER_ORDER;
}

static inline unsigned int
drm_buddy_block_state(struct drm_buddy_block *block)
{
	return block->header & DRM_BUDDY_HEADER_STATE;
}

static inline bool
drm_buddy_block_is_allocated(struct drm_buddy_block *block)
{
	return drm_buddy_block_state(block) == DRM_BUDDY_ALLOCATED;
}

static inline bool
drm_buddy_block_is_clear(struct drm_buddy_block *block)
{
	return block->header & DRM_BUDDY_HEADER_CLEAR;
}

static inline bool
drm_buddy_block_is_free(struct drm_buddy_block *block)
{
	return drm_buddy_block_state(block) == DRM_BUDDY_FREE;
}

static inline bool
drm_buddy_block_is_split(struct drm_buddy_block *block)
{
	return drm_buddy_block_state(block) == DRM_BUDDY_SPLIT;
}

static inline u64
drm_buddy_block_size(struct drm_buddy *mm,
		     struct drm_buddy_block *block)
{
	return mm->chunk_size << drm_buddy_block_order(block);
}

int drm_buddy_init(struct drm_buddy *mm, u64 size, u64 chunk_size);

void drm_buddy_fini(struct drm_buddy *mm);

struct drm_buddy_block *
drm_get_buddy(struct drm_buddy_block *block);

int drm_buddy_alloc_blocks(struct drm_buddy *mm,
			   u64 start, u64 end, u64 size,
			   u64 min_page_size,
			   struct list_head *blocks,
			   unsigned long flags);

int drm_buddy_block_trim(struct drm_buddy *mm,
			 u64 *start,
			 u64 new_size,
			 struct list_head *blocks);

void drm_buddy_free_block(struct drm_buddy *mm, struct drm_buddy_block *block);

void drm_buddy_free_list(struct drm_buddy *mm,
			 struct list_head *objects,
			 unsigned int flags);

void drm_buddy_print(struct drm_buddy *mm, struct drm_printer *p);
void drm_buddy_block_print(struct drm_buddy *mm,
			   struct drm_buddy_block *block,
			   struct drm_printer *p);
#endif