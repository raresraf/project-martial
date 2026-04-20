/* SPDX-License-Identifier: MIT */
/*
 * @01c73441-fa52-489c-ae3f-858737e4dd59/include/drm/drm_buddy.h
 * @brief Public API for the DRM Buddy Allocator.
 *
 * Functional Intent: Implements a binary buddy system for managing contiguous 
 * address spaces in graphics drivers. It facilitates efficient allocation and 
 * deallocation of power-of-two sized blocks, while supporting range-constrained 
 * and contiguous multi-block requests.
 *
 * Domain: Graphics Memory Management, Kernel Resource Allocation.
 *
 * Copyright © 2021 Intel Corporation
 */

#ifndef __DRM_BUDDY_H__
#define __DRM_BUDDY_H__

#include <linux/bitops.h>
#include <linux/list.h>
#include <linux/slab.h>
#include <linux/sched.h>

#include <drm/drm_print.h>

/**
 * Functional Utility: Macro to detect if a requested range exceeds boundaries or wraps.
 */
#define range_overflows(start, size, max) ({ \
	typeof(start) start__ = (start); \
	typeof(size) size__ = (size); \
	typeof(max) max__ = (max); \
	(void)(&start__ == &size__); \
	(void)(&start__ == &max__); \
	start__ >= max__ || size__ > max__ - start__; \
})

/**
 * Allocation Flags:
 * DRM_BUDDY_RANGE_ALLOCATION: Restrict allocation to a specific start/end range.
 * DRM_BUDDY_TOPDOWN_ALLOCATION: Prefer higher addresses.
 * DRM_BUDDY_CONTIGUOUS_ALLOCATION: Ensure the entire request is one contiguous block.
 * DRM_BUDDY_CLEAR_ALLOCATION: Request blocks that are already zero-initialized.
 */
#define DRM_BUDDY_RANGE_ALLOCATION		BIT(0)
#define DRM_BUDDY_TOPDOWN_ALLOCATION		BIT(1)
#define DRM_BUDDY_CONTIGUOUS_ALLOCATION		BIT(2)
#define DRM_BUDDY_CLEAR_ALLOCATION		BIT(3)
#define DRM_BUDDY_CLEARED			BIT(4)
#define DRM_BUDDY_TRIM_DISABLE			BIT(5)

/**
 * struct drm_buddy_block - Atomic unit of the buddy system.
 * @header: Encoded metadata containing offset (bits 63-12), state (bits 11-10), 
 *          clear flag (bit 9), and order (bits 5-0).
 * @left: Pointer to the first child in the binary split.
 * @right: Pointer to the second child in the binary split.
 * @parent: Pointer to the node from which this block was split.
 * 
 * Functional Intent: Represents a node in the buddy tree. State can be 
 * ALLOCATED, FREE, or SPLIT (intermediate node).
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
	 * Functional Utility: Tracks block ownership. 
	 * While allocated, 'link' is available for user-defined list management.
	 * While free, it is managed by the buddy allocator's internal free lists.
	 */
	struct list_head link;
	struct list_head tmp_link;
};

/* Order-zero must be at least SZ_4K */
#define DRM_BUDDY_MAX_ORDER (63 - 12)

/**
 * struct drm_buddy - Buddy system manager instance.
 * @free_list: Array of lists containing free blocks for each supported order.
 * @roots: Array of root nodes for the address space (multiple roots handle 
 *         non-power-of-two total sizes).
 * @chunk_size: Smallest allocatable unit (order 0).
 * @avail: Total free capacity remaining.
 * 
 * Functional Intent: Orchestrates the overall state of a contiguous memory region.
 */
struct drm_buddy {
	/* Maintain a free list for each order. */
	struct list_head *free_list;

	/*
	 * Block Logic: Tracking logic.
	 * Maintains explicit binary tree(s) to track the allocation of the
	 * address space. This gives us a simple way of finding a buddy block
	 * and performing the potentially recursive merge step when freeing a
	 * block.
	 */
	struct drm_buddy_block **roots;

	unsigned int n_roots;
	unsigned int max_order;

	/* Must be at least SZ_4K */
	u64 chunk_size;
	u64 size;
	u64 avail;
	u64 clear_avail;
};

/**
 * Block Logic: Metadata Accessors.
 * Logic: Extracts specific fields from the encoded 64-bit header using bitwise masks.
 */
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

/**
 * Functional Utility: Calculates the byte size of a given block based on its order.
 */
static inline u64
drm_buddy_block_size(struct drm_buddy *mm,
		     struct drm_buddy_block *block)
{
	return mm->chunk_size << drm_buddy_block_order(block);
}

/**
 * External API: Lifecycle Management.
 * Logic: drm_buddy_init initializes the tree structure and free lists. 
 * drm_buddy_fini cleans up all allocated nodes.
 */
int drm_buddy_init(struct drm_buddy *mm, u64 size, u64 chunk_size);
void drm_buddy_fini(struct drm_buddy *mm);

/**
 * External API: Buddy Traversal.
 * Returns: The sibling node of the given block if it exists.
 */
struct drm_buddy_block *
drm_get_buddy(struct drm_buddy_block *block);

/**
 * External API: Allocation and Trimming.
 * Logic: drm_buddy_alloc_blocks searches the tree for available buddies, 
 * splitting nodes as necessary to fulfill the request.
 */
int drm_buddy_alloc_blocks(struct drm_buddy *mm,
			   u64 start, u64 end, u64 size,
			   u64 min_page_size,
			   struct list_head *blocks,
			   unsigned long flags);

int drm_buddy_block_trim(struct drm_buddy *mm,
			 u64 *start,
			 u64 new_size,
			 struct list_head *blocks);

/**
 * External API: Deallocation.
 * Logic: Merges freed blocks with their buddies recursively if the buddy is also free.
 */
void drm_buddy_free_block(struct drm_buddy *mm, struct drm_buddy_block *block);
void drm_buddy_free_list(struct drm_buddy *mm,
			 struct list_head *objects,
			 unsigned int flags);

/**
 * External API: Debugging and Visualization.
 */
void drm_buddy_print(struct drm_buddy *mm, struct drm_printer *p);
void drm_buddy_block_print(struct drm_buddy *mm,
			   struct drm_buddy_block *block,
			   struct drm_printer *p);
#endif
