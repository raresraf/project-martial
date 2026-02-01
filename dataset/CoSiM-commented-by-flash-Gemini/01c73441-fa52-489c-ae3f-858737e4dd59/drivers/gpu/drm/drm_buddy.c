/**
 * @file drm_buddy.c
 * @brief Implements a buddy memory allocator for Direct Rendering Manager (DRM) within the Linux kernel.
 * This allocator is designed for efficient management of contiguous memory blocks, primarily for graphics resources.
 * Algorithm: Buddy memory allocation system. Memory blocks are managed in power-of-2 sizes.
 * Key Data Structures: Free lists (arrays of `list_head` for different block orders),
 * `drm_buddy_block` (representing memory blocks with metadata like order, offset, parent, left/right children).
 * Functional Utility: Provides fast allocation and deallocation of fixed-size or contiguous variable-size memory blocks,
 * minimizing external fragmentation. Supports operations like splitting and merging blocks, and tracking allocation state.
 * Time Complexity: Allocation and deallocation are typically O(log N) where N is the total memory size, due to searching free lists.
 * Space Complexity: O(log N) for the free lists, plus O(number_of_allocated_blocks) for block metadata.
 */


#include <kunit/test-bug.h>

#include <linux/kmemleak.h>
#include <linux/module.h>
#include <linux/sizes.h>

#include <drm/drm_buddy.h>

static struct kmem_cache *slab_blocks;

static struct drm_buddy_block *drm_block_alloc(struct drm_buddy *mm,
					       struct drm_buddy_block *parent,
					       unsigned int order,
					       u64 offset)
{
	/// Functional Utility: Allocates a new `drm_buddy_block` structure from the slab allocator and initializes its metadata
	/// (offset, order, parent). This function is a low-level building block for creating and managing memory blocks within the buddy system.
	struct drm_buddy_block *block;

	BUG_ON(order > DRM_BUDDY_MAX_ORDER);

	// Block Logic: Allocate memory for the block from `slab_blocks`.
	// Precondition: `slab_blocks` must be initialized.
	block = kmem_cache_zalloc(slab_blocks, GFP_KERNEL);
	if (!block)
		return NULL;

	// Block Logic: Initialize the block's header with offset and order.
	block->header = offset;
	block->header |= order;
	// Block Logic: Set the parent block.
	block->parent = parent;

	BUG_ON(block->header & DRM_BUDDY_HEADER_UNUSED);
	return block;
}

static void drm_block_free(struct drm_buddy *mm,
			   struct drm_buddy_block *block)
{
	/// Functional Utility: Frees a `drm_buddy_block` structure back to the slab allocator.
	/// This is the counterpart to `drm_block_alloc` and is used to deallocate the metadata associated with a memory block.
	kmem_cache_free(slab_blocks, block);
}

static void list_insert_sorted(struct drm_buddy *mm,
			       struct drm_buddy_block *block)
{
	/// Functional Utility: Inserts a `drm_buddy_block` into the correct position in the appropriate free list,
	/// maintaining the sorted order by offset. This helps in coalescing adjacent free blocks during deallocation.
	struct drm_buddy_block *node;
	struct list_head *head;

	// Block Logic: Get the head of the free list corresponding to the block's order.
	head = &mm->free_list[drm_buddy_block_order(block)];
	// Conditional Logic: If the list is empty, simply add the block.
	if (list_empty(head)) {
		list_add(&block->link, head);
		return;
	}

	// Block Logic: Iterate through the free list to find the correct insertion point to maintain sorted order by offset.
	// Invariant: Blocks in the free list are sorted by their offset.
	list_for_each_entry(node, head, link)
		// Conditional Logic: If the current block's offset is less than the node's offset, we found the insertion point.
		if (drm_buddy_block_offset(block) < drm_buddy_block_offset(node))
			break;

	// Block Logic: Insert the block into the list.
	__list_add(&block->link, node->link.prev, &node->link);
}

static void clear_reset(struct drm_buddy_block *block)
{
	/// Functional Utility: Clears the `DRM_BUDDY_HEADER_CLEAR` flag in a block's header.
	/// This marks the block as "dirty" or requiring initialization.
	block->header &= ~DRM_BUDDY_HEADER_CLEAR;
}

static void mark_cleared(struct drm_buddy_block *block)
{
	/// Functional Utility: Sets the `DRM_BUDDY_HEADER_CLEAR` flag in a block's header.
	/// This marks the block as "cleared" or initialized.
	block->header |= DRM_BUDDY_HEADER_CLEAR;
}

static void mark_allocated(struct drm_buddy_block *block)
{
	/// Functional Utility: Marks a `drm_buddy_block` as allocated by updating its header state
	/// and removing it from its current free list.
	block->header &= ~DRM_BUDDY_HEADER_STATE;
	block->header |= DRM_BUDDY_ALLOCATED;

	list_del(&block->link);
}

static void mark_free(struct drm_buddy *mm,
		      struct drm_buddy_block *block)
{
	/// Functional Utility: Marks a `drm_buddy_block` as free by updating its header state
	/// and inserting it into the appropriate free list in sorted order.
	block->header &= ~DRM_BUDDY_HEADER_STATE;
	block->header |= DRM_BUDDY_FREE;

	list_insert_sorted(mm, block);
}

static void mark_split(struct drm_buddy_block *block)
{
	/// Functional Utility: Marks a `drm_buddy_block` as split (meaning it has been divided into two smaller blocks)
	/// by updating its header state and removing it from its current free list.
	block->header &= ~DRM_BUDDY_HEADER_STATE;
	block->header |= DRM_BUDDY_SPLIT;

	list_del(&block->link);
}

static inline bool overlaps(u64 s1, u64 e1, u64 s2, u64 e2)
{
	/// Functional Utility: Checks if two memory regions, defined by their start and end offsets, overlap.
	/// This is a helper for range-based memory operations.
	return s1 <= e2 && e1 >= s2;
}

static inline bool contains(u64 s1, u64 e1, u64 s2, u64 e2)
{
	/// Functional Utility: Checks if the first memory region, defined by `s1` and `e1`,
	/// completely contains the second memory region, defined by `s2` and `e2`.
	/// This is a helper for range-based memory operations.
	return s1 <= s2 && e1 >= e2;
}

static struct drm_buddy_block *
__get_buddy(struct drm_buddy_block *block)
{
	/// Functional Utility: Retrieves the buddy block for a given `drm_buddy_block`.
	/// In the buddy system, two blocks are "buddies" if they are adjacent and can be merged
	/// to form a larger block of the next higher order.
	struct drm_buddy_block *parent;

	parent = block->parent;
	if (!parent)
		return NULL;

	if (parent->left == block)
		return parent->right;

	return parent->left;
}

static unsigned int __drm_buddy_free(struct drm_buddy *mm,
				     struct drm_buddy_block *block,
				     bool force_merge)
{
	/// Functional Utility: Recursively frees a `drm_buddy_block` and attempts to merge it with its buddy
	/// if both are free and compatible. This function is central to the deallocation and coalescing
	/// process in the buddy system.
	struct drm_buddy_block *parent;
	unsigned int order;

	// Block Logic: Loop upwards through the buddy tree as long as the current block has a parent.
	// Invariant: `block` refers to the block being considered for merging; `parent` is its direct parent.
	while ((parent = block->parent)) {
		struct drm_buddy_block *buddy;

		// Block Logic: Get the buddy of the current block.
		buddy = __get_buddy(block);

		// Conditional Logic: If the buddy is not free, then merging is not possible at this level; break the loop.
		if (!drm_buddy_block_is_free(buddy))
			break;

		// Conditional Logic: If `force_merge` is false, check clear states for compatibility.
		if (!force_merge) {
			/*
			 * Check the block and its buddy clear state and exit
			 * the loop if they both have the dissimilar state.
			 */
			// Conditional Logic: If block and its buddy have dissimilar clear states, break.
			if (drm_buddy_block_is_clear(block) !=
			    drm_buddy_block_is_clear(buddy))
				break;

			// Conditional Logic: If the block is cleared, mark the parent as cleared.
			if (drm_buddy_block_is_clear(block))
				mark_cleared(parent);
		}

		// Block Logic: Remove the buddy from its free list.
		list_del(&buddy->link);
		// Conditional Logic: If force merging and buddy was clear, update `clear_avail`.
		if (force_merge && drm_buddy_block_is_clear(buddy))
			mm->clear_avail -= drm_buddy_block_size(mm, buddy);

		// Block Logic: Free the memory associated with both the current block and its buddy.
		drm_block_free(mm, block);
		drm_block_free(mm, buddy);

		// Block Logic: Move up to the parent block for the next iteration of merging.
		block = parent;
	}

	// Block Logic: Get the order of the final merged block and mark it as free.
	order = drm_buddy_block_order(block);
	mark_free(mm, block);

	return order;
}

static int __force_merge(struct drm_buddy *mm,
			 u64 start,
			 u64 end,
			 unsigned int min_order)
{
	/// Functional Utility: Attempts to force-merge blocks within a specified memory range down to a `min_order`.
	/// This is typically used in cleanup or compaction scenarios where memory needs to be freed up forcefully,
	/// even if clear states are dissimilar.
	unsigned int order;
	int i;

	// Precondition: `min_order` must be non-zero.
	if (!min_order)
		return -ENOMEM;

	// Precondition: `min_order` must not exceed the maximum order.
	if (min_order > mm->max_order)
		return -EINVAL;

	// Block Logic: Iterate downwards through free lists from `min_order - 1` to 0.
	// Invariant: `i` represents the current order of blocks being considered for merging.
	for (i = min_order - 1; i >= 0; i--) {
		struct drm_buddy_block *block, *prev;

		// Block Logic: Iterate safely in reverse through the free list of the current order.
		list_for_each_entry_safe_reverse(block, prev, &mm->free_list[i], link) {
			struct drm_buddy_block *buddy;
			u64 block_start, block_end;

			// Conditional Logic: Skip if the block has no parent (it's a root).
			if (!block->parent)
				continue;

			// Block Logic: Calculate the start and end offsets of the current block.
			block_start = drm_buddy_block_offset(block);
			block_end = block_start + drm_buddy_block_size(mm, block) - 1;

			// Conditional Logic: Skip if the block is not contained within the target range.
			if (!contains(start, end, block_start, block_end))
				continue;

			// Block Logic: Get the buddy of the current block.
			buddy = __get_buddy(block);
			// Conditional Logic: Skip if the buddy is not free.
			if (!drm_buddy_block_is_free(buddy))
				continue;

			// Warning: This indicates a bug if block and buddy have dissimilar clear states during a force merge.
			WARN_ON(drm_buddy_block_is_clear(block) ==
				drm_buddy_block_is_clear(buddy));

			/*
			 * If the prev block is same as buddy, don't access the
			 * block in the next iteration as we would free the
			 * buddy block as part of the free function.
			 */
			// Conditional Logic: Adjust `prev` pointer if it's the same as `buddy` to prevent use-after-free.
			if (prev == buddy)
				prev = list_prev_entry(prev, link);

			// Block Logic: Remove the block from its free list.
			list_del(&block->link);
			// Conditional Logic: If the block was cleared, update `clear_avail`.
			if (drm_buddy_block_is_clear(block))
				mm->clear_avail -= drm_buddy_block_size(mm, block);

			// Block Logic: Recursively free the block, forcing a merge.
			order = __drm_buddy_free(mm, block, true);
			// Conditional Logic: If the merged block's order is greater than or equal to `min_order`, success.
			if (order >= min_order)
				return 0;
		}
	}

	return -ENOMEM;
}

/**
 * drm_buddy_init - init memory manager
 *
 * @mm: DRM buddy manager to initialize
 * @size: size in bytes to manage
 * @chunk_size: minimum page size in bytes for our allocations
 *
 * Initializes the memory manager and its resources.
 *
 * Returns:
 * 0 on success, error code on failure.
 */
int drm_buddy_init(struct drm_buddy *mm, u64 size, u64 chunk_size)
{
	/// Functional Utility: Initializes the `drm_buddy` memory manager. It sets up the free lists,
	/// allocates root blocks covering the entire managed memory region, and splits them into appropriate
	/// power-of-two sizes.
	unsigned int i;
	u64 offset;

	// Precondition: The `size` to manage must be greater than or equal to `chunk_size`.
	if (size < chunk_size)
		return -EINVAL;

	// Precondition: `chunk_size` must be at least 4KB.
	if (chunk_size < SZ_4K)
		return -EINVAL;

	// Precondition: `chunk_size` must be a power of 2.
	if (!is_power_of_2(chunk_size))
		return -EINVAL;

	// Block Logic: Round down the total size to be a multiple of `chunk_size`.
	size = round_down(size, chunk_size);

	// Block Logic: Initialize memory manager fields.
	mm->size = size;
	mm->avail = size;
	mm->clear_avail = 0;
	mm->chunk_size = chunk_size;
	// Block Logic: Calculate the maximum order based on total size and chunk size.
	// Invariant: `max_order` represents the largest possible block size in powers of 2.
	mm->max_order = ilog2(size) - ilog2(chunk_size);

	BUG_ON(mm->max_order > DRM_BUDDY_MAX_ORDER);

	// Block Logic: Allocate memory for the free lists.
	// Invariant: There is one free list for each order up to `max_order`.
	mm->free_list = kmalloc_array(mm->max_order + 1,
				      sizeof(struct list_head),
				      GFP_KERNEL);
	// Conditional Logic: Handle memory allocation failure for free lists.
	if (!mm->free_list)
		return -ENOMEM;

	// Block Logic: Initialize each free list as empty.
	// Invariant: All free lists are properly initialized.
	for (i = 0; i <= mm->max_order; ++i)
		INIT_LIST_HEAD(&mm->free_list[i]);

// Block Logic: Calculate the number of root blocks needed if size is not a power of 2.
	mm->n_roots = hweight64(size);

	// Block Logic: Allocate memory for storing pointers to root blocks.
	mm->roots = kmalloc_array(mm->n_roots,
				  sizeof(struct drm_buddy_block *),
				  GFP_KERNEL);
	// Conditional Logic: Handle memory allocation failure for root pointers.
	if (!mm->roots)
		goto out_free_list;

	offset = 0;
	i = 0;

	/*
	 * Split into power-of-two blocks, in case we are given a size that is
	 * not itself a power-of-two.
	 */
	// Block Logic: Iterate to create and initialize root blocks for the memory region.
	// Invariant: The loop continues until all `size` has been covered by root blocks.
	do {
		struct drm_buddy_block *root;
		unsigned int order;
		u64 root_size;

		// Block Logic: Determine the order and size of the current root block.
		order = ilog2(size) - ilog2(chunk_size);
		root_size = chunk_size << order;

		// Block Logic: Allocate and initialize a new root block.
		root = drm_block_alloc(mm, NULL, order, offset);
		// Conditional Logic: Handle memory allocation failure for a root block.
		if (!root)
			goto out_free_roots;

		// Block Logic: Mark the newly created root block as free.
		mark_free(mm, root);

		BUG_ON(i > mm->max_order);
		BUG_ON(drm_buddy_block_size(mm, root) < chunk_size);

		// Block Logic: Store the root block and update remaining size and offset.
		mm->roots[i] = root;

		offset += root_size;
		size -= root_size;
		i++;
	} while (size);

	return 0;

out_free_roots:
	// Error Handling: Free partially allocated root blocks on failure.
	// Invariant: Ensures no memory leaks if root block allocation fails mid-loop.
	while (i--)
		drm_block_free(mm, mm->roots[i]);
	kfree(mm->roots);
out_free_list:
	// Error Handling: Free free list on failure.
	kfree(mm->free_list);
	return -ENOMEM;
}
EXPORT_SYMBOL(drm_buddy_init);

/**
 * drm_buddy_fini - tear down the memory manager
 *
 * @mm: DRM buddy manager to free
 *
 * Cleanup memory manager resources and the freelist
 */
void drm_buddy_fini(struct drm_buddy *mm)
{
	/// Functional Utility: Tears down the `drm_buddy` memory manager, freeing all allocated resources.
	/// It ensures that all memory blocks are merged and returned to the system, verifying that all
	/// initially managed memory is available again.
	u64 root_size, size, start;
	unsigned int order;
	int i;

	size = mm->size;

	// Block Logic: Iterate through all root blocks, forcing them to merge and then freeing their metadata.
	// Invariant: Ensures that all memory initially managed by the root blocks is properly deallocated.
	for (i = 0; i < mm->n_roots; ++i) {
		// Block Logic: Determine the order and start of the current root block.
		order = ilog2(size) - ilog2(mm->chunk_size);
		start = drm_buddy_block_offset(mm->roots[i]);
		// Block Logic: Force merge any fragmented blocks within the root block's range.
		__force_merge(mm, start, start + size, order);

		// Warning: This indicates a bug if a root block is not free after force merging.
		if (WARN_ON(!drm_buddy_block_is_free(mm->roots[i])))
			kunit_fail_current_test("buddy_fini() root");

		// Block Logic: Free the metadata structure for the root block.
		drm_block_free(mm, mm->roots[i]);

		// Block Logic: Update `size` for the next iteration to reflect the remaining memory.
		root_size = mm->chunk_size << order;
		size -= root_size;
	}

	// Warning: This indicates a memory leak or corruption if not all memory is available after cleanup.
	WARN_ON(mm->avail != mm->size);

	// Block Logic: Free the memory allocated for root block pointers and free lists.
	kfree(mm->roots);
	kfree(mm->free_list);
}
EXPORT_SYMBOL(drm_buddy_fini);

static int split_block(struct drm_buddy *mm,
		       struct drm_buddy_block *block)
{
	/// Functional Utility: Splits a `drm_buddy_block` into two smaller blocks of half its size (its buddies).
	/// This is a core operation in the buddy allocation system, used when a block of the exact requested size
	/// is not immediately available.
	unsigned int block_order = drm_buddy_block_order(block) - 1;
	u64 offset = drm_buddy_block_offset(block);

	BUG_ON(!drm_buddy_block_is_free(block));
	BUG_ON(!drm_buddy_block_order(block));

	// Block Logic: Allocate and initialize the left child block.
	block->left = drm_block_alloc(mm, block, block_order, offset);
	// Conditional Logic: Handle allocation failure for the left child.
	if (!block->left)
		return -ENOMEM;

	// Block Logic: Allocate and initialize the right child block.
	block->right = drm_block_alloc(mm, block, block_order,
				       offset + (mm->chunk_size << block_order));
	// Conditional Logic: Handle allocation failure for the right child, freeing the left child if necessary.
	if (!block->right) {
		drm_block_free(mm, block->left);
		return -ENOMEM;
	}

	// Block Logic: Mark both newly created child blocks as free.
	mark_free(mm, block->left);
	mark_free(mm, block->right);

	// Conditional Logic: If the parent block was cleared, propagate the cleared state to children and reset parent.
	if (drm_buddy_block_is_clear(block)) {
		mark_cleared(block->left);
		mark_cleared(block->right);
		clear_reset(block);
	}

	// Block Logic: Mark the parent block as split.
	mark_split(block);

	return 0;
}

/**
 * drm_get_buddy - get buddy address
 *
 * @block: DRM buddy block
 *
 * Returns the corresponding buddy block for @block, or NULL
 * if this is a root block and can't be merged further.
 * Requires some kind of locking to protect against
 * any concurrent allocate and free operations.
 */
struct drm_buddy_block *
drm_get_buddy(struct drm_buddy_block *block)
{
	/// Functional Utility: Public interface to retrieve the buddy of a given block,
	/// allowing external modules to query buddy relationships.
	return __get_buddy(block);
}
EXPORT_SYMBOL(drm_get_buddy);

/**
 * drm_buddy_free_block - free a block
 *
 * @mm: DRM buddy manager
 * @block: block to be freed
 */
void drm_buddy_free_block(struct drm_buddy *mm,
			  struct drm_buddy_block *block)
{
	/// Functional Utility: Frees an allocated `drm_buddy_block` and updates the available memory statistics.
	/// It then attempts to merge the block with its buddy using the internal free function.
	BUG_ON(!drm_buddy_block_is_allocated(block));
	// Block Logic: Update available memory statistics.
	mm->avail += drm_buddy_block_size(mm, block);
	// Conditional Logic: If the block was cleared, update `clear_avail`.
	if (drm_buddy_block_is_clear(block))
		mm->clear_avail += drm_buddy_block_size(mm, block);

	// Block Logic: Call the internal freeing function to handle merging with buddies.
	__drm_buddy_free(mm, block, false);
}
EXPORT_SYMBOL(drm_buddy_free_block);

static void __drm_buddy_free_list(struct drm_buddy *mm,
				  struct list_head *objects,
				  bool mark_clear,
				  bool mark_dirty)
{
	/// Functional Utility: Frees a list of `drm_buddy_block`s, optionally marking them as clear or dirty
	/// before freeing. This is an internal helper for bulk deallocation operations.
	struct drm_buddy_block *block, *on;

	WARN_ON(mark_dirty && mark_clear); // Warning: Inconsistent flags, should not mark both dirty and clear.

	// Block Logic: Iterate safely through the list of objects to be freed.
	list_for_each_entry_safe(block, on, objects, link) {
		// Conditional Logic: Mark the block as cleared or dirty based on flags.
		if (mark_clear)
			mark_cleared(block);
		else if (mark_dirty)
			clear_reset(block);
		// Block Logic: Free the individual block.
		drm_buddy_free_block(mm, block);
		// Block Logic: Yield CPU if necessary to avoid preemption issues in a long loop.
		cond_resched();
	}
	// Block Logic: Re-initialize the input list head as empty after freeing all objects.
	INIT_LIST_HEAD(objects);
}

static void drm_buddy_free_list_internal(struct drm_buddy *mm,
					 struct list_head *objects)
{
	/// Functional Utility: Frees a list of blocks internally without modifying their clear/dirty state.
	/// This is used when dealing with temporary allocations or partial allocation failures.
	/*
	 * Don't touch the clear/dirty bit, since allocation is still internal
	 * at this point. For example we might have just failed part of the
	 * allocation.
	 */
	__drm_buddy_free_list(mm, objects, false, false);
}

/**
 * drm_buddy_free_list - free blocks
 *
 * @mm: DRM buddy manager
 * @objects: input list head to free blocks
 * @flags: optional flags like DRM_BUDDY_CLEARED
 */
void drm_buddy_free_list(struct drm_buddy *mm,
			 struct list_head *objects,
			 unsigned int flags)
{
	/// Functional Utility: Public interface to free a list of blocks, allowing external modules
	/// to specify whether the blocks should be marked as cleared or dirty during deallocation.
	bool mark_clear = flags & DRM_BUDDY_CLEARED;

	// Block Logic: Call the internal helper to free the list, applying clear or dirty marking based on flags.
	__drm_buddy_free_list(mm, objects, mark_clear, !mark_clear);
}
EXPORT_SYMBOL(drm_buddy_free_list);

static bool block_incompatible(struct drm_buddy_block *block, unsigned int flags)
{
	/// Functional Utility: Checks if a given `drm_buddy_block` is incompatible with the requested allocation flags,
	/// specifically regarding its clear state.
	bool needs_clear = flags & DRM_BUDDY_CLEAR_ALLOCATION;

	// Block Logic: Compare the `needs_clear` flag with the block's current clear state.
	return needs_clear != drm_buddy_block_is_clear(block);
}

static struct drm_buddy_block *
__alloc_range_bias(struct drm_buddy *mm,
		   u64 start, u64 end,
		   unsigned int order,
		   unsigned long flags,
		   bool fallback)
{
	/// Functional Utility: Attempts to allocate a block of a specific `order` within a given address `range` with a bias.
	/// It performs a depth-first search on the buddy tree, splitting larger blocks as needed, and considering
	/// allocation `flags` and fallback behavior.
	u64 req_size = mm->chunk_size << order;
	struct drm_buddy_block *block;
	struct drm_buddy_block *buddy;
	LIST_HEAD(dfs);
	int err;
	int i;

	end = end - 1;

	// Block Logic: Initialize DFS list with root blocks.
	for (i = 0; i < mm->n_roots; ++i)
		list_add_tail(&mm->roots[i]->tmp_link, &dfs);

	// Block Logic: Depth-first search to find a suitable block.
	// Invariant: The loop continues until a block is found or no more blocks can be processed.
	do {
		u64 block_start;
		u64 block_end;

		// Block Logic: Get the first block from the DFS list.
		block = list_first_entry_or_null(&dfs,
						 struct drm_buddy_block,
						 tmp_link);
		// Conditional Logic: If no more blocks in DFS, break.
		if (!block)
			break;

		// Block Logic: Remove the block from the DFS list.
		list_del(&block->tmp_link);

		// Conditional Logic: Skip if the block's order is too small for the request.
		if (drm_buddy_block_order(block) < order)
			continue;

		// Block Logic: Calculate the start and end offsets of the current block.
		block_start = drm_buddy_block_offset(block);
		block_end = block_start + drm_buddy_block_size(mm, block) - 1;

		// Conditional Logic: Skip if the block does not overlap with the requested range.
		if (!overlaps(start, end, block_start, block_end))
			continue;

		// Conditional Logic: Skip if the block is already allocated.
		if (drm_buddy_block_is_allocated(block))
			continue;

		// Conditional Logic: If the block is partially outside the requested range,
		// and cannot fully contain the requested size with proper alignment, skip.
		if (block_start < start || block_end > end) {
			u64 adjusted_start = max(block_start, start);
			u64 adjusted_end = min(block_end, end);

			if (round_down(adjusted_end + 1, req_size) <=
			    round_up(adjusted_start, req_size))
				continue;
		}

		// Conditional Logic: If not in fallback mode and block is incompatible, skip.
		if (!fallback && block_incompatible(block, flags))
			continue;

		// Conditional Logic: If the block is fully contained within the range and is of the correct order.
		if (contains(start, end, block_start, block_end) &&
		    order == drm_buddy_block_order(block)) {
			/*
			 * Find the free block within the range.
			 */
			// Conditional Logic: If the block is free and suitable, return it.
			if (drm_buddy_block_is_free(block))
				return block;

			continue;
		}

		// Conditional Logic: If the block is not split, split it to find a smaller block.
		if (!drm_buddy_block_is_split(block)) {
			err = split_block(mm, block);
			// Conditional Logic: Handle split failure.
			if (unlikely(err))
				goto err_undo;
		}

		// Block Logic: Add child blocks to DFS list for further exploration.
		list_add(&block->right->tmp_link, &dfs);
		list_add(&block->left->tmp_link, &dfs);
	} while (1);

	return ERR_PTR(-ENOSPC);

err_undo:
	/*
	 * We really don't want to leave around a bunch of split blocks, since
	 * bigger is better, so make sure we merge everything back before we
	 * free the allocated blocks.
	 */
	// Error Handling: Merge split blocks back if an error occurred during allocation.
	buddy = __get_buddy(block);
	if (buddy &&
	    (drm_buddy_block_is_free(block) &&
	     drm_buddy_block_is_free(buddy)))
		__drm_buddy_free(mm, block, false);
	return ERR_PTR(err);
}

static struct drm_buddy_block *
__drm_buddy_alloc_range_bias(struct drm_buddy *mm,
			     u64 start, u64 end,
			     unsigned int order,
			     unsigned long flags)
{
	/// Functional Utility: Serves as a wrapper for `__alloc_range_bias`, providing an initial allocation attempt
	/// without fallback, and then a fallback attempt if the first one fails. This implements a two-pass
	/// allocation strategy for range-biased requests.
	struct drm_buddy_block *block;
	bool fallback = false;

	block = __alloc_range_bias(mm, start, end, order,
				   flags, fallback);
	if (IS_ERR(block))
		return __alloc_range_bias(mm, start, end, order,
					  flags, !fallback);

	return block;
}

static struct drm_buddy_block *
get_maxblock(struct drm_buddy *mm, unsigned int order,
	     unsigned long flags)
{
	/// Functional Utility: Finds the largest available free block (of at least `order`)
	/// that is compatible with the given `flags`, prioritizing higher offsets.
	/// This is used for top-down allocation strategies.
	struct drm_buddy_block *max_block = NULL, *block = NULL;
	unsigned int i;

	// Block Logic: Iterate through free lists from the requested `order` up to the maximum order.
	// Invariant: `i` represents the current order being checked.
	for (i = order; i <= mm->max_order; ++i) {
		struct drm_buddy_block *tmp_block;

		// Block Logic: Iterate in reverse through the free list of the current order.
		list_for_each_entry_reverse(tmp_block, &mm->free_list[i], link) {
			// Conditional Logic: Skip if the block is incompatible with allocation flags.
			if (block_incompatible(tmp_block, flags))
				continue;

			// Block Logic: Found a suitable block, store it and break.
			block = tmp_block;
			break;
		}

		// Conditional Logic: If no block was found in the current order's free list, continue to the next order.
		if (!block)
			continue;

		// Conditional Logic: If `max_block` is not yet set, set the current block as `max_block`.
		if (!max_block) {
			max_block = block;
			continue;
		}

		// Conditional Logic: If the current block's offset is greater than `max_block`'s offset, update `max_block`.
		// This prioritizes blocks with higher addresses (top-down).
		if (drm_buddy_block_offset(block) >
		    drm_buddy_block_offset(max_block)) {
			max_block = block;
		}
	}

	return max_block;
}

static struct drm_buddy_block *
alloc_from_freelist(struct drm_buddy *mm,
		    unsigned int order,
		    unsigned long flags)
{
	/// Functional Utility: Allocates a single block of a specified `order` from the free lists.
	/// It supports both top-down allocation (finding the largest block) and a general allocation strategy,
	/// splitting blocks as necessary to fulfill the request.
	unsigned int tmp;
	int err;

	// Conditional Logic: If `DRM_BUDDY_TOPDOWN_ALLOCATION` flag is set, find the largest suitable block.
	if (flags & DRM_BUDDY_TOPDOWN_ALLOCATION) {
		block = get_maxblock(mm, order, flags);
		// Conditional Logic: If a block is found, store its order.
		if (block)
			/* Store the obtained block order */
			tmp = drm_buddy_block_order(block);
	} else {
		// Block Logic: Iterate upwards through free lists to find the smallest block of sufficient order.
		// Invariant: `tmp` represents the current order being checked.
		for (tmp = order; tmp <= mm->max_order; ++tmp) {
			struct drm_buddy_block *tmp_block;

			// Block Logic: Iterate in reverse through the free list of the current order.
			list_for_each_entry_reverse(tmp_block, &mm->free_list[tmp], link) {
				// Conditional Logic: Skip if the block is incompatible with allocation flags.
				if (block_incompatible(tmp_block, flags))
					continue;

				// Block Logic: Found a suitable block.
				block = tmp_block;
				break;
			}

			// Conditional Logic: If a block is found, break from the outer loop.
			if (block)
				break;
		}
	}

	// Conditional Logic: If no suitable block is found after initial search, try a fallback method.
	if (!block) {
		/* Fallback method */
		// Block Logic: Iterate through free lists to find any available block.
		for (tmp = order; tmp <= mm->max_order; ++tmp) {
			// Conditional Logic: If the free list for `tmp` order is not empty.
			if (!list_empty(&mm->free_list[tmp])) {
				// Block Logic: Get the last block from the free list.
				block = list_last_entry(&mm->free_list[tmp],
							struct drm_buddy_block,
							link);
				// Conditional Logic: If a block is found, break from the outer loop.
				if (block)
					break;
			}
		}

		// Conditional Logic: If no block is found even with fallback, return -ENOSPC.
		if (!block)
			return ERR_PTR(-ENOSPC);
	}

	BUG_ON(!drm_buddy_block_is_free(block)); // Invariant: The chosen block must be free.

	// Block Logic: Split the found block until it reaches the requested `order`.
	// Invariant: `tmp` is the current order of `block`, `order` is the target order.
	while (tmp != order) {
		err = split_block(mm, block);
		// Conditional Logic: Handle split failure.
		if (unlikely(err))
			goto err_undo;

		// Block Logic: Move to the right child and decrement its order.
		block = block->right;
		tmp--;
	}
	return block;

err_undo:
	// Error Handling: Free partially split blocks if an error occurred during splitting.
	if (tmp != order)
		__drm_buddy_free(mm, block, false);
	return ERR_PTR(err);
}

static int __alloc_range(struct drm_buddy *mm,
			 struct list_head *dfs,
			 u64 start, u64 size,
			 struct list_head *blocks,
			 u64 *total_allocated_on_err)
{
	/// Functional Utility: Allocates a contiguous range of memory blocks within a specified `start` and `size`.
	/// It performs a depth-first search on the buddy tree, identifying and marking blocks as allocated if they fit
	/// within the requested range, splitting larger blocks if necessary.
	struct drm_buddy_block *block;
	struct drm_buddy_block *buddy;
	LIST_HEAD(allocated);
	u64 end;
	int err;

	end = start + size - 1;

	// Block Logic: Depth-first search to find and allocate blocks within the specified range.
	// Invariant: The loop continues until the DFS list is empty or an error occurs.
	do {
		u64 block_start;
		u64 block_end;

		// Block Logic: Get the first block from the DFS list.
		block = list_first_entry_or_null(dfs,
						 struct drm_buddy_block,
						 tmp_link);
		// Conditional Logic: If no more blocks in DFS, break.
		if (!block)
			break;

		// Block Logic: Remove the block from the DFS list.
		list_del(&block->tmp_link);

		// Block Logic: Calculate the start and end offsets of the current block.
		block_start = drm_buddy_block_offset(block);
		block_end = block_start + drm_buddy_block_size(mm, block) - 1;

		// Conditional Logic: Skip if the block does not overlap with the requested allocation range.
		if (!overlaps(start, end, block_start, block_end))
			continue;

		// Conditional Logic: If the block is already allocated, cannot use it; return error.
		if (drm_buddy_block_is_allocated(block)) {
			err = -ENOSPC;
			goto err_free;
		}

		// Conditional Logic: If the block fully contains a portion of the requested range.
		if (contains(start, end, block_start, block_end)) {
			// Conditional Logic: If the block is free, allocate it.
			if (drm_buddy_block_is_free(block)) {
				mark_allocated(block);
				total_allocated += drm_buddy_block_size(mm, block);
				mm->avail -= drm_buddy_block_size(mm, block);
				if (drm_buddy_block_is_clear(block))
					mm->clear_avail -= drm_buddy_block_size(mm, block);
				list_add_tail(&block->link, &allocated);
				continue;
			} else if (!mm->clear_avail) {
				err = -ENOSPC;
				goto err_free;
			}
		}

		// Conditional Logic: If the block is not split, split it to find a smaller block.
		if (!drm_buddy_block_is_split(block)) {
			err = split_block(mm, block);
			// Conditional Logic: Handle split failure.
			if (unlikely(err))
				goto err_undo;
		}

		// Block Logic: Add child blocks to DFS list for further exploration.
		list_add(&block->right->tmp_link, dfs);
		list_add(&block->left->tmp_link, dfs);
	} while (1);

	// Conditional Logic: If not enough memory was allocated to satisfy the request, return error.
	if (total_allocated < size) {
		err = -ENOSPC;
		goto err_free;
	}

	// Block Logic: Splice all allocated blocks into the output list.
	list_splice_tail(&allocated, blocks);

	return 0;

err_undo:
	/*
	 * We really don't want to leave around a bunch of split blocks, since
	 * bigger is better, so make sure we merge everything back before we
	 * free the allocated blocks.
	 */
	// Error Handling: Merge split blocks back if an error occurred during allocation.
	buddy = __get_buddy(block);
	if (buddy &&
	    (drm_buddy_block_is_free(block) &&
	     drm_buddy_block_is_free(buddy)))
		__drm_buddy_free(mm, block, false);

err_free:
	// Error Handling: If an error occurred, handle partial allocations.
	if (err == -ENOSPC && total_allocated_on_err) {
		list_splice_tail(&allocated, blocks);
		*total_allocated_on_err = total_allocated;
	} else {
		drm_buddy_free_list_internal(mm, &allocated);
	}

	return err;
}

static int __drm_buddy_alloc_range(struct drm_buddy *mm,
				   u64 start,
				   u64 size,
				   u64 *total_allocated_on_err,
				   struct list_head *blocks)
{
	/// Functional Utility: Initializes a depth-first search (DFS) list with all root blocks
	/// and then calls `__alloc_range` to perform the actual allocation of a memory range.
	LIST_HEAD(dfs);
	int i;

	for (i = 0; i < mm->n_roots; ++i)
		list_add_tail(&mm->roots[i]->tmp_link, &dfs);

	return __alloc_range(mm, &dfs, start, size,
			     blocks, total_allocated_on_err);
}

static int __alloc_contig_try_harder(struct drm_buddy *mm,
				     u64 size,
				     u64 min_block_size,
				     struct list_head *blocks)
{
	/// Functional Utility: Implements a more aggressive strategy for contiguous block allocation when initial attempts fail.
	/// It tries allocating from different positions relative to an existing block, attempting to find a contiguous space.
	u64 rhs_offset, lhs_offset, lhs_size, filled;
	struct drm_buddy_block *block;
	struct list_head *list;
	LIST_HEAD(blocks_lhs);
	unsigned long pages;
	unsigned int order;
	u64 modify_size;
	int err;

	// Block Logic: Calculate the largest power-of-two size less than or equal to the requested size.
	modify_size = rounddown_pow_of_two(size);
	// Block Logic: Convert the size to pages based on `mm->chunk_size`.
	pages = modify_size >> ilog2(mm->chunk_size);
	// Block Logic: Determine the order of the block corresponding to `pages`.
	order = fls(pages) - 1;
	// Conditional Logic: If the order is 0, no block can be allocated, return error.
	if (order == 0)
		return -ENOSPC;

	// Block Logic: Get the free list for the determined order.
	list = &mm->free_list[order];
	// Conditional Logic: If the list is empty, no block of this order is available, return error.
	if (list_empty(list))
		return -ENOSPC;

	// Block Logic: Iterate in reverse through the free list to find a suitable block.
	list_for_each_entry_reverse(block, list, link) {
		/* Allocate blocks traversing RHS */
		// Block Logic: Attempt to allocate blocks starting from the right-hand side (RHS) of the current block.
		rhs_offset = drm_buddy_block_offset(block);
		err =  __drm_buddy_alloc_range(mm, rhs_offset, size,
					       &filled, blocks);
		// Conditional Logic: If allocation is successful or returns an error other than -ENOSPC, return.
		if (!err || err != -ENOSPC)
			return err;

		// Block Logic: Calculate the size needed for the left-hand side (LHS) allocation.
		lhs_size = max((size - filled), min_block_size);
		// Conditional Logic: Align `lhs_size` to `min_block_size` if not already aligned.
		if (!IS_ALIGNED(lhs_size, min_block_size))
			lhs_size = round_up(lhs_size, min_block_size);

		/* Allocate blocks traversing LHS */
		// Block Logic: Attempt to allocate blocks starting from the left-hand side (LHS) of the current block.
		lhs_offset = drm_buddy_block_offset(block) - lhs_size;
		err =  __drm_buddy_alloc_range(mm, lhs_offset, lhs_size,
					       NULL, &blocks_lhs);
		// Conditional Logic: If allocation is successful, splice allocated blocks and return.
		if (!err) {
			list_splice(&blocks_lhs, blocks);
			return 0;
		} else if (err != -ENOSPC) {
			// Error Handling: If an error other than -ENOSPC occurs, free partially allocated blocks and return.
			drm_buddy_free_list_internal(mm, blocks);
			return err;
		}
		/* Free blocks for the next iteration */
		// Block Logic: Free partially allocated blocks before the next iteration.
		drm_buddy_free_list_internal(mm, blocks);
	}

	return -ENOSPC;
}

/**
 * drm_buddy_block_trim - free unused pages
 *
 * @mm: DRM buddy manager
 * @start: start address to begin the trimming.
 * @new_size: original size requested
 * @blocks: Input and output list of allocated blocks.
 * MUST contain single block as input to be trimmed.
 * On success will contain the newly allocated blocks
 * making up the @new_size. Blocks always appear in
 * ascending order
 *
 * For contiguous allocation, we round up the size to the nearest
 * power of two value, drivers consume *actual* size, so remaining
 * portions are unused and can be optionally freed with this function
 *
 * Returns:
 * 0 on success, error code on failure.
 */
int drm_buddy_block_trim(struct drm_buddy *mm,
			 u64 *start,
			 u64 new_size,
			 struct list_head *blocks)
{
	struct drm_buddy_block *parent;
	struct drm_buddy_block *block;
	u64 block_start, block_end;
	LIST_HEAD(dfs);
	u64 new_start;
	int err;

	if (!list_is_singular(blocks))
		return -EINVAL;

	block = list_first_entry(blocks,
				 struct drm_buddy_block,
				 link);

	block_start = drm_buddy_block_offset(block);
	block_end = block_start + drm_buddy_block_size(mm, block);

	if (WARN_ON(!drm_buddy_block_is_allocated(block)))
		return -EINVAL;

	if (new_size > drm_buddy_block_size(mm, block))
		return -EINVAL;

	if (!new_size || !IS_ALIGNED(new_size, mm->chunk_size))
		return -EINVAL;

	if (new_size == drm_buddy_block_size(mm, block))
		return 0;

	new_start = block_start;
	if (start) {
		new_start = *start;

		if (new_start < block_start)
			return -EINVAL;

		if (!IS_ALIGNED(new_start, mm->chunk_size))
			return -EINVAL;

		if (range_overflows(new_start, new_size, block_end))
			return -EINVAL;
	}

	list_del(&block->link);
	mark_free(mm, block);
	mm->avail += drm_buddy_block_size(mm, block);
	if (drm_buddy_block_is_clear(block))
		mm->clear_avail += drm_buddy_block_size(mm, block);

	/* Prevent recursively freeing this node */
	parent = block->parent;
	block->parent = NULL;

	list_add(&block->tmp_link, &dfs);
	err =  __alloc_range(mm, &dfs, new_start, new_size, blocks, NULL);
	if (err) {
		mark_allocated(block);
		mm->avail -= drm_buddy_block_size(mm, block);
		if (drm_buddy_block_is_clear(block))
			mm->clear_avail -= drm_buddy_block_size(mm, block);
		list_add(&block->link, blocks);
	}

	block->parent = parent;
	return err;
}
EXPORT_SYMBOL(drm_buddy_block_trim);

static struct drm_buddy_block *
__drm_buddy_alloc_blocks(struct drm_buddy *mm,
			 u64 start, u64 end,
			 unsigned int order,
			 unsigned long flags)
{
	if (flags & DRM_BUDDY_RANGE_ALLOCATION)
		/* Allocate traversing within the range */
		return  __drm_buddy_alloc_range_bias(mm, start, end,
						     order, flags);
	else
		/* Allocate from freelist */
		return alloc_from_freelist(mm, order, flags);
}

/**
 * drm_buddy_alloc_blocks - allocate power-of-two blocks
 *
 * @mm: DRM buddy manager to allocate from
 * @start: start of the allowed range for this block
 * @end: end of the allowed range for this block
 * @size: size of the allocation in bytes
 * @min_block_size: alignment of the allocation
 * @blocks: output list head to add allocated blocks
 * @flags: DRM_BUDDY_*_ALLOCATION flags
 *
 * alloc_range_bias() called on range limitations, which traverses
 * the tree and returns the desired block.
 *
 * alloc_from_freelist() called when *no* range restrictions
 * are enforced, which picks the block from the freelist.
 *
 * Returns:
 * 0 on success, error code on failure.
 */
int drm_buddy_alloc_blocks(struct drm_buddy *mm,
			   u64 start, u64 end, u64 size,
			   u64 min_block_size,
			   struct list_head *blocks,
			   unsigned long flags)
{
	struct drm_buddy_block *block = NULL;
	u64 original_size, original_min_size;
	unsigned int min_order, order;
	LIST_HEAD(allocated);
	unsigned long pages;
	int err;

	if (size < mm->chunk_size)
		return -EINVAL;

	if (min_block_size < mm->chunk_size)
		return -EINVAL;

	if (!is_power_of_2(min_block_size))
		return -EINVAL;

	if (!IS_ALIGNED(start | end | size, mm->chunk_size))
		return -EINVAL;

	if (end > mm->size)
		return -EINVAL;

	if (range_overflows(start, size, mm->size))
		return -EINVAL;

	/* Actual range allocation */
	if (start + size == end) {
		if (!IS_ALIGNED(start | end, min_block_size))
			return -EINVAL;

		return __drm_buddy_alloc_range(mm, start, size, NULL, blocks);
	}

	original_size = size;
	original_min_size = min_block_size;

	/* Roundup the size to power of 2 */
	if (flags & DRM_BUDDY_CONTIGUOUS_ALLOCATION) {
		size = roundup_pow_of_two(size);
		min_block_size = size;
	/* Align size value to min_block_size */
	} else if (!IS_ALIGNED(size, min_block_size)) {
		size = round_up(size, min_block_size);
	}

	pages = size >> ilog2(mm->chunk_size);
	order = fls(pages) - 1;
	min_order = ilog2(min_block_size) - ilog2(mm->chunk_size);

	do {
		order = min(order, (unsigned int)fls(pages) - 1);
		BUG_ON(order > mm->max_order);
		BUG_ON(order < min_order);

		do {
			block = __drm_buddy_alloc_blocks(mm, start,
							 end,
							 order,
							 flags);
			if (!IS_ERR(block))
				break;

			if (order-- == min_order) {
				/* Try allocation through force merge method */
				if (mm->clear_avail &&
				    !__force_merge(mm, start, end, min_order)) {
					block = __drm_buddy_alloc_blocks(mm, start,
									 end,
									 min_order,
									 flags);
					if (!IS_ERR(block)) {
						order = min_order;
						break;
					}
				}

				/*
				 * Try contiguous block allocation through
				 * try harder method.
				 */
				if (flags & DRM_BUDDY_CONTIGUOUS_ALLOCATION &&
				    !(flags & DRM_BUDDY_RANGE_ALLOCATION))
					return __alloc_contig_try_harder(mm,
									 original_size,
									 original_min_size,
									 blocks);
				err = -ENOSPC;
				goto err_free;
			}
		} while (1);

		mark_allocated(block);
		mm->avail -= drm_buddy_block_size(mm, block);
		if (drm_buddy_block_is_clear(block))
			mm->clear_avail -= drm_buddy_block_size(mm, block);
		kmemleak_update_trace(block);
		list_add_tail(&block->link, &allocated);

		pages -= BIT(order);

		if (!pages)
			break;
	} while (1);

	/* Trim the allocated block to the required size */
	if (!(flags & DRM_BUDDY_TRIM_DISABLE) &&
	    original_size != size) {
		struct list_head *trim_list;
		LIST_HEAD(temp);
		u64 trim_size;

		trim_list = &allocated;
		trim_size = original_size;

		if (!list_is_singular(&allocated)) {
			block = list_last_entry(&allocated, typeof(*block), link);
			list_move(&block->link, &temp);
			trim_list = &temp;
			trim_size = drm_buddy_block_size(mm, block) -
				(size - original_size);
		}

		drm_buddy_block_trim(mm,
				     NULL,
				     trim_size,
				     trim_list);

		if (!list_empty(&temp))
			list_splice_tail(trim_list, &allocated);
	}

	list_splice_tail(&allocated, blocks);
	return 0;

err_free:
	drm_buddy_free_list_internal(mm, &allocated);
	return err;
}
EXPORT_SYMBOL(drm_buddy_alloc_blocks);

/**
 * drm_buddy_block_print - print block information
 *
 * @mm: DRM buddy manager
 * @block: DRM buddy block
 * @p: DRM printer to use
 */
void drm_buddy_block_print(struct drm_buddy *mm,
			   struct drm_buddy_block *block,
			   struct drm_printer *p)
{
	u64 start = drm_buddy_block_offset(block);
	u64 size = drm_buddy_block_size(mm, block);

	drm_printf(p, "%#018llx-%#018llx: %llu\n", start, start + size, size);
}
EXPORT_SYMBOL(drm_buddy_block_print);

/**
 * drm_buddy_print - print allocator state
 *
 * @mm: DRM buddy manager
 * @p: DRM printer to use
 */
void drm_buddy_print(struct drm_buddy *mm, struct drm_printer *p)
{
	int order;

	drm_printf(p, "chunk_size: %lluKiB, total: %lluMiB, free: %lluMiB, clear_free: %lluMiB\n",
		   mm->chunk_size >> 10, mm->size >> 20, mm->avail >> 20, mm->clear_avail >> 20);

	for (order = mm->max_order; order >= 0; order--) {
		struct drm_buddy_block *block;
		u64 count = 0, free;

		list_for_each_entry(block, &mm->free_list[order], link) {
			BUG_ON(!drm_buddy_block_is_free(block));
			count++;
		}

		drm_printf(p, "order-%2d ", order);

		free = count * (mm->chunk_size << order);
		if (free < SZ_1M)
			drm_printf(p, "free: %8llu KiB", free >> 10);
		else
			drm_printf(p, "free: %8llu MiB", free >> 20);

		drm_printf(p, ", blocks: %llu\n", count);
	}
}
EXPORT_SYMBOL(drm_buddy_print);

static void drm_buddy_module_exit(void)
{
	kmem_cache_destroy(slab_blocks);
}

static int __init drm_buddy_module_init(void)
{
	slab_blocks = KMEM_CACHE(drm_buddy_block, 0);
	if (!slab_blocks)
		return -ENOMEM;

	return 0;
}

module_init(drm_buddy_module_init);
module_exit(drm_buddy_module_exit);

MODULE_DESCRIPTION("DRM Buddy Allocator");
MODULE_LICENSE("Dual MIT/GPL");
