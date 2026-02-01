/*
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors: Christian König
 */

/**
 * @file amdgpu_vram_mgr.c
 * @brief AMDGPU VRAM (Video RAM) Manager.
 * This file implements the VRAM manager for the AMDGPU driver, responsible for
 * allocating, managing, and tracking VRAM resources. It integrates with the TTM
 * (Translation Table Manager) framework and uses a buddy allocator for efficient
 * memory management.
 *
 * Functional Utility: Provides core memory management capabilities for GPU VRAM,
 * including allocation, deallocation, tracking usage, and exporting VRAM
 * regions to other devices (via scatter-gather tables). It also exposes VRAM
 * information via sysfs.
 *
 * Key Architectural Components:
 * - **TTM (Translation Table Manager)**: Provides a generic memory management
 *   framework for DRM drivers.
 * - **DRM Buddy Allocator**: Used for efficient allocation and deallocation
 *   of VRAM blocks.
 * - **Sysfs Interface**: Exposes VRAM usage, total size, and vendor information
 *   to userspace.
 */

#include <linux/dma-mapping.h>
#include <drm/ttm/ttm_range_manager.h>
#include <drm/drm_drv.h>

#include "amdgpu.h"
#include "amdgpu_vm.h"
#include "amdgpu_res_cursor.h"
#include "atom.h"

#define AMDGPU_MAX_SG_SEGMENT_SIZE	(2UL << 30)

struct amdgpu_vram_reservation {
	u64 start;
	u64 size;
	struct list_head allocated;
	struct list_head blocks;
};

static inline struct amdgpu_vram_mgr *
to_vram_mgr(struct ttm_resource_manager *man)
{
	return container_of(man, struct amdgpu_vram_mgr, manager);
}

static inline struct amdgpu_device *
to_amdgpu_device(struct amdgpu_vram_mgr *mgr)
{
	return container_of(mgr, struct amdgpu_device, mman.vram_mgr);
}

static inline struct drm_buddy_block *
amdgpu_vram_mgr_first_block(struct list_head *list)
{
	/// Functional Utility: Retrieves the first DRM buddy allocator block from a list.
	/// This helper function is used to access the initial block within a chain of allocated memory.
	return list_first_entry_or_null(list, struct drm_buddy_block, link);
}

static inline bool amdgpu_is_vram_mgr_blocks_contiguous(struct list_head *head)
{
	/// Functional Utility: Checks if a list of DRM buddy allocator blocks forms a contiguous memory region.
	/// This function iterates through the blocks and verifies that each block immediately follows the previous one in memory.
	struct drm_buddy_block *block;
	u64 start, size;

	block = amdgpu_vram_mgr_first_block(head);
	// Conditional Logic: If the list is empty, it's not contiguous.
	if (!block)
		return false;

	// Block Logic: Iterate through the linked list of blocks.
	while (head != block->link.next) {
		start = amdgpu_vram_mgr_block_start(block);
		size = amdgpu_vram_mgr_block_size(block);

		block = list_entry(block->link.next, struct drm_buddy_block, link);
		// Conditional Logic: Check if the current block's end address matches the next block's start address.
		if (start + size != amdgpu_vram_mgr_block_start(block))
			return false;
	}

	return true;
}

static inline u64 amdgpu_vram_mgr_blocks_size(struct list_head *head)
{
	/// Functional Utility: Calculates the total size of memory represented by a list of DRM buddy allocator blocks.
	/// This function sums the size of each block in the list to determine the aggregate memory usage.
	struct drm_buddy_block *block;
	u64 size = 0;

	list_for_each_entry(block, head, link)
		size += amdgpu_vram_mgr_block_size(block);

	return size;
}

/**
 * DOC: mem_info_vram_total
 *
 * The amdgpu driver provides a sysfs API for reporting current total VRAM
 * available on the device
 * The file mem_info_vram_total is used for this and returns the total
 * amount of VRAM in bytes
 */
static ssize_t amdgpu_mem_info_vram_total_show(struct device *dev,
		struct device_attribute *attr, char *buf)
{
	/// Functional Utility: Sysfs callback to display the total amount of VRAM available on the AMDGPU device.
	/// This function reads the `real_vram_size` from the device's GMC (Graphics Memory Controller) and formats it for sysfs output.
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);

	return sysfs_emit(buf, "%llu\n", adev->gmc.real_vram_size);
}

/**
 * DOC: mem_info_vis_vram_total
 *
 * The amdgpu driver provides a sysfs API for reporting current total
 * visible VRAM available on the device
 * The file mem_info_vis_vram_total is used for this and returns the total
 * amount of visible VRAM in bytes
 */
static ssize_t amdgpu_mem_info_vis_vram_total_show(struct device *dev,
		struct device_attribute *attr, char *buf)
{
	/// Functional Utility: Sysfs callback to display the total amount of CPU-visible VRAM available on the AMDGPU device.
	/// This function reports the `visible_vram_size` from the device's GMC, representing the portion of VRAM directly accessible by the CPU.
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);

	return sysfs_emit(buf, "%llu\n", adev->gmc.visible_vram_size);
}

/**
 * DOC: mem_info_vram_used
 *
 * The amdgpu driver provides a sysfs API for reporting current total VRAM
 * available on the device
 * The file mem_info_vram_used is used for this and returns the total
 * amount of currently used VRAM in bytes
 */
static ssize_t amdgpu_mem_info_vram_used_show(struct device *dev,
					      struct device_attribute *attr,
					      char *buf)
{
	/// Functional Utility: Sysfs callback to display the total amount of VRAM currently in use by the AMDGPU driver.
	/// This function queries the TTM resource manager for the overall VRAM usage and formats it for sysfs output.
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);
	struct ttm_resource_manager *man = &adev->mman.vram_mgr.manager;

	return sysfs_emit(buf, "%llu\n", ttm_resource_manager_usage(man));
}

/**
 * DOC: mem_info_vis_vram_used
 *
 * The amdgpu driver provides a sysfs API for reporting current total of
 * used visible VRAM
 * The file mem_info_vis_vram_used is used for this and returns the total
 * amount of currently used visible VRAM in bytes
 */
static ssize_t amdgpu_mem_info_vis_vram_used_show(struct device *dev,
						  struct device_attribute *attr,
						  char *buf)
{
	/// Functional Utility: Sysfs callback to display the amount of CPU-visible VRAM currently in use.
	/// This function reports the portion of allocated VRAM that is accessible by the CPU.
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);

	return sysfs_emit(buf, "%llu\n",
			  amdgpu_vram_mgr_vis_usage(&adev->mman.vram_mgr));
}

/**
 * DOC: mem_info_vram_vendor
 *
 * The amdgpu driver provides a sysfs API for reporting the vendor of the
 * installed VRAM
 * The file mem_info_vram_vendor is used for this and returns the name of the
 * vendor.
 */
static ssize_t amdgpu_mem_info_vram_vendor(struct device *dev,
					   struct device_attribute *attr,
					   char *buf)
{
	/// Functional Utility: Sysfs callback to report the vendor of the installed VRAM.
	/// This function maps the internal VRAM vendor ID to a human-readable string for sysfs output.
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);

	// Block Logic: Use a switch statement to map the VRAM vendor ID to a string.
	switch (adev->gmc.vram_vendor) {
	case SAMSUNG:
		return sysfs_emit(buf, "samsung\n");
	case INFINEON:
		return sysfs_emit(buf, "infineon\n");
	case ELPIDA:
		return sysfs_emit(buf, "elpida\n");
	case ETRON:
		return sysfs_emit(buf, "etron\n");
	case NANYA:
		return sysfs_emit(buf, "nanya\n");
	case HYNIX:
		return sysfs_emit(buf, "hynix\n");
	case MOSEL:
		return sysfs_emit(buf, "mosel\n");
	case WINBOND:
		return sysfs_emit(buf, "winbond\n");
	case ESMT:
		return sysfs_emit(buf, "esmt\n");
	case MICRON:
		return sysfs_emit(buf, "micron\n");
	default:
		return sysfs_emit(buf, "unknown\n");
	}
}

static DEVICE_ATTR(mem_info_vram_total, S_IRUGO,
		   amdgpu_mem_info_vram_total_show, NULL);
static DEVICE_ATTR(mem_info_vis_vram_total, S_IRUGO,
		   amdgpu_mem_info_vis_vram_total_show,NULL);
static DEVICE_ATTR(mem_info_vram_used, S_IRUGO,
		   amdgpu_mem_info_vram_used_show, NULL);
static DEVICE_ATTR(mem_info_vis_vram_used, S_IRUGO,
		   amdgpu_mem_info_vis_vram_used_show, NULL);
static DEVICE_ATTR(mem_info_vram_vendor, S_IRUGO,
		   amdgpu_mem_info_vram_vendor, NULL);

static struct attribute *amdgpu_vram_mgr_attributes[] = {
	&dev_attr_mem_info_vram_total.attr,
	&dev_attr_mem_info_vis_vram_total.attr,
	&dev_attr_mem_info_vram_used.attr,
	&dev_attr_mem_info_vis_vram_used.attr,
	&dev_attr_mem_info_vram_vendor.attr,
	NULL
};

static umode_t amdgpu_vram_attrs_is_visible(struct kobject *kobj,
					    struct attribute *attr, int i)
{
	/// Functional Utility: Determines the visibility of VRAM-related sysfs attributes.
	/// This function conditionally hides the `mem_info_vram_vendor` attribute if the VRAM vendor is unknown (0).
	struct device *dev = kobj_to_dev(kobj);
	struct drm_device *ddev = dev_get_drvdata(dev);
	struct amdgpu_device *adev = drm_to_adev(ddev);

	// Conditional Logic: Hide the vram_vendor attribute if the vendor is unknown.
	if (attr == &dev_attr_mem_info_vram_vendor.attr &&
	    !adev->gmc.vram_vendor)
		return 0;

	return attr->mode;
}

const struct attribute_group amdgpu_vram_mgr_attr_group = {
	.attrs = amdgpu_vram_mgr_attributes,
	.is_visible = amdgpu_vram_attrs_is_visible
};

/**
 * amdgpu_vram_mgr_vis_size - Calculate visible block size
 *
 * @adev: amdgpu_device pointer
 * @block: DRM BUDDY block structure
 *
 * Calculate how many bytes of the DRM BUDDY block are inside visible VRAM
 */
static u64 amdgpu_vram_mgr_vis_size(struct amdgpu_device *adev,
				    struct drm_buddy_block *block)
{
	/// Functional Utility: Calculates the amount of a DRM buddy block that falls within the CPU-visible VRAM region.
	/// This function determines how much of an allocated VRAM block is directly accessible by the CPU.
	u64 start = amdgpu_vram_mgr_block_start(block);
	u64 end = start + amdgpu_vram_mgr_block_size(block);

	// Conditional Logic: If the block starts outside the visible VRAM, it contributes nothing.
	if (start >= adev->gmc.visible_vram_size)
		return 0;

	// Block Logic: Calculate the visible portion of the block.
	return (end > adev->gmc.visible_vram_size ?
		adev->gmc.visible_vram_size : end) - start;
}

/**
 * amdgpu_vram_mgr_bo_visible_size - CPU visible BO size
 *
 * @bo: &amdgpu_bo buffer object (must be in VRAM)
 *
 * Returns:
 * How much of the given &amdgpu_bo buffer object lies in CPU visible VRAM.
 */
u64 amdgpu_vram_mgr_bo_visible_size(struct amdgpu_bo *bo)
{
	/// Functional Utility: Calculates how much of a given AMDGPU buffer object (BO) in VRAM is CPU-visible.
	/// This function iterates through the BO's allocated blocks and sums up the CPU-visible portions,
	/// returning the total CPU-visible size in bytes.
	struct amdgpu_device *adev = amdgpu_ttm_adev(bo->tbo.bdev);
	struct ttm_resource *res = bo->tbo.resource;
	struct amdgpu_vram_mgr_resource *vres = to_amdgpu_vram_mgr_resource(res);
	struct drm_buddy_block *block;
	u64 usage = 0;

	// Conditional Logic: If VRAM is fully visible, return the total BO size.
	if (amdgpu_gmc_vram_full_visible(&adev->gmc))
		return amdgpu_bo_size(bo);

	// Conditional Logic: If the resource starts outside the visible VRAM, it contributes nothing.
	if (res->start >= adev->gmc.visible_vram_size >> PAGE_SHIFT)
		return 0;

	// Block Logic: Iterate through all blocks of the VRAM resource and sum their visible sizes.
	list_for_each_entry(block, &vres->blocks, link)
		usage += amdgpu_vram_mgr_vis_size(adev, block);

	return usage;
}

/* Commit the reservation of VRAM pages */
static void amdgpu_vram_mgr_do_reserve(struct ttm_resource_manager *man)
{
	/// Functional Utility: Commits VRAM reservation requests from the pending list.
	/// This function attempts to allocate specific VRAM ranges that were previously marked for reservation,
	/// updating visible usage and moving successful reservations to the `reserved_pages` list.
	struct amdgpu_vram_mgr *mgr = to_vram_mgr(man);
	struct amdgpu_device *adev = to_amdgpu_device(mgr);
	struct drm_buddy *mm = &mgr->mm;
	struct amdgpu_vram_reservation *rsv, *temp;
	struct drm_buddy_block *block;
	uint64_t vis_usage;

	// Block Logic: Iterate through pending reservations, attempting to allocate blocks for each.
	list_for_each_entry_safe(rsv, temp, &mgr->reservations_pending, blocks) {
		// Conditional Logic: Attempt to allocate blocks from the buddy allocator.
		if (drm_buddy_alloc_blocks(mm, rsv->start, rsv->start + rsv->size,
					   rsv->size, mm->chunk_size, &rsv->allocated,
					   DRM_BUDDY_RANGE_ALLOCATION))
			continue;

		block = amdgpu_vram_mgr_first_block(&rsv->allocated);
		// Conditional Logic: Skip if no blocks were allocated for the reservation.
		if (!block)
			continue;

		dev_dbg(adev->dev, "Reservation 0x%llx - %lld, Succeeded\n",
			rsv->start, rsv->size);

		// Block Logic: Update visible VRAM usage and TTM manager usage.
		vis_usage = amdgpu_vram_mgr_vis_size(adev, block);
		atomic64_add(vis_usage, &mgr->vis_usage);
		spin_lock(&man->bdev->lru_lock);
		man->usage += rsv->size;
		spin_unlock(&man->bdev->lru_lock);
		// Block Logic: Move successful reservation to the reserved_pages list.
		list_move(&rsv->blocks, &mgr->reserved_pages);
	}
}

/**
 * amdgpu_vram_mgr_reserve_range - Reserve a range from VRAM
 *
 * @mgr: amdgpu_vram_mgr pointer
 * @start: start address of the range in VRAM
 * @size: size of the range
 *
 * Reserve memory from start address with the specified size in VRAM
 */
int amdgpu_vram_mgr_reserve_range(struct amdgpu_vram_mgr *mgr,
				  uint64_t start, uint64_t size)
{
	/// Functional Utility: Adds a new VRAM range reservation request to the pending list.
	/// This function allocates a `amdgpu_vram_reservation` structure and initiates the reservation process,
	/// which will be finalized by `amdgpu_vram_mgr_do_reserve`.
	struct amdgpu_vram_reservation *rsv;

	rsv = kzalloc(sizeof(*rsv), GFP_KERNEL);
	// Conditional Logic: Return error if memory allocation for reservation fails.
	if (!rsv)
		return -ENOMEM;

	INIT_LIST_HEAD(&rsv->allocated);
	INIT_LIST_HEAD(&rsv->blocks);

	rsv->start = start;
	rsv->size = size;

	mutex_lock(&mgr->lock);
	// Block Logic: Add the reservation to the pending list and attempt to commit it.
	list_add_tail(&rsv->blocks, &mgr->reservations_pending);
	amdgpu_vram_mgr_do_reserve(&mgr->manager);
	mutex_unlock(&mgr->lock);

	return 0;
}

/**
 * amdgpu_vram_mgr_query_page_status - query the reservation status
 *
 * @mgr: amdgpu_vram_mgr pointer
 * @start: start address of a page in VRAM
 *
 * Returns:
 *	-EBUSY: the page is still hold and in pending list
 *	0: the page has been reserved
 *	-ENOENT: the input page is not a reservation
 */
int amdgpu_vram_mgr_query_page_status(struct amdgpu_vram_mgr *mgr,
				      uint64_t start)
{
	/// Functional Utility: Queries the reservation status of a specific VRAM page.
	/// This function checks both pending and committed reservations to determine if a page is reserved and its current status.
	struct amdgpu_vram_reservation *rsv;
	int ret;

	mutex_lock(&mgr->lock);

	// Block Logic: Iterate through pending reservations to check for the page.
	list_for_each_entry(rsv, &mgr->reservations_pending, blocks) {
		// Conditional Logic: If the page is found in a pending reservation, return -EBUSY.
		if (rsv->start <= start &&
		    (start < (rsv->start + rsv->size))) {
			ret = -EBUSY;
			goto out;
		}
	}

	// Block Logic: Iterate through committed reservations to check for the page.
	list_for_each_entry(rsv, &mgr->reserved_pages, blocks) {
		// Conditional Logic: If the page is found in a committed reservation, return 0.
		if (rsv->start <= start &&
		    (start < (rsv->start + rsv->size))) {
			ret = 0;
			goto out;
		}
	}

	ret = -ENOENT;
out:
	mutex_unlock(&mgr->lock);
	return ret;
}

static void amdgpu_dummy_vram_mgr_debug(struct ttm_resource_manager *man,
				  struct drm_printer *printer)
{
	/// Functional Utility: Dummy debug function for the VRAM manager, used when a full VRAM manager is not set up.
	/// This placeholder prints a debug message indicating that a dummy manager is in use.
	DRM_DEBUG_DRIVER("Dummy vram mgr debug\n");
}

static bool amdgpu_dummy_vram_mgr_compatible(struct ttm_resource_manager *man,
				       struct ttm_resource *res,
				       const struct ttm_place *place,
				       size_t size)
{
	/// Functional Utility: Dummy compatibility check function for the VRAM manager.
	/// This placeholder function always returns false, indicating that a dummy manager does not support compatibility checks.
	DRM_DEBUG_DRIVER("Dummy vram mgr compatible\n");
	return false;
}

static bool amdgpu_dummy_vram_mgr_intersects(struct ttm_resource_manager *man,
				       struct ttm_resource *res,
				       const struct ttm_place *place,
				       size_t size)
{
	/// Functional Utility: Dummy intersection check function for the VRAM manager.
	/// This placeholder function always returns true, indicating that a dummy manager does not perform actual intersection checks.
	DRM_DEBUG_DRIVER("Dummy vram mgr intersects\n");
	return true;
}

static void amdgpu_dummy_vram_mgr_del(struct ttm_resource_manager *man,
				struct ttm_resource *res)
{
	/// Functional Utility: Dummy resource deletion function for the VRAM manager.
	/// This placeholder prints a debug message indicating that a dummy manager is handling resource deletion.
	DRM_DEBUG_DRIVER("Dummy vram mgr deleted\n");
}

static int amdgpu_dummy_vram_mgr_new(struct ttm_resource_manager *man,
			       struct ttm_buffer_object *tbo,
			       const struct ttm_place *place,
			       struct ttm_resource **res)
{
	/// Functional Utility: Dummy allocation function for the VRAM manager.
	/// This placeholder always returns -ENOSPC, indicating that a dummy manager does not support new allocations.
	DRM_DEBUG_DRIVER("Dummy vram mgr new\n");
	return -ENOSPC;
}

/**
 * amdgpu_vram_mgr_new - allocate new ranges
 *
 * @man: TTM memory type manager
 * @tbo: TTM BO we need this range for
 * @place: placement flags and restrictions
 * @res: the resulting mem object
 *
 * Allocate VRAM for the given BO.
 */
static int amdgpu_vram_mgr_new(struct ttm_resource_manager *man,
			       struct ttm_buffer_object *tbo,
			       const struct ttm_place *place,
			       struct ttm_resource **res)
{
	/// Functional Utility: Allocates VRAM resources for a given TTM buffer object (BO).
	/// This function uses the DRM buddy allocator to find and reserve contiguous or fragmented VRAM blocks
	/// based on placement flags, BO properties, and ASIC capabilities.
	struct amdgpu_vram_mgr *mgr = to_vram_mgr(man);
	struct amdgpu_device *adev = to_amdgpu_device(mgr);
	struct amdgpu_bo *bo = ttm_to_amdgpu_bo(tbo);
	u64 vis_usage = 0, max_bytes, min_block_size;
	struct amdgpu_vram_mgr_resource *vres;
	u64 size, remaining_size, lpfn, fpfn;
	unsigned int adjust_dcc_size = 0;
	struct drm_buddy *mm = &mgr->mm;
	struct drm_buddy_block *block;
	unsigned long pages_per_block;
	int r;

	lpfn = (u64)place->lpfn << PAGE_SHIFT;
	// Conditional Logic: Adjust lpfn if it's 0 or greater than manager size.
	if (!lpfn || lpfn > man->size)
		lpfn = man->size;

	fpfn = (u64)place->fpfn << PAGE_SHIFT;

	max_bytes = adev->gmc.mc_vram_size;
	// Conditional Logic: Adjust max_bytes if it's not a kernel TBO.
	if (tbo->type != ttm_bo_type_kernel)
		max_bytes -= AMDGPU_VM_RESERVED_VRAM;

	// Conditional Logic: Set pages_per_block for contiguous VRAM.
	if (bo->flags & AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS) {
		pages_per_block = ~0ul;
	} else {
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
		pages_per_block = HPAGE_PMD_NR;
#else
		/* default to 2MB */
		pages_per_block = 2UL << (20UL - PAGE_SHIFT);
#endif
		pages_per_block = max_t(u32, pages_per_block,
					tbo->page_alignment);
	}

	vres = kzalloc(sizeof(*vres), GFP_KERNEL);
	// Conditional Logic: Return error if memory allocation for vres fails.
	if (!vres)
		return -ENOMEM;

	ttm_resource_init(tbo, place, &vres->base);

	/* bail out quickly if there's likely not enough VRAM for this BO */
	// Conditional Logic: Return error if VRAM usage exceeds maximum allowed bytes.
	if (ttm_resource_manager_usage(man) > max_bytes) {
		r = -ENOSPC;
		goto error_fini;
	}

	INIT_LIST_HEAD(&vres->blocks);

	// Conditional Logic: Set topdown allocation flag if specified.
	if (place->flags & TTM_PL_FLAG_TOPDOWN)
		vres->flags |= DRM_BUDDY_TOPDOWN_ALLOCATION;

	// Conditional Logic: Set contiguous allocation flag if specified.
	if (bo->flags & AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS)
		vres->flags |= DRM_BUDDY_CONTIGUOUS_ALLOCATION;

	// Conditional Logic: Set cleared allocation flag if specified.
	if (bo->flags & AMDGPU_GEM_CREATE_VRAM_CLEARED)
		vres->flags |= DRM_BUDDY_CLEAR_ALLOCATION;

	// Conditional Logic: Set range allocation flag if fpfn is set or lpfn is not mgr->mm.size.
	if (fpfn || lpfn != mgr->mm.size)
		/* Allocate blocks in desired range */
		vres->flags |= DRM_BUDDY_RANGE_ALLOCATION;

	// Conditional Logic: Adjust DCC size if GFX12 DCC is enabled and get_dcc_alignment is available.
	if (bo->flags & AMDGPU_GEM_CREATE_GFX12_DCC &&
	    adev->gmc.gmc_funcs->get_dcc_alignment)
		adjust_dcc_size = amdgpu_gmc_get_dcc_alignment(adev);

	remaining_size = (u64)vres->base.size;
	// Conditional Logic: Adjust remaining size for contiguous VRAM with DCC.
	if (bo->flags & AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS && adjust_dcc_size) {
		unsigned int dcc_size;

		dcc_size = roundup_pow_of_two(vres->base.size + adjust_dcc_size);
		remaining_size = (u64)dcc_size;

		vres->flags |= DRM_BUDDY_TRIM_DISABLE;
	}

	mutex_lock(&mgr->lock);
	// Block Logic: Loop until all remaining size is allocated.
	while (remaining_size) {
		// Conditional Logic: Determine minimum block size based on page alignment or default.
		if (tbo->page_alignment)
			min_block_size = (u64)tbo->page_alignment << PAGE_SHIFT;
		else
			min_block_size = mgr->default_page_size;

		size = remaining_size;

		// Conditional Logic: Adjust min_block_size for contiguous VRAM with DCC or pages_per_block.
		if (bo->flags & AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS && adjust_dcc_size)
			min_block_size = size;
		else if ((size >= (u64)pages_per_block << PAGE_SHIFT) &&
			 !(size & (((u64)pages_per_block << PAGE_SHIFT) - 1)))
			min_block_size = (u64)pages_per_block << PAGE_SHIFT;

		BUG_ON(min_block_size < mm->chunk_size);

		// Block Logic: Attempt to allocate blocks from the buddy allocator.
		r = drm_buddy_alloc_blocks(mm, fpfn,
					   lpfn,
					   size,
					   min_block_size,
					   &vres->blocks,
					   vres->flags);

		// Conditional Logic: Handle allocation failure and retry for non-contiguous if applicable.
		if (unlikely(r == -ENOSPC) && pages_per_block == ~0ul &&
		    !(place->flags & TTM_PL_FLAG_CONTIGUOUS)) {
			vres->flags &= ~DRM_BUDDY_CONTIGUOUS_ALLOCATION;
			pages_per_block = max_t(u32, 2UL << (20UL - PAGE_SHIFT),
						tbo->page_alignment);

			continue;
		}

		// Conditional Logic: If allocation fails, jump to error handling.
		if (unlikely(r))
			goto error_free_blocks;

		// Conditional Logic: Update remaining size after allocation.
		if (size > remaining_size)
			remaining_size = 0;
		else
			remaining_size -= size;
	}

	// Conditional Logic: Adjust block for contiguous VRAM with DCC.
	if (bo->flags & AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS && adjust_dcc_size) {
		struct drm_buddy_block *dcc_block;
		unsigned long dcc_start;
		u64 trim_start;

		dcc_block = amdgpu_vram_mgr_first_block(&vres->blocks);
		/* Adjust the start address for DCC buffers only */
		dcc_start =
			roundup((unsigned long)amdgpu_vram_mgr_block_start(dcc_block),
				adjust_dcc_size);
		trim_start = (u64)dcc_start;
		drm_buddy_block_trim(mm, &trim_start,
				     (u64)vres->base.size,
				     &vres->blocks);
	}
	mutex_unlock(&mgr->lock);

	vres->base.start = 0;
	size = max_t(u64, amdgpu_vram_mgr_blocks_size(&vres->blocks),
		     vres->base.size);
	// Block Logic: Determine the base start address and calculate visible usage.
	list_for_each_entry(block, &vres->blocks, link) {
		unsigned long start;

		start = amdgpu_vram_mgr_block_start(block) +
			amdgpu_vram_mgr_block_size(block);
		start >>= PAGE_SHIFT;

		if (start > PFN_UP(size))
			start -= PFN_UP(size);
		else
			start = 0;
		vres->base.start = max(vres->base.start, start);

		vis_usage += amdgpu_vram_mgr_vis_size(adev, block);
	}

	// Conditional Logic: Set contiguous flag if blocks are contiguous.
	if (amdgpu_is_vram_mgr_blocks_contiguous(&vres->blocks))
		vres->base.placement |= TTM_PL_FLAG_CONTIGUOUS;

	// Conditional Logic: Set caching type based on XGMI connection.
	if (adev->gmc.xgmi.connected_to_cpu)
		vres->base.bus.caching = ttm_cached;
	else
		vres->base.bus.caching = ttm_write_combined;

	atomic64_add(vis_usage, &mgr->vis_usage);
	*res = &vres->base;
	return 0;

error_free_blocks:
	drm_buddy_free_list(mm, &vres->blocks, 0);
	mutex_unlock(&mgr->lock);
error_fini:
	ttm_resource_fini(man, &vres->base);
	kfree(vres);

	return r;
}

/**
 * amdgpu_vram_mgr_del - free ranges
 *
 * @man: TTM memory type manager
 * @res: TTM memory object
 *
 * Free the allocated VRAM again.
 */
static void amdgpu_vram_mgr_del(struct ttm_resource_manager *man,
				struct ttm_resource *res)
{
	/// Functional Utility: Deallocates VRAM resources previously allocated for a TTM resource.
	/// This function frees the DRM buddy allocator blocks associated with the resource and updates VRAM usage statistics.
	struct amdgpu_vram_mgr_resource *vres = to_amdgpu_vram_mgr_resource(res);
	struct amdgpu_vram_mgr *mgr = to_vram_mgr(man);
	struct amdgpu_device *adev = to_amdgpu_device(mgr);
	struct drm_buddy *mm = &mgr->mm;
	struct drm_buddy_block *block;
	uint64_t vis_usage = 0;

	mutex_lock(&mgr->lock);
	// Block Logic: Calculate visible usage of the blocks to be freed.
	list_for_each_entry(block, &vres->blocks, link)
		vis_usage += amdgpu_vram_mgr_vis_size(adev, block);

	amdgpu_vram_mgr_do_reserve(man);

	// Block Logic: Free the DRM buddy allocator blocks.
	drm_buddy_free_list(mm, &vres->blocks, vres->flags);
	mutex_unlock(&mgr->lock);

	// Block Logic: Decrement visible VRAM usage.
	atomic64_sub(vis_usage, &mgr->vis_usage);

	ttm_resource_fini(man, res);
	kfree(vres);
}

/**
 * amdgpu_vram_mgr_alloc_sgt - allocate and fill a sg table
 *
 * @adev: amdgpu device pointer
 * @res: TTM memory object
 * @offset: byte offset from the base of VRAM BO
 * @length: number of bytes to export in sg_table
 * @dev: the other device
 * @dir: dma direction
 * @sgt: resulting sg table
 *
 * Allocate and fill a sg table from a VRAM allocation.
 */
int amdgpu_vram_mgr_alloc_sgt(struct amdgpu_device *adev,
			      struct ttm_resource *res,
			      u64 offset, u64 length,
			      struct device *dev,
			      enum dma_data_direction dir,
			      struct sg_table **sgt)
{
	/// Functional Utility: Allocates and populates a scatter-gather table (SGT) for a given VRAM resource.
	/// This function maps portions of VRAM into DMA-addressable memory, enabling data transfer to/from other devices.
	struct amdgpu_res_cursor cursor;
	struct scatterlist *sg;
	int num_entries = 0;
	int i, r;

	*sgt = kmalloc(sizeof(**sgt), GFP_KERNEL);
	// Conditional Logic: Return error if SGT allocation fails.
	if (!*sgt)
		return -ENOMEM;

	/* Determine the number of DRM_BUDDY blocks to export */
	// Block Logic: Iterate through the resource to count the number of scatterlist entries needed.
	amdgpu_res_first(res, offset, length, &cursor);
	while (cursor.remaining) {
		num_entries++;
		amdgpu_res_next(&cursor, min(cursor.size, AMDGPU_MAX_SG_SEGMENT_SIZE));
	}

	r = sg_alloc_table(*sgt, num_entries, GFP_KERNEL);
	// Conditional Logic: Return error if scatter-gather table allocation fails.
	if (r)
		goto error_free;

	/* Initialize scatterlist nodes of sg_table */
	// Block Logic: Initialize scatterlist lengths to 0.
	for_each_sgtable_sg((*sgt), sg, i)
		sg->length = 0;

	/*
	 * Walk down DRM_BUDDY blocks to populate scatterlist nodes
	 * @note: Use iterator api to get first the DRM_BUDDY block
	 * and the number of bytes from it. Access the following
	 * DRM_BUDDY block(s) if more buffer needs to exported
	 */
	// Block Logic: Populate scatterlist entries with DMA-mapped VRAM blocks.
	amdgpu_res_first(res, offset, length, &cursor);
	for_each_sgtable_sg((*sgt), sg, i) {
		phys_addr_t phys = cursor.start + adev->gmc.aper_base;
		unsigned long size = min(cursor.size, AMDGPU_MAX_SG_SEGMENT_SIZE);
		dma_addr_t addr;

		// Block Logic: DMA map the VRAM segment.
		addr = dma_map_resource(dev, phys, size, dir,
					DMA_ATTR_SKIP_CPU_SYNC);
		// Conditional Logic: Handle DMA mapping errors.
		r = dma_mapping_error(dev, addr);
		if (r)
			goto error_unmap;

		// Block Logic: Set scatterlist entry details.
		sg_set_page(sg, NULL, size, 0);
		sg_dma_address(sg) = addr;
		sg_dma_len(sg) = size;

		amdgpu_res_next(&cursor, size);
	}

	return 0;

error_unmap:
	// Error Handling: Unmap DMA resources on error.
	for_each_sgtable_sg((*sgt), sg, i) {
		if (!sg->length)
			continue;

		dma_unmap_resource(dev, sg->dma_address,
				   sg->length, dir,
				   DMA_ATTR_SKIP_CPU_SYNC);
	}
	sg_free_table(*sgt);

error_free:
	// Error Handling: Free SGT on error.
	kfree(*sgt);
	return r;
}

/**
 * amdgpu_vram_mgr_free_sgt - allocate and fill a sg table
 *
 * @dev: device pointer
 * @dir: data direction of resource to unmap
 * @sgt: sg table to free
 *
 * Free a previously allocate sg table.
 */
void amdgpu_vram_mgr_free_sgt(struct device *dev,
			      enum dma_data_direction dir,
			      struct sg_table *sgt)
{
	/// Functional Utility: Frees a previously allocated scatter-gather table (SGT) and unmaps its DMA resources.
	/// This function reverses the operations performed by `amdgpu_vram_mgr_alloc_sgt`, releasing the mapped memory.
	struct scatterlist *sg;
	int i;

	// Block Logic: Iterate through scatterlist entries and unmap DMA resources.
	for_each_sgtable_sg(sgt, sg, i)
		dma_unmap_resource(dev, sg->dma_address,
				   sg->length, dir,
				   DMA_ATTR_SKIP_CPU_SYNC);
	// Block Logic: Free the scatter-gather table.
	sg_free_table(sgt);
	kfree(sgt);
}

/**
 * amdgpu_vram_mgr_vis_usage - how many bytes are used in the visible part
 *
 * @mgr: amdgpu_vram_mgr pointer
 *
 * Returns how many bytes are used in the visible part of VRAM
 */
uint64_t amdgpu_vram_mgr_vis_usage(struct amdgpu_vram_mgr *mgr)
{
	/// Functional Utility: Reports the total amount of VRAM that is currently CPU-visible and in use.
	/// This function reads an atomic counter to provide the current visible VRAM usage.
	return atomic64_read(&mgr->vis_usage);
}

/**
 * amdgpu_vram_mgr_intersects - test each drm buddy block for intersection
 *
 * @man: TTM memory type manager
 * @res: The resource to test
 * @place: The place to test against
 * @size: Size of the new allocation
 *
 * Test each drm buddy block for intersection for eviction decision.
 */
static bool amdgpu_vram_mgr_intersects(struct ttm_resource_manager *man,
				       struct ttm_resource *res,
				       const struct ttm_place *place,
				       size_t size)
{
	/// Functional Utility: Checks if a VRAM resource (composed of DRM buddy blocks) intersects with a specified TTM placement range.
	/// This function is used by the TTM eviction logic to determine if an existing resource needs to be moved.
	struct amdgpu_vram_mgr_resource *mgr = to_amdgpu_vram_mgr_resource(res);
	struct drm_buddy_block *block;

	/* Check each drm buddy block individually */
	// Block Logic: Iterate through each DRM buddy block of the resource.
	list_for_each_entry(block, &mgr->blocks, link) {
		unsigned long fpfn =
			amdgpu_vram_mgr_block_start(block) >> PAGE_SHIFT;
		unsigned long lpfn = fpfn +
			(amdgpu_vram_mgr_block_size(block) >> PAGE_SHIFT);

		// Conditional Logic: Check for intersection between the block's range and the placement's range.
		if (place->fpfn < lpfn &&
		    (!place->lpfn || place->lpfn > fpfn))
			return true;
	}

	return false;
}

/**
 * amdgpu_vram_mgr_compatible - test each drm buddy block for compatibility
 *
 * @man: TTM memory type manager
 * @res: The resource to test
 * @place: The place to test against
 * @size: Size of the new allocation
 *
 * Test each drm buddy block for placement compatibility.
 */
static bool amdgpu_vram_mgr_compatible(struct ttm_resource_manager *man,
				       struct ttm_resource *res,
				       const struct ttm_place *place,
				       size_t size)
{
	/// Functional Utility: Checks if a VRAM resource (composed of DRM buddy blocks) is compatible with a specified TTM placement range.
	/// This function is used by the TTM eviction logic to determine if an existing resource can be kept in its current location.
	struct amdgpu_vram_mgr_resource *mgr = to_amdgpu_vram_mgr_resource(res);
	struct drm_buddy_block *block;

	/* Check each drm buddy block individually */
	// Block Logic: Iterate through each DRM buddy block of the resource.
	list_for_each_entry(block, &mgr->blocks, link) {
		unsigned long fpfn =
			amdgpu_vram_mgr_block_start(block) >> PAGE_SHIFT;
		unsigned long lpfn = fpfn +
			(amdgpu_vram_mgr_block_size(block) >> PAGE_SHIFT);

		// Conditional Logic: Check if the block's range falls within the placement's range.
		if (fpfn < place->fpfn ||
		    (place->lpfn && lpfn > place->lpfn))
			return false;
	}

	return true;
}

/**
 * amdgpu_vram_mgr_debug - dump VRAM table
 *
 * @man: TTM memory type manager
 * @printer: DRM printer to use
 *
 * Dump the table content using printk.
 */
static void amdgpu_vram_mgr_debug(struct ttm_resource_manager *man,
				  struct drm_printer *printer)
{
	/// Functional Utility: Dumps the current state of the VRAM manager to a DRM printer.
	/// This function provides detailed debug information about VRAM usage, default page size,
	/// buddy allocator state, and reserved VRAM ranges.
	struct amdgpu_vram_mgr *mgr = to_vram_mgr(man);
	struct drm_buddy *mm = &mgr->mm;
	struct amdgpu_vram_reservation *rsv;

	drm_printf(printer, "  vis usage:%llu\n",
		   amdgpu_vram_mgr_vis_usage(mgr));

	mutex_lock(&mgr->lock);
	drm_printf(printer, "default_page_size: %lluKiB\n",
		   mgr->default_page_size >> 10);

	drm_buddy_print(mm, printer);

	drm_printf(printer, "reserved:\n");
	// Block Logic: Iterate through reserved pages and print their ranges.
	list_for_each_entry(rsv, &mgr->reserved_pages, blocks)
		drm_printf(printer, "%#018llx-%#018llx: %llu\n",
			rsv->start, rsv->start + rsv->size, rsv->size);
	mutex_unlock(&mgr->lock);
}

static const struct ttm_resource_manager_func amdgpu_dummy_vram_mgr_func = {
	.alloc	= amdgpu_dummy_vram_mgr_new,
	.free	= amdgpu_dummy_vram_mgr_del,
	.intersects = amdgpu_dummy_vram_mgr_intersects,
	.compatible = amdgpu_dummy_vram_mgr_compatible,
	.debug	= amdgpu_dummy_vram_mgr_debug
};

static const struct ttm_resource_manager_func amdgpu_vram_mgr_func = {
	.alloc	= amdgpu_vram_mgr_new,
	.free	= amdgpu_vram_mgr_del,
	.intersects = amdgpu_vram_mgr_intersects,
	.compatible = amdgpu_vram_mgr_compatible,
	.debug	= amdgpu_vram_mgr_debug
};

/**
 * amdgpu_vram_mgr_init - init VRAM manager and DRM MM
 *
 * @adev: amdgpu_device pointer
 *
 * Allocate and initialize the VRAM manager.
 */
int amdgpu_vram_mgr_init(struct amdgpu_device *adev)
{
	/// Functional Utility: Initializes the VRAM manager and the DRM buddy allocator for the AMDGPU device.
	/// This function sets up the data structures and callbacks required for VRAM allocation and management.
	struct amdgpu_vram_mgr *mgr = &adev->mman.vram_mgr;
	struct ttm_resource_manager *man = &mgr->manager;
	int err;

	man->cg = drmm_cgroup_register_region(adev_to_drm(adev), "vram", adev->gmc.real_vram_size);
	// Conditional Logic: Handle cgroup registration failure.
	if (IS_ERR(man->cg))
		return PTR_ERR(man->cg);
	ttm_resource_manager_init(man, &adev->mman.bdev,
				  adev->gmc.real_vram_size);

	mutex_init(&mgr->lock);
	INIT_LIST_HEAD(&mgr->reservations_pending);
	INIT_LIST_HEAD(&mgr->reserved_pages);
	mgr->default_page_size = PAGE_SIZE;

	// Conditional Logic: Initialize the VRAM manager based on whether it's an APU application.
	if (!adev->gmc.is_app_apu) {
		man->func = &amdgpu_vram_mgr_func;

		err = drm_buddy_init(&mgr->mm, man->size, PAGE_SIZE);
		// Conditional Logic: Handle DRM buddy allocator initialization failure.
		if (err)
			return err;
	} else {
		// Block Logic: Use dummy VRAM manager functions for APU applications.
		man->func = &amdgpu_dummy_vram_mgr_func;
		DRM_INFO("Setup dummy vram mgr\n");
	}

	ttm_set_driver_manager(&adev->mman.bdev, TTM_PL_VRAM, &mgr->manager);
	ttm_resource_manager_set_used(man, true);
	return 0;
}

/**
 * amdgpu_vram_mgr_fini - free and destroy VRAM manager
 *
 * @adev: amdgpu_device pointer
 *
 * Destroy and free the VRAM manager, returns -EBUSY if ranges are still
 * allocated inside it.
 */
void amdgpu_vram_mgr_fini(struct amdgpu_device *adev)
{
	/// Functional Utility: Destroys and frees the VRAM manager, releasing all allocated resources.
	/// This function cleans up the TTM resource manager, DRM buddy allocator, and any pending or reserved VRAM ranges.
	struct amdgpu_vram_mgr *mgr = &adev->mman.vram_mgr;
	struct ttm_resource_manager *man = &mgr->manager;
	int ret;
	struct amdgpu_vram_reservation *rsv, *temp;

	ttm_resource_manager_set_used(man, false);

	ret = ttm_resource_manager_evict_all(&adev->mman.bdev, man);
	// Conditional Logic: Return if eviction fails.
	if (ret)
		return;

	mutex_lock(&mgr->lock);
	// Block Logic: Free pending reservations.
	list_for_each_entry_safe(rsv, temp, &mgr->reservations_pending, blocks)
		kfree(rsv);

	// Block Logic: Free reserved pages and their allocated blocks.
	list_for_each_entry_safe(rsv, temp, &mgr->reserved_pages, blocks) {
		drm_buddy_free_list(&mgr->mm, &rsv->allocated, 0);
		kfree(rsv);
	}
	// Conditional Logic: Finalize DRM buddy allocator if not an APU application.
	if (!adev->gmc.is_app_apu)
		drm_buddy_fini(&mgr->mm);
	mutex_unlock(&mgr->lock);

	ttm_resource_manager_cleanup(man);
	ttm_set_driver_manager(&adev->mman.bdev, TTM_PL_VRAM, NULL);
}
