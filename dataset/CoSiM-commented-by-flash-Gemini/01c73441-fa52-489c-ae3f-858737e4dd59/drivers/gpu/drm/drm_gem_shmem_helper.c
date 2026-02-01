/**
 * @file drm_gem_shmem_helper.c
 * @brief Provides helper functions for managing DRM GEM objects backed by shmem (shared memory) buffers.
 * These helpers are crucial for efficient memory sharing and mapping within the DRM subsystem,
 * often used for userspace visible buffers.
 *
 * Functional Utility: Simplifies the creation, management, and sharing of GEM objects
 * that use anonymous pageable memory as their backing store. This includes pinning/unpinning
 * pages, creating scatter/gather tables, virtual memory mapping, and handling madvise states.
 *
 * Key Concepts:
 * - **GEM (Graphics Execution Manager) Objects**: Generic buffer objects managed by DRM.
 * - **shmem (Shared Memory)**: Anonymous, pageable memory used as backing store.
 * - **Pinning/Unpinning**: Managing the resident state of memory pages.
 * - **Scatter/Gather Table**: Describes memory layout for DMA operations.
 * - **VMA (Virtual Memory Area)**: Kernel structures for managing memory mappings.
 */

// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright 2018 Noralf Trønnes
 */

#include <linux/dma-buf.h>
#include <linux/export.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/shmem_fs.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

#ifdef CONFIG_X86
#include <asm/set_memory.h>
#endif

#include <drm/drm.h>
#include <drm/drm_device.h>
#include <drm/drm_drv.h>
#include <drm/drm_gem_shmem_helper.h>
#include <drm/drm_prime.h>
#include <drm/drm_print.h>

MODULE_IMPORT_NS("DMA_BUF");

/**
 * DOC: overview
 *
 * This library provides helpers for GEM objects backed by shmem buffers
 * allocated using anonymous pageable memory.
 *
 * Functions that operate on the GEM object receive struct &drm_gem_shmem_object.
 * For GEM callback helpers in struct &drm_gem_object functions, see likewise
 * named functions with an _object_ infix (e.g., drm_gem_shmem_object_vmap() wraps
 * drm_gem_shmem_vmap()). These helpers perform the necessary type conversion.
 */

static const struct drm_gem_object_funcs drm_gem_shmem_funcs = {
	.free = drm_gem_shmem_object_free,
	.print_info = drm_gem_shmem_object_print_info,
	.pin = drm_gem_shmem_object_pin,
	.unpin = drm_gem_shmem_object_unpin,
	.get_sg_table = drm_gem_shmem_object_get_sg_table,
	.vmap = drm_gem_shmem_object_vmap,
	.vunmap = drm_gem_shmem_object_vunmap,
	.mmap = drm_gem_shmem_object_mmap,
	.vm_ops = &drm_gem_shmem_vm_ops,
};

static struct drm_gem_shmem_object *
__drm_gem_shmem_create(struct drm_device *dev, size_t size, bool private,
		       struct vfsmount *gemfs)
{
	/// Functional Utility: Creates and initializes a shmem GEM object structure without allocating its backing memory.
	/// This is an internal helper that sets up the base GEM object, assigns default functions, and prepares for memory allocation.
	struct drm_gem_shmem_object *shmem;
	struct drm_gem_object *obj;
	int ret = 0;

	// Block Logic: Align the requested size to page boundaries.
	size = PAGE_ALIGN(size);

	// Conditional Logic: Check if the driver provides a custom GEM object creation function.
	if (dev->driver->gem_create_object) {
		// Block Logic: Use the driver's custom function to create the GEM object.
		obj = dev->driver->gem_create_object(dev, size);
		// Conditional Logic: Handle potential errors from the custom creation function.
		if (IS_ERR(obj))
			return ERR_CAST(obj);
		// Block Logic: Convert the generic GEM object to a shmem GEM object.
		shmem = to_drm_gem_shmem_obj(obj);
	} else {
		// Block Logic: Allocate memory for the shmem GEM object structure.
		shmem = kzalloc(sizeof(*shmem), GFP_KERNEL);
		// Conditional Logic: Handle memory allocation failure.
		if (!shmem)
			return ERR_PTR(-ENOMEM);
		// Block Logic: Point the generic GEM object to the base of the shmem GEM object.
		obj = &shmem->base;
	}

	// Conditional Logic: If GEM object functions are not set, assign default shmem GEM functions.
	if (!obj->funcs)
		obj->funcs = &drm_gem_shmem_funcs;

	// Conditional Logic: Initialize the GEM object based on whether it's for private use.
	if (private) {
		// Block Logic: Initialize a private GEM object.
		drm_gem_private_object_init(dev, obj, size);
		// Block Logic: For private objects, writecombine mapping is disabled as dma-buf mappings always use it.
		shmem->map_wc = false; /* dma-buf mappings use always writecombine */
	} else {
		// Block Logic: Initialize a regular GEM object with a given mountpoint.
		ret = drm_gem_object_init_with_mnt(dev, obj, size, gemfs);
	}
	// Conditional Logic: Handle initialization errors.
	if (ret) {
		drm_gem_private_object_fini(obj);
		goto err_free;
	}

	// Block Logic: Create a memory-map offset for the GEM object.
	ret = drm_gem_create_mmap_offset(obj);
	// Conditional Logic: Handle memory-map offset creation errors.
	if (ret)
		goto err_release;

	// Block Logic: Initialize the madvise list.
	INIT_LIST_HEAD(&shmem->madv_list);

	// Conditional Logic: If not a private object, set GFP mask for backing pages.
	if (!private) {
		/*
		 * Our buffers are kept pinned, so allocating them
		 * from the MOVABLE zone is a really bad idea, and
		 * conflicts with CMA. See comments above new_inode()
		 * why this is required _and_ expected if you're
		 * going to pin these pages.
		 */
		// Block Logic: Set the GFP mask for the file mapping to prevent allocation from MOVABLE zone.
		mapping_set_gfp_mask(obj->filp->f_mapping, GFP_HIGHUSER |
				     __GFP_RETRY_MAYFAIL | __GFP_NOWARN);
	}

	return shmem;

err_release:
	// Error Handling: Release the GEM object upon error.
	drm_gem_object_release(obj);
err_free:
	// Error Handling: Free the shmem object structure upon error.
	kfree(obj);

	return ERR_PTR(ret);
}
/**
 * drm_gem_shmem_create - Allocate an object with the given size
 * @dev: DRM device
 * @size: Size of the object to allocate
 *
 * This function creates a shmem GEM object.
 *
 * Returns:
 * A struct drm_gem_shmem_object * on success or an ERR_PTR()-encoded negative
 * error code on failure.
 */
struct drm_gem_shmem_object *drm_gem_shmem_create(struct drm_device *dev, size_t size)
{
	/// Functional Utility: Allocates a new shmem GEM object with a given size.
	/// This is a public interface for creating a general-purpose GEM object backed by shared memory.
	return __drm_gem_shmem_create(dev, size, false, NULL);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_create);

/**
 * drm_gem_shmem_create_with_mnt - Allocate an object with the given size in a
 * given mountpoint
 * @dev: DRM device
 * @size: Size of the object to allocate
 * @gemfs: tmpfs mount where the GEM object will be created
 *
 * This function creates a shmem GEM object in a given tmpfs mountpoint.
 *
 * Returns:
 * A struct drm_gem_shmem_object * on success or an ERR_PTR()-encoded negative
 * error code on failure.
 */
struct drm_gem_shmem_object *drm_gem_shmem_create_with_mnt(struct drm_device *dev,
							   size_t size,
							   struct vfsmount *gemfs)
{
	/// Functional Utility: Allocates a new shmem GEM object within a specified `tmpfs` mountpoint.
	/// This allows for fine-grained control over where the shared memory object resides in the filesystem hierarchy.
	return __drm_gem_shmem_create(dev, size, false, gemfs);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_create_with_mnt);

/**
 * drm_gem_shmem_free - Free resources associated with a shmem GEM object
 * @shmem: shmem GEM object to free
 *
 * This function cleans up the GEM object state and frees the memory used to
 * store the object itself.
 */
void drm_gem_shmem_free(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Frees all resources associated with a shmem GEM object, including its backing pages and GEM object state.
	/// It handles both locally allocated and imported shmem buffers, and performs necessary DMA unmapping and table freeing.
	struct drm_gem_object *obj = &shmem->base;

	// Conditional Logic: If the GEM object is imported, destroy it as a PRIME GEM object.
	if (drm_gem_is_imported(obj)) {
		drm_prime_gem_destroy(obj, shmem->sgt);
	} else {
		// Block Logic: Acquire DMA reservation lock.
		dma_resv_lock(shmem->base.resv, NULL);

		drm_WARN_ON(obj->dev, refcount_read(&shmem->vmap_use_count));

		// Conditional Logic: If a scatter/gather table exists, unmap and free it.
		if (shmem->sgt) {
			dma_unmap_sgtable(obj->dev->dev, shmem->sgt,
					  DMA_BIDIRECTIONAL, 0);
			sg_free_table(shmem->sgt);
			kfree(shmem->sgt);
		}
		// Conditional Logic: If backing pages exist, put their references.
		if (shmem->pages)
			drm_gem_shmem_put_pages_locked(shmem);

		drm_WARN_ON(obj->dev, refcount_read(&shmem->pages_use_count));
		drm_WARN_ON(obj->dev, refcount_read(&shmem->pages_pin_count));

		// Block Logic: Release DMA reservation lock.
		dma_resv_unlock(shmem->base.resv);
	}

	// Block Logic: Release the base GEM object.
	drm_gem_object_release(obj);
	// Block Logic: Free the shmem GEM object structure.
	kfree(shmem);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_free);

static int drm_gem_shmem_get_pages_locked(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Acquires and pins the backing pages for a shmem GEM object, making them resident in memory.
	/// This is a locked internal helper that ensures pages are available for mapping or DMA operations.
	struct drm_gem_object *obj = &shmem->base;
	struct page **pages;

	dma_resv_assert_held(shmem->base.resv);

	// Conditional Logic: If pages are already in use, increment use count and return.
	if (refcount_inc_not_zero(&shmem->pages_use_count))
		return 0;

	// Block Logic: Get the backing pages for the GEM object.
	pages = drm_gem_get_pages(obj);
	// Conditional Logic: Handle errors during page acquisition.
	if (IS_ERR(pages)) {
		drm_dbg_kms(obj->dev, "Failed to get pages (%ld)\n",
			    PTR_ERR(pages));
		return PTR_ERR(pages);
	}

	/*
	 * TODO: Allocating WC pages which are correctly flushed is only
	 * supported on x86. Ideal solution would be a GFP_WC flag, which also
	 * ttm_pool.c could use.
	 */
#ifdef CONFIG_X86
	// Conditional Logic: If write-combine mapping is enabled, set pages array to write-combine.
	if (shmem->map_wc)
		set_pages_array_wc(pages, obj->size >> PAGE_SHIFT);
#endif

	// Block Logic: Store the acquired pages.
	shmem->pages = pages;

	// Block Logic: Set the initial use count for the pages.
	refcount_set(&shmem->pages_use_count, 1);

	return 0;
}

/*
 * drm_gem_shmem_put_pages_locked - Decrease use count on the backing pages for a shmem GEM object
 * @shmem: shmem GEM object
 *
 * This function decreases the use count and puts the backing pages when use drops to zero.
 */
void drm_gem_shmem_put_pages_locked(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Decreases the use count of the backing pages for a shmem GEM object and releases them when the count drops to zero.
	/// This is a locked internal helper that handles page unpinning and flushing (if write-combined).
	struct drm_gem_object *obj = &shmem->base;

	dma_resv_assert_held(shmem->base.resv);

	// Conditional Logic: If the use count drops to zero, release the pages.
	if (refcount_dec_and_test(&shmem->pages_use_count)) {
#ifdef CONFIG_X86
		// Conditional Logic: If write-combine mapping was used, set pages array back to write-back.
		if (shmem->map_wc)
			set_pages_array_wb(shmem->pages, obj->size >> PAGE_SHIFT);
#endif

		// Block Logic: Put the pages back to the system.
		drm_gem_put_pages(obj, shmem->pages,
				  shmem->pages_mark_dirty_on_put,
				  shmem->pages_mark_accessed_on_put);
		// Block Logic: Clear the pages pointer.
		shmem->pages = NULL;
	}
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_put_pages_locked);

int drm_gem_shmem_pin_locked(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Pins the backing pages of a shmem GEM object in memory and increments a pin count.
	/// This is a locked internal helper to prevent pages from being swapped out while in use.
	int ret;

	dma_resv_assert_held(shmem->base.resv);

	drm_WARN_ON(shmem->base.dev, drm_gem_is_imported(&shmem->base));

	// Conditional Logic: If pages are already pinned, increment pin count and return.
	if (refcount_inc_not_zero(&shmem->pages_pin_count))
		return 0;

	// Block Logic: Get the backing pages if not already obtained.
	ret = drm_gem_shmem_get_pages_locked(shmem);
	// Conditional Logic: If successful, set pin count to 1.
	if (!ret)
		refcount_set(&shmem->pages_pin_count, 1);

	return ret;
}
EXPORT_SYMBOL(drm_gem_shmem_pin_locked);

void drm_gem_shmem_unpin_locked(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Unpins the backing pages of a shmem GEM object from memory and decrements a pin count.
	/// This is a locked internal helper that releases pages when the pin count drops to zero.
	dma_resv_assert_held(shmem->base.resv);

	// Conditional Logic: If the pin count drops to zero, put the pages.
	if (refcount_dec_and_test(&shmem->pages_pin_count))
		drm_gem_shmem_put_pages_locked(shmem);
}
EXPORT_SYMBOL(drm_gem_shmem_unpin_locked);

/**
 * drm_gem_shmem_pin - Pin backing pages for a shmem GEM object
 * @shmem: shmem GEM object
 *
 * This function makes sure the backing pages are pinned in memory while the
 * buffer is exported.
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_shmem_pin(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Public interface to pin the backing pages of a shmem GEM object, ensuring they remain resident in memory.
	/// This function acquires a reservation lock and calls the internal locked pinning helper.
	struct drm_gem_object *obj = &shmem->base;
	int ret;

	drm_WARN_ON(obj->dev, drm_gem_is_imported(obj));

	// Conditional Logic: If pages are already pinned, increment pin count and return.
	if (refcount_inc_not_zero(&shmem->pages_pin_count))
		return 0;

	// Block Logic: Acquire DMA reservation lock.
	ret = dma_resv_lock_interruptible(shmem->base.resv, NULL);
	// Conditional Logic: Handle lock acquisition failure.
	if (ret)
		return ret;
	// Block Logic: Call internal locked pinning function.
	ret = drm_gem_shmem_pin_locked(shmem);
	// Block Logic: Release DMA reservation lock.
	dma_resv_unlock(shmem->base.resv);

	return ret;
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_pin);

/**
 * drm_gem_shmem_unpin - Unpin backing pages for a shmem GEM object
 * @shmem: shmem GEM object
 *
 * This function removes the requirement that the backing pages are pinned in
 * memory.
 */
void drm_gem_shmem_unpin(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Public interface to unpin the backing pages of a shmem GEM object, allowing them to be swapped out if not actively used.
	/// This function acquires a reservation lock and calls the internal locked unpinning helper.
	struct drm_gem_object *obj = &shmem->base;

	drm_WARN_ON(obj->dev, drm_gem_is_imported(obj));

	// Conditional Logic: If pin count is not one, simply decrement and return.
	if (refcount_dec_not_one(&shmem->pages_pin_count))
		return;

	// Block Logic: Acquire DMA reservation lock.
	dma_resv_lock(shmem->base.resv, NULL);
	// Block Logic: Call internal locked unpinning function.
	drm_gem_shmem_unpin_locked(shmem);
	// Block Logic: Release DMA reservation lock.
	dma_resv_unlock(shmem->base.resv);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_unpin);

/*
 * drm_gem_shmem_vmap_locked - Create a virtual mapping for a shmem GEM object
 * @shmem: shmem GEM object
 * @map: Returns the kernel virtual address of the SHMEM GEM object's backing
 *       store.
 *
 * This function makes sure that a contiguous kernel virtual address mapping
 * exists for the buffer backing the shmem GEM object. It hides the differences
 * between dma-buf imported and natively allocated objects.
 *
 * Acquired mappings should be cleaned up by calling drm_gem_shmem_vunmap_locked().
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_shmem_vmap_locked(struct drm_gem_shmem_object *shmem,
			      struct iosys_map *map)
{
	/// Functional Utility: Creates a kernel virtual address mapping for a shmem GEM object's backing store.
	/// This function handles both imported and natively allocated objects, providing a contiguous virtual address view.
	struct drm_gem_object *obj = &shmem->base;
	int ret = 0;

	// Conditional Logic: If the GEM object is imported, use `dma_buf_vmap`.
	if (drm_gem_is_imported(obj)) {
		ret = dma_buf_vmap(obj->dma_buf, map);
	} else {
		pgprot_t prot = PAGE_KERNEL;

		dma_resv_assert_held(shmem->base.resv);

		// Conditional Logic: If a vmap already exists, increment use count and return.
		if (refcount_inc_not_zero(&shmem->vmap_use_count)) {
			iosys_map_set_vaddr(map, shmem->vaddr);
			return 0;
		}

		// Block Logic: Pin the backing pages.
		ret = drm_gem_shmem_pin_locked(shmem);
		// Conditional Logic: Handle pinning failure.
		if (ret)
			return ret;

		// Conditional Logic: If write-combine mapping is enabled, set page protection to write-combine.
		if (shmem->map_wc)
			prot = pgprot_writecombine(prot);
		// Block Logic: Create a kernel virtual mapping for the pages.
		shmem->vaddr = vmap(shmem->pages, obj->size >> PAGE_SHIFT,
				    VM_MAP, prot);
		// Conditional Logic: Handle vmap failure.
		if (!shmem->vaddr) {
			ret = -ENOMEM;
		} else {
			// Block Logic: Set the virtual address in the map and set vmap use count.
			iosys_map_set_vaddr(map, shmem->vaddr);
			refcount_set(&shmem->vmap_use_count, 1);
		}
	}

	// Conditional Logic: If an error occurred during vmap, log it.
	if (ret) {
		drm_dbg_kms(obj->dev, "Failed to vmap pages, error %d\n", ret);
		goto err_put_pages;
	}

	return 0;

err_put_pages:
	// Error Handling: If not an imported object, unpin pages upon error.
	if (!drm_gem_is_imported(obj))
		drm_gem_shmem_unpin_locked(shmem);

	return ret;
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_vmap_locked);

/*
 * drm_gem_shmem_vunmap_locked - Unmap a virtual mapping for a shmem GEM object
 * @shmem: shmem GEM object
 * @map: Kernel virtual address where the SHMEM GEM object was mapped
 *
 * This function cleans up a kernel virtual address mapping acquired by
 * drm_gem_shmem_vmap_locked(). The mapping is only removed when the use count
 * drops to zero.
 *
 * This function hides the differences between dma-buf imported and natively
 * allocated objects.
 */
void drm_gem_shmem_vunmap_locked(struct drm_gem_shmem_object *shmem,
				 struct iosys_map *map)
{
	/// Functional Utility: Unmaps a kernel virtual address mapping for a shmem GEM object.
	/// This function is the counterpart to `drm_gem_shmem_vmap_locked`, decrementing the vmap use count
	/// and unmapping the memory when the count reaches zero.
	struct drm_gem_object *obj = &shmem->base;

	// Conditional Logic: If the GEM object is imported, use `dma_buf_vunmap`.
	if (drm_gem_is_imported(obj)) {
		dma_buf_vunmap(obj->dma_buf, map);
	} else {
		dma_resv_assert_held(shmem->base.resv);

		// Conditional Logic: If the vmap use count drops to zero, unmap the virtual address.
		if (refcount_dec_and_test(&shmem->vmap_use_count)) {
			// Block Logic: Unmap the virtual address.
			vunmap(shmem->vaddr);
			shmem->vaddr = NULL;

			// Block Logic: Unpin the backing pages.
			drm_gem_shmem_unpin_locked(shmem);
		}
	}
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_vunmap_locked);

static int
drm_gem_shmem_create_with_handle(struct drm_file *file_priv,
				 struct drm_device *dev, size_t size,
				 uint32_t *handle)
{
	/// Functional Utility: Creates a shmem GEM object, allocates its backing shared memory, and registers a GEM handle for it.
	/// This makes the shared memory object accessible to userspace via a handle.
	struct drm_gem_shmem_object *shmem;
	int ret;

	// Block Logic: Create the shmem GEM object.
	shmem = drm_gem_shmem_create(dev, size);
	// Conditional Logic: Handle errors from shmem object creation.
	if (IS_ERR(shmem))
		return PTR_ERR(shmem);

	/*
	 * Allocate an id of idr table where the obj is registered
	 * and handle has the id what user can see.
	 */
	// Block Logic: Create a GEM handle for the object, making it visible to userspace.
	ret = drm_gem_handle_create(file_priv, &shmem->base, handle);
	/* drop reference from allocate - handle holds it now. */
	// Block Logic: Release the initial reference, as the handle now holds a reference.
	drm_gem_object_put(&shmem->base);

	return ret;
}

/* Update madvise status, returns true if not purged, else
 * false or -errno.
 */
int drm_gem_shmem_madvise_locked(struct drm_gem_shmem_object *shmem, int madv)
{
	/// Functional Utility: Updates the madvise status of a shmem GEM object, indicating its expected future access pattern.
	/// This locked internal helper is used to optimize memory management by hinting to the kernel about page reclaim priority.
	dma_resv_assert_held(shmem->base.resv);

	// Conditional Logic: If `shmem->madv` is not negative (i.e., not already purged), update its value.
	if (shmem->madv >= 0)
		shmem->madv = madv;

	// Block Logic: Use the current `madv` value for the return.
	madv = shmem->madv;

	// Conditional Logic: Return true if the object is not purged (madv >= 0).
	return (madv >= 0);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_madvise_locked);

void drm_gem_shmem_purge_locked(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Purges the backing pages of a shmem GEM object, returning as much memory as possible to the system.
	/// This locked internal helper is typically called under memory pressure (e.g., OOM conditions) to reclaim resources.
	struct drm_gem_object *obj = &shmem->base;
	struct drm_device *dev = obj->dev;

	dma_resv_assert_held(shmem->base.resv);

	drm_WARN_ON(obj->dev, !drm_gem_shmem_is_purgeable(shmem));

	// Block Logic: Unmap and free the scatter/gather table if it exists.
	dma_unmap_sgtable(dev->dev, shmem->sgt, DMA_BIDIRECTIONAL, 0);
	sg_free_table(shmem->sgt);
	kfree(shmem->sgt);
	shmem->sgt = NULL;

	// Block Logic: Put the backing pages.
	drm_gem_shmem_put_pages_locked(shmem);

	// Block Logic: Mark the object as purged.
	shmem->madv = -1;

	// Block Logic: Unmap VMA node and free mmap offset.
	drm_vma_node_unmap(&obj->vma_node, dev->anon_inode->i_mapping);
	drm_gem_free_mmap_offset(obj);

	/* Our goal here is to return as much of the memory as
	 * is possible back to the system as we are called from OOM.
	 * To do this we must instruct the shmfs to drop all of its
	 * backing pages, *now*.
	 */
	// Block Logic: Truncate the shmem file to release all backing pages.
	shmem_truncate_range(file_inode(obj->filp), 0, (loff_t)-1);

	// Block Logic: Invalidate page cache entries for the shmem file.
	invalidate_mapping_pages(file_inode(obj->filp)->i_mapping, 0, (loff_t)-1);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_purge_locked);

/**
 * drm_gem_shmem_dumb_create - Create a dumb shmem buffer object
 * @file: DRM file structure to create the dumb buffer for
 * @dev: DRM device
 * @args: IOCTL data
 *
 * This function computes the pitch of the dumb buffer and rounds it up to an
 * integer number of bytes per pixel. Drivers for hardware that doesn't have
 * any additional restrictions on the pitch can directly use this function as
 * their &drm_driver.dumb_create callback.
 *
 * For hardware with additional restrictions, drivers can adjust the fields
 * set up by userspace before calling into this function.
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_shmem_dumb_create(struct drm_file *file, struct drm_device *dev,
			      struct drm_mode_create_dumb *args)
{
	/// Functional Utility: Creates a "dumb" shmem buffer object, aligning its pitch and size as needed.
	/// This public interface is used by drivers to create simple shared memory buffers for userspace.
	u32 min_pitch = DIV_ROUND_UP(args->width * args->bpp, 8);

	// Conditional Logic: If pitch or size are not provided, calculate them.
	if (!args->pitch || !args->size) {
		args->pitch = min_pitch;
		args->size = PAGE_ALIGN(args->pitch * args->height);
	} else {
		/* ensure sane minimum values */
		// Conditional Logic: Ensure pitch and size meet sane minimum values.
		if (args->pitch < min_pitch)
			args->pitch = min_pitch;
		if (args->size < args->pitch * args->height)
			args->size = PAGE_ALIGN(args->pitch * args->height);
	}

	// Block Logic: Create the shmem GEM object with a handle.
	return drm_gem_shmem_create_with_handle(file, dev, args->size, &args->handle);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_dumb_create);

static vm_fault_t drm_gem_shmem_fault(struct vm_fault *vmf)
{
	/// Functional Utility: Handles page faults for shmem GEM objects when memory-mapped into userspace.
	/// It ensures that the correct physical page is inserted into the VMA, handling cases of purged or invalid pages.
	struct vm_area_struct *vma = vmf->vma;
	struct drm_gem_object *obj = vma->vm_private_data;
	struct drm_gem_shmem_object *shmem = to_drm_gem_shmem_obj(obj);
	loff_t num_pages = obj->size >> PAGE_SHIFT;
	vm_fault_t ret;
	struct page *page;
	pgoff_t page_offset;

	/* We don't use vmf->pgoff since that has the fake offset */
	// Block Logic: Calculate the page offset within the object from the fault address.
	page_offset = (vmf->address - vma->vm_start) >> PAGE_SHIFT;

	// Block Logic: Acquire DMA reservation lock.
	dma_resv_lock(shmem->base.resv, NULL);

	// Conditional Logic: Check for out-of-bounds access, missing pages, or purged object.
	if (page_offset >= num_pages ||
	    drm_WARN_ON_ONCE(obj->dev, !shmem->pages) ||
	    shmem->madv < 0) {
		ret = VM_FAULT_SIGBUS;
	} else {
		// Block Logic: Get the physical page corresponding to the page offset.
		page = shmem->pages[page_offset];

		// Block Logic: Insert the physical page into the VMA.
		ret = vmf_insert_pfn(vma, vmf->address, page_to_pfn(page));
	}

	// Block Logic: Release DMA reservation lock.
	dma_resv_unlock(shmem->base.resv);

	return ret;
}

static void drm_gem_shmem_vm_open(struct vm_area_struct *vma)
{
	/// Functional Utility: Handles the `open` operation for a memory-mapped shmem GEM object.
	/// It increments the use count for the backing pages, typically when a VMA is copied (e.g., on fork).
	struct drm_gem_object *obj = vma->vm_private_data;
	struct drm_gem_shmem_object *shmem = to_drm_gem_shmem_obj(obj);

	drm_WARN_ON(obj->dev, drm_gem_is_imported(obj));

	// Block Logic: Acquire DMA reservation lock.
	dma_resv_lock(shmem->base.resv, NULL);

	/*
	 * We should have already pinned the pages when the buffer was first
	 * mmap'd, vm_open() just grabs an additional reference for the new
	 * mm the vma is getting copied into (ie. on fork()).
	 */
	// Block Logic: Increment the page use count if it's not zero.
	drm_WARN_ON_ONCE(obj->dev,
			 !refcount_inc_not_zero(&shmem->pages_use_count));

	// Block Logic: Release DMA reservation lock.
	dma_resv_unlock(shmem->base.resv);

	drm_gem_vm_open(vma);
}

static void drm_gem_shmem_vm_close(struct vm_area_struct *vma)
{
	/// Functional Utility: Handles the `close` operation for a memory-mapped shmem GEM object.
	/// It decrements the use count for the backing pages and potentially unpins them when the count reaches zero.
	struct drm_gem_object *obj = vma->vm_private_data;
	struct drm_gem_shmem_object *shmem = to_drm_gem_shmem_obj(obj);

	// Block Logic: Acquire DMA reservation lock.
	dma_resv_lock(shmem->base.resv, NULL);
	// Block Logic: Put pages, which will decrement use count and potentially unpin.
	drm_gem_shmem_put_pages_locked(shmem);
	// Block Logic: Release DMA reservation lock.
	dma_resv_unlock(shmem->base.resv);

	drm_gem_vm_close(vma);
}

const struct vm_operations_struct drm_gem_shmem_vm_ops = {
	.fault = drm_gem_shmem_fault,
	.open = drm_gem_shmem_vm_open,
	.close = drm_gem_shmem_vm_close,
};
EXPORT_SYMBOL_GPL(drm_gem_shmem_vm_ops);

/**
 * drm_gem_shmem_mmap - Memory-map a shmem GEM object
 * @shmem: shmem GEM object
 * @vma: VMA for the area to be mapped
 *
 * This function implements an augmented version of the GEM DRM file mmap
 * operation for shmem objects.
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_shmem_mmap(struct drm_gem_shmem_object *shmem, struct vm_area_struct *vma)
{
	/// Functional Utility: Memory-maps a shmem GEM object into a userspace process's address space.
	/// It handles both imported and natively allocated objects, setting up appropriate VMA flags and protections.
	struct drm_gem_object *obj = &shmem->base;
	int ret;

	// Conditional Logic: If the GEM object is imported, use `dma_buf_mmap`.
	if (drm_gem_is_imported(obj)) {
		/* Reset both vm_ops and vm_private_data, so we don't end up with
		 * vm_ops pointing to our implementation if the dma-buf backend
		 * doesn't set those fields.
		 */
		// Block Logic: Reset VMA ops and private data to avoid conflicts with dma-buf backend.
		vma->vm_private_data = NULL;
		vma->vm_ops = NULL;

		// Block Logic: Memory-map the dma-buf.
		ret = dma_buf_mmap(obj->dma_buf, vma, 0);

		/* Drop the reference drm_gem_mmap_obj() acquired.*/
		// Conditional Logic: If mapping is successful, drop the GEM object reference.
		if (!ret)
			drm_gem_object_put(obj);

		return ret;
	}

	// Conditional Logic: Return error if it's a copy-on-write mapping.
	if (is_cow_mapping(vma->vm_flags))
		return -EINVAL;

	// Block Logic: Acquire DMA reservation lock.
	dma_resv_lock(shmem->base.resv, NULL);
	// Block Logic: Get and pin pages for the shmem object.
	ret = drm_gem_shmem_get_pages_locked(shmem);
	// Block Logic: Release DMA reservation lock.
	dma_resv_unlock(shmem->base.resv);

	// Conditional Logic: Handle page acquisition failure.
	if (ret)
		return ret;

	// Block Logic: Set VMA flags and page protections.
	vm_flags_set(vma, VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);
	vma->vm_page_prot = vm_get_page_prot(vma->vm_flags);
	// Conditional Logic: Apply write-combine protection if enabled.
	if (shmem->map_wc)
		vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);

	return 0;
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_mmap);

/**
 * drm_gem_shmem_print_info() - Print &drm_gem_shmem_object info for debugfs
 * @shmem: shmem GEM object
 * @p: DRM printer
 * @indent: Tab indentation level
 */
void drm_gem_shmem_print_info(const struct drm_gem_shmem_object *shmem,
			      struct drm_printer *p, unsigned int indent)
{
	/// Functional Utility: Prints debugging information for a shmem GEM object to a DRM printer.
	/// This includes reference counts for pages, vmap use, and the virtual address, useful for debugging.
	if (drm_gem_is_imported(&shmem->base))
		return;

	// Block Logic: Print the pin count of backing pages.
	drm_printf_indent(p, indent, "pages_pin_count=%u\n", refcount_read(&shmem->pages_pin_count));
	// Block Logic: Print the use count of backing pages.
	drm_printf_indent(p, indent, "pages_use_count=%u\n", refcount_read(&shmem->pages_use_count));
	// Block Logic: Print the virtual mapping use count.
	drm_printf_indent(p, indent, "vmap_use_count=%u\n", refcount_read(&shmem->vmap_use_count));
	// Block Logic: Print the virtual address.
	drm_printf_indent(p, indent, "vaddr=%p\n", shmem->vaddr);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_print_info);

/**
 * drm_gem_shmem_get_sg_table - Provide a scatter/gather table of pinned
 *                              pages for a shmem GEM object
 * @shmem: shmem GEM object
 *
 * This function exports a scatter/gather table suitable for PRIME usage by
 * calling the standard DMA mapping API.
 *
 * Drivers who need to acquire an scatter/gather table for objects need to call
 * drm_gem_shmem_get_pages_sgt() instead.
 *
 * Returns:
 * A pointer to the scatter/gather table of pinned pages or error pointer on failure.
 */
struct sg_table *drm_gem_shmem_get_sg_table(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Generates a scatter/gather table for a shmem GEM object.
	/// This function is intended for PRIME usage, describing the physical memory layout for DMA operations.
	struct drm_gem_object *obj = &shmem->base;

	drm_WARN_ON(obj->dev, drm_gem_is_imported(obj));

	// Block Logic: Convert the GEM object's pages into a scatter/gather table.
	return drm_prime_pages_to_sg(obj->dev, shmem->pages, obj->size >> PAGE_SHIFT);
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_get_sg_table);

static struct sg_table *drm_gem_shmem_get_pages_sgt_locked(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Retrieves a scatter/gather table for a shmem GEM object, creating and DMA-mapping it if necessary.
	/// This locked internal helper ensures the underlying pages are pinned and ready for hardware access.
	struct drm_gem_object *obj = &shmem->base;
	int ret;
	struct sg_table *sgt;

	// Conditional Logic: If a scatter/gather table already exists, return it.
	if (shmem->sgt)
		return shmem->sgt;

	drm_WARN_ON(obj->dev, drm_gem_is_imported(obj));

	// Block Logic: Get and pin the backing pages.
	ret = drm_gem_shmem_get_pages_locked(shmem);
	// Conditional Logic: Handle page acquisition failure.
	if (ret)
		return ERR_PTR(ret);

	// Block Logic: Create a new scatter/gather table from the pages.
	sgt = drm_gem_shmem_get_sg_table(shmem);
	// Conditional Logic: Handle scatter/gather table creation failure.
	if (IS_ERR(sgt)) {
		ret = PTR_ERR(sgt);
		goto err_put_pages;
	}
	/* Map the pages for use by the h/w. */
	// Block Logic: DMA map the scatter/gather table for hardware access.
	ret = dma_map_sgtable(obj->dev->dev, sgt, DMA_BIDIRECTIONAL, 0);
	// Conditional Logic: Handle DMA mapping failure.
	if (ret)
		goto err_free_sgt;

	// Block Logic: Store the newly created scatter/gather table.
	shmem->sgt = sgt;

	return sgt;

err_free_sgt:
	// Error Handling: Free the scatter/gather table upon error.
	sg_free_table(sgt);
	kfree(sgt);
err_put_pages:
	// Error Handling: Put the backing pages upon error.
	drm_gem_shmem_put_pages_locked(shmem);
	return ERR_PTR(ret);
}

/**
 * drm_gem_shmem_get_pages_sgt - Pin pages, dma map them, and return a
 *				 scatter/gather table for a shmem GEM object.
 * @shmem: shmem GEM object
 *
 * This function returns a scatter/gather table suitable for driver usage. If
 * the sg table doesn't exist, the pages are pinned, dma-mapped, and a sg
 * table created.
 *
 * This is the main function for drivers to get at backing storage, and it hides
 * and difference between dma-buf imported and natively allocated objects.
 * drm_gem_shmem_get_sg_table() should not be directly called by drivers.
 *
 * Returns:
 * A pointer to the scatter/gather table of pinned pages or errno on failure.
 */
struct sg_table *drm_gem_shmem_get_pages_sgt(struct drm_gem_shmem_object *shmem)
{
	/// Functional Utility: Public interface to get a scatter/gather table for a shmem GEM object, handling locking and internal creation.
	/// This is the primary function for drivers to obtain the physical layout of the backing store for DMA.
	int ret;
	struct sg_table *sgt;

	// Block Logic: Acquire DMA reservation lock in an interruptible manner.
	ret = dma_resv_lock_interruptible(shmem->base.resv, NULL);
	// Conditional Logic: Handle lock acquisition failure.
	if (ret)
		return ERR_PTR(ret);
	// Block Logic: Get the scatter/gather table using the locked internal helper.
	sgt = drm_gem_shmem_get_pages_sgt_locked(shmem);
	// Block Logic: Release DMA reservation lock.
	dma_resv_unlock(shmem->base.resv);

	return sgt;
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_get_pages_sgt);

/**
 * drm_gem_shmem_prime_import_sg_table - Produce a shmem GEM object from
 *                 another driver's scatter/gather table of pinned pages
 * @dev: Device to import into
 * @attach: DMA-BUF attachment
 * @sgt: Scatter/gather table of pinned pages
 *
 * This function imports a scatter/gather table exported via DMA-BUF by
 * another driver. Drivers that use the shmem helpers should set this as their
 * &drm_driver.gem_prime_import_sg_table callback.
 *
 * Returns:
 * A pointer to a newly created GEM object or an ERR_PTR-encoded negative
 * error code on failure.
 */
struct drm_gem_object *
drm_gem_shmem_prime_import_sg_table(struct drm_device *dev,
				    struct dma_buf_attachment *attach,
				    struct sg_table *sgt)
{
	/// Functional Utility: Imports a scatter/gather table from another driver (via DMA-BUF) and creates a shmem GEM object from it.
	/// This allows sharing of shared memory buffers between different DRM drivers or other kernel components.
	size_t size = PAGE_ALIGN(attach->dmabuf->size);
	struct drm_gem_shmem_object *shmem;

	shmem = __drm_gem_shmem_create(dev, size, true, NULL);
	if (IS_ERR(shmem))
		return ERR_CAST(shmem);

	// Block Logic: Store the imported scatter/gather table.
	shmem->sgt = sgt;

	drm_dbg_prime(dev, "size = %zu\n", size);

	return &shmem->base;
}
EXPORT_SYMBOL_GPL(drm_gem_shmem_prime_import_sg_table);

MODULE_DESCRIPTION("DRM SHMEM memory-management helpers");
MODULE_IMPORT_NS("DMA_BUF");
MODULE_LICENSE("GPL v2");
