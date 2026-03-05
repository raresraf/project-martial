// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2014-2018 Etnaviv Project
 */

/**
 * @file etnaviv_gem_prime.c
 * @brief Implements PRIME buffer sharing for the Etnaviv DRM GEM driver.
 *
 * This file contains the implementation of functions necessary for Direct Rendering Manager (DRM)
 * Graphics Execution Manager (GEM) objects to support PRIME buffer sharing. PRIME is a
 * mechanism in the Linux kernel for efficient sharing of memory buffers between different
 * DRM drivers or between a DRM driver and another kernel subsystem (e.g., display controller,
 * video encoder/decoder).
 *
 * The core functionality includes:
 * - Exporting GEM buffers as dma-bufs.
 * - Importing dma-bufs as GEM buffers.
 * - Managing Scatter-Gather (SG) tables for DMA operations.
 * - Handling virtual memory mapping and pinning/unpinning of pages for shared buffers.
 *
 * This enables zero-copy buffer sharing, improving performance and reducing memory overhead
 * in graphics and multimedia pipelines.
 */

#include <drm/drm_prime.h>
#include <linux/dma-buf.h>
#include <linux/module.h>

#include "etnaviv_drv.h"
#include "etnaviv_gem.h"

MODULE_IMPORT_NS("DMA_BUF");

static struct lock_class_key etnaviv_prime_lock_class;

/**
 * @brief Retrieves the Scatter-Gather (SG) table for a GEM object.
 * @param obj Pointer to the DRM GEM object.
 * @return A pointer to the Scatter-Gather table if successful, or an ERR_PTR on failure.
 *
 * This function is used to export a GEM object as a dma-buf, making its memory
 * accessible to other devices via DMA. It converts the pinned pages of the
 * GEM object into an SG table, which describes the physical memory layout.
 *
 * Preconditions:
 * - The `etnaviv_obj->pages` must have been successfully pinned previously.
 * Invariant:
 * - The returned `sg_table` accurately represents the physical memory pages
 *   backing the GEM object.
 */
struct sg_table *etnaviv_gem_prime_get_sg_table(struct drm_gem_object *obj)
{
	struct etnaviv_gem_object *etnaviv_obj = to_etnaviv_bo(obj);
	unsigned int npages = obj->size >> PAGE_SHIFT;

	if (WARN_ON(!etnaviv_obj->pages))  /* should have already pinned! */
		return ERR_PTR(-EINVAL);

	return drm_prime_pages_to_sg(obj->dev, etnaviv_obj->pages, npages);
}

/**
 * @brief Performs a virtual memory map for an imported GEM object.
 * @param obj Pointer to the DRM GEM object.
 * @param map Pointer to a `iosys_map` structure to store the virtual address.
 * @return 0 on success, or a negative error code on failure.
 *
 * This function provides a virtual address mapping for the memory associated
 * with an imported GEM object, allowing CPU access to the buffer.
 * It leverages the `etnaviv_gem_vmap` function for the actual mapping.
 *
 * Preconditions:
 * - `obj` must be a valid DRM GEM object, likely imported from a dma-buf.
 * Invariant:
 * - Upon successful return, `map->vaddr` points to the CPU-accessible virtual
 *   address of the buffer.
 */
int etnaviv_gem_prime_vmap(struct drm_gem_object *obj, struct iosys_map *map)
{
	void *vaddr;

	vaddr = etnaviv_gem_vmap(obj);
	if (!vaddr)
		return -ENOMEM;
	iosys_map_set_vaddr(map, vaddr);

	return 0;
}

/**
 * @brief Pins the pages of a GEM object, ensuring they are resident in memory.
 * @param obj Pointer to the DRM GEM object.
 * @return 0 on success, or a negative error code on failure.
 *
 * For GEM objects not imported from a dma-buf (i.e., native Etnaviv GEM objects),
 * this function acquires references to the physical pages backing the object,
 * preventing them from being swapped out. Imported dma-bufs manage their own
 * pinning, so this operation is skipped for them.
 *
 * Preconditions:
 * - `etnaviv_obj->lock` must be available for locking if the object is not imported.
 * Invariant:
 * - If the object is not imported, its physical pages are pinned in memory.
 */
int etnaviv_gem_prime_pin(struct drm_gem_object *obj)
{
	if (!drm_gem_is_imported(obj)) {
		struct etnaviv_gem_object *etnaviv_obj = to_etnaviv_bo(obj);

		mutex_lock(&etnaviv_obj->lock);
		// Block Logic: Acquire references to the physical pages.
		// etnaviv_gem_get_pages ensures the backing pages are allocated and mapped.
		etnaviv_gem_get_pages(etnaviv_obj);
		mutex_unlock(&etnaviv_obj->lock);
	}
	return 0;
}

/**
 * @brief Unpins the pages of a GEM object, releasing acquired references.
 * @param obj Pointer to the DRM GEM object.
 *
 * For GEM objects not imported from a dma-buf (i.e., native Etnaviv GEM objects),
 * this function releases the references to the physical pages backing the object,
 * allowing them to be potentially swapped out. Imported dma-bufs manage their own
 * unpinning, so this operation is skipped for them.
 *
 * Preconditions:
 * - `etnaviv_obj->lock` must be available for locking if the object is not imported.
 * Invariant:
 * - If the object is not imported, its physical pages are released, decrementing
 *   their reference count.
 */
void etnaviv_gem_prime_unpin(struct drm_gem_object *obj)
{
	if (!drm_gem_is_imported(obj)) {
		struct etnaviv_gem_object *etnaviv_obj = to_etnaviv_bo(obj);

		mutex_lock(&etnaviv_obj->lock);
		// Block Logic: Release references to the physical pages.
		// etnaviv_gem_put_pages ensures the backing pages are deallocated if no longer used.
		etnaviv_gem_put_pages(to_etnaviv_bo(obj));
		mutex_unlock(&etnaviv_obj->lock);
	}
}

/**
 * @brief Releases resources associated with a PRIME-imported GEM object.
 * @param etnaviv_obj Pointer to the Etnaviv GEM object.
 *
 * This function is called when a PRIME-imported GEM object is destroyed.
 * It unmaps any CPU-accessible virtual addresses and frees the page array
 * allocated for the scatter-gather table, then calls the generic DRM PRIME
 * GEM destruction function.
 *
 * Preconditions:
 * - `etnaviv_obj` must be a valid Etnaviv GEM object previously imported via PRIME.
 * Invariant:
 * - All resources held by the `etnaviv_obj` related to PRIME import (vmap, pages, sgt)
 *   are released.
 */
static void etnaviv_gem_prime_release(struct etnaviv_gem_object *etnaviv_obj)
{
	struct iosys_map map = IOSYS_MAP_INIT_VADDR(etnaviv_obj->vaddr);

	// Block Logic: Unmap CPU-accessible virtual address if it exists.
	if (etnaviv_obj->vaddr)
		dma_buf_vunmap_unlocked(etnaviv_obj->base.dma_buf, &map);

	/* Don't drop the pages for imported dmabuf, as they are not
	 * ours, just free the array we allocated:
	 */
	kvfree(etnaviv_obj->pages);

	drm_prime_gem_destroy(&etnaviv_obj->base, etnaviv_obj->sgt);
}

/**
 * @brief Internal function to perform virtual memory mapping for a PRIME-imported GEM object.
 * @param etnaviv_obj Pointer to the Etnaviv GEM object.
 * @return A pointer to the virtual address on success, or NULL on failure.
 *
 * This function provides the underlying mechanism for mapping the dma-buf
 * backing a PRIME-imported GEM object into the kernel's virtual address space,
 * allowing the CPU to access its content.
 *
 * Preconditions:
 * - `etnaviv_obj->lock` must be held by the caller.
 * - `etnaviv_obj->base.dma_buf` must be a valid dma-buf.
 * Invariant:
 * - Upon successful return, the dma-buf's memory is mapped into the kernel
 *   and its virtual address is returned.
 */
static void *etnaviv_gem_prime_vmap_impl(struct etnaviv_gem_object *etnaviv_obj)
{
	struct iosys_map map;
	int ret;

	lockdep_assert_held(&etnaviv_obj->lock);

	ret = dma_buf_vmap(etnaviv_obj->base.dma_buf, &map);
	if (ret)
		return NULL;
	return map.vaddr;
}

/**
 * @brief Performs memory mapping for a PRIME-imported GEM object into user space.
 * @param etnaviv_obj Pointer to the Etnaviv GEM object.
 * @param vma Pointer to the `vm_area_struct` representing the user-space memory region.
 * @return 0 on success, or a negative error code on failure.
 *
 * This function enables user-space processes to mmap (memory map) the underlying
 * dma-buf of a PRIME-imported GEM object, facilitating direct access from
 * user-space applications.
 *
 * Preconditions:
 * - `etnaviv_obj->base.dma_buf` must be a valid dma-buf.
 * Invariant:
 * - Upon successful return, the dma-buf's memory is mapped into the provided
 *   `vm_area_struct`.
 */
static int etnaviv_gem_prime_mmap_obj(struct etnaviv_gem_object *etnaviv_obj,
		struct vm_area_struct *vma)
{
	int ret;

	ret = dma_buf_mmap(etnaviv_obj->base.dma_buf, vma, 0);
	if (!ret) {
		/* Drop the reference acquired by drm_gem_mmap_obj(). */
		drm_gem_object_put(&etnaviv_obj->base);
	}

	return ret;
}

/**
 * @brief GEM operations for PRIME-imported objects.
 *
 * This structure defines the set of operations (release, vmap, mmap) that
 * are specific to GEM objects that have been imported via the PRIME buffer
 * sharing mechanism. It overrides some of the default GEM operations
 * to handle the specifics of dma-buf backed memory.
 */
static const struct etnaviv_gem_ops etnaviv_gem_prime_ops = {
	/* .get_pages should never be called */
	.release = etnaviv_gem_prime_release,
	.vmap = etnaviv_gem_prime_vmap_impl,
	.mmap = etnaviv_gem_prime_mmap_obj,
};

/**
 * @brief Imports a Scatter-Gather (SG) table as a new Etnaviv GEM object.
 * @param dev Pointer to the DRM device.
 * @param attach Pointer to the dma-buf attachment.
 * @param sgt Pointer to the Scatter-Gather table representing the imported memory.
 * @return A pointer to the newly created DRM GEM object on success, or an ERR_PTR on failure.
 *
 * This function is the entry point for importing an external dma-buf (represented
 * by its SG table) into the Etnaviv GEM subsystem. It allocates a new Etnaviv
 * GEM object, populates it with information from the SG table, and adds it
 * to the DRM GEM object list.
 *
 * Preconditions:
 * - `dev` must be a valid DRM device.
 * - `attach` must be a valid dma-buf attachment.
 * - `sgt` must be a valid Scatter-Gather table.
 * Invariant:
 * - A new Etnaviv GEM object is created, backed by the imported SG table,
 *   and registered with the DRM subsystem.
 */
struct drm_gem_object *etnaviv_gem_prime_import_sg_table(struct drm_device *dev,
	struct dma_buf_attachment *attach, struct sg_table *sgt)
{
	struct etnaviv_gem_object *etnaviv_obj;
	size_t size = PAGE_ALIGN(attach->dmabuf->size);
	int ret, npages;

	// Block Logic: Allocate a new private Etnaviv GEM object.
	// This object will encapsulate the imported dma-buf.
	ret = etnaviv_gem_new_private(dev, size, ETNA_BO_WC,
				      &etnaviv_gem_prime_ops, &etnaviv_obj);
	if (ret < 0)
		return ERR_PTR(ret);

	lockdep_set_class(&etnaviv_obj->lock, &etnaviv_prime_lock_class);

	npages = size / PAGE_SIZE;

	etnaviv_obj->sgt = sgt;
	// Block Logic: Allocate an array to store page pointers for the imported buffer.
	// GFP_KERNEL is used for kernel memory allocation that can sleep.
	etnaviv_obj->pages = kvmalloc_array(npages, sizeof(struct page *), GFP_KERNEL);
	if (!etnaviv_obj->pages) {
		ret = -ENOMEM;
		goto fail;
	}

	// Block Logic: Convert the scatter-gather list into a page array.
	// This populates `etnaviv_obj->pages` with references to the physical pages.
	ret = drm_prime_sg_to_page_array(sgt, etnaviv_obj->pages, npages);
	if (ret)
		goto fail;

	etnaviv_gem_obj_add(dev, &etnaviv_obj->base);

	return &etnaviv_obj->base;

fail:
	drm_gem_object_put(&etnaviv_obj->base);

	return ERR_PTR(ret);
}