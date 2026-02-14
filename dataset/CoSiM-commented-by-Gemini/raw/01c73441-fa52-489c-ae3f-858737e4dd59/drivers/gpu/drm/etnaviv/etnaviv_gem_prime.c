/**
 * @file etnaviv_gem_prime.c
 * @brief Etnaviv GEM PRIME (DMA-BUF) implementation
 *
 * This file implements the PRIME interface for the Etnaviv GEM buffer
 * manager. PRIME is the Linux kernel's framework for sharing buffer
 * objects between different devices and subsystems, commonly known as DMA-BUF.
 * The functions in this file handle the logic for exporting Etnaviv GEM
 * objects as DMA-BUF file descriptors and importing DMA-BUFs from other
 * devices into a format that Etnaviv can use.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2014-2018 Etnaviv Project
 */

#include <drm/drm_prime.h>
#include <linux/dma-buf.h>
#include <linux/module.h>

#include "etnaviv_drv.h"
#include "etnaviv_gem.h"

MODULE_IMPORT_NS("DMA_BUF");

static struct lock_class_key etnaviv_prime_lock_class;

/**
 * etnaviv_gem_prime_get_sg_table - Provides the scatter-gather table for a GEM object.
 * @obj: The GEM object to get the sg_table for.
 *
 * This function is a wrapper around drm_prime_pages_to_sg to convert the
 * page array of a given Etnaviv GEM object into a scatter-gather table, which
 * is a standard representation for non-contiguous memory regions used by DMA-BUF.
 *
 * Return: A pointer to the sg_table on success, or an ERR_PTR on failure.
 */
struct sg_table *etnaviv_gem_prime_get_sg_table(struct drm_gem_object *obj)
{
	struct etnaviv_gem_object *etnaviv_obj = to_etnaviv_bo(obj);
	unsigned int npages = obj->size >> PAGE_SHIFT;

	/* Pre-condition: The pages for the GEM object must have been allocated
	 * and pinned already, a prerequisite for buffer exporting.
	 */
	if (WARN_ON(!etnaviv_obj->pages))  /* should have already pinned! */
		return ERR_PTR(-EINVAL);

	return drm_prime_pages_to_sg(obj->dev, etnaviv_obj->pages, npages);
}

/**
 * etnaviv_gem_prime_vmap - Maps an exported GEM object into the kernel's virtual address space.
 * @obj: The GEM object to be mapped.
 * @map: The iosys_map structure to store the mapping information.
 *
 * This function provides a CPU-accessible virtual address for the buffer's
 * contents. It is a necessary step for CPU-based access to the buffer memory.
 *
 * Return: 0 on success, or a negative error code on failure.
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
 * etnaviv_gem_prime_pin - Pins the pages of a GEM object, preventing them from being swapped out.
 * @obj: The GEM object to pin.
 *
 * Pinning ensures that the physical pages backing the GEM object remain resident
 * in memory, which is essential before initiating DMA operations. For objects
 * that are not imported, this function explicitly acquires their page backing.
 *
 * Return: 0 on success.
 */
int etnaviv_gem_prime_pin(struct drm_gem_object *obj)
{
	/* Invariant: Only pin pages for locally-created buffers. Imported buffers
	 * are managed by the exporter.
	 */
	if (!drm_gem_is_imported(obj)) {
		struct etnaviv_gem_object *etnaviv_obj = to_etnaviv_bo(obj);

		mutex_lock(&etnaviv_obj->lock);
		etnaviv_gem_get_pages(etnaviv_obj);
		mutex_unlock(&etnaviv_obj->lock);
	}
	return 0;
}

/**
 * etnaviv_gem_prime_unpin - Unpins the pages of a GEM object, allowing them to be swapped.
 * @obj: The GEM object to unpin.
 *
 * This function releases the pin on the physical pages, allowing the memory
 * manager to move them if necessary. It is called when DMA access is no longer
 * required.
 */
void etnaviv_gem_prime_unpin(struct drm_gem_object *obj)
{
	/* Invariant: Only unpin pages for locally-created buffers. Imported buffers
	 * are managed by the exporter.
	 */
	if (!drm_gem_is_imported(obj)) {
		struct etnaviv_gem_object *etnaviv_obj = to_etnaviv_bo(obj);

		mutex_lock(&etnaviv_obj->lock);
		etnaviv_gem_put_pages(to_etnaviv_bo(obj));
		mutex_unlock(&etnaviv_obj->lock);
	}
}

/**
 * etnaviv_gem_prime_release - Releases resources associated with an imported GEM object.
 * @etnaviv_obj: The Etnaviv GEM object that was created from an import.
 *
 * This function is the release callback for GEM objects that were created by
 * importing a dma_buf. It handles unmapping any virtual addresses and freeing
 * the structures associated with the import process.
 */
static void etnaviv_gem_prime_release(struct etnaviv_gem_object *etnaviv_obj)
{
	struct iosys_map map = IOSYS_MAP_INIT_VADDR(etnaviv_obj->vaddr);

	/* If the buffer was virtually mapped, unmap it now. */
	if (etnaviv_obj->vaddr)
		dma_buf_vunmap_unlocked(etnaviv_obj->base.dma_buf, &map);

	/* Don't drop the pages for imported dmabuf, as they are not
	 * ours, just free the array we allocated to store page pointers.
	 */
	kvfree(etnaviv_obj->pages);

	/* Perform the generic PRIME GEM destruction. */
	drm_prime_gem_destroy(&etnaviv_obj->base, etnaviv_obj->sgt);
}

/**
 * etnaviv_gem_prime_vmap_impl - Internal implementation for vmapping an imported object.
 * @etnaviv_obj: The imported Etnaviv GEM object to map.
 *
 * This function calls the dma_buf's vmap operation to get a kernel virtual
 * address for the imported buffer.
 *
 * Return: A virtual address on success, or NULL on failure.
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
 * etnaviv_gem_prime_mmap_obj - Memory maps an imported GEM object into user space.
 * @etnaviv_obj: The imported Etnaviv GEM object to mmap.
 * @vma: The user space virtual memory area.
 *
 * This function allows a user-space process to directly map the memory of an
 * imported buffer into its own address space, enabling CPU access.
 *
 * Return: 0 on success, or a negative error code on failure.
 */
static int etnaviv_gem_prime_mmap_obj(struct etnaviv_gem_object *etnaviv_obj,
		struct vm_area_struct *vma)
{
	int ret;

	/* Pre-condition: The VMA must be valid and the object must be a DMA-BUF. */
	ret = dma_buf_mmap(etnaviv_obj->base.dma_buf, vma, 0);
	if (!ret) {
		/* Drop the reference acquired by drm_gem_mmap_obj(), as dma_buf_mmap()
		 * takes its own reference.
		 */
		drm_gem_object_put(&etnaviv_obj->base);
	}

	return ret;
}

/**
 * @brief Operations table for PRIME-imported GEM objects.
 *
 * This structure assigns the specialized functions required for handling
 * GEM objects that are created from imported DMA-BUFs. These functions
 * correctly manage the lifetime and mapping of external buffers.
 */
static const struct etnaviv_gem_ops etnaviv_gem_prime_ops = {
	/* .get_pages should never be called on imported objects */
	.release = etnaviv_gem_prime_release,
	.vmap = etnaviv_gem_prime_vmap_impl,
	.mmap = etnaviv_gem_prime_mmap_obj,
};

/**
 * etnaviv_gem_prime_import_sg_table - Imports a scatter-gather table as a GEM object.
 * @dev: The DRM device.
 * @attach: The DMA-BUF attachment.
 * @sgt: The scatter-gather table representing the buffer's memory.
 *
 * This is the core import function. It takes a scatter-gather table from a
 * DMA-BUF and wraps it in an Etnaviv GEM object. This allows the Etnaviv
 * driver to use a buffer allocated by another device.
 *
 * Return: A pointer to the newly created GEM object on success, or an ERR_PTR
 * on failure.
 */
struct drm_gem_object *etnaviv_gem_prime_import_sg_table(struct drm_device *dev,
	struct dma_buf_attachment *attach, struct sg_table *sgt)
{
	struct etnaviv_gem_object *etnaviv_obj;
	size_t size = PAGE_ALIGN(attach->dmabuf->size);
	int ret, npages;

	/* Create a new private GEM object to represent the imported buffer. */
	ret = etnaviv_gem_new_private(dev, size, ETNA_BO_WC,
				      &etnaviv_gem_prime_ops, &etnaviv_obj);
	if (ret < 0)
		return ERR_PTR(ret);

	lockdep_set_class(&etnaviv_obj->lock, &etnaviv_prime_lock_class);

	npages = size / PAGE_SIZE;

	etnaviv_obj->sgt = sgt;
	/* Allocate memory to hold the array of page pointers for the imported buffer. */
	etnaviv_obj->pages = kvmalloc_array(npages, sizeof(struct page *), GFP_KERNEL);
	if (!etnaviv_obj->pages) {
		ret = -ENOMEM;
		goto fail;
	}

	/* Convert the sg_table into a flat array of pages. */
	ret = drm_prime_sg_to_page_array(sgt, etnaviv_obj->pages, npages);
	if (ret)
		goto fail;

	etnaviv_gem_obj_add(dev, &etnaviv_obj->base);

	return &etnaviv_obj->base;

fail:
	/* Clean up the partially created GEM object on failure. */
	drm_gem_object_put(&etnaviv_obj->base);

	return ERR_PTR(ret);
}
