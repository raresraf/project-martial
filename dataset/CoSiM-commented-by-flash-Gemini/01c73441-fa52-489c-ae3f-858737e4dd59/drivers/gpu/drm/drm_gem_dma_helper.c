// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * drm gem DMA helper functions
 *
 * Copyright (C) 2012 Sascha Hauer, Pengutronix
 *
 * Based on Samsung Exynos code
 *
 * Copyright (c) 2011 Samsung Electronics Co., Ltd.
 */

#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/export.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/slab.h>

#include <drm/drm.h>
#include <drm/drm_device.h>
#include <drm/drm_drv.h>
#include <drm/drm_gem_dma_helper.h>
#include <drm/drm_vma_manager.h>

/**
 * @file drm_gem_dma_helper.c
 * @brief Provides helper functions for managing DRM GEM (Graphics Execution Manager) objects
 * backed by DMA (Direct Memory Access) contiguous memory. These helpers are essential
 * for graphics drivers in the Linux kernel, especially for devices that require
 * memory to appear contiguous in bus address space.
 *
 * Functional Utility: Facilitates the allocation, mapping, and sharing of graphics
 * memory as contiguous DMA buffers, supporting both physically contiguous (via CMA)
 * and IOVA-contiguous (via IOMMU) scenarios. This abstraction simplifies memory
 * management for various hardware architectures and driver needs.
 *
 * Key Concepts:
 * - **DMA Contiguity**: Ensures memory appears as a single contiguous block to hardware devices.
 * - **CMA (Contiguous Memory Allocator)**: Used for devices that need physically contiguous memory.
 * - **IOVA (IOMMU Virtual Address)**: For devices behind an IOMMU, memory may be physically scattered
 *   but appear contiguous in the IOVA space.
 * - **Scatter-Gather DMA**: Handled where possible, but these helpers primarily focus on contiguous views.
 */

// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * drm gem DMA helper functions
 *
 * Copyright (C) 2012 Sascha Hauer, Pengutronix
 *
 * Based on Samsung Exynos code
 *
 * Copyright (c) 2011 Samsung Electronics Co., Ltd.
 */

static const struct drm_gem_object_funcs drm_gem_dma_default_funcs = {
	.free = drm_gem_dma_object_free,
	.print_info = drm_gem_dma_object_print_info,
	.get_sg_table = drm_gem_dma_object_get_sg_table,
	.vmap = drm_gem_dma_object_vmap,
	.mmap = drm_gem_dma_object_mmap,
	.vm_ops = &drm_gem_dma_vm_ops,
};

/**
 * __drm_gem_dma_create - Create a GEM DMA object without allocating memory
 * @drm: DRM device
 * @size: size of the object to allocate
 * @private: true if used for internal purposes
 *
 * This function creates and initializes a GEM DMA object of the given size,
 * but doesn't allocate any memory to back the object.
 *
 * Returns:
 * A struct drm_gem_dma_object * on success or an ERR_PTR()-encoded negative
 * error code on failure.
 */
static struct drm_gem_dma_object *
__drm_gem_dma_create(struct drm_device *drm, size_t size, bool private)
{
	/// Functional Utility: Creates and initializes a GEM DMA object of a given size without allocating its backing memory.
	/// This serves as a base for subsequent memory allocation or import operations.
	struct drm_gem_dma_object *dma_obj;
	struct drm_gem_object *gem_obj;
	int ret = 0;

	// Conditional Logic: Check if the driver provides a custom GEM object creation function.
	if (drm->driver->gem_create_object) {
		// Block Logic: Use the driver's custom function to create the GEM object.
		gem_obj = drm->driver->gem_create_object(drm, size);
		// Conditional Logic: Handle potential errors from the custom creation function.
		if (IS_ERR(gem_obj))
			return ERR_CAST(gem_obj);
		// Block Logic: Convert the generic GEM object to a DMA GEM object.
		dma_obj = to_drm_gem_dma_obj(gem_obj);
	} else {
		// Block Logic: Allocate memory for the DMA GEM object structure.
		dma_obj = kzalloc(sizeof(*dma_obj), GFP_KERNEL);
		// Conditional Logic: Handle memory allocation failure.
		if (!dma_obj)
			return ERR_PTR(-ENOMEM);
		// Block Logic: Point the generic GEM object to the base of the DMA GEM object.
		gem_obj = &dma_obj->base;
	}

	// Conditional Logic: If GEM object functions are not set, assign default DMA GEM functions.
	if (!gem_obj->funcs)
		gem_obj->funcs = &drm_gem_dma_default_funcs;

	// Conditional Logic: Initialize the GEM object based on whether it's for private use.
	if (private) {
		// Block Logic: Initialize a private GEM object.
		drm_gem_private_object_init(drm, gem_obj, size);

		/* Always use writecombine for dma-buf mappings */
		// Block Logic: For private objects, non-coherent mapping is disabled.
		dma_obj->map_noncoherent = false;
	} else {
		// Block Logic: Initialize a regular GEM object.
		ret = drm_gem_object_init(drm, gem_obj, size);
	}
	// Conditional Logic: Handle initialization errors.
	if (ret)
		goto error;

	// Block Logic: Create a memory-map offset for the GEM object.
	ret = drm_gem_create_mmap_offset(gem_obj);
	// Conditional Logic: Handle memory-map offset creation errors.
	if (ret) {
		drm_gem_object_release(gem_obj);
		goto error;
	}

	return dma_obj;

error:
	// Error Handling: Free the DMA GEM object structure upon error.
	kfree(dma_obj);
	return ERR_PTR(ret);
}

/**
 * drm_gem_dma_create - allocate an object with the given size
 * @drm: DRM device
 * @size: size of the object to allocate
 *
 * This function creates a DMA GEM object and allocates memory as backing store.
 * The allocated memory will occupy a contiguous chunk of bus address space.
 *
 * For devices that are directly connected to the memory bus then the allocated
 * memory will be physically contiguous. For devices that access through an
 * IOMMU, then the allocated memory is not expected to be physically contiguous
 * because having contiguous IOVAs is sufficient to meet a devices DMA
 * requirements.
 *
 * Returns:
 * A struct drm_gem_dma_object * on success or an ERR_PTR()-encoded negative
 * error code on failure.
 */
struct drm_gem_dma_object *drm_gem_dma_create(struct drm_device *drm,
					      size_t size)
{
	/// Functional Utility: Creates a DMA GEM object and allocates physically (or IOVA) contiguous memory as its backing store.
	/// This function ensures the memory is suitable for DMA operations by the device.
	struct drm_gem_dma_object *dma_obj;
	int ret;

	// Block Logic: Round up the requested size to the nearest page size.
	size = round_up(size, PAGE_SIZE);

	// Block Logic: Create the base GEM DMA object structure.
	dma_obj = __drm_gem_dma_create(drm, size, false);
	// Conditional Logic: Handle errors from base object creation.
	if (IS_ERR(dma_obj))
		return dma_obj;

	// Conditional Logic: Allocate memory based on whether non-coherent mapping is required.
	if (dma_obj->map_noncoherent) {
		// Block Logic: Allocate non-coherent DMA memory.
		dma_obj->vaddr = dma_alloc_noncoherent(drm->dev, size,
						       &dma_obj->dma_addr,
						       DMA_TO_DEVICE,
						       GFP_KERNEL | __GFP_NOWARN);
	} else {
		// Block Logic: Allocate write-combined DMA memory.
		dma_obj->vaddr = dma_alloc_wc(drm->dev, size,
					      &dma_obj->dma_addr,
					      GFP_KERNEL | __GFP_NOWARN);
	}
	// Conditional Logic: Handle memory allocation failure.
	if (!dma_obj->vaddr) {
		drm_dbg(drm, "failed to allocate buffer with size %zu\n",
			 size);
		ret = -ENOMEM;
		goto error;
	}

	return dma_obj;

error:
	// Error Handling: Put a reference on the GEM object base to clean up upon error.
	drm_gem_object_put(&dma_obj->base);
	return ERR_PTR(ret);
}
EXPORT_SYMBOL_GPL(drm_gem_dma_create);

/**
 * drm_gem_dma_create_with_handle - allocate an object with the given size and
 *     return a GEM handle to it
 * @file_priv: DRM file-private structure to register the handle for
 * @drm: DRM device
 * @size: size of the object to allocate
 * @handle: return location for the GEM handle
 *
 * This function creates a DMA GEM object, allocating a chunk of memory as
 * backing store. The GEM object is then added to the list of object associated
 * with the given file and a handle to it is returned.
 *
 * The allocated memory will occupy a contiguous chunk of bus address space.
 * See drm_gem_dma_create() for more details.
 *
 * Returns:
 * A struct drm_gem_dma_object * on success or an ERR_PTR()-encoded negative
 * error code on failure.
 */
static struct drm_gem_dma_object *
drm_gem_dma_create_with_handle(struct drm_file *file_priv,
			       struct drm_device *drm, size_t size,
			       uint32_t *handle)
{
	/// Functional Utility: Creates a DMA GEM object, allocates its backing memory, and registers a GEM handle for it
	/// with the given DRM file-private structure. This makes the GEM object accessible to userspace.
	struct drm_gem_dma_object *dma_obj;
	struct drm_gem_object *gem_obj;
	int ret;

	// Block Logic: Create the DMA GEM object.
	dma_obj = drm_gem_dma_create(drm, size);
	// Conditional Logic: Handle errors from object creation.
	if (IS_ERR(dma_obj))
		return dma_obj;

	gem_obj = &dma_obj->base;

	/*
	 * allocate a id of idr table where the obj is registered
	 * and handle has the id what user can see.
	 */
	// Block Logic: Create a GEM handle for the object, making it visible to userspace.
	ret = drm_gem_handle_create(file_priv, gem_obj, handle);
	/* drop reference from allocate - handle holds it now. */
	// Block Logic: Release the initial reference, as the handle now holds a reference.
	drm_gem_object_put(gem_obj);
	// Conditional Logic: Handle handle creation errors.
	if (ret)
		return ERR_PTR(ret);

	return dma_obj;
}

/**
 * drm_gem_dma_free - free resources associated with a DMA GEM object
 * @dma_obj: DMA GEM object to free
 *
 * This function frees the backing memory of the DMA GEM object, cleans up the
 * GEM object state and frees the memory used to store the object itself.
 * If the buffer is imported and the virtual address is set, it is released.
 */
void drm_gem_dma_free(struct drm_gem_dma_object *dma_obj)
{
	/// Functional Utility: Frees all resources associated with a DMA GEM object, including its backing memory and GEM object state.
	/// It handles both locally allocated and imported DMA buffers.
	struct drm_gem_object *gem_obj = &dma_obj->base;
	struct iosys_map map = IOSYS_MAP_INIT_VADDR(dma_obj->vaddr);

	// Conditional Logic: Check if the GEM object is imported (i.e., shared from another driver).
	if (drm_gem_is_imported(gem_obj)) {
		// Conditional Logic: If a virtual address is mapped, unmap it.
		if (dma_obj->vaddr)
			dma_buf_vunmap_unlocked(gem_obj->dma_buf, &map);
		// Block Logic: Destroy the PRIME GEM object.
		drm_prime_gem_destroy(gem_obj, dma_obj->sgt);
	} else if (dma_obj->vaddr) {
		// Conditional Logic: If the object is not imported and has a virtual address.
		// Block Logic: Free the DMA memory based on whether it was non-coherent.
		if (dma_obj->map_noncoherent)
			dma_free_noncoherent(gem_obj->dev->dev, dma_obj->base.size,
					     dma_obj->vaddr, dma_obj->dma_addr,
					     DMA_TO_DEVICE);
		else
			dma_free_wc(gem_obj->dev->dev, dma_obj->base.size,
				    dma_obj->vaddr, dma_obj->dma_addr);
	}

	// Block Logic: Release the GEM object.
	drm_gem_object_release(gem_obj);

	// Block Logic: Free the DMA GEM object structure itself.
	kfree(dma_obj);
}
EXPORT_SYMBOL_GPL(drm_gem_dma_free);

/**
 * drm_gem_dma_dumb_create_internal - create a dumb buffer object
 * @file_priv: DRM file-private structure to create the dumb buffer for
 * @drm: DRM device
 * @args: IOCTL data
 *
 * This aligns the pitch and size arguments to the minimum required. This is
 * an internal helper that can be wrapped by a driver to account for hardware
 * with more specific alignment requirements. It should not be used directly
 * as their &drm_driver.dumb_create callback.
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_dma_dumb_create_internal(struct drm_file *file_priv,
				     struct drm_device *drm,
				     struct drm_mode_create_dumb *args)
{
	/// Functional Utility: Creates a "dumb" buffer object (a simple, unmanaged GEM object) with DMA backing.
	/// It adjusts the pitch and size arguments to meet minimum alignment requirements and returns a GEM handle.
	unsigned int min_pitch = DIV_ROUND_UP(args->width * args->bpp, 8);
	struct drm_gem_dma_object *dma_obj;

	// Block Logic: Ensure the requested pitch meets the minimum requirements.
	// Invariant: `args->pitch` will be at least `min_pitch`.
	if (args->pitch < min_pitch)
		args->pitch = min_pitch;

	// Block Logic: Ensure the requested size accommodates the calculated pitch and height.
	// Invariant: `args->size` will be at least `args->pitch * args->height`.
	if (args->size < args->pitch * args->height)
		args->size = args->pitch * args->height;

	// Block Logic: Create the DMA GEM object with a handle.
	dma_obj = drm_gem_dma_create_with_handle(file_priv, drm, args->size,
						 &args->handle);
	// Block Logic: Return 0 on success or an ERR_PTR()-encoded negative error code.
	return PTR_ERR_OR_ZERO(dma_obj);
}
EXPORT_SYMBOL_GPL(drm_gem_dma_dumb_create_internal);

/**
 * drm_gem_dma_dumb_create - create a dumb buffer object
 * @file_priv: DRM file-private structure to create the dumb buffer for
 * @drm: DRM device
 * @args: IOCTL data
 *
 * This function computes the pitch of the dumb buffer and rounds it up to an
 * integer number of bytes per pixel. Drivers for hardware that doesn't have
 * any additional restrictions on the pitch can directly use this function as
 * their &drm_driver.dumb_create callback.
 *
 * For hardware with additional restrictions, drivers can adjust the fields
 * set up by userspace and pass the IOCTL data along to the
 * drm_gem_dma_dumb_create_internal() function.
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_dma_dumb_create(struct drm_file *file_priv,
			    struct drm_device *drm,
			    struct drm_mode_create_dumb *args)
{
	/// Functional Utility: Public interface for creating a "dumb" buffer object with DMA backing.
	/// It calculates and aligns the buffer's pitch and size, then calls `drm_gem_dma_create_with_handle`
	/// to create the GEM object and obtain a handle.
	struct drm_gem_dma_object *dma_obj;

	// Block Logic: Calculate the pitch, rounding up to an integer number of bytes per pixel.
	args->pitch = DIV_ROUND_UP(args->width * args->bpp, 8);
	// Block Logic: Calculate the total size required for the buffer.
	args->size = args->pitch * args->height;

	// Block Logic: Create the DMA GEM object with a handle.
	dma_obj = drm_gem_dma_create_with_handle(file_priv, drm, args->size,
						 &args->handle);
	// Block Logic: Return 0 on success or an ERR_PTR()-encoded negative error code.
	return PTR_ERR_OR_ZERO(dma_obj);
}
EXPORT_SYMBOL_GPL(drm_gem_dma_dumb_create);

const struct vm_operations_struct drm_gem_dma_vm_ops = {
	/// Functional Utility: Defines the virtual memory operations for DMA GEM objects,
	/// specifying how the kernel should handle `open` and `close` operations on memory-mapped regions.
	.open = drm_gem_vm_open,
	.close = drm_gem_vm_close,
};
EXPORT_SYMBOL_GPL(drm_gem_dma_vm_ops);

#ifndef CONFIG_MMU
/**
 * drm_gem_dma_get_unmapped_area - propose address for mapping in noMMU cases
 * @filp: file object
 * @addr: memory address
 * @len: buffer size
 * @pgoff: page offset
 * @flags: memory flags
 *
 * This function is used in noMMU platforms to propose address mapping
 * for a given buffer.
 * It's intended to be used as a direct handler for the struct
 * &file_operations.get_unmapped_area operation.
 *
 * Returns:
 * mapping address on success or a negative error code on failure.
 */
unsigned long drm_gem_dma_get_unmapped_area(struct file *filp,
					    unsigned long addr,
					    unsigned long len,
					    unsigned long pgoff,
					    unsigned long flags)
{
	/// Functional Utility: Proposes a suitable address for memory-mapping a DMA GEM object in noMMU environments.
	/// It looks up the GEM object associated with the requested offset and ensures its validity before returning
	/// the virtual address of its backing store.
	struct drm_gem_dma_object *dma_obj;
	struct drm_gem_object *obj = NULL;
	struct drm_file *priv = filp->private_data;
	struct drm_device *dev = priv->minor->dev;
	struct drm_vma_offset_node *node;

	// Conditional Logic: Return error if the DRM device is unplugged.
	if (drm_dev_is_unplugged(dev))
		return -ENODEV;

	// Block Logic: Acquire a lock for VMA offset manager lookup.
	drm_vma_offset_lock_lookup(dev->vma_offset_manager);
	// Block Logic: Look up the VMA node corresponding to the requested page offset and length.
	node = drm_vma_offset_exact_lookup_locked(dev->vma_offset_manager,
						  pgoff,
						  len >> PAGE_SHIFT);
	// Conditional Logic: If a VMA node is found.
	if (likely(node)) {
		// Block Logic: Get the GEM object from the VMA node.
		obj = container_of(node, struct drm_gem_object, vma_node);
		/*
		 * When the object is being freed, after it hits 0-refcnt it
		 * proceeds to tear down the object. In the process it will
		 * attempt to remove the VMA offset and so acquire this
		 * mgr->vm_lock.  Therefore if we find an object with a 0-refcnt
		 * that matches our range, we know it is in the process of being
		 * destroyed and will be freed as soon as we release the lock -
		 * so we have to check for the 0-refcnted object and treat it as
		 * invalid.
		 */
		// Conditional Logic: Check if the object's refcount is non-zero, otherwise treat as invalid.
		if (!kref_get_unless_zero(&obj->refcount))
			obj = NULL;
	}

	// Block Logic: Release the lock for VMA offset manager lookup.
	drm_vma_offset_unlock_lookup(dev->vma_offset_manager);

	// Conditional Logic: If no valid GEM object is found, return error.
	if (!obj)
		return -EINVAL;

	// Conditional Logic: Check if the VMA node is allowed for the given file private.
	if (!drm_vma_node_is_allowed(node, priv)) {
		drm_gem_object_put(obj);
		return -EACCES;
	}

	// Block Logic: Convert the generic GEM object to a DMA GEM object.
	dma_obj = to_drm_gem_dma_obj(obj);

	// Block Logic: Put a reference on the GEM object.
	drm_gem_object_put(obj);

	// Conditional Logic: Return the virtual address if available, otherwise return error.
	return dma_obj->vaddr ? (unsigned long)dma_obj->vaddr : -EINVAL;
}
EXPORT_SYMBOL_GPL(drm_gem_dma_get_unmapped_area);
#endif

/**
 * drm_gem_dma_print_info() - Print &drm_gem_dma_object info for debugfs
 * @dma_obj: DMA GEM object
 * @p: DRM printer
 * @indent: Tab indentation level
 *
 * This function prints dma_addr and vaddr for use in e.g. debugfs output.
 */
void drm_gem_dma_print_info(const struct drm_gem_dma_object *dma_obj,
			    struct drm_printer *p, unsigned int indent)
{
	/// Functional Utility: Prints debugging information for a DMA GEM object, including its DMA address and virtual address.
	/// This is useful for debugfs output and driver diagnostics.
	drm_printf_indent(p, indent, "dma_addr=%pad\n", &dma_obj->dma_addr);
	drm_printf_indent(p, indent, "vaddr=%p\n", dma_obj->vaddr);
}
EXPORT_SYMBOL(drm_gem_dma_print_info);

/**
 * drm_gem_dma_get_sg_table - provide a scatter/gather table of pinned
 *     pages for a DMA GEM object
 * @dma_obj: DMA GEM object
 *
 * This function exports a scatter/gather table by calling the standard
 * DMA mapping API.
 *
 * Returns:
 * A pointer to the scatter/gather table of pinned pages or NULL on failure.
 */
struct sg_table *drm_gem_dma_get_sg_table(struct drm_gem_dma_object *dma_obj)
{
	/// Functional Utility: Generates a scatter/gather table for a DMA GEM object, describing its physical memory layout.
	/// This table can then be used for DMA operations, especially when dealing with devices that support scatter-gather.
	struct drm_gem_object *obj = &dma_obj->base;
	struct sg_table *sgt;
	int ret;

	// Block Logic: Allocate memory for the scatter/gather table structure.
	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	// Conditional Logic: Handle memory allocation failure.
	if (!sgt)
		return ERR_PTR(-ENOMEM);

	// Block Logic: Populate the scatter/gather table using DMA mapping API.
	ret = dma_get_sgtable(obj->dev->dev, sgt, dma_obj->vaddr,
			      dma_obj->dma_addr, obj->size);
	// Conditional Logic: Handle error during scatter/gather table creation.
	if (ret < 0)
		goto out;

	return sgt;

out:
	// Error Handling: Free the scatter/gather table structure upon error.
	kfree(sgt);
	return ERR_PTR(ret);
}
EXPORT_SYMBOL_GPL(drm_gem_dma_get_sg_table);

/**
 * drm_gem_dma_prime_import_sg_table - produce a DMA GEM object from another
 *     driver's scatter/gather table of pinned pages
 * @dev: device to import into
 * @attach: DMA-BUF attachment
 * @sgt: scatter/gather table of pinned pages
 *
 * This function imports a scatter/gather table exported via DMA-BUF by
 * another driver. Imported buffers must be physically contiguous in memory
 * (i.e. the scatter/gather table must contain a single entry). Drivers that
 * use the DMA helpers should set this as their
 * &drm_driver.gem_prime_import_sg_table callback.
 *
 * Returns:
 * A pointer to a newly created GEM object or an ERR_PTR-encoded negative
 * error code on failure.
 */
struct drm_gem_object *
drm_gem_dma_prime_import_sg_table(struct drm_device *dev,
				  struct dma_buf_attachment *attach,
				  struct sg_table *sgt)
{
	/// Functional Utility: Imports a scatter/gather table from another driver (via DMA-BUF)
	/// and creates a DMA GEM object from it. This allows sharing of memory buffers between
	/// different DRM drivers or other kernel components.
	struct drm_gem_dma_object *dma_obj;

	/* check if the entries in the sg_table are contiguous */
	// Conditional Logic: Check if the scatter/gather table represents a contiguous block of memory.
	// Precondition: Imported buffers must be physically contiguous.
	if (drm_prime_get_contiguous_size(sgt) < attach->dmabuf->size)
		return ERR_PTR(-EINVAL);

	/* Create a DMA GEM buffer. */
	// Block Logic: Create the base DMA GEM object for the imported buffer.
	dma_obj = __drm_gem_dma_create(dev, attach->dmabuf->size, true);
	// Conditional Logic: Handle errors from object creation.
	if (IS_ERR(dma_obj))
		return ERR_CAST(dma_obj);

	// Block Logic: Store the DMA address and scatter/gather table in the DMA GEM object.
	dma_obj->dma_addr = sg_dma_address(sgt->sgl);
	dma_obj->sgt = sgt;

	drm_dbg_prime(dev, "dma_addr = %pad, size = %zu\n", &dma_obj->dma_addr,
		      attach->dmabuf->size);

	return &dma_obj->base;
}
EXPORT_SYMBOL_GPL(drm_gem_dma_prime_import_sg_table);

/**
 * drm_gem_dma_vmap - map a DMA GEM object into the kernel's virtual
 *     address space
 * @dma_obj: DMA GEM object
 * @map: Returns the kernel virtual address of the DMA GEM object's backing
 *       store.
 *
 * This function maps a buffer into the kernel's virtual address space.
 * Since the DMA buffers are already mapped into the kernel virtual address
 * space this simply returns the cached virtual address.
 *
 * Returns:
 * 0 on success, or a negative error code otherwise.
 */
int drm_gem_dma_vmap(struct drm_gem_dma_object *dma_obj,
		     struct iosys_map *map)
{
	/// Functional Utility: Maps a DMA GEM object into the kernel's virtual address space,
	/// providing a kernel-accessible pointer to its backing memory.
	iosys_map_set_vaddr(map, dma_obj->vaddr);

	return 0;
}
EXPORT_SYMBOL_GPL(drm_gem_dma_vmap);

/**
 * drm_gem_dma_mmap - memory-map an exported DMA GEM object
 * @dma_obj: DMA GEM object
 * @vma: VMA for the area to be mapped
 *
 * This function maps a buffer into a userspace process's address space.
 * In addition to the usual GEM VMA setup it immediately faults in the entire
 * object instead of using on-demand faulting.
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_dma_mmap(struct drm_gem_dma_object *dma_obj, struct vm_area_struct *vma)
{
	/// Functional Utility: Memory-maps an exported DMA GEM object into a userspace process's address space.
	/// This enables userspace applications to directly access the graphics memory.
	int ret;

	/*
	 * Clear the VM_PFNMAP flag that was set by drm_gem_mmap(), and set the
	 * vm_pgoff (used as a fake buffer offset by DRM) to 0 as we want to map
	 * the whole buffer.
	 */
	// Block Logic: Adjust `vma->vm_pgoff` and clear `VM_PFNMAP` flag for mapping the entire buffer.
	vma->vm_pgoff -= drm_vma_node_start(&obj->vma_node);
	vm_flags_mod(vma, VM_DONTEXPAND, VM_PFNMAP);

	// Conditional Logic: Perform memory mapping based on whether non-coherent mapping is enabled.
	if (dma_obj->map_noncoherent) {
		// Block Logic: Get page protection flags for the VMA.
		vma->vm_page_prot = vm_get_page_prot(vma->vm_flags);

		// Block Logic: Map pages for non-coherent DMA.
		ret = dma_mmap_pages(dma_obj->base.dev->dev,
				     vma, vma->vm_end - vma->vm_start,
				     virt_to_page(dma_obj->vaddr));
	} else {
		// Block Logic: Map pages for write-combined DMA.
		ret = dma_mmap_wc(dma_obj->base.dev->dev, vma, dma_obj->dma_addr,
				  dma_obj->dma_addr,
				  vma->vm_end - vma->vm_start);
	}
	// Conditional Logic: If mapping fails, close the VMA.
	if (ret)
		drm_gem_vm_close(vma);

	return ret;
}
EXPORT_SYMBOL_GPL(drm_gem_dma_mmap);

/**
 * drm_gem_dma_prime_import_sg_table_vmap - PRIME import another driver's
 *	scatter/gather table and get the virtual address of the buffer
 * @dev: DRM device
 * @attach: DMA-BUF attachment
 * @sgt: Scatter/gather table of pinned pages
 *
 * This function imports a scatter/gather table using
 * drm_gem_dma_prime_import_sg_table() and uses dma_buf_vmap() to get the kernel
 * virtual address. This ensures that a DMA GEM object always has its virtual
 * address set. This address is released when the object is freed.
 *
 * This function can be used as the &drm_driver.gem_prime_import_sg_table
 * callback. The &DRM_GEM_DMA_DRIVER_OPS_VMAP macro provides a shortcut to set
 * the necessary DRM driver operations.
 *
 * Returns:
 * A pointer to a newly created GEM object or an ERR_PTR-encoded negative
 * error code on failure.
 */
struct drm_gem_object *
drm_gem_dma_prime_import_sg_table_vmap(struct drm_device *dev,
				       struct dma_buf_attachment *attach,
				       struct sg_table *sgt)
{
	/// Functional Utility: Imports a scatter/gather table from another driver via DMA-BUF and then maps it
	/// into the kernel's virtual address space, ensuring the DMA GEM object always has a valid virtual address.
	struct drm_gem_dma_object *dma_obj;
	struct drm_gem_object *obj;
	struct iosys_map map;
	int ret;

	// Block Logic: Virtually map the DMA-BUF attachment.
	ret = dma_buf_vmap_unlocked(attach->dmabuf, &map);
	// Conditional Logic: Handle failure to virtually map the buffer.
	if (ret) {
		DRM_ERROR("Failed to vmap PRIME buffer\n");
		return ERR_PTR(ret);
	}

	// Block Logic: Import the scatter/gather table to create a GEM object.
	obj = drm_gem_dma_prime_import_sg_table(dev, attach, sgt);
	// Conditional Logic: Handle errors during GEM object creation.
	if (IS_ERR(obj)) {
		// Error Handling: Unmap the buffer if GEM object creation fails.
		dma_buf_vunmap_unlocked(attach->dmabuf, &map);
		return obj;
	}

	// Block Logic: Store the virtual address in the DMA GEM object.
	dma_obj = to_drm_gem_dma_obj(obj);
	dma_obj->vaddr = map.vaddr;

	return obj;
}
EXPORT_SYMBOL(drm_gem_dma_prime_import_sg_table_vmap);

MODULE_DESCRIPTION("DRM DMA memory-management helpers");
MODULE_IMPORT_NS("DMA_BUF");
MODULE_LICENSE("GPL");
