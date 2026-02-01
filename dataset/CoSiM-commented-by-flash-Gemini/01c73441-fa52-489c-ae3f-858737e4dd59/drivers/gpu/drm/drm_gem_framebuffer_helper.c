/**
 * @file drm_gem_framebuffer_helper.c
 * @brief Provides helper functions for creating and managing DRM framebuffers
 * backed by GEM (Graphics Execution Manager) objects. This includes general
 * framebuffer management and specific support for AFBC (ARM Frame Buffer Compression)
 * formats.
 *
 * Functional Utility: Simplifies the process of creating, initializing, and
 * managing framebuffers for DRM drivers that use GEM objects for their backing
 * storage. It handles details like GEM object lookup, framebuffer initialization,
 * buffer size validation, and provides callbacks for framebuffer destruction
 * and handle creation.
 *
 * Key Data Structures:
 * - `drm_framebuffer`: Represents a framebuffer in the DRM subsystem.
 * - `drm_gem_object`: Generic buffer objects managed by DRM.
 * - `drm_mode_fb_cmd2`: Userspace command for creating framebuffers.
 * - `drm_afbc_framebuffer`: Extended framebuffer structure for AFBC formats.
 */

// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * drm gem framebuffer helper functions
 *
 * Copyright (C) 2017 Noralf Trønnes
 */

#include <linux/slab.h>
#include <linux/module.h>

#include <drm/drm_damage_helper.h>
#include <drm/drm_drv.h>
#include <drm/drm_fourcc.h>
#include <drm/drm_framebuffer.h>
#include <drm/drm_gem.h>
#include <drm/drm_gem_framebuffer_helper.h>
#include <drm/drm_modeset_helper.h>

#include "drm_internal.h"

MODULE_IMPORT_NS("DMA_BUF");

#define AFBC_HEADER_SIZE		16
#define AFBC_TH_LAYOUT_ALIGNMENT	8
#define AFBC_HDR_ALIGN			64
#define AFBC_SUPERBLOCK_PIXELS		256
#define AFBC_SUPERBLOCK_ALIGNMENT	128
#define AFBC_TH_BODY_START_ALIGNMENT	4096

/**
 * DOC: overview
 *
 * This library provides helpers for drivers that don't subclass
 * &drm_framebuffer and use &drm_gem_object for their backing storage.
 *
 * Drivers without additional needs to validate framebuffers can simply use
 * drm_gem_fb_create() and everything is wired up automatically. Other drivers
 * can use all parts independently.
 */

/**
 * drm_gem_fb_get_obj() - Get GEM object backing the framebuffer
 * @fb: Framebuffer
 * @plane: Plane index
 *
 * No additional reference is taken beyond the one that the &drm_frambuffer
 * already holds.
 *
 * Returns:
 * Pointer to &drm_gem_object for the given framebuffer and plane index or NULL
 * if it does not exist.
 */
struct drm_gem_object *drm_gem_fb_get_obj(struct drm_framebuffer *fb,
					  unsigned int plane)
{
	/// Functional Utility: Retrieves the GEM object that backs a specific plane of the given framebuffer.
	/// This allows access to the underlying buffer object without increasing its reference count.
	struct drm_device *dev = fb->dev;

	// Conditional Logic: Warn and return NULL if the plane index is out of bounds.
	if (drm_WARN_ON_ONCE(dev, plane >= ARRAY_SIZE(fb->obj)))
		return NULL;
	// Conditional Logic: Warn and return NULL if the GEM object for the specified plane does not exist.
	else if (drm_WARN_ON_ONCE(dev, !fb->obj[plane]))
		return NULL;

	return fb->obj[plane];
}
EXPORT_SYMBOL_GPL(drm_gem_fb_get_obj);

static int
drm_gem_fb_init(struct drm_device *dev,
		 struct drm_framebuffer *fb,
		 const struct drm_mode_fb_cmd2 *mode_cmd,
		 struct drm_gem_object **obj, unsigned int num_planes,
		 const struct drm_framebuffer_funcs *funcs)
{
	/// Functional Utility: Initializes a `drm_framebuffer` structure with common metadata and backing GEM objects.
	/// This is an internal helper used by other framebuffer creation functions.
	unsigned int i;
	int ret;

	// Block Logic: Fill the basic framebuffer structure using helper function.
	drm_helper_mode_fill_fb_struct(dev, fb, mode_cmd);

	// Block Logic: Assign the provided GEM objects to the framebuffer's object array.
	for (i = 0; i < num_planes; i++)
		fb->obj[i] = obj[i];

	// Block Logic: Initialize the DRM framebuffer.
	ret = drm_framebuffer_init(dev, fb, funcs);
	// Conditional Logic: Log an error if framebuffer initialization fails.
	if (ret)
		drm_err(dev, "Failed to init framebuffer: %d\n", ret);

	return ret;
}

/**
 * drm_gem_fb_destroy - Free GEM backed framebuffer
 * @fb: Framebuffer
 *
 * Frees a GEM backed framebuffer with its backing buffer(s) and the structure
 * itself. Drivers can use this as their &drm_framebuffer_funcs->destroy
 * callback.
 */
void drm_gem_fb_destroy(struct drm_framebuffer *fb)
{
	/// Functional Utility: Destroys a GEM-backed framebuffer, releasing its backing GEM objects and freeing the framebuffer structure itself.
	/// This function serves as a common `destroy` callback for `drm_framebuffer_funcs`.
	unsigned int i;

	// Block Logic: Put references on all backing GEM objects to allow them to be freed.
	for (i = 0; i < fb->format->num_planes; i++)
		drm_gem_object_put(fb->obj[i]);

	// Block Logic: Clean up the framebuffer resources.
	drm_framebuffer_cleanup(fb);
	// Block Logic: Free the framebuffer structure memory.
	kfree(fb);
}
EXPORT_SYMBOL(drm_gem_fb_destroy);

/**
 * drm_gem_fb_create_handle - Create handle for GEM backed framebuffer
 * @fb: Framebuffer
 * @file: DRM file to register the handle for
 * @handle: Pointer to return the created handle
 *
 * This function creates a handle for the GEM object backing the framebuffer.
 * Drivers can use this as their &drm_framebuffer_funcs->create_handle
 * callback. The GETFB IOCTL calls into this callback.
 *
 * Returns:
 * 0 on success or a negative error code on failure.
 */
int drm_gem_fb_create_handle(struct drm_framebuffer *fb, struct drm_file *file,
			     unsigned int *handle)
{
	/// Functional Utility: Creates a GEM handle for the primary GEM object backing the framebuffer.
	/// This allows userspace to refer to the framebuffer's main buffer via a handle.
	return drm_gem_handle_create(file, fb->obj[0], handle);
}
EXPORT_SYMBOL(drm_gem_fb_create_handle);

/**
 * drm_gem_fb_init_with_funcs() - Helper function for implementing
 *				  &drm_mode_config_funcs.fb_create
 *				  callback in cases when the driver
 *				  allocates a subclass of
 *				  struct drm_framebuffer
 * @dev: DRM device
 * @fb: framebuffer object
 * @file: DRM file that holds the GEM handle(s) backing the framebuffer
 * @mode_cmd: Metadata from the userspace framebuffer creation request
 * @funcs: vtable to be used for the new framebuffer object
 *
 * This function can be used to set &drm_framebuffer_funcs for drivers that need
 * custom framebuffer callbacks. Use drm_gem_fb_create() if you don't need to
 * change &drm_framebuffer_funcs. The function does buffer size validation.
 * The buffer size validation is for a general case, though, so users should
 * pay attention to the checks being appropriate for them or, at least,
 * non-conflicting.
 *
 * Returns:
 * Zero or a negative error code.
 */
int drm_gem_fb_init_with_funcs(struct drm_device *dev,
			       struct drm_framebuffer *fb,
			       struct drm_file *file,
			       const struct drm_mode_fb_cmd2 *mode_cmd,
			       const struct drm_framebuffer_funcs *funcs)
{
	/// Functional Utility: Initializes a framebuffer with custom functions and validates its backing GEM objects.
	/// This helper is for drivers requiring specific framebuffer callbacks beyond the default ones.
	const struct drm_format_info *info;
	struct drm_gem_object *objs[DRM_FORMAT_MAX_PLANES];
	unsigned int i;
	int ret;

	// Block Logic: Retrieve format information for the framebuffer.
	info = drm_get_format_info(dev, mode_cmd);
	// Conditional Logic: Handle failure to get format info.
	if (!info) {
		drm_dbg_kms(dev, "Failed to get FB format info\n");
		return -EINVAL;
	}

	// Conditional Logic: If atomic modeset is used, validate the pixel format and modifier.
	if (drm_drv_uses_atomic_modeset(dev) &&
	    !drm_any_plane_has_format(dev, mode_cmd->pixel_format,
				      mode_cmd->modifier[0])) {
		drm_dbg_kms(dev, "Unsupported pixel format %p4cc / modifier 0x%llx\n",
			    &mode_cmd->pixel_format, mode_cmd->modifier[0]);
		return -EINVAL;
	}

	// Block Logic: Iterate through each plane to look up GEM objects and validate their sizes.
	for (i = 0; i < info->num_planes; i++) {
		unsigned int width = mode_cmd->width / (i ? info->hsub : 1);
		unsigned int height = mode_cmd->height / (i ? info->vsub : 1);
		unsigned int min_size;

		// Block Logic: Look up the GEM object by handle.
		objs[i] = drm_gem_object_lookup(file, mode_cmd->handles[i]);
		// Conditional Logic: Handle failure to lookup GEM object.
		if (!objs[i]) {
			drm_dbg_kms(dev, "Failed to lookup GEM object\n");
			ret = -ENOENT;
			goto err_gem_object_put;
		}

		// Block Logic: Calculate the minimum required size for the current plane's buffer.
		min_size = (height - 1) * mode_cmd->pitches[i]
			 + drm_format_info_min_pitch(info, i, width)
			 + mode_cmd->offsets[i];

		// Conditional Logic: Validate that the GEM object's size is sufficient.
		if (objs[i]->size < min_size) {
			drm_dbg_kms(dev,
				    "GEM object size (%zu) smaller than minimum size (%u) for plane %d\n",
				    objs[i]->size, min_size, i);
			drm_gem_object_put(objs[i]);
			ret = -EINVAL;
			goto err_gem_object_put;
		}
	}

	// Block Logic: Initialize the framebuffer with the validated GEM objects and custom functions.
	ret = drm_gem_fb_init(dev, fb, mode_cmd, objs, i, funcs);
	// Conditional Logic: Handle framebuffer initialization failure.
	if (ret)
		goto err_gem_object_put;

	return 0;

err_gem_object_put:
	// Error Handling: Put references on any looked-up GEM objects in case of an error during initialization.
	while (i > 0) {
		--i;
		drm_gem_object_put(objs[i]);
	}
	return ret;
}
EXPORT_SYMBOL_GPL(drm_gem_fb_init_with_funcs);

/**
 * drm_gem_fb_create_with_funcs() - Helper function for the
 *                                  &drm_mode_config_funcs.fb_create
 *                                  callback
 * @dev: DRM device
 * @file: DRM file that holds the GEM handle(s) backing the framebuffer
 * @mode_cmd: Metadata from the userspace framebuffer creation request
 * @funcs: vtable to be used for the new framebuffer object
 *
 * This function can be used to set &drm_framebuffer_funcs for drivers that need
 * custom framebuffer callbacks. Use drm_gem_fb_create() if you don't need to
 * change &drm_framebuffer_funcs. The function does buffer size validation.
 *
 * Returns:
 * Pointer to a &drm_framebuffer on success or an error pointer on failure.
 */
struct drm_framebuffer *
drm_gem_fb_create_with_funcs(struct drm_device *dev, struct drm_file *file,
			     const struct drm_mode_fb_cmd2 *mode_cmd,
			     const struct drm_framebuffer_funcs *funcs)
{
	/// Functional Utility: Allocates a new `drm_framebuffer` structure and initializes it using custom functions.
	/// This serves as a flexible framebuffer creation entry point for drivers with specific needs.
	struct drm_framebuffer *fb;
	int ret;

	// Block Logic: Allocate memory for the framebuffer structure.
	fb = kzalloc(sizeof(*fb), GFP_KERNEL);
	// Conditional Logic: Handle memory allocation failure.
	if (!fb)
		return ERR_PTR(-ENOMEM);

	// Block Logic: Initialize the framebuffer with the provided metadata and functions.
	ret = drm_gem_fb_init_with_funcs(dev, fb, file, mode_cmd, funcs);
	// Conditional Logic: Handle initialization failure, freeing the allocated framebuffer.
	if (ret) {
		kfree(fb);
		return ERR_PTR(ret);
	}

	return fb;
}
EXPORT_SYMBOL_GPL(drm_gem_fb_create_with_funcs);

static const struct drm_framebuffer_funcs drm_gem_fb_funcs = {
	.destroy	= drm_gem_fb_destroy,
	.create_handle	= drm_gem_fb_create_handle,
};

/**
 * drm_gem_fb_create() - Helper function for the
 *                       &drm_mode_config_funcs.fb_create callback
 * @dev: DRM device
 * @file: DRM file that holds the GEM handle(s) backing the framebuffer
 * @mode_cmd: Metadata from the userspace framebuffer creation request
 *
 * This function creates a new framebuffer object described by
 * &drm_mode_fb_cmd2. This description includes handles for the buffer(s)
 * backing the framebuffer.
 *
 * If your hardware has special alignment or pitch requirements these should be
 * checked before calling this function. The function does buffer size
 * validation. Use drm_gem_fb_create_with_dirty() if you need framebuffer
 * flushing.
 *
 * Drivers can use this as their &drm_mode_config_funcs.fb_create callback.
 * The ADDFB2 IOCTL calls into this callback.
 *
 * Returns:
 * Pointer to a &drm_framebuffer on success or an error pointer on failure.
 */
struct drm_framebuffer *
drm_gem_fb_create(struct drm_device *dev, struct drm_file *file,
		  const struct drm_mode_fb_cmd2 *mode_cmd)
{
	/// Functional Utility: Creates a new framebuffer object with default GEM-backed functions.
	/// This is a simplified entry point for drivers without special requirements for framebuffer callbacks.
	return drm_gem_fb_create_with_funcs(dev, file, mode_cmd,
					    &drm_gem_fb_funcs);
}
EXPORT_SYMBOL_GPL(drm_gem_fb_create);

static const struct drm_framebuffer_funcs drm_gem_fb_funcs_dirtyfb = {
	.destroy	= drm_gem_fb_destroy,
	.create_handle	= drm_gem_fb_create_handle,
	.dirty		= drm_atomic_helper_dirtyfb,
};

/**
 * drm_gem_fb_create_with_dirty() - Helper function for the
 *                       &drm_mode_config_funcs.fb_create callback
 * @dev: DRM device
 * @file: DRM file that holds the GEM handle(s) backing the framebuffer
 * @mode_cmd: Metadata from the userspace framebuffer creation request
 *
 * This function creates a new framebuffer object described by
 * &drm_mode_fb_cmd2. This description includes handles for the buffer(s)
 * backing the framebuffer. drm_atomic_helper_dirtyfb() is used for the dirty
 * callback giving framebuffer flushing through the atomic machinery. Use
 * drm_gem_fb_create() if you don't need the dirty callback.
 * The function does buffer size validation.
 *
 * Drivers should also call drm_plane_enable_fb_damage_clips() on all planes
 * to enable userspace to use damage clips also with the ATOMIC IOCTL.
 *
 * Drivers can use this as their &drm_mode_config_funcs.fb_create callback.
 * The ADDFB2 IOCTL calls into this callback.
 *
 * Returns:
 * Pointer to a &drm_framebuffer on success or an error pointer on failure.
 */
struct drm_framebuffer *
drm_gem_fb_create_with_dirty(struct drm_device *dev, struct drm_file *file,
			     const struct drm_mode_fb_cmd2 *mode_cmd)
{
	/// Functional Utility: Creates a new framebuffer object with GEM backing and includes the `drm_atomic_helper_dirtyfb` callback.
	/// This function is used when framebuffer flushing through the atomic modesetting machinery is required.
	return drm_gem_fb_create_with_funcs(dev, file, mode_cmd,
					    &drm_gem_fb_funcs_dirtyfb);
}
EXPORT_SYMBOL_GPL(drm_gem_fb_create_with_dirty);

/**
 * drm_gem_fb_vmap - maps all framebuffer BOs into kernel address space
 * @fb: the framebuffer
 * @map: returns the mapping's address for each BO
 * @data: returns the data address for each BO, can be NULL
 *
 * This function maps all buffer objects of the given framebuffer into
 * kernel address space and stores them in struct iosys_map. If the
 * mapping operation fails for one of the BOs, the function unmaps the
 * already established mappings automatically.
 *
 * Callers that want to access a BO's stored data should pass @data.
 * The argument returns the addresses of the data stored in each BO. This
 * is different from @map if the framebuffer's offsets field is non-zero.
 *
 * Both, @map and @data, must each refer to arrays with at least
 * fb->format->num_planes elements.
 *
 * See drm_gem_fb_vunmap() for unmapping.
 *
 * Returns:
 * 0 on success, or a negative errno code otherwise.
 */
int drm_gem_fb_vmap(struct drm_framebuffer *fb, struct iosys_map *map,
		    struct iosys_map *data)
{
	/// Functional Utility: Maps all GEM buffer objects of a given framebuffer into the kernel's virtual address space.
	/// It provides kernel-accessible pointers to the framebuffer data, handling multiple planes and potential offsets.
	struct drm_gem_object *obj;
	unsigned int i;
	int ret;

	// Block Logic: Iterate through each plane to map its backing GEM object.
	for (i = 0; i < fb->format->num_planes; ++i) {
		// Block Logic: Get the GEM object for the current plane.
		obj = drm_gem_fb_get_obj(fb, i);
		// Conditional Logic: Handle missing GEM object for a plane.
		if (!obj) {
			ret = -EINVAL;
			goto err_drm_gem_vunmap;
		}
		// Block Logic: Map the GEM object into kernel virtual address space.
		ret = drm_gem_vmap(obj, &map[i]);
		// Conditional Logic: Handle mapping failure.
		if (ret)
			goto err_drm_gem_vunmap;
	}

	// Conditional Logic: If `data` is provided, copy and adjust the mappings with framebuffer offsets.
	if (data) {
		for (i = 0; i < fb->format->num_planes; ++i) {
			memcpy(&data[i], &map[i], sizeof(data[i]));
			// Conditional Logic: Skip if the mapping is null.
			if (iosys_map_is_null(&data[i]))
				continue;
			// Block Logic: Adjust the data address by the framebuffer offset.
			iosys_map_incr(&data[i], fb->offsets[i]);
		}
	}

	return 0;

err_drm_gem_vunmap:
	// Error Handling: Unmap any already mapped GEM objects in case of an error during mapping.
	while (i) {
		--i;
		obj = drm_gem_fb_get_obj(fb, i);
		if (!obj)
			continue;
		drm_gem_vunmap(obj, &map[i]);
	}
	return ret;
}
EXPORT_SYMBOL(drm_gem_fb_vmap);

/**
 * drm_gem_fb_vunmap - unmaps framebuffer BOs from kernel address space
 * @fb: the framebuffer
 * @map: mapping addresses as returned by drm_gem_fb_vmap()
 *
 * This function unmaps all buffer objects of the given framebuffer.
 *
 * See drm_gem_fb_vmap() for more information.
 */
void drm_gem_fb_vunmap(struct drm_framebuffer *fb, struct iosys_map *map)
{
	/// Functional Utility: Unmaps all GEM buffer objects of a given framebuffer from the kernel's virtual address space.
	/// This function reverses the mapping performed by `drm_gem_fb_vmap`.
	unsigned int i = fb->format->num_planes;
	struct drm_gem_object *obj;

	// Block Logic: Iterate backwards through the planes to unmap their backing GEM objects.
	while (i) {
		--i;
		// Block Logic: Get the GEM object for the current plane.
		obj = drm_gem_fb_get_obj(fb, i);
		// Conditional Logic: Skip if the GEM object for the plane is missing.
		if (!obj)
			continue;
		// Conditional Logic: Skip if the mapping for the current plane is null.
		if (iosys_map_is_null(&map[i]))
			continue;
		// Block Logic: Unmap the GEM object from kernel virtual address space.
		drm_gem_vunmap(obj, &map[i]);
	}
}
EXPORT_SYMBOL(drm_gem_fb_vunmap);

static void __drm_gem_fb_end_cpu_access(struct drm_framebuffer *fb, enum dma_data_direction dir,
					unsigned int num_planes)
{
	/// Functional Utility: Signals the end of CPU access for a specified number of planes in a framebuffer's GEM buffer objects.
	/// It iterates through the planes and calls `dma_buf_end_cpu_access` for imported buffers.
	struct drm_gem_object *obj;
	int ret;

	// Block Logic: Iterate backwards through the specified number of planes.
	while (num_planes) {
		--num_planes;
		// Block Logic: Get the GEM object for the current plane.
		obj = drm_gem_fb_get_obj(fb, num_planes);
		// Conditional Logic: Skip if the GEM object is missing.
		if (!obj)
			continue;
		// Conditional Logic: Only process imported GEM objects.
		if (!drm_gem_is_imported(obj))
			continue;
		// Block Logic: Signal the end of CPU access for the DMA buffer.
		ret = dma_buf_end_cpu_access(obj->dma_buf, dir);
		// Conditional Logic: Log an error if signalling end of CPU access fails.
		if (ret)
			drm_err(fb->dev, "dma_buf_end_cpu_access(%u, %d) failed: %d\n",
				ret, num_planes, dir);
	}
}

/**
 * drm_gem_fb_begin_cpu_access - prepares GEM buffer objects for CPU access
 * @fb: the framebuffer
 * @dir: access mode
 *
 * Prepares a framebuffer's GEM buffer objects for CPU access. This function
 * must be called before accessing the BO data within the kernel. For imported
 * BOs, the function calls dma_buf_begin_cpu_access().
 *
 * See drm_gem_fb_end_cpu_access() for signalling the end of CPU access.
 *
 * Returns:
 * 0 on success, or a negative errno code otherwise.
 */
int drm_gem_fb_begin_cpu_access(struct drm_framebuffer *fb, enum dma_data_direction dir)
{
	/// Functional Utility: Prepares a framebuffer's GEM buffer objects for CPU access.
	/// For imported buffers, it calls `dma_buf_begin_cpu_access` to synchronize CPU and DMA access.
	struct drm_gem_object *obj;
	unsigned int i;
	int ret;

	// Block Logic: Iterate through each plane to prepare its backing GEM object for CPU access.
	for (i = 0; i < fb->format->num_planes; ++i) {
		// Block Logic: Get the GEM object for the current plane.
		obj = drm_gem_fb_get_obj(fb, i);
		// Conditional Logic: Handle missing GEM object.
		if (!obj) {
			ret = -EINVAL;
			goto err___drm_gem_fb_end_cpu_access;
		}
		// Conditional Logic: Only process imported GEM objects.
		if (!drm_gem_is_imported(obj))
			continue;
		// Block Logic: Signal beginning of CPU access for the DMA buffer.
		ret = dma_buf_begin_cpu_access(obj->dma_buf, dir);
		// Conditional Logic: Handle failure to begin CPU access.
		if (ret)
			goto err___drm_gem_fb_end_cpu_access;
	}

	return 0;

err___drm_gem_fb_end_cpu_access:
	// Error Handling: Call `__drm_gem_fb_end_cpu_access` to clean up partially prepared objects.
	__drm_gem_fb_end_cpu_access(fb, dir, i);
	return ret;
}
EXPORT_SYMBOL(drm_gem_fb_begin_cpu_access);

/**
 * drm_gem_fb_end_cpu_access - signals end of CPU access to GEM buffer objects
 * @fb: the framebuffer
 * @dir: access mode
 *
 * Signals the end of CPU access to the given framebuffer's GEM buffer objects. This
 * function must be paired with a corresponding call to drm_gem_fb_begin_cpu_access().
 * For imported BOs, the function calls dma_buf_end_cpu_access().
 *
 * See also drm_gem_fb_begin_cpu_access().
 */
void drm_gem_fb_end_cpu_access(struct drm_framebuffer *fb, enum dma_data_direction dir)
{
	/// Functional Utility: Signals the end of CPU access to a framebuffer's GEM buffer objects.
	/// This function is the counterpart to `drm_gem_fb_begin_cpu_access` and performs necessary synchronization for imported buffers.
	__drm_gem_fb_end_cpu_access(fb, dir, fb->format->num_planes);
}
EXPORT_SYMBOL(drm_gem_fb_end_cpu_access);

// TODO Drop this function and replace by drm_format_info_bpp() once all
// DRM_FORMAT_* provide proper block info in drivers/gpu/drm/drm_fourcc.c
static __u32 drm_gem_afbc_get_bpp(struct drm_device *dev,
				  const struct drm_mode_fb_cmd2 *mode_cmd)
{
	/// Functional Utility: Determines the bits per pixel (BPP) for a given AFBC framebuffer format.
	/// It handles specific AFBC formats and falls back to generic DRM format BPP calculation.
	const struct drm_format_info *info;

	info = drm_get_format_info(dev, mode_cmd);

	// Block Logic: Use a switch statement to return BPP for known AFBC formats.
	// Invariant: For unsupported formats, it falls back to a generic BPP calculation.
	switch (info->format) {
	case DRM_FORMAT_YUV420_8BIT:
		return 12;
	case DRM_FORMAT_YUV420_10BIT:
		return 15;
	case DRM_FORMAT_VUY101010:
		return 30;
	default:
		return drm_format_info_bpp(info, 0);
	}
}

static int drm_gem_afbc_min_size(struct drm_device *dev,
				 const struct drm_mode_fb_cmd2 *mode_cmd,
				 struct drm_afbc_framebuffer *afbc_fb)
{
	/// Functional Utility: Calculates the minimum required size for an AFBC framebuffer based on its dimensions,
	/// block size, and tiling properties. This ensures sufficient memory is allocated for the compressed format.
	__u32 n_blocks, w_alignment, h_alignment, hdr_alignment;
	/* remove bpp when all users properly encode cpp in drivers/gpu/drm/drm_fourcc.c */
	__u32 bpp;

	// Block Logic: Determine AFBC block dimensions based on modifier.
	switch (mode_cmd->modifier[0] & AFBC_FORMAT_MOD_BLOCK_SIZE_MASK) {
	case AFBC_FORMAT_MOD_BLOCK_SIZE_16x16:
		afbc_fb->block_width = 16;
		afbc_fb->block_height = 16;
		break;
	case AFBC_FORMAT_MOD_BLOCK_SIZE_32x8:
		afbc_fb->block_width = 32;
		afbc_fb->block_height = 8;
		break;
	/* no user exists yet - fall through */
	case AFBC_FORMAT_MOD_BLOCK_SIZE_64x4:
	case AFBC_FORMAT_MOD_BLOCK_SIZE_32x8_64x4:
	default:
		drm_dbg_kms(dev, "Invalid AFBC_FORMAT_MOD_BLOCK_SIZE: %lld.\n",
			    mode_cmd->modifier[0]
			    & AFBC_FORMAT_MOD_BLOCK_SIZE_MASK);
		return -EINVAL;
	}

	/* tiled header afbc */
	// Block Logic: Determine alignment requirements based on AFBC tiling.
	w_alignment = afbc_fb->block_width;
	h_alignment = afbc_fb->block_height;
	hdr_alignment = AFBC_HDR_ALIGN;
	if (mode_cmd->modifier[0] & AFBC_FORMAT_MOD_TILED) {
		w_alignment *= AFBC_TH_LAYOUT_ALIGNMENT;
		h_alignment *= AFBC_TH_LAYOUT_ALIGNMENT;
		hdr_alignment = AFBC_TH_BODY_START_ALIGNMENT;
	}

	afbc_fb->aligned_width = ALIGN(mode_cmd->width, w_alignment);
	afbc_fb->aligned_height = ALIGN(mode_cmd->height, h_alignment);
	afbc_fb->offset = mode_cmd->offsets[0];

	// Block Logic: Get BPP for the AFBC format.
	bpp = drm_gem_afbc_get_bpp(dev, mode_cmd);
	// Conditional Logic: Handle invalid BPP.
	if (!bpp) {
		drm_dbg_kms(dev, "Invalid AFBC bpp value: %d\n", bpp);
		return -EINVAL;
	}

	// Block Logic: Calculate total AFBC size based on number of blocks and alignment.
	n_blocks = (afbc_fb->aligned_width * afbc_fb->aligned_height)
		   / AFBC_SUPERBLOCK_PIXELS;
	afbc_fb->afbc_size = ALIGN(n_blocks * AFBC_HEADER_SIZE, hdr_alignment);
	afbc_fb->afbc_size += n_blocks * ALIGN(bpp * AFBC_SUPERBLOCK_PIXELS / 8,
					       AFBC_SUPERBLOCK_ALIGNMENT);

	return 0;
}

/**
 * drm_gem_fb_afbc_init() - Helper function for drivers using afbc to
 *			    fill and validate all the afbc-specific
 *			    struct drm_afbc_framebuffer members
 *
 * @dev: DRM device
 * @afbc_fb: afbc-specific framebuffer
 * @mode_cmd: Metadata from the userspace framebuffer creation request
 * @afbc_fb: afbc framebuffer
 *
 * This function can be used by drivers which support afbc to complete
 * the preparation of struct drm_afbc_framebuffer. It must be called after
 * allocating the said struct and calling drm_gem_fb_init_with_funcs().
 * It is caller's responsibility to put afbc_fb->base.obj objects in case
 * the call is unsuccessful.
 *
 * Returns:
 * Zero on success or a negative error value on failure.
 */
int drm_gem_fb_afbc_init(struct drm_device *dev,
			 const struct drm_mode_fb_cmd2 *mode_cmd,
			 struct drm_afbc_framebuffer *afbc_fb)
{
	/// Functional Utility: Completes the initialization of an AFBC-specific framebuffer,
	/// including validation of its size against the calculated minimum AFBC size.
	const struct drm_format_info *info;
	struct drm_gem_object **objs;
	int ret;

	objs = afbc_fb->base.obj;
	info = drm_get_format_info(dev, mode_cmd);
	if (!info)
		return -EINVAL;

	// Block Logic: Calculate the minimum required size for the AFBC framebuffer.
	ret = drm_gem_afbc_min_size(dev, mode_cmd, afbc_fb);
	// Conditional Logic: Handle errors during minimum size calculation.
	if (ret < 0)
		return ret;

	// Conditional Logic: Validate that the allocated GEM object size is sufficient for the AFBC format.
	if (objs[0]->size < afbc_fb->afbc_size)
		return -EINVAL;

	return 0;
}
EXPORT_SYMBOL_GPL(drm_gem_fb_afbc_init);
