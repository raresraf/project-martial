// SPDX-License-Identifier: GPL-2.0-only
/**
 * @file qat_bl.c
 * @brief Provides Scatter-Gather List to Buffer List conversion for the Intel QAT driver.
 *
 * @details This file contains helper functions for converting the Linux kernel's
 * `scatterlist` (SGL) format into a QAT-specific "Buffer List" (BL) format. The QAT
 * hardware requires a physically contiguous list of buffer descriptors to perform DMA
 * on data that may be fragmented in system memory. This file bridges the gap between
 * the kernel's representation of scattered memory (SGLs) and the hardware's requirements.
 *
 * Algorithm:
 * The main function, `__qat_bl_sgl_to_bufl`, iterates through each entry in a given
 * `scatterlist`. For each entry, it performs the following:
 * 1.  It maps the virtual memory page of the SGL entry into a DMA-able physical address
 *     using `dma_map_single`. This is a critical step for allowing the hardware device
 *     to access system memory.
 * 2.  The resulting physical address and length are stored in a flat array within a
 *     `qat_alg_buf_list` structure.
 * 3.  Finally, the `qat_alg_buf_list` structure itself is also DMA-mapped, because the
 *     QAT hardware needs to read this list of pointers to find the actual data.
 * The `qat_bl_free_bufl` function performs the reverse operation, unmapping all DMA'd
 * regions and freeing allocated memory.
 *
 * Performance Optimization (C/Rust):
 * - To reduce overhead for small requests, the conversion functions use a pre-allocated
 *   `qat_sgl_hdr` within the `qat_request_buffs` struct. This avoids a potentially costly
 *   `kzalloc` call if the number of scatter-gather entries is below a certain threshold
 *   (`QAT_MAX_BUFF_DESC`). This is a common optimization in high-performance kernel drivers.
 */
/* Copyright(c) 2014 - 2022 Intel Corporation */
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/pci.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/types.h>
#include "adf_accel_devices.h"
#include "qat_bl.h"
#include "qat_crypto.h"

/**
 * @brief Frees a buffer list and unmaps its associated DMA regions.
 * @param accel_dev Pointer to the accelerator device.
 * @param buf       Pointer to the qat_request_buffs structure containing the
 *                  buffer lists to be freed.
 *
 * This function iterates through all the buffers in the provided buffer lists
 * (`bl` and `blout`), unmaps them from the device's DMA address space, and
 * then frees the memory allocated for the lists themselves.
 */
void qat_bl_free_bufl(struct adf_accel_dev *accel_dev,
		      struct qat_request_buffs *buf)
{
	struct device *dev = &GET_DEV(accel_dev);
	struct qat_alg_buf_list *bl = buf->bl;
	struct qat_alg_buf_list *blout = buf->blout;
	dma_addr_t blp = buf->blp;
	dma_addr_t blpout = buf->bloutp;
	size_t sz = buf->sz;
	size_t sz_out = buf->sz_out;
	int bl_dma_dir;
	int i;

	// Determine DMA direction for unmapping. If source and destination are
	// different, the source was mapped as TO_DEVICE only.
	bl_dma_dir = blp != blpout ? DMA_TO_DEVICE : DMA_BIDIRECTIONAL;

	// Invariant: Unmap every buffer in the source buffer list.
	for (i = 0; i < bl->num_bufs; i++)
		dma_unmap_single(dev, bl->buffers[i].addr,
				 bl->buffers[i].len, bl_dma_dir);

	// Unmap the source buffer list itself.
	dma_unmap_single(dev, blp, sz, DMA_TO_DEVICE);

	// Free the source buffer list if it was dynamically allocated.
	if (!buf->sgl_src_valid)
		kfree(bl);

	// Block Logic: If this was an out-of-place operation, also free the destination buffer list.
	if (blp != blpout) {
		for (i = 0; i < blout->num_mapped_bufs; i++) {
			dma_unmap_single(dev, blout->buffers[i].addr,
					 blout->buffers[i].len,
					 DMA_FROM_DEVICE);
		}
		dma_unmap_single(dev, blpout, sz_out, DMA_TO_DEVICE);

		if (!buf->sgl_dst_valid)
			kfree(blout);
	}
}

/**
 * @brief Core worker function to convert scatterlists to a QAT buffer list.
 * @param accel_dev Pointer to the accelerator device.
 * @param sgl       Source scatterlist.
 * @param sglout    Destination scatterlist (can be the same as `sgl` for in-place).
 * @param buf       The qat_request_buffs struct to populate.
 * @param extra_dst_buff An optional, pre-mapped extra destination buffer address.
 * @param sz_extra_dst_buff Size of the optional extra destination buffer.
 * @param sskip     Bytes to skip at the beginning of the source SGL.
 * @param dskip     Bytes to skip at the beginning of the destination SGL.
 * @param flags     Memory allocation flags (e.g., GFP_KERNEL).
 * @return 0 on success, negative error code on failure.
 */
static int __qat_bl_sgl_to_bufl(struct adf_accel_dev *accel_dev,
				struct scatterlist *sgl,
				struct scatterlist *sglout,
				struct qat_request_buffs *buf,
				dma_addr_t extra_dst_buff,
				size_t sz_extra_dst_buff,
				unsigned int sskip,
				unsigned int dskip,
				gfp_t flags)
{
	struct device *dev = &GET_DEV(accel_dev);
	int i, sg_nctr = 0;
	int n = sg_nents(sgl);
	struct qat_alg_buf_list *bufl;
	struct qat_alg_buf_list *buflout = NULL;
	dma_addr_t blp = DMA_MAPPING_ERROR;
	dma_addr_t bloutp = DMA_MAPPING_ERROR;
	struct scatterlist *sg;
	size_t sz_out, sz = struct_size(bufl, buffers, n);
	int node = dev_to_node(&GET_DEV(accel_dev));
	unsigned int left;
	int bufl_dma_dir;

	if (unlikely(!n))
		return -EINVAL;

	buf->sgl_src_valid = false;
	buf->sgl_dst_valid = false;

	// Optimization: For small SGLs, use a pre-allocated buffer to avoid kzalloc.
	if (n > QAT_MAX_BUFF_DESC) {
		bufl = kzalloc_node(sz, flags, node);
		if (unlikely(!bufl))
			return -ENOMEM;
	} else {
		bufl = container_of(&buf->sgl_src.sgl_hdr,
				    struct qat_alg_buf_list, hdr);
		memset(bufl, 0, sizeof(struct qat_alg_buf_list));
		buf->sgl_src_valid = true;
	}

	bufl_dma_dir = sgl != sglout ? DMA_TO_DEVICE : DMA_BIDIRECTIONAL;

	for (i = 0; i < n; i++)
		bufl->buffers[i].addr = DMA_MAPPING_ERROR;

	left = sskip;

	// Invariant: Iterate through all segments of the source scatterlist.
	for_each_sg(sgl, sg, n, i) {
		int y = sg_nctr;

		if (!sg->length)
			continue;

		if (left >= sg->length) {
			left -= sg->length;
			continue;
		}
		// Map the SGL segment for DMA and store the physical address.
		bufl->buffers[y].addr = dma_map_single(dev, sg_virt(sg) + left,
						       sg->length - left,
						       bufl_dma_dir);
		bufl->buffers[y].len = sg->length;
		if (unlikely(dma_mapping_error(dev, bufl->buffers[y].addr)))
			goto err_in;
		sg_nctr++;
		if (left) {
			bufl->buffers[y].len -= left;
			left = 0;
		}
	}
	bufl->num_bufs = sg_nctr;
	// Map the buffer list itself so the hardware can read it.
	blp = dma_map_single(dev, bufl, sz, DMA_TO_DEVICE);
	if (unlikely(dma_mapping_error(dev, blp)))
		goto err_in;
	buf->bl = bufl;
	buf->blp = blp;
	buf->sz = sz;

	// Block Logic: Handle out-of-place operations where src and dst SGLs are different.
	if (sgl != sglout) {
		struct qat_alg_buf *buffers;
		int extra_buff = extra_dst_buff ? 1 : 0;
		int n_sglout = sg_nents(sglout);

		n = n_sglout + extra_buff;
		sz_out = struct_size(buflout, buffers, n);
		left = dskip;

		sg_nctr = 0;

		if (n > QAT_MAX_BUFF_DESC) {
			buflout = kzalloc_node(sz_out, flags, node);
			if (unlikely(!buflout))
				goto err_in;
		} else {
			buflout = container_of(&buf->sgl_dst.sgl_hdr,
					       struct qat_alg_buf_list, hdr);
			memset(buflout, 0, sizeof(struct qat_alg_buf_list));
			buf->sgl_dst_valid = true;
		}

		buffers = buflout->buffers;
		for (i = 0; i < n; i++)
			buffers[i].addr = DMA_MAPPING_ERROR;

		// Invariant: Iterate and map all segments of the destination scatterlist.
		for_each_sg(sglout, sg, n_sglout, i) {
			int y = sg_nctr;

			if (!sg->length)
				continue;

			if (left >= sg->length) {
				left -= sg->length;
				continue;
			}
			buffers[y].addr = dma_map_single(dev, sg_virt(sg) + left,
							 sg->length - left,
							 DMA_FROM_DEVICE);
			if (unlikely(dma_mapping_error(dev, buffers[y].addr)))
				goto err_out;
			buffers[y].len = sg->length;
			sg_nctr++;
			if (left) {
				buffers[y].len -= left;
				left = 0;
			}
		}
		if (extra_buff) {
			buffers[sg_nctr].addr = extra_dst_buff;
			buffers[sg_nctr].len = sz_extra_dst_buff;
		}

		buflout->num_bufs = sg_nctr;
		buflout->num_bufs += extra_buff;
		buflout->num_mapped_bufs = sg_nctr;
		bloutp = dma_map_single(dev, buflout, sz_out, DMA_TO_DEVICE);
		if (unlikely(dma_mapping_error(dev, bloutp)))
			goto err_out;
		buf->blout = buflout;
		buf->bloutp = bloutp;
		buf->sz_out = sz_out;
	} else {
		// For in-place, the destination buffer list points to the source.
		buf->bloutp = buf->blp;
		buf->sz_out = 0;
	}
	return 0;

err_out: // Error handling for destination SGL mapping failure.
	if (!dma_mapping_error(dev, bloutp))
		dma_unmap_single(dev, bloutp, sz_out, DMA_TO_DEVICE);

	n = sg_nents(sglout);
	for (i = 0; i < n; i++) {
		if (buflout->buffers[i].addr == extra_dst_buff)
			break;
		if (!dma_mapping_error(dev, buflout->buffers[i].addr))
			dma_unmap_single(dev, buflout->buffers[i].addr,
					 buflout->buffers[i].len,
					 DMA_FROM_DEVICE);
	}

	if (!buf->sgl_dst_valid)
		kfree(buflout);

err_in: // Error handling for source SGL mapping failure.
	if (!dma_mapping_error(dev, blp))
		dma_unmap_single(dev, blp, sz, DMA_TO_DEVICE);

	n = sg_nents(sgl);
	for (i = 0; i < n; i++)
		if (!dma_mapping_error(dev, bufl->buffers[i].addr))
			dma_unmap_single(dev, bufl->buffers[i].addr,
					 bufl->buffers[i].len,
					 bufl_dma_dir);

	if (!buf->sgl_src_valid)
		kfree(bufl);

	dev_err(dev, "Failed to map buf for dma
");
	return -ENOMEM;
}

/**
 * @brief Public wrapper function for SGL to buffer list conversion.
 * @param accel_dev Pointer to the accelerator device.
 * @param sgl       Source scatterlist.
 * @param sglout    Destination scatterlist.
 * @param buf       The qat_request_buffs struct to populate.
 * @param params    Optional parameters for skipping bytes or adding extra buffers.
 * @param flags     Memory allocation flags.
 * @return 0 on success, negative error code on failure.
 *
 * This function provides a cleaner API by unpacking the optional `params`
 * struct before calling the main `__qat_bl_sgl_to_bufl` worker function.
 */
int qat_bl_sgl_to_bufl(struct adf_accel_dev *accel_dev,
		       struct scatterlist *sgl,
		       struct scatterlist *sglout,
		       struct qat_request_buffs *buf,
		       struct qat_sgl_to_bufl_params *params,
		       gfp_t flags)
{
	dma_addr_t extra_dst_buff = 0;
	size_t sz_extra_dst_buff = 0;
	unsigned int sskip = 0;
	unsigned int dskip = 0;

	if (params) {
		extra_dst_buff = params->extra_dst_buff;
		sz_extra_dst_buff = params->sz_extra_dst_buff;
		sskip = params->sskip;
		dskip = params->dskip;
	}

	return __qat_bl_sgl_to_bufl(accel_dev, sgl, sglout, buf,
				    extra_dst_buff, sz_extra_dst_buff,
				    sskip, dskip, flags);
}
