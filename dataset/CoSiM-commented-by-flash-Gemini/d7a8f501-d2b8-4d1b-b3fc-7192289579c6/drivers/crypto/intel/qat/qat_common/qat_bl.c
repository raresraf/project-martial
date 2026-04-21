// SPDX-License-Identifier: GPL-2.0-only
/* Copyright(c) 2014 - 2022 Intel Corporation */

/**
 * @file qat_bl.c
 * @brief Buffer list translation and DMA mapping for Intel QAT hardware acceleration.
 * 
 * Functional Intent: Provides the transformation logic to convert Linux kernel 
 * scatter-gather lists (SGL) into the structured descriptor format required 
 * by QAT hardware. It orchestrates the mapping of physical memory pages into 
 * the device's DMA space, supporting complex sub-buffer offsets and auxiliary 
 * descriptor appending for authentication or status metadata.
 * 
 * Domain: Production Systems, Kernel Drivers, DMA Memory Management.
 */

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
 * qat_bl_free_bufl - Performs surgical cleanup of DMA mappings and descriptor memory.
 * 
 * Logic: Iterates through source and destination descriptor chains, unmapping 
 * every individual scatter-gather entry from the physical device window before 
 * releasing the host memory containers.
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

	// Optimization: Determine DMA intent based on buffer configuration.
	bl_dma_dir = blp != blpout ? DMA_TO_DEVICE : DMA_BIDIRECTIONAL;

	// Block Logic: Source chain unmapping.
	for (i = 0; i < bl->num_bufs; i++)
		dma_unmap_single(dev, bl->buffers[i].addr,
				 bl->buffers[i].len, bl_dma_dir);

	dma_unmap_single(dev, blp, sz, DMA_TO_DEVICE);

	if (!buf->sgl_src_valid)
		kfree(bl);

	// Block Logic: Destination chain unmapping (Out-of-place).
	if (blp != blpout) {
		for (i = 0; i < blout->num_mapped_bufs; i++) {
			// Logic: Uses FROM_DEVICE as this chain receives hardware output.
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
 * __qat_bl_sgl_to_bufl - Core algorithmic engine for SGL to QAT-Bufl mapping.
 * 
 * Algorithm: Linear SGL-to-Bufl translation.
 * 1. Computes required descriptor memory and allocates (with pre-allocated static fallback).
 * 2. Iteratively maps kernel virtual addresses to physical DMA addresses.
 * 3. Implements byte-skipping to support sub-buffer cryptographical operations.
 * 4. Manages out-of-place destination chains with specialized "FROM_DEVICE" mapping.
 * 
 * Invariant: Maintains strict transactional integrity; any mapping failure 
 * triggers an immediate full rollback of all allocated DMA resources.
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

	// Optimization: Use pre-allocated static header for low-complexity SGLs to avoid slab pressure.
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

	// Block Logic: Source traversal and mapping.
	for_each_sg(sgl, sg, n, i) {
		int y = sg_nctr;

		if (!sg->length)
			continue;

		// Logic: Handle sub-buffer Cryptographic start offsets.
		if (left >= sg->length) {
			left -= sg->length;
			continue;
		}
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
	// Synchronization: Atomic map of the descriptor structure itself for device consumption.
	blp = dma_map_single(dev, bufl, sz, DMA_TO_DEVICE);
	if (unlikely(dma_mapping_error(dev, blp)))
		goto err_in;
	buf->bl = bufl;
	buf->blp = blp;
	buf->sz = sz;

	/* Block Logic: Out-of-place destination handling. */
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

		for_each_sg(sglout, sg, n_sglout, i) {
			int y = sg_nctr;

			if (!sg->length)
				continue;

			if (left >= sg->length) {
				left -= sg->length;
				continue;
			}
			// Logic: Destination mapped as FROM_DEVICE for data recovery from acceleration.
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
		// Logic: Append auxiliary output buffer (e.g. status codes or auth tags).
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
		/* In-place Logic: Share source and destination DMA addresses. */
		buf->bloutp = buf->blp;
		buf->sz_out = 0;
	}
	return 0;

err_out:
	// Rollback: Teardown destination chain.
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

err_in:
	// Rollback: Teardown source chain.
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

	dev_err(dev, "Failed to map buf for dma\n");
	return -ENOMEM;
}

/**
 * qat_bl_sgl_to_bufl - High-level interface for creating device-consumable DMA descriptors.
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
