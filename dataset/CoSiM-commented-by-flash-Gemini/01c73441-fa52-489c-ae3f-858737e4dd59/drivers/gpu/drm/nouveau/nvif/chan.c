/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 */

/**
 * @file chan.c
 * @brief Implements GPU command channel management for the Nouveau driver (nvif).
 *
 * This file provides the core logic for managing GPU command channels, specifically
 * focusing on GPFIFO (Graphics Processing Unit FIFO) operations. It defines functions
 * for pushing commands to the GPU, handling synchronization, and managing the
 * associated push buffers and GPFIFO entries.
 *
 * Key functionalities include:
 * - Managing the GPU's command submission queue (GPFIFO).
 * - Handling the push buffer where commands are prepared before submission.
 * - Implementing kick and wait mechanisms for command processing.
 * - Constructor for initializing channel-related data structures.
 */
#include <nvif/chan.h>

/**
 * @brief Kicks the GPU to process commands in the push buffer.
 * @param push Pointer to the nvif_push structure.
 *
 * This function signals the GPU that new commands are available in the push buffer
 * for processing. It updates the GPFIFO entries and calls the channel-specific
 * kick function to initiate command execution on the GPU.
 * If a `post` operation is defined for the GPFIFO, it will execute it.
 *
 * Preconditions:
 * - `push->bgn` and `push->cur` are valid pointers within the push buffer.
 * Invariant:
 * - The `put` pointer in the GPFIFO is updated, indicating new commands,
 *   and the GPU is notified to process them.
 */
static void
nvif_chan_gpfifo_push_kick(struct nvif_push *push)
{
	struct nvif_chan *chan = container_of(push, typeof(*chan), push);
	u32 put = push->bgn - (u32 *)chan->push.mem.object.map.ptr;
	u32 cnt;

	// Block Logic: Handle potential post-processing if defined for the GPFIFO.
	// This ensures any necessary cleanup or finalization occurs after command submission.
	if (chan->func->gpfifo.post) {
		if (push->end - push->cur < chan->func->gpfifo.post_size)
			push->end = push->cur + chan->func->gpfifo.post_size;

		// Inline: Warn on failure of the GPFIFO post operation.
		WARN_ON(nvif_chan_gpfifo_post(chan));
	}

	cnt = push->cur - push->bgn;

	// Block Logic: Update GPFIFO entry to indicate new commands and kick the GPU.
	// The `push` function adds the push buffer entry to the GPFIFO.
	// The `kick` function signals the hardware that new entries are present.
	chan->func->gpfifo.push(chan, true, chan->push.addr + (put << 2), cnt << 2, false);
	chan->func->gpfifo.kick(chan);
}

/**
 * @brief Waits for GPFIFO entries to be processed.
 * @param push Pointer to the nvif_push structure.
 * @param push_nr The number of push buffer entries to wait for.
 * @return 0 on success, or a negative error code on timeout.
 *
 * This function is a wrapper around `nvif_chan_gpfifo_wait` to integrate with
 * the `nvif_push` mechanism, allowing the system to block until a certain
 * number of push buffer entries have been consumed by the GPU.
 *
 * Preconditions:
 * - `chan` must be a valid nvif_chan structure.
 * Invariant:
 * - The function returns when the specified number of push buffer entries
 *   have been processed or a timeout occurs.
 */
static int
nvif_chan_gpfifo_push_wait(struct nvif_push *push, u32 push_nr)
{
	struct nvif_chan *chan = container_of(push, typeof(*chan), push);

	return nvif_chan_gpfifo_wait(chan, 1, push_nr);
}

/**
 * @brief Posts a GPFIFO entry.
 * @param chan Pointer to the nvif_chan structure.
 * @return 0 on success, or a negative error code on failure.
 *
 * This function adds a new entry to the GPU's GPFIFO, signaling that a push buffer
 * containing commands is ready to be consumed by the GPU.
 *
 * Preconditions:
 * - `chan->func->gpfifo.post` must be a valid function pointer.
 * Invariant:
 * - A GPFIFO entry is added, linking to the current push buffer content.
 */
int
nvif_chan_gpfifo_post(struct nvif_chan *chan)
{
	const u32 *map = chan->push.mem.object.map.ptr;
	const u32 pbptr = (chan->push.cur - map) + chan->func->gpfifo.post_size;
	const u32 gpptr = (chan->gpfifo.cur + 1) & chan->gpfifo.max;

	return chan->func->gpfifo.post(chan, gpptr, pbptr);
}

/**
 * @brief Pushes a command buffer address and size to the GPFIFO.
 * @param chan Pointer to the nvif_chan structure.
 * @param addr The GPU physical address of the command buffer.
 * @param size The size of the command buffer in bytes.
 * @param no_prefetch Boolean indicating whether prefetching should be disabled.
 *
 * This function adds an entry to the GPFIFO that points to a command buffer
 * located at `addr` with a specified `size`. The `no_prefetch` flag can
 * influence GPU's prefetching behavior.
 *
 * Preconditions:
 * - `chan->func->gpfifo.push` must be a valid function pointer.
 * Invariant:
 * - A new entry is written into the GPFIFO, directing the GPU to fetch
 *   and execute the specified command buffer.
 */
void
nvif_chan_gpfifo_push(struct nvif_chan *chan, u64 addr, u32 size, bool no_prefetch)
{
	chan->func->gpfifo.push(chan, false, addr, size, no_prefetch);
}

/**
 * @brief Waits for space to become available in the GPFIFO and push buffer.
 * @param chan Pointer to the nvif_chan structure.
 * @param gpfifo_nr The number of GPFIFO entries required.
 * @param push_nr The number of push buffer entries required.
 * @return 0 on success, or -ETIMEDOUT if the wait times out.
 *
 * This function ensures that there is enough space in both the main push buffer
 * and the GPFIFO for new commands. It accounts for space needed by potential
 * `post` operations.
 *
 * Preconditions:
 * - `chan` must be a valid nvif_chan structure with initialized push and gpfifo members.
 * Invariant:
 * - Upon successful return, sufficient space is available in both the push buffer
 *   and the GPFIFO.
 */
int
nvif_chan_gpfifo_wait(struct nvif_chan *chan, u32 gpfifo_nr, u32 push_nr)
{
	struct nvif_push *push = &chan->push;
	int ret = 0, time = 1000000;

	// Block Logic: Account for additional space required by GPFIFO post operations.
	if (gpfifo_nr) {
		/* Account for pushbuf space needed by nvif_chan_gpfifo_post(),
		 * if used after pushing userspace GPFIFO entries.
		 */
		if (chan->func->gpfifo.post)
			push_nr += chan->func->gpfifo.post_size;
	}

	/* Account for the GPFIFO entry needed to submit pushbuf. */
	if (push_nr)
		gpfifo_nr++;

	// Block Logic: Wait for sufficient space in the main push buffer.
	// This loop ensures that the push buffer has enough room for new commands,
	// potentially kicking the GPU if the buffer is full.
	if (push->cur + push_nr > push->end) {
		ret = nvif_chan_dma_wait(chan, push_nr);
		if (ret)
			return ret;
	}

	// Block Logic: Wait for sufficient space in the GPFIFO.
	// This loop continuously checks the available GPFIFO space and delays
	// if necessary, until enough space is free or a timeout occurs.
	while (chan->gpfifo.free < gpfifo_nr) {
		chan->gpfifo.free = chan->func->gpfifo.read_get(chan) - chan->gpfifo.cur - 1;
		if (chan->gpfifo.free < 0)
			chan->gpfifo.free += chan->gpfifo.max + 1;

		if (chan->gpfifo.free < gpfifo_nr) {
			if (!time--)
				return -ETIMEDOUT;
			udelay(1);
		}
	}

	return 0;
}

/**
 * @brief Constructor for initializing an nvif_chan GPFIFO channel.
 * @param func Pointer to channel-specific function table.
 * @param userd Pointer to the user-space doorbell region.
 * @param gpfifo Pointer to the GPFIFO memory.
 * @param gpfifo_size Size of the GPFIFO memory in bytes.
 * @param push Pointer to the push buffer memory.
 * @param push_addr GPU physical address of the push buffer.
 * @param push_size Size of the push buffer in bytes.
 * @param chan Pointer to the nvif_chan structure to initialize.
 *
 * This function initializes the `nvif_chan` structure with the provided
 * function pointers, memory mappings, and buffer sizes for GPFIFO-based
 * command submission. It sets up the push buffer and GPFIFO management.
 *
 * Preconditions:
 * - All input pointers (`func`, `userd`, `gpfifo`, `push`) must be valid.
 * - `gpfifo_size` and `push_size` must be non-zero.
 * Invariant:
 * - The `nvif_chan` structure is fully initialized for GPFIFO command submission.
 */
void
nvif_chan_gpfifo_ctor(const struct nvif_chan_func *func, void *userd, void *gpfifo, u32 gpfifo_size,
		      void *push, u64 push_addr, u32 push_size, struct nvif_chan *chan)
{
	chan->func = func;

	chan->userd.map.ptr = userd;

	chan->gpfifo.map.ptr = gpfifo;
	chan->gpfifo.max = (gpfifo_size >> 3) - 1;
	chan->gpfifo.free = chan->gpfifo.max;

	chan->push.mem.object.map.ptr = push;
	chan->push.wait = nvif_chan_gpfifo_push_wait;
	chan->push.kick = nvif_chan_gpfifo_push_kick;
	chan->push.addr = push_addr;
	chan->push.hw.max = push_size >> 2;
	chan->push.bgn = chan->push.cur = chan->push.end = push;
}

/**
 * @brief Waits for space in the DMA push buffer.
 * @param chan Pointer to the nvif_chan structure.
 * @param nr The number of required push buffer entries.
 * @return 0 on success, or -ETIMEDOUT if the wait times out.
 *
 * This function ensures that there is enough free space in the DMA push buffer
 * for `nr` new entries. It may involve kicking the GPU if the buffer is full
 * and then waiting for the GPU to consume some entries.
 *
 * Preconditions:
 * - `chan` must be a valid nvif_chan structure with initialized push buffer.
 * Invariant:
 * - Upon successful return, `nr` entries worth of space is available in the
 *   DMA push buffer.
 */
int
nvif_chan_dma_wait(struct nvif_chan *chan, u32 nr)
{
	struct nvif_push *push = &chan->push;
	u32 cur = push->cur - (u32 *)push->mem.object.map.ptr;
	u32 free, time = 1000000;

	// Inline: Account for additional post-processing size in push buffer if applicable.
	nr += chan->func->gpfifo.post_size;

	// Block Logic: Continuously check for free space in the push buffer.
	// If the current `put` pointer approaches the end of the buffer,
	// the GPU is kicked, and the CPU waits for the GPU to consume entries.
	do {
		u32 get = chan->func->push.read_get(chan);

		if (get <= cur) {
			free = push->hw.max - cur;
			if (free >= nr)
				break;

			// Inline: Kick the GPU to process commands and free up space.
			PUSH_KICK(push);

			// Block Logic: Wait for the GPU to make progress and free up space.
			// This inner loop waits for the `get` pointer to advance,
			// indicating that the GPU has consumed commands.
			while (get == 0) {
				get = chan->func->push.read_get(chan);
				if (get == 0) {
					if (!time--)
						return -ETIMEDOUT;
					udelay(1);
				}
			}

			cur = 0;
		}

		free = get - cur - 1;

		if (free < nr) {
			if (!time--)
				return -ETIMEDOUT;
			udelay(1);
		}
	} while (free < nr);

	// Block Logic: Update push buffer pointers to reflect newly available space.
	// Adjusts `bgn`, `cur`, and `end` pointers for the push buffer.
	push->bgn = (u32 *)push->mem.object.map.ptr + cur;
	push->cur = push->bgn;
	push->end = push->bgn + free - chan->func->gpfifo.post_size;
	return 0;
}