/**
 * @file chan.c
 * @brief NVIF channel management
 *
 * This file implements the channel management functions for the NVIF (Nouveau
 * Interface) layer. A channel is a command submission context that includes a
 * push buffer (for writing commands) and a GPFIFO (for queuing command buffers
 * to the hardware).
 *
 * This implementation handles the low-level details of command submission,
 * including:
 * - Kicking off command buffer execution.
 * - Waiting for space in the push buffer and GPFIFO.
 * - Posting GPFIFO entries to the hardware.
 */
/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 */
#include <nvif/chan.h>

static void
nvif_chan_gpfifo_push_kick(struct nvif_push *push)
{
	struct nvif_chan *chan = container_of(push, typeof(*chan), push);
	u32 put = push->bgn - (u32 *)chan->push.mem.object.map.ptr;
	u32 cnt;

	if (chan->func->gpfifo.post) {
		if (push->end - push->cur < chan->func->gpfifo.post_size)
			push->end = push->cur + chan->func->gpfifo.post_size;

		WARN_ON(nvif_chan_gpfifo_post(chan));
	}

	cnt = push->cur - push->bgn;

	chan->func->gpfifo.push(chan, true, chan->push.addr + (put << 2), cnt << 2, false);
	chan->func->gpfifo.kick(chan);
}

static int
nvif_chan_gpfifo_push_wait(struct nvif_push *push, u32 push_nr)
{
	struct nvif_chan *chan = container_of(push, typeof(*chan), push);

	return nvif_chan_gpfifo_wait(chan, 1, push_nr);
}

int
nvif_chan_gpfifo_post(struct nvif_chan *chan)
{
	const u32 *map = chan->push.mem.object.map.ptr;
	const u32 pbptr = (chan->push.cur - map) + chan->func->gpfifo.post_size;
	const u32 gpptr = (chan->gpfifo.cur + 1) & chan->gpfifo.max;

	return chan->func->gpfifo.post(chan, gpptr, pbptr);
}

/**
 * nvif_chan_gpfifo_push() - Push a command buffer to the GPFIFO.
 * @chan: The channel to push to.
 * @addr: The GPU virtual address of the command buffer.
 * @size: The size of the command buffer in bytes.
 * @no_prefetch: Disable prefetching of the command buffer.
 *
 * This function queues a command buffer for execution by writing its address
 * and size to the GPFIFO.
 */
void
nvif_chan_gpfifo_push(struct nvif_chan *chan, u64 addr, u32 size, bool no_prefetch)
{
	chan->func->gpfifo.push(chan, false, addr, size, no_prefetch);
}

/**
 * nvif_chan_gpfifo_wait() - Wait for space in the GPFIFO and push buffer.
 * @chan: The channel to wait on.
 * @gpfifo_nr: The number of GPFIFO entries required.
 * @push_nr: The number of dwords required in the push buffer.
 *
 * This function blocks until there is enough space available in both the
 * GPFIFO and the main push buffer to submit new commands.
 *
 * Returns: 0 on success, or -ETIMEDOUT if the wait times out.
 */
int
nvif_chan_gpfifo_wait(struct nvif_chan *chan, u32 gpfifo_nr, u32 push_nr)
{
	struct nvif_push *push = &chan->push;
	int ret = 0, time = 1000000;

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

	/* Wait for space in main push buffer. */
	if (push->cur + push_nr > push->end) {
		ret = nvif_chan_dma_wait(chan, push_nr);
		if (ret)
			return ret;
	}

	/* Wait for GPFIFO space. */
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
 * nvif_chan_gpfifo_ctor() - Constructor for a GPFIFO channel.
 * @func: A pointer to the channel function table.
 * @userd: Pointer to the user-mapped data area.
 * @gpfifo: Pointer to the GPFIFO buffer.
 * @gpfifo_size: The size of the GPFIFO buffer.
 * @push: Pointer to the push buffer.
 * @push_addr: The GPU virtual address of the push buffer.
 * @push_size: The size of the push buffer.
 * @chan: The channel structure to be initialized.
 *
 * This function initializes a `nvif_chan` structure with the memory and
 * function pointers necessary for it to operate as a GPFIFO channel.
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

int
nvif_chan_dma_wait(struct nvif_chan *chan, u32 nr)
{
	struct nvif_push *push = &chan->push;
	u32 cur = push->cur - (u32 *)push->mem.object.map.ptr;
	u32 free, time = 1000000;

	nr += chan->func->gpfifo.post_size;

	do {
		u32 get = chan->func->push.read_get(chan);

		if (get <= cur) {
			free = push->hw.max - cur;
			if (free >= nr)
				break;

			PUSH_KICK(push);

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

	push->bgn = (u32 *)push->mem.object.map.ptr + cur;
	push->cur = push->bgn;
	push->end = push->bgn + free - chan->func->gpfifo.post_size;
	return 0;
}