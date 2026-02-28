/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file blk-integrity.h
 * @brief Block Layer Data Integrity Support.
 *
 * This header file defines the structures, flags, and function prototypes for
 * the Linux block layer's data integrity framework. This framework provides
 * end-to-end data protection, guarding against silent data corruption between
 * the OS and the storage device, often implementing standards like T10 DIF/DIX.
 */
#ifndef _LINUX_BLK_INTEGRITY_H
#define _LINUX_BLK_INTEGRITY_H

#include <linux/blk-mq.h>
#include <linux/bio-integrity.h>

struct request;

/**
 * enum blk_integrity_flags - Flags to control integrity processing behavior.
 * @BLK_INTEGRITY_NOVERIFY: Skip integrity verification for a read operation.
 * @BLK_INTEGRITY_NOGENERATE: Skip integrity metadata generation for a write.
 * @BLK_INTEGRITY_DEVICE_CAPABLE: Indicates the device itself can perform
 *                                integrity checks.
 * @BLK_INTEGRITY_REF_TAG: The integrity metadata includes a reference tag,
 *                         typically the starting LBA of the I/O.
 * @BLK_INTEGRITY_STACKED: Indicates that this integrity profile is stacked on
 *                         top of another device that may also have integrity.
 */
enum blk_integrity_flags {
	BLK_INTEGRITY_NOVERIFY		= 1 << 0,
	BLK_INTEGRITY_NOGENERATE	= 1 << 1,
	BLK_INTEGRITY_DEVICE_CAPABLE	= 1 << 2,
	BLK_INTEGRITY_REF_TAG		= 1 << 3,
	BLK_INTEGRITY_STACKED		= 1 << 4,
};

const char *blk_integrity_profile_name(struct blk_integrity *bi);
bool queue_limits_stack_integrity(struct queue_limits *t,
		struct queue_limits *b);
static inline bool queue_limits_stack_integrity_bdev(struct queue_limits *t,
		struct block_device *bdev)
{
	return queue_limits_stack_integrity(t, &bdev->bd_disk->queue->limits);
}

#ifdef CONFIG_BLK_DEV_INTEGRITY
/**
 * blk_rq_map_integrity_sg - Map integrity metadata for a request to a scatterlist.
 * @rq: The block I/O request.
 * @sg: The scatterlist to map the integrity data into.
 * Functional Utility: Prepares the integrity metadata for DMA by mapping its
 * memory layout into a scatter-gather list, which hardware can process.
 */
int blk_rq_map_integrity_sg(struct request *, struct scatterlist *);

/**
 * blk_rq_count_integrity_sg - Count scatterlist segments for integrity metadata.
 * @q: The request queue for the device.
 * @bio: The bio containing the data.
 * Functional Utility: Calculates how many scatter-gather segments are needed
 * to represent the integrity metadata for a given bio. This is used for
 * allocating resources before mapping.
 */
int blk_rq_count_integrity_sg(struct request_queue *, struct bio *);

/**
 * blk_rq_integrity_map_user - Map integrity metadata from a user-space buffer.
 * @rq: The block I/O request.
 * @ubuf: Pointer to the user-space buffer containing the metadata.
 * @bytes: The size of the user-space buffer.
 * Functional Utility: Maps integrity metadata provided by a user-space
 * application into the kernel for an I/O operation.
 */
int blk_rq_integrity_map_user(struct request *rq, void __user *ubuf,
			      ssize_t bytes);
int blk_get_meta_cap(struct block_device *bdev, unsigned int cmd,
		     struct logical_block_metadata_cap __user *argp);

/**
 * blk_integrity_queue_supports_integrity - Check if a queue has integrity enabled.
 * @q: The request queue to check.
 * Functional Utility: Returns true if the device associated with the queue has
 * an active data integrity profile.
 */
static inline bool
blk_integrity_queue_supports_integrity(struct request_queue *q)
{
	return q->limits.integrity.metadata_size;
}

/**
 * blk_get_integrity - Get the integrity profile for a disk.
 * @disk: The generic disk object.
 * Functional Utility: Retrieves the integrity profile (`struct blk_integrity`)
 * associated with a disk, which contains parameters like metadata size and
 * interval. Returns NULL if integrity is not configured.
 */
static inline struct blk_integrity *blk_get_integrity(struct gendisk *disk)
{
	if (!blk_integrity_queue_supports_integrity(disk->queue))
		return NULL;
	return &disk->queue->limits.integrity;
}

/**
 * bdev_get_integrity - Get the integrity profile for a block device.
 * @bdev: The block device.
 * Functional Utility: A convenience wrapper around blk_get_integrity to get
 * the integrity profile directly from a block device object.
 */
static inline struct blk_integrity *
bdev_get_integrity(struct block_device *bdev)
{
	return blk_get_integrity(bdev->bd_disk);
}

static inline unsigned short
queue_max_integrity_segments(const struct request_queue *q)
{
	return q->limits.max_integrity_segments;
}

/**
 * bio_integrity_intervals - Calculate the number of integrity intervals for a bio.
 * @bi: The blk_integrity profile for the device.
 * @sectors: The size of the bio in 512-byte sectors.
 * Functional Utility: Converts the bio's size from standard 512-byte sectors
 * into the number of device-specific integrity intervals. This is a critical
 * unit conversion, as a device's data integrity interval can be different
 * from the standard sector size.
 */
static inline unsigned int bio_integrity_intervals(struct blk_integrity *bi,
						   unsigned int sectors)
{
	return sectors >> (bi->interval_exp - 9);
}

/**
 * bio_integrity_bytes - Calculate the total byte size of integrity metadata.
 * @bi: The blk_integrity profile for the device.
 * @sectors: The size of the bio in 512-byte sectors.
 * Functional Utility: Calculates the total number of bytes required to store
 * the integrity metadata for a given amount of data.
 */
static inline unsigned int bio_integrity_bytes(struct blk_integrity *bi,
					       unsigned int sectors)
{
	return bio_integrity_intervals(bi, sectors) * bi->metadata_size;
}

/**
 * blk_integrity_rq - Check if a request has integrity data.
 * @rq: The block I/O request.
 * Functional Utility: Returns true if the REQ_INTEGRITY flag is set on the
 * request, indicating it carries integrity metadata.
 */
static inline bool blk_integrity_rq(struct request *rq)
{
	return rq->cmd_flags & REQ_INTEGRITY;
}

/*
 * Return the current bvec that contains the integrity data. bip_iter may be
 * advanced to iterate over the integrity data.
 */
static inline struct bio_vec rq_integrity_vec(struct request *rq)
{
	return mp_bvec_iter_bvec(rq->bio->bi_integrity->bip_vec,
				 rq->bio->bi_integrity->bip_iter);
}
#else /* CONFIG_BLK_DEV_INTEGRITY */

/*
 * When CONFIG_BLK_DEV_INTEGRITY is disabled, the core functions are replaced
 * with static inline stubs. These stubs return error codes or default values,
 * allowing code that calls these functions to compile correctly while
 * effectively disabling the data integrity feature at build time.
 */
static inline int blk_get_meta_cap(struct block_device *bdev, unsigned int cmd,
				   struct logical_block_metadata_cap __user *argp)
{
	return -ENOIOCTLCMD;
}
static inline int blk_rq_count_integrity_sg(struct request_queue *q,
					    struct bio *b)
{
	return 0;
}
static inline int blk_rq_map_integrity_sg(struct request *q,
					  struct scatterlist *s)
{
	return 0;
}
static inline int blk_rq_integrity_map_user(struct request *rq,
					    void __user *ubuf,
					    ssize_t bytes)
{
	return -EINVAL;
}
static inline struct blk_integrity *bdev_get_integrity(struct block_device *b)
{
	return NULL;
}
static inline struct blk_integrity *blk_get_integrity(struct gendisk *disk)
{
	return NULL;
}
static inline bool
blk_integrity_queue_supports_integrity(struct request_queue *q)
{
	return false;
}
static inline unsigned short
queue_max_integrity_segments(const struct request_queue *q)
{
	return 0;
}

static inline unsigned int bio_integrity_intervals(struct blk_integrity *bi,
						   unsigned int sectors)
{
	return 0;
}

static inline unsigned int bio_integrity_bytes(struct blk_integrity *bi,
					       unsigned int sectors)
{
	return 0;
}
static inline int blk_integrity_rq(struct request *rq)
{
	return 0;
}

static inline struct bio_vec rq_integrity_vec(struct request *rq)
{
	/* the optimizer will remove all calls to this function */
	return (struct bio_vec){ };
}
#endif /* CONFIG_BLK_DEV_INTEGRITY */

#endif /* _LINUX_BLK_INTEGRITY_H */
