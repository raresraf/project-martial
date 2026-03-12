/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file blk-integrity.h
 * @brief Defines interfaces and data structures for Linux kernel's block layer
 * data integrity (DIF/DIX - Data Integrity Field/Extension) features.
 *
 * This header provides definitions for managing data integrity metadata
 * associated with block device I/O operations. It includes flags for integrity
 * operations, functions for mapping integrity data with requests, and utilities
 * for calculating integrity-related buffer sizes. The functionality is
 * conditionally compiled based on `CONFIG_BLK_DEV_INTEGRITY`.
 */
#ifndef _LINUX_BLK_INTEGRITY_H
#define _LINUX_BLK_INTEGRITY_H

#include <linux/blk-mq.h>
#include <linux/bio-integrity.h>

struct request;

/**
 * @enum blk_integrity_flags
 * @brief Defines various flags that control or describe block integrity operations and capabilities.
 */
enum blk_integrity_flags {
	/** Skip integrity verification for this request. */
	BLK_INTEGRITY_NOVERIFY		= 1 << 0,
	/** Skip integrity metadata generation for this request. */
	BLK_INTEGRITY_NOGENERATE	= 1 << 1,
	/** Indicates if the underlying device is capable of integrity operations. */
	BLK_INTEGRITY_DEVICE_CAPABLE	= 1 << 2,
	/** Indicates that a reference tag is used for integrity checking. */
	BLK_INTEGRITY_REF_TAG		= 1 << 3,
	/** Indicates that integrity metadata is managed in a stacked manner (e.g., in a layered device). */
	BLK_INTEGRITY_STACKED		= 1 << 4,
};

const char *blk_integrity_profile_name(struct blk_integrity *bi);
/**
 * @brief Checks if integrity can be stacked between two queue limits.
 * @param t The top queue_limits structure.
 * @param b The bottom queue_limits structure.
 * @return True if integrity can be stacked, False otherwise.
 */
bool queue_limits_stack_integrity(struct queue_limits *t,
		struct queue_limits *b);
/**
 * @brief Inline helper to check if integrity can be stacked from a top queue_limits
 * to a specific block device.
 * @param t The top queue_limits structure.
 * @param bdev The block_device to check against.
 * @return True if integrity can be stacked, False otherwise.
 */
static inline bool queue_limits_stack_integrity_bdev(struct queue_limits *t,
		struct block_device *bdev)
{
	return queue_limits_stack_integrity(t, &bdev->bd_disk->queue->limits);
}

#ifdef CONFIG_BLK_DEV_INTEGRITY
/**
 * @brief This section provides the actual implementations for block layer integrity
 * functions when `CONFIG_BLK_DEV_INTEGRITY` is enabled in the kernel configuration.
 * These functions handle the mapping, counting, user interaction, and querying
 * of integrity metadata for block devices.
 */
int blk_rq_map_integrity_sg(struct request *, struct scatterlist *);
/**
 * @brief Counts the number of integrity scatter/gather segments required for a bio.
 * @param q The request queue.
 * @param bio The bio structure.
 * @return The number of integrity scatter/gather segments.
 */
int blk_rq_count_integrity_sg(struct request_queue *, struct bio *);
/**
 * @brief Maps user-provided integrity metadata to a request.
 * @param rq The request structure.
 * @param ubuf User-space buffer containing integrity metadata.
 * @param bytes Size of the user-space buffer.
 * @return 0 on success, negative error code on failure.
 */
int blk_rq_integrity_map_user(struct request *rq, void __user *ubuf,
			      ssize_t bytes);
/**
 * @brief Retrieves metadata capabilities from a block device.
 * @param bdev The block device.
 * @param cmd The ioctl command.
 * @param argp User-space pointer to a `logical_block_metadata_cap` structure.
 * @return 0 on success, negative error code on failure.
 */
int blk_get_meta_cap(struct block_device *bdev, unsigned int cmd,
		     struct logical_block_metadata_cap __user *argp);

/**
 * @brief Checks if a request queue supports block integrity.
 * @param q The request queue to check.
 * @return True if the queue supports integrity, False otherwise.
 */
static inline bool
blk_integrity_queue_supports_integrity(struct request_queue *q)
{
	return q->limits.integrity.metadata_size;
}

/**
 * @brief Retrieves the block integrity profile for a given gendisk.
 * @param disk The gendisk structure.
 * @return A pointer to the `blk_integrity` structure if supported, or NULL.
 */
static inline struct blk_integrity *blk_get_integrity(struct gendisk *disk)
{
	if (!blk_integrity_queue_supports_integrity(disk->queue))
		return NULL;
	return &disk->queue->limits.integrity;
}

/**
 * @brief Retrieves the block integrity profile for a given block device.
 * @param bdev The block device structure.
 * @return A pointer to the `blk_integrity` structure if supported, or NULL.
 */
static inline struct blk_integrity *
bdev_get_integrity(struct block_device *bdev)
{
	return blk_get_integrity(bdev->bd_disk);
}

/**
 * @brief Returns the maximum number of integrity segments supported by a request queue.
 * @param q The request queue.
 * @return The maximum number of integrity segments.
 */
static inline unsigned short
queue_max_integrity_segments(const struct request_queue *q)
{
	return q->limits.max_integrity_segments;
}

/**
 * @brief Calculates the number of integrity intervals for a bio based on its sectors.
 * The block layer works with 512-byte sectors, while integrity metadata uses
 * device-specific integrity interval sizes.
 * @param bi The `blk_integrity` profile for the device.
 * @param sectors The size of the bio in 512-byte sectors.
 * @return The number of integrity intervals.
 */
static inline unsigned int bio_integrity_intervals(struct blk_integrity *bi,
						   unsigned int sectors)
{
	return sectors >> (bi->interval_exp - 9);
}

/**
 * @brief Calculates the total size of integrity metadata bytes for a bio.
 * @param bi The `blk_integrity` profile for the device.
 * @param sectors The size of the bio in 512-byte sectors.
 * @return The total size of integrity metadata in bytes.
 */
static inline unsigned int bio_integrity_bytes(struct blk_integrity *bi,
					       unsigned int sectors)
{
	return bio_integrity_intervals(bi, sectors) * bi->metadata_size;
}

/**
 * @brief Checks if a given request involves integrity operations.
 * @param rq The request structure.
 * @return True if the request has integrity flags set, False otherwise.
 */
static inline bool blk_integrity_rq(struct request *rq)
{
	return rq->cmd_flags & REQ_INTEGRITY;
}

/**
 * @brief Returns the current `bio_vec` that contains the integrity data
 * for a request. The `bip_iter` within the bio integrity profile may be
 * advanced to iterate over subsequent integrity data segments.
 * @param rq The request structure.
 * @return A `bio_vec` structure containing the integrity data.
 */
static inline struct bio_vec rq_integrity_vec(struct request *rq)
{
	return mp_bvec_iter_bvec(rq->bio->bi_integrity->bip_vec,
				 rq->bio->bi_integrity->bip_iter);
}
#else /* CONFIG_BLK_DEV_INTEGRITY */
/**
 * @brief This section provides stub (dummy) implementations for block layer integrity
 * functions when `CONFIG_BLK_DEV_INTEGRITY` is disabled in the kernel configuration.
 * These stubs typically return error codes or default values, indicating that
 * integrity features are not available.
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
