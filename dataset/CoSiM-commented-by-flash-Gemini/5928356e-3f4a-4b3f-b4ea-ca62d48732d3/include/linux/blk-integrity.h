/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @5928356e-3f4a-4b3f-b4ea-ca62d48732d3/include/linux/blk-integrity.h
 * @brief Public API and data structures for the Block Layer Data Integrity (DI) framework.
 * 
 * Functional Intent: Manages end-to-end data integrity metadata (e.g., T10 PI, 
 * checksums) as it flows through the block I/O stack. It provides abstraction for 
 * hardware-assisted integrity generation and verification, allowing the kernel 
 * to detect silent data corruption.
 * 
 * Domain: Storage Stack, Reliability & Serviceability (RAS), Kernel Block Layer.
 */
#ifndef _LINUX_BLK_INTEGRITY_H
#define _LINUX_BLK_INTEGRITY_H

#include <linux/blk-mq.h>
#include <linux/bio-integrity.h>

struct request;

/**
 * enum blk_integrity_flags - Operational flags for block integrity profiles.
 * BLK_INTEGRITY_NOVERIFY: Disable verification check (trust the data).
 * BLK_INTEGRITY_NOGENERATE: Disable metadata generation (source provides it).
 * BLK_INTEGRITY_DEVICE_CAPABLE: Hardware supports offloading integrity operations.
 * BLK_INTEGRITY_REF_TAG: Profile includes a reference tag (e.g., LBA-based).
 * BLK_INTEGRITY_STACKED: Profile is an aggregate of multiple underlying devices.
 */
enum blk_integrity_flags {
	BLK_INTEGRITY_NOVERIFY		= 1 << 0,
	BLK_INTEGRITY_NOGENERATE	= 1 << 1,
	BLK_INTEGRITY_DEVICE_CAPABLE	= 1 << 2,
	BLK_INTEGRITY_REF_TAG		= 1 << 3,
	BLK_INTEGRITY_STACKED		= 1 << 4,
};

const char *blk_integrity_profile_name(struct blk_integrity *bi);

/**
 * Functional Utility: Propagates integrity constraints across stacked block devices.
 */
bool queue_limits_stack_integrity(struct queue_limits *t,
		struct queue_limits *b);

static inline bool queue_limits_stack_integrity_bdev(struct queue_limits *t,
		struct block_device *bdev)
{
	return queue_limits_stack_integrity(t, &bdev->bd_disk->queue->limits);
}

#ifdef CONFIG_BLK_DEV_INTEGRITY
int blk_rq_map_integrity_sg(struct request *, struct scatterlist *);
int blk_rq_count_integrity_sg(struct request_queue *, struct bio *);
int blk_rq_integrity_map_user(struct request *rq, void __user *ubuf,
			      ssize_t bytes);
int blk_get_meta_cap(struct block_device *bdev, unsigned int cmd,
		     struct logical_block_metadata_cap __user *argp);

/**
 * Block Logic: Integrity Support Predicates.
 * Logic: Checks the registered metadata size to determine if a queue is 
 * configured for integrity operations.
 */
static inline bool
blk_integrity_queue_supports_integrity(struct request_queue *q)
{
	return q->limits.integrity.metadata_size;
}

static inline struct blk_integrity *blk_get_integrity(struct gendisk *disk)
{
	if (!blk_integrity_queue_supports_integrity(disk->queue))
		return NULL;
	return &disk->queue->limits.integrity;
}

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
 * bio_integrity_intervals - Converts sectors to integrity-native units.
 * @bi:   Active integrity profile for the target device.
 * @sectors: Size of the I/O in 512-byte blocks.
 *
 * Description: The block layer calculates everything in 512 byte
 * sectors but integrity metadata is done in terms of the data integrity
 * interval size of the storage device (e.g., 4096 bytes).
 */
static inline unsigned int bio_integrity_intervals(struct blk_integrity *bi,
						   unsigned int sectors)
{
	return sectors >> (bi->interval_exp - 9);
}

/**
 * bio_integrity_bytes - Calculates the total metadata payload size in bytes.
 */
static inline unsigned int bio_integrity_bytes(struct blk_integrity *bi,
					       unsigned int sectors)
{
	return bio_integrity_intervals(bi, sectors) * bi->metadata_size;
}

/**
 * blk_integrity_rq - Predicate to check if a request has the integrity bit set.
 */
static inline bool blk_integrity_rq(struct request *rq)
{
	return rq->cmd_flags & REQ_INTEGRITY;
}

/**
 * rq_integrity_vec - Retrieves the active scatter-gather element for integrity data.
 */
static inline struct bio_vec rq_integrity_vec(struct request *rq)
{
	return mp_bvec_iter_bvec(rq->bio->bi_integrity->bip_vec,
				 rq->bio->bi_integrity->bip_iter);
}
#else /* CONFIG_BLK_DEV_INTEGRITY */
/* Stub implementations for kernels without block integrity enabled. */
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
