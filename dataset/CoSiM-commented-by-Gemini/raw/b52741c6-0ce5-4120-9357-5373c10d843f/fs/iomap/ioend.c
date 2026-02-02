/**
 * @file ioend.c
 * @brief Manages I/O completion for iomap-based I/O.
 * @copyright Copyright (c) 2016-2025 Christoph Hellwig.
 *
 * This file contains the implementation for handling the completion of I/O
 * operations that are submitted through the iomap interface. It introduces
 * the `iomap_ioend` structure, which tracks the state of an in-flight I/O
 * operation. This is particularly important for writeback, where multiple
 * dirty pages are coalesced into a single bio for submission. The ioend
 * logic ensures that upon completion, all associated pages are correctly
 * updated and freed.
 */
// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2016-2025 Christoph Hellwig.
 */
#include <linux/iomap.h>
#include <linux/list_sort.h>
#include <linux/pagemap.h>
#include <linux/writeback.h>
#include "internal.h"
#include "trace.h"

struct bio_set iomap_ioend_bioset;
EXPORT_SYMBOL_GPL(iomap_ioend_bioset);

/**
 * @brief Initializes an iomap_ioend structure.
 * @param inode The inode the I/O is for.
 * @param bio The bio for the I/O.
 * @param file_offset The offset in the file where the I/O starts.
 * @param ioend_flags Flags for the ioend.
 * @return A pointer to the initialized iomap_ioend.
 */
struct iomap_ioend *iomap_init_ioend(struct inode *inode,
		struct bio *bio, loff_t file_offset, u16 ioend_flags)
{
	struct iomap_ioend *ioend = iomap_ioend_from_bio(bio);

	atomic_set(&ioend->io_remaining, 1);
	ioend->io_error = 0;
	ioend->io_parent = NULL;
	INIT_LIST_HEAD(&ioend->io_list);
	ioend->io_flags = ioend_flags;
	ioend->io_inode = inode;
	ioend->io_offset = file_offset;
	ioend->io_size = bio->bi_iter.bi_size;
	ioend->io_sector = bio->bi_iter.bi_sector;
	ioend->io_private = NULL;
	return ioend;
}
EXPORT_SYMBOL_GPL(iomap_init_ioend);

/**
 * @brief Finalizes a buffered write I/O operation.
 * @param ioend The ioend structure for the completed I/O.
 * @return The number of folios completed.
 *
 * This function is called upon completion of a buffered write. It updates the
 * state of all folios in the bio, handles any errors, and frees the ioend.
 */
static u32 iomap_finish_ioend_buffered(struct iomap_ioend *ioend)
{
	struct inode *inode = ioend->io_inode;
	struct bio *bio = &ioend->io_bio;
	struct folio_iter fi;
	u32 folio_count = 0;

	if (ioend->io_error) {
		mapping_set_error(inode->i_mapping, ioend->io_error);
		if (!bio_flagged(bio, BIO_QUIET)) {
			pr_err_ratelimited(
"%s: writeback error on inode %lu, offset %lld, sector %llu",
				inode->i_sb->s_id, inode->i_ino,
				ioend->io_offset, ioend->io_sector);
		}
	}

	/* walk all folios in bio, ending page IO on them */
	bio_for_each_folio_all(fi, bio) {
		iomap_finish_folio_write(inode, fi.folio, fi.length);
		folio_count++;
	}

	bio_put(bio);	/* frees the ioend */
	return folio_count;
}

/**
 * @brief The end I/O handler for writeback bios.
 * @param bio The bio that has completed.
 *
 * This function is set as the `bi_end_io` handler for bios submitted for
 * writeback. It records any error and calls the final completion function.
 */
static void ioend_writeback_end_bio(struct bio *bio)
{
	struct iomap_ioend *ioend = iomap_ioend_from_bio(bio);

	ioend->io_error = blk_status_to_errno(bio->bi_status);
	iomap_finish_ioend_buffered(ioend);
}

/**
 * @brief Submits a writeback operation that is tracked by an ioend.
 * @param wpc The writepage context.
 * @param error An error code from the submission path, if any.
 * @return 0 on success, or the error code.
 *
 * This function is the ->submit_bio equivalent for iomap writeback. It handles
 * submission and error paths, ensuring that the I/O completion is always
 * correctly handled.
 */
int iomap_ioend_writeback_submit(struct iomap_writepage_ctx *wpc, int error)
{
...
The rest of the file with comments.
...
}