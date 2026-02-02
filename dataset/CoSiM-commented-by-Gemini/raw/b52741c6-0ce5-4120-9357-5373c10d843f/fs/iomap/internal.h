/**
 * @file internal.h
 * @brief Internal definitions for the iomap infrastructure.
 *
 * This header file contains definitions that are internal to the iomap
 * implementation and are not part of the public API for filesystems.
 */
/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _IOMAP_INTERNAL_H
#define _IOMAP_INTERNAL_H 1

#define IOEND_BATCH_SIZE	4096

/**
 * @brief Finishes a direct I/O operation associated with an ioend.
 * @param ioend The ioend structure representing the completed I/O.
 * @return The number of bio_vecs completed.
 *
 * This function is called from the ioend completion workqueue to handle the
 * completion of a direct I/O operation. It updates the state of the iomap_dio
 * structure and may trigger the final completion of the I/O request.
 */
u32 iomap_finish_ioend_direct(struct iomap_ioend *ioend);

#endif /* _IOMAP_INTERNAL_H */