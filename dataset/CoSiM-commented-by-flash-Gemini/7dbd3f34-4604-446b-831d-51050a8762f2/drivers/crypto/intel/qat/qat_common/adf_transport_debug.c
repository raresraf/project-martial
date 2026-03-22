// SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0-only)
/* Copyright(c) 2014 - 2020 Intel Corporation */
/**
 * @file adf_transport_debug.c
 * @brief This file provides debugfs functionality for the QAT (QuickAssist Technology)
 *        transport layer. It exposes internal states of rings and banks of QAT
 *        accelerator devices through the Linux debugfs interface, allowing
 *        developers and system administrators to inspect the device's operational
 *        status and data flow for diagnostic purposes.
 *        The debugfs entries provide insights into ring configuration, head/tail pointers,
 *        and raw ring buffer data.
 */
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/seq_file.h>
#include "adf_accel_devices.h"
#include "adf_transport_internal.h"
#include "adf_transport_access_macros.h"

/**
 * @brief Mutex to protect concurrent reads from ring debugfs entries.
 * Prevents race conditions when multiple users try to read ring contents simultaneously.
 */
static DEFINE_MUTEX(ring_read_lock);
/**
 * @brief Mutex to protect concurrent reads from bank debugfs entries.
 * Prevents race conditions when multiple users try to read bank configuration simultaneously.
 */
static DEFINE_MUTEX(bank_read_lock);

/**
 * @brief Implements the .start operation for seq_file to iterate over ring data.
 *
 * This function is called to start an iteration through the ring's message buffer.
 * It acquires a mutex to ensure exclusive access during the read operation.
 *
 * @param sfile Pointer to the seq_file structure, containing private data (ring).
 * @param pos Current position within the sequence.
 * @return void* A pointer to the first message buffer entry, or SEQ_START_TOKEN for header,
 *               or NULL if out of bounds.
 */
static void *adf_ring_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private;

	mutex_lock(&ring_read_lock);
	/* Conditional Logic: If position is 0, return a token indicating the start of the sequence. */
	if (*pos == 0)
		return SEQ_START_TOKEN;

	/* Conditional Logic: Check if the current position is beyond the end of the ring. */
	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size)))
		return NULL;

	/* Return a pointer to the message buffer at the current position and increment position. */
	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

/**
 * @brief Implements the .next operation for seq_file to continue iterating over ring data.
 *
 * This function is called to get the next message buffer entry in the sequence.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the previous element returned by .start or .next.
 * @param pos Current position within the sequence.
 * @return void* A pointer to the next message buffer entry, or NULL if out of bounds.
 */
static void *adf_ring_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private;

	/* Block Logic: Increment the position and check if it's beyond the ring's capacity. */
	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size)))
		return NULL;

	/* Return a pointer to the message buffer at the new current position and increment position. */
	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

/**
 * @brief Implements the .show operation for seq_file to format and display ring data.
 *
 * This function formats and prints the ring's configuration or its raw data
 * for debugfs output.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the current element being processed.
 * @return int 0 on success.
 */
static int adf_ring_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_ring_data *ring = sfile->private;
	struct adf_etr_bank_data *bank = ring->bank;
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev);
	void __iomem *csr = ring->bank->csr_addr;

	/* Conditional Logic: If this is the start token, print ring configuration header. */
	if (v == SEQ_START_TOKEN) {
		int head, tail, empty;

		/* Read current hardware ring head, tail, and empty status. */
		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
						   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
						   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number);

		seq_puts(sfile, "------- Ring configuration -------\n");
		seq_printf(sfile, "ring name: %s\n",
			   ring->ring_debug->ring_name);
		seq_printf(sfile, "ring num %d, bank num %d\n",
			   ring->ring_number, ring->bank->bank_number);
		seq_printf(sfile, "head %x, tail %x, empty: %d\n",
			   head, tail, (empty & (1 << ring->ring_number))
			   >> ring->ring_number); /* Inline: Extracts the 'empty' bit for the specific ring. */
		seq_printf(sfile, "ring size %lld, msg size %d\n",
			   (long long)ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size),
			   ADF_MSG_SIZE_TO_BYTES(ring->msg_size));
		seq_puts(sfile, "----------- Ring data ------------\n");
		return 0;
	}
	/* Block Logic: For non-header elements, dump raw hexadecimal content of the ring message. */
	seq_hex_dump(sfile, "", DUMP_PREFIX_ADDRESS, 32, 4,
		     v, ADF_MSG_SIZE_TO_BYTES(ring->msg_size), false);
	return 0;
}

/**
 * @brief Implements the .stop operation for seq_file to clean up after iterating.
 *
 * This function is called at the end of a seq_file read operation for a ring.
 * It releases the mutex acquired in `adf_ring_start`.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the last element processed (ignored).
 */
static void adf_ring_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&ring_read_lock);
}

/**
 * @brief Sequence operations structure for ring debugfs entries.
 * Defines the callback functions for iterating and displaying ring data.
 */
static const struct seq_operations adf_ring_debug_sops = {
	.start = adf_ring_start,
	.next = adf_ring_next,
	.stop = adf_ring_stop,
	.show = adf_ring_show
};

/**
 * @brief Macro to define a seq_file attribute for ring debugfs.
 * This helper macro creates the necessary file_operations structure
 * based on the seq_operations defined above.
 */
DEFINE_SEQ_ATTRIBUTE(adf_ring_debug);

/**
 * @brief Adds a debugfs entry for a specific QAT ETR ring.
 *
 * This function creates a debugfs file named "ring_XX" within the parent
 * bank's debugfs directory, allowing inspection of the ring's state.
 *
 * @param ring Pointer to the adf_etr_ring_data structure for the ring.
 * @param name Custom name for the ring (e.g., "tx" or "rx").
 * @return int 0 on success, or negative errno on failure (e.g., -ENOMEM).
 */
int adf_ring_debugfs_add(struct adf_etr_ring_data *ring, const char *name)
{
	struct adf_etr_ring_debug_entry *ring_debug;
	char entry_name[16];

	/* Block Logic: Allocate memory for the ring debug entry structure. */
	ring_debug = kzalloc(sizeof(*ring_debug), GFP_KERNEL);
	/* Conditional Logic: Check if memory allocation failed. */
	if (!ring_debug)
		return -ENOMEM;

	strscpy(ring_debug->ring_name, name, sizeof(ring_debug->ring_name));
	/* Inline: Formats the debugfs entry name as "ring_XX". */
	snprintf(entry_name, sizeof(entry_name), "ring_%02d",
		 ring->ring_number);

	/* Create the debugfs file. */
	ring_debug->debug = debugfs_create_file(entry_name, S_IRUSR,
						ring->bank->bank_debug_dir,
						ring, &adf_ring_debug_fops);
	ring->ring_debug = ring_debug;
	return 0;
}

/**
 * @brief Removes the debugfs entry for a QAT ETR ring.
 *
 * This function cleans up the debugfs file and frees associated memory
 * when a ring is no longer active or the device is being removed.
 *
 * @param ring Pointer to the adf_etr_ring_data structure for the ring.
 */
void adf_ring_debugfs_rm(struct adf_etr_ring_data *ring)
{
	/* Conditional Logic: Check if debugfs entry exists. */
	if (ring->ring_debug) {
		debugfs_remove(ring->ring_debug->debug); /* Remove the debugfs file. */
		kfree(ring->ring_debug);               /* Free the allocated memory. */
		ring->ring_debug = NULL;               /* Clear the pointer. */
	}
}

/**
 * @brief Implements the .start operation for seq_file to iterate over bank ring configuration.
 *
 * This function is called to start an iteration through a bank's ring configurations.
 * It acquires a mutex to ensure exclusive access during the read operation.
 *
 * @param sfile Pointer to the seq_file structure, containing private data (bank).
 * @param pos Current position within the sequence.
 * @return void* A pointer to the current position in the iteration, or SEQ_START_TOKEN for header,
 *               or NULL if out of bounds.
 */
static void *adf_bank_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private;
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev);

	mutex_lock(&bank_read_lock);
	/* Conditional Logic: If position is 0, return a token indicating the start of the sequence. */
	if (*pos == 0)
		return SEQ_START_TOKEN;

	/* Conditional Logic: Check if the current position is beyond the number of rings per bank. */
	if (*pos >= num_rings_per_bank)
		return NULL;

	/* Return a pointer to the current position (used as an index) for iteration. */
	return pos;
}

/**
 * @brief Implements the .next operation for seq_file to continue iterating over bank ring configuration.
 *
 * This function is called to get the next ring's configuration in the sequence.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the previous element returned by .start or .next (ignored).
 * @param pos Current position within the sequence.
 * @return void* A pointer to the next position in the iteration, or NULL if out of bounds.
 */
static void *adf_bank_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private;
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev);

	/* Block Logic: Increment the position and check if it's beyond the number of rings per bank. */
	if (++(*pos) >= num_rings_per_bank)
		return NULL;

	/* Return a pointer to the new current position for iteration. */
	return pos;
}

/**
 * @brief Implements the .show operation for seq_file to format and display bank data.
 *
 * This function formats and prints the bank's overall configuration header or
 * individual ring configurations within the bank for debugfs output.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the current element being processed.
 * @return int 0 on success.
 */
static int adf_bank_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_bank_data *bank = sfile->private;
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev);

	/* Conditional Logic: If this is the start token, print bank configuration header. */
	if (v == SEQ_START_TOKEN) {
		seq_printf(sfile, "------- Bank %d configuration -------\n",
			   bank->bank_number);
	} else {
		/* Block Logic: For non-header elements, print detailed configuration for an individual ring. */
		int ring_id = *((int *)v); /* Inline: Dereferences `v` to get the current ring index. */
		struct adf_etr_ring_data *ring = &bank->rings[ring_id];
		void __iomem *csr = bank->csr_addr;
		int head, tail, empty;

		/* Conditional Logic: Check if this ring is enabled in the bank's mask. */
		if (!(bank->ring_mask & (1 << ring_id))) /* Inline: Bitwise check for ring enable. */
			return 0;

		/* Read current hardware ring head, tail, and empty status for the specific ring. */
		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
						   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
						   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number);

		seq_printf(sfile,
			   "ring num %02d, head %04x, tail %04x, empty: %d\n",
			   ring->ring_number, head, tail,
			   (empty & (1 << ring->ring_number)) >> /* Inline: Extracts the 'empty' bit for the specific ring. */
			   ring->ring_number);
	}
	return 0;
}

/**
 * @brief Implements the .stop operation for seq_file to clean up after iterating.
 *
 * This function is called at the end of a seq_file read operation for a bank.
 * It releases the mutex acquired in `adf_bank_start`.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the last element processed (ignored).
 */
static void adf_bank_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&bank_read_lock);
}

/**
 * @brief Sequence operations structure for bank debugfs entries.
 * Defines the callback functions for iterating and displaying bank configuration and ring summaries.
 */
static const struct seq_operations adf_bank_debug_sops = {
	.start = adf_bank_start,
	.next = adf_bank_next,
	.stop = adf_bank_stop,
	.show = adf_bank_show
};

/**
 * @brief Macro to define a seq_file attribute for bank debugfs.
 * This helper macro creates the necessary file_operations structure
 * based on the seq_operations defined above.
 */
DEFINE_SEQ_ATTRIBUTE(adf_bank_debug);

/**
 * @brief Adds debugfs entries for a specific QAT ETR bank.
 *
 * This function creates a debugfs directory named "bank_XX" and a "config" file
 * within it for a given bank, allowing inspection of the bank's state and its rings.
 *
 * @param bank Pointer to the adf_etr_bank_data structure for the bank.
 * @return int 0 on success, or negative errno on failure.
 */
int adf_bank_debugfs_add(struct adf_etr_bank_data *bank)
{
	struct adf_accel_dev *accel_dev = bank->accel_dev;
	struct dentry *parent = accel_dev->transport->debug;
	char name[16];

	/* Inline: Formats the debugfs directory name as "bank_XX". */
	snprintf(name, sizeof(name), "bank_%02d", bank->bank_number);
	/* Create the debugfs directory for the bank. */
	bank->bank_debug_dir = debugfs_create_dir(name, parent);
	/* Create the "config" file within the bank's debugfs directory. */
	bank->bank_debug_cfg = debugfs_create_file("config", S_IRUSR,
						   bank->bank_debug_dir, bank,
						   &adf_bank_debug_fops);
	return 0;
}

/**
 * @brief Removes debugfs entries for a QAT ETR bank.
 *
 * This function cleans up the debugfs directory and config file
 * when a bank is no longer active or the device is being removed.
 *
 * @param bank Pointer to the adf_etr_bank_data structure for the bank.
 */
void adf_bank_debugfs_rm(struct adf_etr_bank_data *bank)
{
	debugfs_remove(bank->bank_debug_cfg); /* Remove the "config" debugfs file. */
	debugfs_remove(bank->bank_debug_dir); /* Remove the bank's debugfs directory. */
}
