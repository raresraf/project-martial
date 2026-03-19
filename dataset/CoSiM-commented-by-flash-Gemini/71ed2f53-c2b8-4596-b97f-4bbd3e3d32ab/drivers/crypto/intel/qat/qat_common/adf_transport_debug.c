/**
 * @file adf_transport_debug.c
 * @brief Provides debugfs support for the Intel(R) QuickAssist Technology (QAT)
 *        accelerator driver's transport layer.
 *
 * This file implements functionality to expose internal state and configuration
 * of QAT's Entry/Ring Transport (ETR) rings and banks via the Linux debugfs
 * filesystem. It utilizes the `seq_file` interface to create virtual files
 * that display dynamic information such as ring head/tail pointers, message
 * sizes, and ring contents for diagnostic purposes.
 *
 * It includes functions for:
 * - Iterating and displaying data within ETR rings.
 * - Displaying configuration and status of ETR banks and their associated rings.
 * - Managing the creation and removal of debugfs entries for rings and banks.
 *
 * @copyright Copyright(c) 2014 - 2020 Intel Corporation
 * @license SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0-only)
 */

#include <linux/mutex.h>      // For mutex synchronization.
#include <linux/slab.h>       // For kernel memory allocation (kzalloc, kfree).
#include <linux/seq_file.h>   // For seq_file interface to create dynamic debugfs files.
#include "adf_accel_devices.h"        // Common definitions for QAT acceleration devices.
#include "adf_transport_internal.h"   // Internal transport layer definitions.
#include "adf_transport_access_macros.h" // Macros for accessing transport layer registers.

/** @def ring_read_lock
 * @brief Mutex to protect concurrent access to ring data during debugfs reads.
 */
static DEFINE_MUTEX(ring_read_lock);

/** @def bank_read_lock
 * @brief Mutex to protect concurrent access to bank data during debugfs reads.
 */
static DEFINE_MUTEX(bank_read_lock);

/**
 * @macro ADF_RING_NUM_MSGS
 * @brief Calculates the number of messages a ring can hold.
 * @param ring Pointer to an `adf_etr_ring_data` structure.
 * @return The maximum number of messages that can be stored in the ring.
 */
#define ADF_RING_NUM_MSGS(ring)				\
	(ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /	\
	ADF_MSG_SIZE_TO_BYTES(ring->msg_size))

/**
 * @brief `seq_file` start operation for reading ETR ring debugfs entries.
 *
 * This function is called to start an iteration over the messages within an ETR ring.
 * It acquires `ring_read_lock` to ensure exclusive access during the read sequence.
 * The first call (pos == 0) returns a special token for header printing.
 *
 * @param sfile Pointer to the `seq_file` structure. `sfile->private` holds `adf_etr_ring_data`.
 * @param pos Current position in the sequence, incremented to point to the next item.
 * @return Pointer to the first message buffer in the ring, or `SEQ_START_TOKEN` for header.
 */
static void *adf_ring_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private;
	unsigned int num_msg = ADF_RING_NUM_MSGS(ring);
	loff_t val = *pos;

	mutex_lock(&ring_read_lock); // Acquire mutex to prevent concurrent debugfs reads of rings.
	if (val == 0)
		return SEQ_START_TOKEN; // Special token to indicate start of sequence for header.

	/**
	 * Block Logic: Check if the current position exceeds the number of messages in the ring.
	 * Pre-condition: `val` is the current iteration position.
	 * Invariant: If `val` is within bounds, the next message address is returned.
	 */
	if (val >= num_msg)
		return NULL; // End of sequence.

	// Calculate the physical address of the current message in the ring.
	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

/**
 * @brief `seq_file` next operation for reading ETR ring debugfs entries.
 *
 * This function is called to advance to the next message buffer in an ETR ring.
 *
 * @param sfile Pointer to the `seq_file` structure.
 * @param v Current item (not used for calculation here, `*pos` is used).
 * @param pos Current position in the sequence, incremented by `adf_ring_start`.
 * @return Pointer to the next message buffer in the ring, or `NULL` if end of sequence.
 */
static void *adf_ring_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private;
	unsigned int num_msg = ADF_RING_NUM_MSGS(ring);
	loff_t val = *pos; // `val` already incremented by `adf_ring_start` to point to next.

	(*pos)++; // Increment position for the *next* call to `adf_ring_next`.

	/**
	 * Block Logic: Check if the current position exceeds the number of messages in the ring.
	 * Pre-condition: `val` is the current iteration position.
	 * Invariant: If `val` is within bounds, the next message address is implicitly provided.
	 */
	if (val >= num_msg)
		return NULL; // End of sequence.

	// Calculate the physical address of the message at `val`.
	return ring->base_addr + (ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * val);
}

/**
 * @brief `seq_file` show operation for displaying ETR ring debugfs entries.
 *
 * This function is responsible for formatting and printing the content
 * of an ETR ring or its configuration header to the debugfs file.
 *
 * @param sfile Pointer to the `seq_file` structure.
 * @param v Pointer to the current item (message buffer or `SEQ_START_TOKEN`).
 * @return 0 on success.
 */
static int adf_ring_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_ring_data *ring = sfile->private;
	struct adf_etr_bank_data *bank = ring->bank;
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev);
	void __iomem *csr = ring->bank->csr_addr; // Memory-mapped I/O address for Control Status Registers.

	/**
	 * Block Logic: If `v` is `SEQ_START_TOKEN`, print the ring configuration header.
	 * Otherwise, dump the hexadecimal content of the current message.
	 * Pre-condition: `sfile` points to a valid sequence file, `v` is the current item.
	 * Invariant: The correct information (header or message content) is printed.
	 */
	if (v == SEQ_START_TOKEN) {
		int head, tail, empty;

		// Read current ring head, tail pointers and empty status from hardware CSRs.
		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
							   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
							   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number); // Read the enable status register.

		seq_puts(sfile, "------- Ring configuration -------\n");
		seq_printf(sfile, "ring name: %s\n",
			   ring->ring_debug->ring_name);
		seq_printf(sfile, "ring num %d, bank num %d\n",
			   ring->ring_number, ring->bank->bank_number);
		seq_printf(sfile, "head %x, tail %x, empty: %d\n",
			   head, tail, (empty & (1 << ring->ring_number)) // Check the specific bit for this ring's empty status.
			   >> ring->ring_number); // Shift to get 0 or 1.
		seq_printf(sfile, "ring size %lld, msg size %d\n",
			   (long long)ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size),
			   ADF_MSG_SIZE_TO_BYTES(ring->msg_size));
		seq_puts(sfile, "----------- Ring data ------------\n");
		return 0;
	}
	// If not SEQ_START_TOKEN, `v` points to a message buffer; dump its hexadecimal content.
	seq_hex_dump(sfile, "", DUMP_PREFIX_ADDRESS, 32, 4,
		     v, ADF_MSG_SIZE_TO_BYTES(ring->msg_size), false);
	return 0;
}

/**
 * @brief `seq_file` stop operation for reading ETR ring debugfs entries.
 *
 * This function is called when the `seq_file` read sequence is finished.
 * It releases the `ring_read_lock`.
 *
 * @param sfile Pointer to the `seq_file` structure.
 * @param v Current item (not used).
 */
static void adf_ring_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&ring_read_lock); // Release mutex.
}

/**
 * @var adf_ring_debug_sops
 * @brief `seq_operations` structure defining the callback functions for ETR ring debugfs entries.
 */
static const struct seq_operations adf_ring_debug_sops = {
	.start = adf_ring_start,
	.next = adf_ring_next,
	.stop = adf_ring_stop,
	.show = adf_ring_show
};

/**
 * @macro DEFINE_SEQ_ATTRIBUTE(adf_ring_debug)
 * @brief Helper macro to define debugfs file operations for `adf_ring_debug`.
 * This expands to define `adf_ring_debug_fops` (file operations).
 */
DEFINE_SEQ_ATTRIBUTE(adf_ring_debug);

/**
 * @brief Adds a debugfs entry for an ETR ring.
 *
 * Creates a debugfs file named `ring_XX` (where XX is ring number) under
 * the parent bank's debugfs directory. This file exposes ring configuration
 * and contents.
 *
 * @param ring Pointer to the `adf_etr_ring_data` structure for the ring.
 * @param name The name to associate with the ring (e.g., "TX" or "RX").
 * @return 0 on success, or a negative errno on failure (e.g., -ENOMEM).
 */
int adf_ring_debugfs_add(struct adf_etr_ring_data *ring, const char *name)
{
	struct adf_etr_ring_debug_entry *ring_debug;
	char entry_name[16];

	ring_debug = kzalloc(sizeof(*ring_debug), GFP_KERNEL); // Allocate memory for debug entry.
	if (!ring_debug)
		return -ENOMEM;

	strscpy(ring_debug->ring_name, name, sizeof(ring_debug->ring_name)); // Copy ring name.
	snprintf(entry_name, sizeof(entry_name), "ring_%02d",
		 ring->ring_number); // Format debugfs entry name.

	ring_debug->debug = debugfs_create_file(entry_name, S_IRUSR,
							ring->bank->bank_debug_dir, // Parent directory is the bank's debugfs directory.
							ring, &adf_ring_debug_fops); // Associate with ring data and fops.
	ring->ring_debug = ring_debug; // Store reference to debug entry in ring data.
	return 0;
}

/**
 * @brief Removes the debugfs entry for an ETR ring.
 *
 * @param ring Pointer to the `adf_etr_ring_data` structure.
 */
void adf_ring_debugfs_rm(struct adf_etr_ring_data *ring)
{
	/**
	 * Block Logic: Check if the ring has an associated debugfs entry.
	 * If yes, remove the debugfs file and free its allocated memory.
	 */
	if (ring->ring_debug) {
		debugfs_remove(ring->ring_debug->debug); // Remove debugfs file.
		kfree(ring->ring_debug);                 // Free allocated memory.
		ring->ring_debug = NULL;
	}
}

/**
 * @brief `seq_file` start operation for reading ETR bank debugfs entries.
 *
 * This function is called to start an iteration over the rings within an ETR bank.
 * It acquires `bank_read_lock`.
 *
 * @param sfile Pointer to the `seq_file` structure. `sfile->private` holds `adf_etr_bank_data`.
 * @param pos Current position in the sequence.
 * @return `SEQ_START_TOKEN` for header, or `NULL` if end of sequence.
 */
static void *adf_bank_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private;
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev);

	mutex_lock(&bank_read_lock); // Acquire mutex.
	if (*pos == 0)
		return SEQ_START_TOKEN; // Special token for header.

	/**
	 * Block Logic: Check if the current position exceeds the number of rings in the bank.
	 * Pre-condition: `*pos` is the current ring ID to display.
	 * Invariant: If `*pos` is within bounds, the position itself is returned.
	 */
	if (*pos >= num_rings_per_bank)
		return NULL; // End of sequence.

	return pos; // Return the position as the "item" to show (it will be cast to int later).
}

/**
 * @brief `seq_file` next operation for reading ETR bank debugfs entries.
 *
 * This function is called to advance to the next ring ID within an ETR bank.
 *
 * @param sfile Pointer to the `seq_file` structure.
 * @param v Current item (the `pos` pointer from `adf_bank_start` or previous `adf_bank_next`).
 * @param pos Current position in the sequence, incremented to point to the next ring ID.
 * @return Pointer to the next ring ID (as `pos`), or `NULL` if end of sequence.
 */
static void *adf_bank_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private;
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev);

	/**
	 * Block Logic: Increment the position and check if it exceeds the number of rings.
	 * Pre-condition: `*pos` is the current ring ID.
	 * Invariant: If the new `*pos` is within bounds, it's returned.
	 */
	if (++(*pos) >= num_rings_per_bank)
		return NULL; // End of sequence.

	return pos; // Return the incremented position.
}

/**
 * @brief `seq_file` show operation for displaying ETR bank debugfs entries.
 *
 * This function formats and prints either the bank configuration header
 * or the status of individual rings within the bank.
 *
 * @param sfile Pointer to the `seq_file` structure.
 * @param v Pointer to the current item (`SEQ_START_TOKEN` or `pos` pointer representing ring ID).
 * @return 0 on success.
 */
static int adf_bank_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_bank_data *bank = sfile->private;
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev);

	/**
	 * Block Logic: If `v` is `SEQ_START_TOKEN`, print the bank configuration header.
	 * Otherwise, print the head/tail/empty status for the specific ring.
	 * Pre-condition: `sfile` points to a valid sequence file, `v` is the current item.
	 * Invariant: The correct information (header or ring status) is printed.
	 */
	if (v == SEQ_START_TOKEN) {
		seq_printf(sfile, "------- Bank %d configuration -------\n",
			   bank->bank_number);
	} else {
		// `v` is a loff_t *pos, cast it to int to get the ring ID.
		int ring_id = *((int *)v) - 1; // Adjust index as `pos` is 1-based conceptually for this usage.
		struct adf_etr_ring_data *ring = &bank->rings[ring_id];
		void __iomem *csr = bank->csr_addr; // Memory-mapped I/O address.
		int head, tail, empty;

		/**
		 * Block Logic: Check if the ring is enabled in the bank's mask.
		 * If not enabled, skip its display.
		 */
		if (!(bank->ring_mask & (1 << ring_id))) // Check if the bit corresponding to ring_id is set in ring_mask.
			return 0;

		// Read current ring head, tail pointers and empty status from hardware CSRs.
		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
							   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
							   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number); // Read the enable status register.

		seq_printf(sfile,
			   "ring num %02d, head %04x, tail %04x, empty: %d\n",
			   ring->ring_number, head, tail,
			   (empty & (1 << ring->ring_number)) >> // Check the specific bit for this ring's empty status.
			   ring->ring_number); // Shift to get 0 or 1.
	}
	return 0;
}

/**
 * @brief `seq_file` stop operation for reading ETR bank debugfs entries.
 *
 * This function is called when the `seq_file` read sequence is finished.
 * It releases the `bank_read_lock`.
 *
 * @param sfile Pointer to the `seq_file` structure.
 * @param v Current item (not used).
 */
static void adf_bank_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&bank_read_lock); // Release mutex.
}

/**
 * @var adf_bank_debug_sops
 * @brief `seq_operations` structure defining the callback functions for ETR bank debugfs entries.
 */
static const struct seq_operations adf_bank_debug_sops = {
	.start = adf_bank_start,
	.next = adf_bank_next,
	.stop = adf_bank_stop,
	.show = adf_bank_show
};

/**
 * @macro DEFINE_SEQ_ATTRIBUTE(adf_bank_debug)
 * @brief Helper macro to define debugfs file operations for `adf_bank_debug`.
 * This expands to define `adf_bank_debug_fops` (file operations).
 */
DEFINE_SEQ_ATTRIBUTE(adf_bank_debug);

/**
 * @brief Adds debugfs entries for an ETR bank.
 *
 * Creates a debugfs directory named `bank_XX` (where XX is bank number)
 * and within it, a `config` file that exposes bank and ring status.
 *
 * @param bank Pointer to the `adf_etr_bank_data` structure for the bank.
 * @return 0 on success.
 */
int adf_bank_debugfs_add(struct adf_etr_bank_data *bank)
{
	struct adf_accel_dev *accel_dev = bank->accel_dev;
	struct dentry *parent = accel_dev->transport->debug; // Parent debugfs directory.
	char name[16];

	snprintf(name, sizeof(name), "bank_%02d", bank->bank_number); // Format directory name.
	bank->bank_debug_dir = debugfs_create_dir(name, parent);      // Create debugfs directory.
	bank->bank_debug_cfg = debugfs_create_file("config", S_IRUSR,
							   bank->bank_debug_dir, bank,
						   &adf_bank_debug_fops); // Create config file within.
	return 0;
}

/**
 * @brief Removes the debugfs entries for an ETR bank.
 *
 * @param bank Pointer to the `adf_etr_bank_data` structure.
 */
void adf_bank_debugfs_rm(struct adf_etr_bank_data *bank)
{
	debugfs_remove(bank->bank_debug_cfg); // Remove config file.
	debugfs_remove(bank->bank_debug_dir); // Remove bank directory.
}