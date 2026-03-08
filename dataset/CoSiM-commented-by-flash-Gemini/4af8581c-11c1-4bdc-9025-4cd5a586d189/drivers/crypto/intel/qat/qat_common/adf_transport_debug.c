// SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0-only)
/* Copyright(c) 2014 - 2020 Intel Corporation */
/**
 * @4af8581c-11c1-4bdc-9025-4cd5a586d189/drivers/crypto/intel/qat/qat_common/adf_transport_debug.c
 * @brief Implements debugfs entries for Intel QuickAssist Technology (QAT)
 *        accelerator transport rings and banks.
 * Architectural Intent: To provide a user-accessible interface (via debugfs)
 *        for monitoring the internal state and configuration of QAT hardware
 *        rings and banks. This facilitates debugging, performance analysis,
 *        and operational introspection of the QAT driver's transport layer.
 */
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/seq_file.h>
#include "adf_accel_devices.h"
#include "adf_transport_internal.h"
#include "adf_transport_access_macros.h"

static DEFINE_MUTEX(ring_read_lock); // Mutex to protect concurrent reads from ring debugfs entries.
static DEFINE_MUTEX(bank_read_lock); // Mutex to protect concurrent reads from bank debugfs entries.

static void *adf_ring_start(struct seq_file *sfile, loff_t *pos)
{
	/**
	 * Functional Utility: Initiates the iteration over a QAT ring's data for `seq_file` output.
	 * This function is part of the `seq_operations` structure, providing a starting point
	 * for reading the contents of a QAT ring buffer via debugfs.
	 *
	 * @param sfile: Pointer to the `seq_file` structure.
	 * @param pos: Pointer to the file offset, used to track iteration progress.
	 * @return: A pointer to the first message in the ring buffer, `SEQ_START_TOKEN` for header, or NULL if out of bounds.
	 */
	struct adf_etr_ring_data *ring = sfile->private; // Retrieve the ring data from `seq_file` private data.

	mutex_lock(&ring_read_lock); // Acquire mutex to prevent concurrent reads to the ring.
	if (*pos == 0)
		return SEQ_START_TOKEN; // Special token to indicate printing a header first.

	// Check if the current position is within the bounds of the ring buffer.
	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size)))
		return NULL; // Out of bounds, no more data to read.

	// Calculate the address of the message at the current position and increment position for the next call.
	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

static void *adf_ring_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	/**
	 * Functional Utility: Advances the iteration over a QAT ring's data for `seq_file` output.
	 * This function is part of the `seq_operations` structure, providing the next item
	 * in the ring buffer. It is called repeatedly after `adf_ring_start` until it returns NULL.
	 *
	 * @param sfile: Pointer to the `seq_file` structure.
	 * @param v: The previously returned item from `adf_ring_start` or `adf_ring_next`.
	 * @param pos: Pointer to the file offset, updated to track iteration progress.
	 * @return: A pointer to the next message in the ring buffer, or NULL if there are no more messages.
	 */
	struct adf_etr_ring_data *ring = sfile->private; // Retrieve the ring data.

	// Check if the current position (which was incremented in adf_ring_start or previous adf_ring_next)
	// is beyond the last valid message in the ring.
	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size))) {
		(*pos)++; // Increment position to ensure `adf_ring_stop` is eventually called.
		return NULL; // Out of bounds, no more data to read.
	}

	// Calculate the address of the message at the current position and increment position for the next call.
	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

static int adf_ring_show(struct seq_file *sfile, void *v)
{
	/**
	 * Functional Utility: Displays the configuration and contents of a QAT ring.
	 * This function is responsible for formatting and printing both summary
	 * information (like head, tail, and size) and raw message data from the ring
	 * buffer to the `seq_file`.
	 *
	 * @param sfile: Pointer to the `seq_file` structure for output.
	 * @param v: Pointer to the current item being processed (either `SEQ_START_TOKEN` or a ring message).
	 * @return: 0 on success.
	 */
	struct adf_etr_ring_data *ring = sfile->private;       // Retrieve the ring data.
	struct adf_etr_bank_data *bank = ring->bank;           // Retrieve the bank data associated with the ring.
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev); // Get CSR (Control/Status Register) operations.
	void __iomem *csr = ring->bank->csr_addr;              // Get the base address for CSR access.

	if (v == SEQ_START_TOKEN) { // If it's the start token, print ring configuration header.
		int head, tail, empty;

		// Read the hardware ring head and tail pointers, and empty status.
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
			   head, tail, (empty & 1 << ring->ring_number)
			   >> ring->ring_number); // Extract empty status for this specific ring.
		seq_printf(sfile, "ring size %lld, msg size %d\n",
			   (long long)ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size),
			   ADF_MSG_SIZE_TO_BYTES(ring->msg_size));
		seq_puts(sfile, "----------- Ring data ------------\n");
		return 0; // Header printed.
	}
	// If not the start token, it's a pointer to a message in the ring; print a hex dump of the message.
	seq_hex_dump(sfile, "", DUMP_PREFIX_ADDRESS, 32, 4,
		     v, ADF_MSG_SIZE_TO_BYTES(ring->msg_size), false);
	return 0; // Message data printed.
}

static void adf_ring_stop(struct seq_file *sfile, void *v)
{
	/**
	 * Functional Utility: Concludes the iteration over a QAT ring's data for `seq_file` output.
	 * This function is part of the `seq_operations` structure and is called after the
	 * `seq_file` has finished reading from the ring, or when an error occurs.
	 * Its primary responsibility is to release any held resources, specifically the mutex.
	 *
	 * @param sfile: Pointer to the `seq_file` structure.
	 * @param v: The last item returned by `adf_ring_next`, or NULL.
	 */
	mutex_unlock(&ring_read_lock); // Release the mutex acquired in `adf_ring_start`.
}

static const struct seq_operations adf_ring_debug_sops = {
	/**
	 * @brief Defines the `seq_operations` for a QAT ring debugfs entry.
	 * This structure specifies the functions used by the `seq_file` interface
	 * to iterate through and display the contents of a QAT ring.
	 * @field start: Function to initialize iteration and return the first element.
	 * @field next: Function to advance iteration and return the next element.
	 * @field stop: Function to clean up resources after iteration.
	 * @field show: Function to display a single element.
	 */
	.start = adf_ring_start,
	.next = adf_ring_next,
	.stop = adf_ring_stop,
	.show = adf_ring_show
};

DEFINE_SEQ_ATTRIBUTE(adf_ring_debug);

int adf_ring_debugfs_add(struct adf_etr_ring_data *ring, const char *name)
{
	/**
	 * Functional Utility: Creates a debugfs entry for a QAT ring.
	 * This function allocates memory for debug-specific ring data,
	 * assigns a name, and registers a file in debugfs that, when read,
	 * will expose the ring's configuration and contents via `seq_file`.
	 *
	 * @param ring: Pointer to the `adf_etr_ring_data` structure for which to create the debugfs entry.
	 * @param name: The name to assign to the debugfs entry.
	 * @return: 0 on success, or a negative errno on failure.
	 */
	struct adf_etr_ring_debug_entry *ring_debug;
	char entry_name[16];

	// Allocate memory for the ring's debug entry structure.
	ring_debug = kzalloc(sizeof(*ring_debug), GFP_KERNEL);
	if (!ring_debug)
		return -ENOMEM; // Return error if memory allocation fails.

	strscpy(ring_debug->ring_name, name, sizeof(ring_debug->ring_name)); // Copy the provided name.
	snprintf(entry_name, sizeof(entry_name), "ring_%02d",
		 ring->ring_number); // Create a unique name for the debugfs file.

	// Create the debugfs file. When this file is read, `adf_ring_debug_fops` (which uses `adf_ring_debug_sops`)
	// will be used to generate its content.
	ring_debug->debug = debugfs_create_file(entry_name, S_IRUSR,
						ring->bank->bank_debug_dir, // Parent directory for the debugfs entry.
						ring, &adf_ring_debug_fops); // Associate ring data and file operations.
	ring->ring_debug = ring_debug; // Store the debug entry in the ring data structure.
	return 0;                      // Success.
}

void adf_ring_debugfs_rm(struct adf_etr_ring_data *ring)
{
	/**
	 * Functional Utility: Removes the debugfs entry for a QAT ring and frees associated memory.
	 * This function cleans up the debugfs entry created by `adf_ring_debugfs_add`
	 * and releases the memory allocated for the `ring_debug` structure.
	 *
	 * @param ring: Pointer to the `adf_etr_ring_data` structure.
	 */
	if (ring->ring_debug) { // Check if a debugfs entry exists for this ring.
		debugfs_remove(ring->ring_debug->debug); // Remove the debugfs file.
		kfree(ring->ring_debug);                 // Free the memory for the debug structure.
		ring->ring_debug = NULL;                 // Clear the pointer to prevent stale references.
	}
}

static void *adf_bank_start(struct seq_file *sfile, loff_t *pos)
{
	/**
	 * Functional Utility: Initiates the iteration over a QAT bank's rings for `seq_file` output.
	 * This function is part of the `seq_operations` structure, providing a starting point
	 * for displaying configuration information about a QAT bank and its rings via debugfs.
	 *
	 * @param sfile: Pointer to the `seq_file` structure.
	 * @param pos: Pointer to the file offset, used to track iteration progress.
	 * @return: A pointer to `SEQ_START_TOKEN` for header, or `pos` for actual ring data, or NULL if out of bounds.
	 */
	struct adf_etr_bank_data *bank = sfile->private; // Retrieve the bank data from `seq_file` private data.
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev); // Get the number of rings per bank.

	mutex_lock(&bank_read_lock); // Acquire mutex to prevent concurrent reads to the bank.
	if (*pos == 0)
		return SEQ_START_TOKEN; // Special token to indicate printing a header first.

	// Check if the current position is within the bounds of the rings for this bank.
	if (*pos >= num_rings_per_bank)
		return NULL; // Out of bounds, no more rings to read.

	return pos; // Return the position itself to be passed to `adf_bank_next` and `adf_bank_show`.
}

static void *adf_bank_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	/**
	 * Functional Utility: Advances the iteration over a QAT bank's rings for `seq_file` output.
	 * This function is part of the `seq_operations` structure, providing the next item
	 * (which is the next ring's data) in the bank. It is called repeatedly after `adf_bank_start`
	 * until it returns NULL.
	 *
	 * @param sfile: Pointer to the `seq_file` structure.
	 * @param v: The previously returned item from `adf_bank_start` or `adf_bank_next` (the `pos` pointer).
	 * @param pos: Pointer to the file offset, updated to track iteration progress.
	 * @return: A pointer to the updated `pos` for the next ring, or NULL if there are no more rings.
	 */
	struct adf_etr_bank_data *bank = sfile->private; // Retrieve the bank data.
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev); // Get the number of rings per bank.

	if (++(*pos) >= num_rings_per_bank) // Increment position and check if it's within bounds.
		return NULL; // Out of bounds, no more rings to read.

	return pos; // Return the updated position for the next ring.
}

static int adf_bank_show(struct seq_file *sfile, void *v)
{
	/**
	 * Functional Utility: Displays configuration and status information for a QAT bank and its rings.
	 * This function formats and prints either a header for the bank or detailed
	 * status for individual rings within the bank, including head, tail, and empty status.
	 *
	 * @param sfile: Pointer to the `seq_file` structure for output.
	 * @param v: Pointer to the current item being processed (either `SEQ_START_TOKEN` or a pointer to the ring index).
	 * @return: 0 on success.
	 */
	struct adf_etr_bank_data *bank = sfile->private; // Retrieve the bank data.
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev); // Get CSR operations.

	if (v == SEQ_START_TOKEN) { // If it's the start token, print bank configuration header.
		seq_printf(sfile, "------- Bank %d configuration -------\n",
			   bank->bank_number);
	} else { // Otherwise, 'v' points to the current ring number to display.
		int ring_id = *((int *)v) - 1; // Retrieve the ring ID (adjust for 0-based indexing if necessary).
		struct adf_etr_ring_data *ring = &bank->rings[ring_id]; // Get the specific ring data.
		void __iomem *csr = bank->csr_addr; // Get the base address for CSR access.
		int head, tail, empty;

		// Check if this specific ring is enabled in the bank's ring mask.
		if (!(bank->ring_mask & 1 << ring_id))
			return 0; // If not enabled, skip and return.

		// Read the hardware ring head and tail pointers, and empty status.
		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
						   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
						   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number);

		// Print detailed information for the individual ring.
		seq_printf(sfile,
			   "ring num %02d, head %04x, tail %04x, empty: %d\n",
			   ring->ring_number, head, tail,
			   (empty & 1 << ring->ring_number) >>
			   ring->ring_number); // Extract empty status for this specific ring.
	}
	return 0; // Indicate success.
}

static void adf_bank_stop(struct seq_file *sfile, void *v)
{
	/**
	 * Functional Utility: Concludes the iteration over a QAT bank's rings for `seq_file` output.
	 * This function is part of the `seq_operations` structure and is called after the
	 * `seq_file` has finished reading from the bank, or when an error occurs.
	 * Its primary responsibility is to release any held resources, specifically the mutex.
	 *
	 * @param sfile: Pointer to the `seq_file` structure.
	 * @param v: The last item returned by `adf_bank_next`, or NULL.
	 */
	mutex_unlock(&bank_read_lock); // Release the mutex acquired in `adf_bank_start`.
}

static const struct seq_operations adf_bank_debug_sops = {
	/**
	 * @brief Defines the `seq_operations` for a QAT bank debugfs entry.
	 * This structure specifies the functions used by the `seq_file` interface
	 * to iterate through and display the configuration and status of a QAT bank.
	 * @field start: Function to initialize iteration and return the first element.
	 * @field next: Function to advance iteration and return the next element.
	 * @field stop: Function to clean up resources after iteration.
	 * @field show: Function to display a single element.
	 */
	.start = adf_bank_start,
	.next = adf_bank_next,
	.stop = adf_bank_stop,
	.show = adf_bank_show
};

DEFINE_SEQ_ATTRIBUTE(adf_bank_debug);

int adf_bank_debugfs_add(struct adf_etr_bank_data *bank)
{
	/**
	 * Functional Utility: Creates debugfs entries for a QAT bank.
	 * This function creates a directory for the bank and a configuration file
	 * within that directory. The configuration file, when read, will expose
	 * the bank's operational parameters and the status of its rings.
	 *
	 * @param bank: Pointer to the `adf_etr_bank_data` structure for which to create debugfs entries.
	 * @return: 0 on success.
	 */
	struct adf_accel_dev *accel_dev = bank->accel_dev; // Get the accelerator device.
	struct dentry *parent = accel_dev->transport->debug; // Get the parent debugfs directory.
	char name[16];

	snprintf(name, sizeof(name), "bank_%02d", bank->bank_number); // Create a unique name for the bank's debugfs directory.
	bank->bank_debug_dir = debugfs_create_dir(name, parent);      // Create the debugfs directory for the bank.
	// Create the "config" file within the bank's directory.
	// When this file is read, `adf_bank_debug_fops` (which uses `adf_bank_debug_sops`)
	// will be used to generate its content.
	bank->bank_debug_cfg = debugfs_create_file("config", S_IRUSR,
						   bank->bank_debug_dir, bank,
						   &adf_bank_debug_fops);
	return 0; // Success.
}

void adf_bank_debugfs_rm(struct adf_etr_bank_data *bank)
{
	/**
	 * Functional Utility: Removes the debugfs entries for a QAT bank.
	 * This function cleans up the debugfs files and directories created
	 * by `adf_bank_debugfs_add`, ensuring proper resource release.
	 *
	 * @param bank: Pointer to the `adf_etr_bank_data` structure.
	 */
	debugfs_remove(bank->bank_debug_cfg); // Remove the debugfs config file.
	debugfs_remove(bank->bank_debug_dir); // Remove the debugfs directory for the bank.
}
