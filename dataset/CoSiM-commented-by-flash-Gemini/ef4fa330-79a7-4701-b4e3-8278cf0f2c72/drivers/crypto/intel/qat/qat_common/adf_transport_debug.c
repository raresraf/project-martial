/**
 * @file adf_transport_debug.c
 * @brief This file provides debugfs functionality for Intel QuickAssist Technology (QAT)
 * accelerator transport layer. It allows users to inspect the state of QAT rings and banks
 * through the Linux debugfs interface, aiding in debugging and monitoring QAT operations.
 * It's part of the Linux kernel driver for QAT devices.
 *
 * @license SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0-only)
 * @copyright Copyright(c) 2014 - 2020 Intel Corporation
 */
#include <linux/mutex.h>      // For mutual exclusion locks in the kernel.
#include <linux/slab.h>       // For kernel memory allocation (kzalloc, kfree).
#include <linux/seq_file.h>   // For iterating and displaying data through debugfs.
#include "adf_accel_devices.h"        // Common QAT accelerator device definitions.
#include "adf_transport_internal.h"   // Internal transport layer definitions.
#include "adf_transport_access_macros.h" // Macros for accessing transport layer registers.

/**
 * @brief Static mutex to protect concurrent reads from ring debugfs entries.
 * Ensures that only one reader can access ring data at a time to maintain data consistency.
 */
static DEFINE_MUTEX(ring_read_lock);
/**
 * @brief Static mutex to protect concurrent reads from bank debugfs entries.
 * Ensures that only one reader can access bank data at a time.
 */
static DEFINE_MUTEX(bank_read_lock);

/**
 * @brief Initializes the iteration for a ring's debugfs entry.
 * This function is called when a user reads from a debugfs file representing a QAT ring.
 * It acquires a lock to prevent concurrent access during the read operation.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param pos Current position in the iteration (offset).
 * @return void*: Pointer to the first item to display, or SEQ_START_TOKEN if at the beginning.
 */
static void *adf_ring_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private; // Retrieves ring data from seq_file private data.

	mutex_lock(&ring_read_lock); // Acquires a lock for safe access to ring data.
	// Conditional Logic: If starting a new iteration (pos == 0), return a special token.
	if (*pos == 0)
		return SEQ_START_TOKEN;

	// Conditional Logic: Checks if the current position is beyond the end of the ring data.
	// ADF_SIZE_TO_RING_SIZE_IN_BYTES converts ring size units to bytes.
	// ADF_MSG_SIZE_TO_BYTES converts message size units to bytes.
	// The division gives the total number of messages in the ring.
	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size)))
		return NULL; // End of iteration.

	// Calculates the address of the message at the current position.
	// Increments 'pos' for the next call to adf_ring_next.
	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

/**
 * @brief Advances the iteration for a ring's debugfs entry.
 * This function is called repeatedly to get the next item in the ring.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the previous item (not directly used here for calculation, *pos is).
 * @param pos Current position in the iteration (offset).
 * @return void*: Pointer to the next item to display, or NULL if at the end.
 */
static void *adf_ring_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private; // Retrieves ring data.

	// Conditional Logic: Checks if the next position is beyond the end of the ring data.
	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size))) {
		(*pos)++; // Increment pos even if out of bounds to correctly signal end.
		return NULL; // End of iteration.
	}

	// Calculates the address of the message at the current position.
	// Increments 'pos' for the next call to adf_ring_next.
	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

/**
 * @brief Displays the data for a ring's debugfs entry.
 * This function is called for each item returned by adf_ring_start and adf_ring_next.
 * It either prints the ring's configuration or a hex dump of a ring message.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the current item (or SEQ_START_TOKEN for header).
 * @return int: 0 on success.
 */
static int adf_ring_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_ring_data *ring = sfile->private; // Retrieves ring data.
	struct adf_etr_bank_data *bank = ring->bank;     // Retrieves bank data.
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev); // Hardware CSR operations.
	void __iomem *csr = ring->bank->csr_addr;       // Base address of Control Status Registers.

	// Conditional Logic: If it's the start token, print ring configuration header.
	if (v == SEQ_START_TOKEN) {
		int head, tail, empty;

		// Reads ring head, tail, and empty status from hardware CSRs.
		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
						   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
						   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number); // Empty status register.

		seq_puts(sfile, "------- Ring configuration -------\n"); // Prints a header.
		seq_printf(sfile, "ring name: %s\n",
			   ring->ring_debug->ring_name); // Prints ring debug name.
		seq_printf(sfile, "ring num %d, bank num %d\n",
			   ring->ring_number, ring->bank->bank_number); // Prints ring and bank numbers.
		seq_printf(sfile, "head %x, tail %x, empty: %d\n",
			   head, tail, (empty & 1 << ring->ring_number) // Extracts specific ring's empty status bit.
			   >> ring->ring_number);
		seq_printf(sfile, "ring size %lld, msg size %d\n",
			   (long long)ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size), // Ring size in bytes.
			   ADF_MSG_SIZE_TO_BYTES(ring->msg_size)); // Message size in bytes.
		seq_puts(sfile, "----------- Ring data ------------\n"); // Prints another header.
		return 0;
	}
	// If 'v' is not SEQ_START_TOKEN, it's a pointer to ring data.
	// Prints a hexadecimal dump of the ring message content.
	seq_hex_dump(sfile, "", DUMP_PREFIX_ADDRESS, 32, 4,
		     v, ADF_MSG_SIZE_TO_BYTES(ring->msg_size), false);
	return 0;
}

/**
 * @brief Cleans up after iterating through a ring's debugfs entry.
 * This function is called when the debugfs file reading is complete.
 * It releases the lock acquired in adf_ring_start.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the last item processed.
 */
static void adf_ring_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&ring_read_lock); // Releases the lock.
}

/**
 * @brief Defines the sequence operations for ring debugfs entries.
 * This structure links the generic seq_file operations to the specific
 * functions for QAT ring debugging.
 */
static const struct seq_operations adf_ring_debug_sops = {
	.start = adf_ring_start, // Function to start the iteration.
	.next = adf_ring_next,   // Function to get the next item.
	.stop = adf_ring_stop,   // Function to stop the iteration and cleanup.
	.show = adf_ring_show    // Function to display an item.
};

/**
 * @brief Macro to define seq_file attributes for adf_ring_debug.
 * This macro generates helper functions and structures needed to register
 * seq_file operations with the kernel's debugfs framework.
 */
DEFINE_SEQ_ATTRIBUTE(adf_ring_debug);

/**
 * @brief Adds a debugfs entry for a specific QAT ring.
 * This function creates a debugfs file that allows users to inspect
 * the configuration and data of a QAT ring.
 *
 * @param ring Pointer to the adf_etr_ring_data structure for the ring.
 * @param name The base name for the ring (e.g., "tx_ring").
 * @return int: 0 on success, or a negative error code.
 */
int adf_ring_debugfs_add(struct adf_etr_ring_data *ring, const char *name)
{
	struct adf_etr_ring_debug_entry *ring_debug;
	char entry_name[16];

	// Allocates memory for the ring debug entry.
	ring_debug = kzalloc(sizeof(*ring_debug), GFP_KERNEL);
	if (!ring_debug)
		return -ENOMEM; // Returns error if memory allocation fails.

	strscpy(ring_debug->ring_name, name, sizeof(ring_debug->ring_name)); // Copies the ring name.
	// Formats the debugfs entry name (e.g., "ring_00", "ring_01").
	snprintf(entry_name, sizeof(entry_name), "ring_%02d",
		 ring->ring_number);

	// Creates the debugfs file for the ring.
	// S_IRUSR specifies read permissions for the user.
	// ring->bank->bank_debug_dir is the parent directory.
	// ring is the private data passed to seq_file operations.
	// &adf_ring_debug_fops is the file operations structure generated by DEFINE_SEQ_ATTRIBUTE.
	ring_debug->debug = debugfs_create_file(entry_name, S_IRUSR,
						ring->bank->bank_debug_dir,
						ring, &adf_ring_debug_fops);
	ring->ring_debug = ring_debug; // Stores the debug entry pointer in the ring data structure.
	return 0;
}

/**
 * @brief Removes the debugfs entry for a specific QAT ring.
 *
 * @param ring Pointer to the adf_etr_ring_data structure for the ring.
 */
void adf_ring_debugfs_rm(struct adf_etr_ring_data *ring)
{
	// Conditional Logic: Checks if a debugfs entry exists for the ring.
	if (ring->ring_debug) {
		debugfs_remove(ring->ring_debug->debug); // Removes the debugfs file.
		kfree(ring->ring_debug); // Frees the allocated memory.
		ring->ring_debug = NULL; // Clears the pointer.
	}
}

/**
 * @brief Initializes the iteration for a bank's debugfs entry.
 * Similar to adf_ring_start, but for a QAT bank (which contains multiple rings).
 *
 * @param sfile Pointer to the seq_file structure.
 * @param pos Current position in the iteration (offset).
 * @return void*: Pointer to the first item to display, or SEQ_START_TOKEN if at the beginning.
 */
static void *adf_bank_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private; // Retrieves bank data.
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev); // Number of rings in the bank.

	mutex_lock(&bank_read_lock); // Acquires a lock for safe access to bank data.
	// Conditional Logic: If starting a new iteration (pos == 0), return a special token.
	if (*pos == 0)
		return SEQ_START_TOKEN;

	// Conditional Logic: Checks if the current position is beyond the number of rings in the bank.
	if (*pos >= num_rings_per_bank)
		return NULL; // End of iteration.

	return pos; // Returns the current position as the "item" to process.
}

/**
 * @brief Advances the iteration for a bank's debugfs entry.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the previous item.
 * @param pos Current position in the iteration (offset).
 * @return void*: Pointer to the next item to display, or NULL if at the end.
 */
static void *adf_bank_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private; // Retrieves bank data.
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev); // Number of rings in the bank.

	// Conditional Logic: Increments position and checks if it's beyond the number of rings.
	if (++(*pos) >= num_rings_per_bank)
		return NULL; // End of iteration.

	return pos; // Returns the current position as the "item" to process.
}

/**
 * @brief Displays the data for a bank's debugfs entry.
 * This function displays either a bank's configuration header or individual ring statuses.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the current item (or SEQ_START_TOKEN for header, or pos for ring_id).
 * @return int: 0 on success.
 */
static int adf_bank_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_bank_data *bank = sfile->private; // Retrieves bank data.
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev); // Hardware CSR operations.

	// Conditional Logic: If it's the start token, print bank configuration header.
	if (v == SEQ_START_TOKEN) {
		seq_printf(sfile, "------- Bank %d configuration -------\n",
			   bank->bank_number); // Prints bank number.
	} else {
		// If 'v' is a position, derive the ring ID from it.
		int ring_id = *((int *)v) - 1;
		struct adf_etr_ring_data *ring = &bank->rings[ring_id]; // Retrieves ring data.
		void __iomem *csr = bank->csr_addr; // Base address of CSRs.
		int head, tail, empty;

		// Conditional Logic: Checks if the ring is enabled in the bank's mask.
		if (!(bank->ring_mask & 1 << ring_id))
			return 0; // Skip if ring is not active.

		// Reads ring head, tail, and empty status from hardware CSRs.
		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
						   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
						   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number); // Empty status register.

		seq_printf(sfile,
			   "ring num %02d, head %04x, tail %04x, empty: %d\n",
			   ring->ring_number, head, tail,
			   (empty & 1 << ring->ring_number) >> // Extracts specific ring's empty status bit.
			   ring->ring_number);
	}
	return 0;
}

/**
 * @brief Cleans up after iterating through a bank's debugfs entry.
 * Releases the lock acquired in adf_bank_start.
 *
 * @param sfile Pointer to the seq_file structure.
 * @param v Pointer to the last item processed.
 */
static void adf_bank_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&bank_read_lock); // Releases the lock.
}

/**
 * @brief Defines the sequence operations for bank debugfs entries.
 * This structure links the generic seq_file operations to the specific
 * functions for QAT bank debugging.
 */
static const struct seq_operations adf_bank_debug_sops = {
	.start = adf_bank_start, // Function to start the iteration.
	.next = adf_bank_next,   // Function to get the next item.
	.stop = adf_bank_stop,   // Function to stop the iteration and cleanup.
	.show = adf_bank_show    // Function to display an item.
};

/**
 * @brief Macro to define seq_file attributes for adf_bank_debug.
 * This macro generates helper functions and structures needed to register
 * seq_file operations with the kernel's debugfs framework.
 */
DEFINE_SEQ_ATTRIBUTE(adf_bank_debug);

/**
 * @brief Adds debugfs entries for a specific QAT bank.
 * This function creates a directory and a "config" file within debugfs
 * that allows users to inspect the configuration and ring statuses of a QAT bank.
 *
 * @param bank Pointer to the adf_etr_bank_data structure for the bank.
 * @return int: 0 on success, or a negative error code.
 */
int adf_bank_debugfs_add(struct adf_etr_bank_data *bank)
{
	struct adf_accel_dev *accel_dev = bank->accel_dev; // Accelerator device.
	struct dentry *parent = accel_dev->transport->debug; // Parent debugfs directory.
	char name[16];

	// Formats the debugfs directory name (e.g., "bank_00", "bank_01").
	snprintf(name, sizeof(name), "bank_%02d", bank->bank_number);
	// Creates the debugfs directory for the bank.
	bank->bank_debug_dir = debugfs_create_dir(name, parent);
	// Creates the "config" file within the bank's debugfs directory.
	bank->bank_debug_cfg = debugfs_create_file("config", S_IRUSR,
						   bank->bank_debug_dir, bank,
						   &adf_bank_debug_fops);
	return 0;
}

/**
 * @brief Removes the debugfs entries for a specific QAT bank.
 *
 * @param bank Pointer to the adf_etr_bank_data structure for the bank.
 */
void adf_bank_debugfs_rm(struct adf_etr_bank_data *bank)
{
	debugfs_remove(bank->bank_debug_cfg); // Removes the "config" file.
	debugfs_remove(bank->bank_debug_dir); // Removes the bank's debugfs directory.
}
