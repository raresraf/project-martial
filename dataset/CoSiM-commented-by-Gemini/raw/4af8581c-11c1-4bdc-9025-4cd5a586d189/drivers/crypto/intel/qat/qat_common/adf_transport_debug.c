// SPDX-License-Identifier: (BSD-3-Clause OR GPL-2.0-only)
/* Copyright(c) 2014 - 2020 Intel Corporation */
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/seq_file.h>
#include "adf_accel_devices.h"
#include "adf_transport_internal.h"
#include "adf_transport_access_macros.h"

/* Mutexes to protect against concurrent reads of debugfs files. */
static DEFINE_MUTEX(ring_read_lock);
static DEFINE_MUTEX(bank_read_lock);

/**
 * adf_ring_start() - seq_file start operation for ring debugging.
 * @sfile: The sequence file.
 * @pos: The current position in the sequence.
 *
 * This function is called at the beginning of a read from the debugfs file.
 * It locks the ring for reading and returns the first element for the sequence,
 * which is either a special start token or a pointer to a message in the ring.
 *
 * Return: A pointer to the current message or a token, NULL if at the end.
 */
static void *adf_ring_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private;

	mutex_lock(&ring_read_lock);
	if (*pos == 0)
		return SEQ_START_TOKEN;

	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size)))
		return NULL;

	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

/**
 * adf_ring_next() - seq_file next operation for ring debugging.
 * @sfile: The sequence file.
 * @v: The current element in the sequence.
 * @pos: The current position in the sequence.
 *
 * This function moves to the next message in the ring for sequential reading.
 *
 * Return: A pointer to the next message, or NULL if at the end of the ring.
 */
static void *adf_ring_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_ring_data *ring = sfile->private;

	if (*pos >= (ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size) /
		     ADF_MSG_SIZE_TO_BYTES(ring->msg_size))) {
		(*pos)++;
		return NULL;
	}

	return ring->base_addr +
		(ADF_MSG_SIZE_TO_BYTES(ring->msg_size) * (*pos)++);
}

/**
 * adf_ring_show() - seq_file show operation for ring debugging.
 * @sfile: The sequence file.
 * @v: The current element in the sequence to be shown.
 *
 * This function formats the output for the debugfs file. For the first
 * element, it prints a header with ring configuration details read from
 * hardware. For subsequent elements, it prints a hex dump of the ring message.
 *
 * Return: 0 on success.
 */
static int adf_ring_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_ring_data *ring = sfile->private;
	struct adf_etr_bank_data *bank = ring->bank;
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev);
	void __iomem *csr = ring->bank->csr_addr;

	if (v == SEQ_START_TOKEN) {
		int head, tail, empty;

		/* Read CSRs to get the current state of the ring. */
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
			   >> ring->ring_number);
		seq_printf(sfile, "ring size %lld, msg size %d\n",
			   (long long)ADF_SIZE_TO_RING_SIZE_IN_BYTES(ring->ring_size),
			   ADF_MSG_SIZE_TO_BYTES(ring->msg_size));
		seq_puts(sfile, "----------- Ring data ------------\n");
		return 0;
	}
	/* Dump the content of the ring message. */
	seq_hex_dump(sfile, "", DUMP_PREFIX_ADDRESS, 32, 4,
		     v, ADF_MSG_SIZE_TO_BYTES(ring->msg_size), false);
	return 0;
}

/**
 * adf_ring_stop() - seq_file stop operation for ring debugging.
 * @sfile: The sequence file.
 * @v: The current element in the sequence.
 *
 * This function is called at the end of a read from the debugfs file.
 * It releases the lock taken in adf_ring_start().
 */
static void adf_ring_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&ring_read_lock);
}

/* seq_operations structure for ring debugfs files. */
static const struct seq_operations adf_ring_debug_sops = {
	.start = adf_ring_start,
	.next = adf_ring_next,
	.stop = adf_ring_stop,
	.show = adf_ring_show
};

DEFINE_SEQ_ATTRIBUTE(adf_ring_debug);

/**
 * adf_ring_debugfs_add() - Creates a debugfs file for a transport ring.
 * @ring: The ring data structure to create the debugfs file for.
 * @name: A descriptive name for the ring.
 *
 * This function creates a file (e.g., "ring_02") in the appropriate bank
 * directory in debugfs, allowing the inspection of the ring's state and content.
 *
 * Return: 0 on success, or a negative error code on failure.
 */
int adf_ring_debugfs_add(struct adf_etr_ring_data *ring, const char *name)
{
	struct adf_etr_ring_debug_entry *ring_debug;
	char entry_name[16];

	ring_debug = kzalloc(sizeof(*ring_debug), GFP_KERNEL);
	if (!ring_debug)
		return -ENOMEM;

	strscpy(ring_debug->ring_name, name, sizeof(ring_debug->ring_name));
	snprintf(entry_name, sizeof(entry_name), "ring_%02d",
		 ring->ring_number);

	ring_debug->debug = debugfs_create_file(entry_name, S_IRUSR,
						ring->bank->bank_debug_dir,
						ring, &adf_ring_debug_fops);
	ring->ring_debug = ring_debug;
	return 0;
}

/**
 * adf_ring_debugfs_rm() - Removes the debugfs file for a transport ring.
 * @ring: The ring data structure whose debugfs file should be removed.
 */
void adf_ring_debugfs_rm(struct adf_etr_ring_data *ring)
{
	if (ring->ring_debug) {
		debugfs_remove(ring->ring_debug->debug);
		kfree(ring->ring_debug);
		ring->ring_debug = NULL;
	}
}

/**
 * adf_bank_start() - seq_file start operation for bank debugging.
 * @sfile: The sequence file.
 * @pos: The current position in the sequence.
 *
 * Prepares for iterating over the rings within a bank.
 *
 * Return: A pointer-like object for the iterator, or NULL if at the end.
 */
static void *adf_bank_start(struct seq_file *sfile, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private;
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev);

	mutex_lock(&bank_read_lock);
	if (*pos == 0)
		return SEQ_START_TOKEN;

	if (*pos >= num_rings_per_bank)
		return NULL;

	return pos;
}

/**
 * adf_bank_next() - seq_file next operation for bank debugging.
 * @sfile: The sequence file.
 * @v: The current element in the sequence.
 * @pos: The current position in the sequence.
 *
 * Moves the iterator to the next ring in the bank.
 *
 * Return: A pointer-like object for the iterator, or NULL if at the end.
 */
static void *adf_bank_next(struct seq_file *sfile, void *v, loff_t *pos)
{
	struct adf_etr_bank_data *bank = sfile->private;
	u8 num_rings_per_bank = GET_NUM_RINGS_PER_BANK(bank->accel_dev);

	if (++(*pos) >= num_rings_per_bank)
		return NULL;

	return pos;
}

/**
 * adf_bank_show() - seq_file show operation for bank debugging.
 * @sfile: The sequence file.
 * @v: The current element (ring) to be shown.
 *
 * Formats the output for the bank's "config" debugfs file. It prints a
 * header and then a one-line summary for each active ring in the bank,
 * showing its head, tail, and empty status.
 *
 * Return: 0 on success.
 */
static int adf_bank_show(struct seq_file *sfile, void *v)
{
	struct adf_etr_bank_data *bank = sfile->private;
	struct adf_hw_csr_ops *csr_ops = GET_CSR_OPS(bank->accel_dev);

	if (v == SEQ_START_TOKEN) {
		seq_printf(sfile, "------- Bank %d configuration -------\n",
			   bank->bank_number);
	} else {
		int ring_id = *((int *)v) - 1;
		struct adf_etr_ring_data *ring = &bank->rings[ring_id];
		void __iomem *csr = bank->csr_addr;
		int head, tail, empty;

		if (!(bank->ring_mask & 1 << ring_id))
			return 0;

		head = csr_ops->read_csr_ring_head(csr, bank->bank_number,
						   ring->ring_number);
		tail = csr_ops->read_csr_ring_tail(csr, bank->bank_number,
						   ring->ring_number);
		empty = csr_ops->read_csr_e_stat(csr, bank->bank_number);

		seq_printf(sfile,
			   "ring num %02d, head %04x, tail %04x, empty: %d\n",
			   ring->ring_number, head, tail,
			   (empty & 1 << ring->ring_number) >>
			   ring->ring_number);
	}
	return 0;
}

/**
 * adf_bank_stop() - seq_file stop operation for bank debugging.
 * @sfile: The sequence file.
 * @v: The current element in the sequence.
 *
 * Releases the lock taken in adf_bank_start().
 */
static void adf_bank_stop(struct seq_file *sfile, void *v)
{
	mutex_unlock(&bank_read_lock);
}

/* seq_operations structure for bank debugfs files. */
static const struct seq_operations adf_bank_debug_sops = {
	.start = adf_bank_start,
	.next = adf_bank_next,
	.stop = adf_bank_stop,
	.show = adf_bank_show
};

DEFINE_SEQ_ATTRIBUTE(adf_bank_debug);

/**
 * adf_bank_debugfs_add() - Creates debugfs entries for a transport bank.
 * @bank: The bank data structure.
 *
 * This function creates a directory for the bank (e.g., "bank_01") and a
 * "config" file inside it that provides a summary of all rings in the bank.
 *
 * Return: 0 on success.
 */
int adf_bank_debugfs_add(struct adf_etr_bank_data *bank)
{
	struct adf_accel_dev *accel_dev = bank->accel_dev;
	struct dentry *parent = accel_dev->transport->debug;
	char name[16];

	snprintf(name, sizeof(name), "bank_%02d", bank->bank_number);
	bank->bank_debug_dir = debugfs_create_dir(name, parent);
	bank->bank_debug_cfg = debugfs_create_file("config", S_IRUSR,
						   bank->bank_debug_dir, bank,
						   &adf_bank_debug_fops);
	return 0;
}

/**
 * adf_bank_debugfs_rm() - Removes the debugfs entries for a transport bank.
 * @bank: The bank data structure.
 */
void adf_bank_debugfs_rm(struct adf_etr_bank_data *bank)
{
	debugfs_remove(bank->bank_debug_cfg);
	debugfs_remove(bank->bank_debug_dir);
}
