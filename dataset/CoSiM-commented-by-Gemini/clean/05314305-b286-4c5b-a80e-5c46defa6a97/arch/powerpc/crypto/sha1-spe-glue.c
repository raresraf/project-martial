// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file sha1-spe-glue.c
 * @brief Glue code for SHA-1 implementation using PowerPC SPE instructions.
 * @details This file provides the C-level interface (glue code) to integrate
 * the highly optimized SHA-1 (Secure Hash Algorithm 1) implementation,
 * which leverages the PowerPC Signal Processing Engine (SPE) instruction set,
 * with the Linux kernel's cryptographic API (`shash`). Its primary
 * responsibilities include:
 * - **SPE Context Management**: Ensuring safe and proper saving/restoring of
 *   the SPE unit's context when entering and exiting SPE-accelerated sections
 *   (`spe_begin()` and `spe_end()`).
 * - **Block Processing**: Orchestrating the processing of input data in chunks
 *   to the SPE assembly routine, while respecting kernel preemption limits.
 * - **Integration with Crypto API**: Providing the standard `update` and `finup`
 *   operations required by the `shash` API, delegating the core work to the
 *   SPE-accelerated assembly.
 * This implementation aims to provide a high-performance SHA-1 solution for
 * PowerPC platforms with SPE capabilities.
 */
/*
 * Glue code for SHA-1 implementation for SPE instructions (PPC)
 *
 * Based on generic implementation.
 *
 * Copyright (c) 2015 Markus Stockhausen <stockhausen@collogia.de>
 */

#include <asm/switch_to.h>
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/kernel.h>
#include <linux/preempt.h>
#include <linux/module.h>

/**
 * @brief Defines the maximum number of bytes allowed to be processed in a critical section.
 * @details This constant (`MAX_BYTES`) specifies the maximum amount of input data
 * that can be processed by the PowerPC SPE-accelerated SHA-1 implementation
 * within a `preempt_disable()` / `preempt_enable()` block. This is a critical
 * performance and latency consideration in a real-time operating system like
 * Linux. Although SPE operations are fast, processing very large chunks of
 * data with preemption disabled can introduce undesirable latency. SHA-1
 * typically takes ~1000 operations per 64 bytes. For e500 cores, which can
 * issue two arithmetic instructions per clock cycle, 2KB of input data
 * (32 blocks) results in approximately 18,000 cycles. On a 667 MHz core, this
 * translates to a critical window of less than 27 microseconds, which is an
 * acceptable maximum latency for such critical sections.
 */
#define MAX_BYTES 2048

/**
 * @brief External declaration for the PowerPC SPE SHA-1 transform function.
 * @details This function is implemented in assembly (`sha1-spe-asm.S`) and
 * performs the core SHA-1 block transformation using PowerPC SPE instructions.
 * It takes the SHA-1 state, input data, and the number of blocks to process.
 *
 * @param state Pointer to the SHA-1 state array (H0-H4).
 * @param src Pointer to the input data blocks.
 * @param blocks Number of 64-byte blocks to process.
 * Functional Utility: Dispatches to the PowerPC SPE assembly routine for hardware-accelerated SHA-1 block processing.
 */
asmlinkage void ppc_spe_sha1_transform(u32 *state, const u8 *src, u32 blocks);

/**
 * @brief Prepares the system for PowerPC SPE operations.
 * @details This function is called before initiating any PowerPC SPE-accelerated
 * cryptographic operations. It first disables kernel preemption to ensure that
 * the CPU context remains stable during the SPE execution. Then, it enables
 * kernel-mode access to the SPE unit, which might involve saving the current
 * SPE registers (if a user-space process was using it) and preparing the hardware
 * for kernel usage.
 * Functional Utility: Disables preemption and enables kernel access to the PowerPC SPE unit.
 */
static void spe_begin(void)
{
	// Functional Utility: Disables kernel preemption to prevent context switches during SPE operations.
	preempt_disable();
	// Functional Utility: Enables the PowerPC SPE unit for kernel mode access.
	enable_kernel_spe();
}

/**
 * @brief Cleans up and restores the system after PowerPC SPE operations.
 * @details This function is called after completing PowerPC SPE-accelerated
 * cryptographic operations. It first disables kernel-mode access to the SPE unit,
 * which might involve restoring any user-space SPE context that was saved. Then,
 * it re-enables kernel preemption, allowing the scheduler to resume normal
 * operation.
 * Functional Utility: Disables kernel SPE access and re-enables preemption.
 */
static void spe_end(void)
{
	// Functional Utility: Disables the PowerPC SPE unit for kernel mode access.
	disable_kernel_spe();
	// Functional Utility: Re-enables kernel preemption, allowing context switches.
	preempt_enable();
}

/**
 * @brief Processes SHA-1 blocks in chunks, managing SPE context.
 * @details This function processes input data in chunks, ensuring that each
 * block of data passed to the PowerPC SPE assembly routine (`ppc_spe_sha1_transform`)
 * respects the `MAX_BYTES` limit for critical sections. It repeatedly calls
 * `spe_begin()` and `spe_end()` to manage the SPE unit's context around each
 * chunk of processing, thereby preventing preemption-related latency issues
 * and ensuring safe hardware access.
 *
 * @param sctx Pointer to the SHA-1 state structure.
 * @param src Pointer to the input data.
 * @param blocks The total number of 64-byte blocks to process.
 * Functional Utility: Orchestrates the processing of SHA-1 blocks using PowerPC SPE, respecting preemption limits.
 */
static void ppc_spe_sha1_block(struct sha1_state *sctx, const u8 *src,
			       int blocks)
{
	// Block Logic: Processes input data in chunks of `MAX_BYTES` or less.
	do {
		// Functional Utility: Determines the number of blocks to process in the current chunk, limited by MAX_BYTES.
		int unit = min(blocks, MAX_BYTES / SHA1_BLOCK_SIZE);

		// Functional Utility: Prepares the system for SPE operations (disables preemption, enables SPE).
		spe_begin();
		// Functional Utility: Invokes the PowerPC SPE assembly routine to transform a chunk of SHA-1 blocks.
		ppc_spe_sha1_transform(sctx->state, src, unit);
		// Functional Utility: Cleans up after SPE operations (disables SPE, re-enables preemption).
		spe_end();

		// Functional Utility: Advances the source pointer and decrements the remaining block count.
		src += unit * SHA1_BLOCK_SIZE;
		blocks -= unit;
	} while (blocks); // Functional Utility: Continues looping until all blocks are processed.
}

/**
 * @brief Implements the `shash` 'update' operation for PowerPC SPE SHA-1.
 * @details This function integrates the PowerPC SPE-accelerated SHA-1 block
 * processing into the kernel's generic `shash` API for incremental updates.
 * It acts as a straightforward wrapper that delegates data buffering and
 * partial block processing to the `sha1_base_do_update_blocks` helper function,
 * providing `ppc_spe_sha1_block` as the callback for core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-1 hash state incrementally, leveraging PowerPC SPE for core block processing.
 */
static int ppc_spe_sha1_update(struct shash_desc *desc, const u8 *data,
			unsigned int len)
{
	// Functional Utility: Delegates data buffering and block processing to the base SHA-1 update function,
	// using the SPE-accelerated `ppc_spe_sha1_block` as the core block transformation.
	return sha1_base_do_update_blocks(desc, data, len, ppc_spe_sha1_block);
}

/**
 * @brief Implements the `shash` 'finup' (finalize and update) operation for PowerPC SPE SHA-1.
 * @details This function handles the final steps of the SHA-1 algorithm. It applies
 * the necessary padding to the last data block, appends the total message length,
 * and then processes this final block using the PowerPC SPE for hardware-
 * accelerated compression. It utilizes the `sha1_base_do_finup` and
 * `sha1_base_finish` helper functions, providing `ppc_spe_sha1_block` for the
 * core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param src Pointer to any remaining partial input data.
 * @param len The length of the remaining partial input data.
 * @param out The buffer to store the final 20-byte SHA-1 hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-1 hashing process, including padding and length encoding, using PowerPC SPE hardware.
 */
static int ppc_spe_sha1_finup(struct shash_desc *desc, const u8 *src,
			      unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-1 padding using the SPE-accelerated block transformation.
	sha1_base_do_finup(desc, src, len, ppc_spe_sha1_block);
	// Functional Utility: Writes the final computed SHA-1 hash digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the PowerPC SPE-accelerated SHA-1 algorithm for the crypto API.
 * @details This structure registers the SHA-1 algorithm implementation that leverages
 * the PowerPC Signal Processing Engine (SPE). It specifies the algorithm's
 * properties (digest size, block size), associates the core operations (`init`,
 * `update`, `finup`) with their respective handler functions, and sets a high
 * priority (`300`) to ensure this hardware-accelerated version is preferred
 * over generic software implementations.
 * Functional Utility: Registers the PowerPC SPE hardware-accelerated SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	ppc_spe_sha1_update,
	.finup		=	ppc_spe_sha1_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"sha1-ppc-spe",
		.cra_priority	=	300,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Module initialization function for PowerPC SPE SHA-1.
 * @details This function is the entry point when the kernel module is loaded.
 * It registers the PowerPC SPE-accelerated SHA-1 algorithm (`alg` structure)
 * with the Linux kernel's cryptographic API. This makes the `sha1-ppc-spe`
 * driver available to the system, enabling high-performance SHA-1 hashing
 * on PowerPC platforms with SPE capabilities.
 * Functional Utility: Registers the PowerPC SPE hardware-accelerated SHA-1 algorithm with the kernel crypto API.
 * @return 0 on successful registration, or an error code on failure.
 */
static int __init ppc_spe_sha1_mod_init(void)
{
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for PowerPC SPE SHA-1.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the PowerPC SPE-accelerated SHA-1 algorithm
 * (`alg` structure) from the Linux kernel's cryptographic API. This cleanly
 * removes the `sha1-ppc-spe` driver from the system, releasing associated resources
 * and preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters the PowerPC SPE hardware-accelerated SHA-1 algorithm from the kernel crypto API.
 */
static void __exit ppc_spe_sha1_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

module_init(ppc_spe_sha1_mod_init);
module_exit(ppc_spe_sha1_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm, SPE optimized");

MODULE_ALIAS_CRYPTO("sha1");
MODULE_ALIAS_CRYPTO("sha1-ppc-spe");
