/**
 * @file sha512-neon-glue.c
 * @brief Glue code for NEON-accelerated SHA-384/SHA-512 implementations.
 * @details This file provides the C-level interface functions required to
 * integrate the NEON-optimized SHA-512/384 assembly implementation with the
 * Linux kernel's cryptographic API. Its primary responsibility is to wrap the
 * high-performance assembly functions and to manage the NEON/FPU hardware
 * context safely using `kernel_neon_begin()` and `kernel_neon_end()`. This
 * ensures that the sensitive Floating-Point Unit (FPU) state is preserved
 * across cryptographic operations, maintaining system stability. This file
 * is conditionally registered by `sha512-glue.c` when NEON support is detected
 * at runtime, providing a tiered approach to performance optimization.
 */
// SPDX-License-Identifier: GPL-2.0-only
/*
 * sha512-neon-glue.c - accelerated SHA-384/512 for ARM NEON
 *
 * Copyright (C) 2015 Linaro Ltd <ard.biesheuvel@linaro.org>
 */

#include <asm/neon.h>
#include <crypto/internal/hash.h>
#include <crypto/sha2.h>
#include <crypto/sha512_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include "sha512.h"

MODULE_ALIAS_CRYPTO("sha384-neon");
MODULE_ALIAS_CRYPTO("sha512-neon");

/**
 * @brief Entry point for the NEON-accelerated SHA-512 block transformation.
 * @details This function serves as the direct interface to the highly optimized
 * assembly implementation (e.g., in `sha512-neon.S`) that performs the core
 * SHA-512 compression using the ARM NEON SIMD instruction set. It is designed
 * for high performance, leveraging parallel processing capabilities for
 * cryptographic computations.
 *
 * @param state  Pointer to the SHA-512 state structure.
 * @param src    Pointer to the source data blocks.
 * @param blocks Number of 128-byte blocks to process.
 * Functional Utility: Dispatches to the NEON-optimized assembly routine for high-speed SHA-512 block processing.
 */
asmlinkage void sha512_block_data_order_neon(struct sha512_state *state,
					     const u8 *src, int blocks);

/**
 * @brief Implements the `shash` 'update' operation using NEON acceleration.
 * @details This wrapper function integrates the NEON-accelerated SHA-512/384
 * block processing into the kernel's generic cryptographic update mechanism.
 * Its primary responsibility is to ensure safe execution by managing the NEON
 * FPU context. `kernel_neon_begin()` is called to save the FPU state before
 * the high-performance assembly routine executes, and `kernel_neon_end()`
 * restores it afterward. This is crucial for preventing FPU state corruption
 * and maintaining system stability within the kernel environment.
 *
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return The number of bytes not processed.
 * Functional Utility: Processes data incrementally for SHA-512/384 hashing, integrating NEON acceleration with safe FPU context management.
 */
static int sha512_neon_update(struct shash_desc *desc, const u8 *data,
			      unsigned int len)
{
	int remain;

	// Pre-condition: Saves the current FPU state and enables NEON operations for kernel mode.
	kernel_neon_begin();
	// Functional Utility: Delegates the data buffering and block-wise update logic to a common SHA-512 base helper,
	// passing the NEON-accelerated assembly function for the actual block transformation.
	remain = sha512_base_do_update_blocks(desc, data, len,
					      sha512_block_data_order_neon);
	// Post-condition: Restores the FPU state, making it available for other kernel tasks.
	kernel_neon_end();
	return remain;
}

/**
 * @brief Implements the `shash` 'finup' (finalize and update) operation using NEON acceleration.
 * @details This function handles the final processing of data for SHA-512/384,
 * including padding any remaining input and computing the final hash digest.
 * It wraps the call to the NEON-accelerated assembly function (`sha512_block_data_order_neon`)
 * and ensures proper FPU context management by using `kernel_neon_begin()`
 * and `kernel_neon_end()`. This guarantees the integrity of the FPU state
 * during the critical final cryptographic computations.
 *
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-512/384 hashing process for the final data block, including padding and final digest generation, with NEON acceleration and safe FPU context management.
 */
static int sha512_neon_finup(struct shash_desc *desc, const u8 *data,
			     unsigned int len, u8 *out)
{
	// Pre-condition: Saves the FPU state, enabling safe use of NEON for the final cryptographic operations.
	kernel_neon_begin();
	// Functional Utility: Processes any remaining input data and applies SHA-512 padding using the NEON-accelerated block transformation.
	sha512_base_do_finup(desc, data, len, sha512_block_data_order_neon);
	// Post-condition: Restores the FPU state after the NEON-accelerated operations are complete.
	kernel_neon_end();
	return sha512_base_finish(desc, out);
}

/**
 * @brief Algorithm definitions for NEON-accelerated SHA-384 and SHA-512.
 * @details This array of `shash_alg` structures defines the NEON-accelerated
 * implementations for both SHA-384 (`sha384-neon`) and SHA-512 (`sha512-neon`).
 * It is designed for registration with the kernel's cryptographic API and is
 * exported for use by `sha512-glue.c`. The `cra_priority` of 300 is
 * intentionally set high to ensure that these highly optimized NEON versions
 * are preferred over generic C implementations, as well as over scalar ARM
 * assembly versions, on NEON-capable hardware.
 * Functional Utility: Defines and registers the high-priority, NEON-accelerated SHA-512/384 algorithms for the kernel crypto API.
 */
struct shash_alg sha512_neon_algs[] = { {
	.init			= sha384_base_init, @ Functional Role: Initializes SHA-384 hash state.
	.update			= sha512_neon_update, @ Functional Role: Updates SHA-384 hash state with new data using NEON.
	.finup			= sha512_neon_finup, @ Functional Role: Finalizes SHA-384 hash computation and outputs digest using NEON.
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA384_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha384",
		.cra_driver_name	= "sha384-neon",
		.cra_priority		= 300, @ Functional Role: Sets highest priority for NEON implementation.
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA384_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,

	}
},  {
	.init			= sha512_base_init, @ Functional Role: Initializes SHA-512 hash state.
	.update			= sha512_neon_update, @ Functional Role: Updates SHA-512 hash state with new data using NEON.
	.finup			= sha512_neon_finup, @ Functional Role: Finalizes SHA-512 hash computation and outputs digest using NEON.
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA512_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha512",
		.cra_driver_name	= "sha512-neon",
		.cra_priority		= 300, @ Functional Role: Sets highest priority for NEON implementation.
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA512_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
} };
