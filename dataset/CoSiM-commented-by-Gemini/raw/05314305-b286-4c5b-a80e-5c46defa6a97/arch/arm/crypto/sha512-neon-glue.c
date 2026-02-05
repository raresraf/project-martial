/**
 * @file sha512-neon-glue.c
 * @brief Glue code for NEON-accelerated SHA-384/SHA-512 implementations.
 * @details This file provides the C-level interface functions required to
 * integrate the NEON-optimized SHA-512/384 assembly implementation with the
 * Linux kernel's cryptographic API. Its primary responsibility is to wrap the
 * assembly functions and manage the NEON/FPU hardware context safely using
 * kernel_neon_begin() and kernel_neon_end(). This file is conditionally
 * registered by sha512-glue.c when NEON support is detected at runtime.
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
 * @param state  Pointer to the SHA-512 state structure.
 * @param src    Pointer to the source data blocks.
 * @param blocks Number of 128-byte blocks to process.
 *
 * This function, defined in assembly, performs the core SHA-512 compression
 * using the ARM NEON SIMD instruction set for high performance.
 */
asmlinkage void sha512_block_data_order_neon(struct sha512_state *state,
					     const u8 *src, int blocks);

/**
 * @brief Implements the shash 'update' operation using NEON.
 * @param desc The shash descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return The number of bytes not processed.
 *
 * This wrapper function ensures that the NEON FPU context is saved before
 * calling the high-performance assembly routine and restored afterward, which
 * is essential for safe execution within the kernel.
 */
static int sha512_neon_update(struct shash_desc *desc, const u8 *data,
			      unsigned int len)
{
	int remain;

	// Pre-condition: Save FPU state and enable NEON for kernel mode.
	kernel_neon_begin();
	// Functional Utility: Delegates block processing to the NEON assembly function.
	remain = sha512_base_do_update_blocks(desc, data, len,
					      sha512_block_data_order_neon);
	// Post-condition: Restore FPU state.
	kernel_neon_end();
	return remain;
}

/**
 * @brief Implements the shash 'finup' operation using NEON.
 * @param desc The shash descriptor.
 * @param data The final data chunk.
 * @param len  The length of the final data.
 * @param out  The buffer for the resulting hash digest.
 * @return 0 on success.
 *
 * This function processes the final data segment and computes the hash digest,
 * wrapping the call to the NEON assembly function with FPU context management.
 */
static int sha512_neon_finup(struct shash_desc *desc, const u8 *data,
			     unsigned int len, u8 *out)
{
	// Pre-condition: Save FPU state.
	kernel_neon_begin();
	// Functional Utility: Perform final update and padding with the NEON implementation.
	sha512_base_do_finup(desc, data, len, sha512_block_data_order_neon);
	// Post-condition: Restore FPU state.
	kernel_neon_end();
	return sha512_base_finish(desc, out);
}

/**
 * @brief Algorithm definitions for NEON-accelerated SHA-384 and SHA-512.
 *
 * This structure array defines the NEON-accelerated algorithms for registration
 * with the crypto API. It is exported for use by sha512-glue.c. The high
 * priority (300) ensures this implementation is preferred over both the generic
 * C and scalar ARM assembly versions on NEON-capable hardware.
 */
struct shash_alg sha512_neon_algs[] = { {
	.init			= sha384_base_init,
	.update			= sha512_neon_update,
	.finup			= sha512_neon_finup,
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA384_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha384",
		.cra_driver_name	= "sha384-neon",
		.cra_priority		= 300,
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA384_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,

	}
},  {
	.init			= sha512_base_init,
	.update			= sha512_neon_update,
	.finup			= sha512_neon_finup,
	.descsize		= SHA512_STATE_SIZE,
	.digestsize		= SHA512_DIGEST_SIZE,
	.base			= {
		.cra_name		= "sha512",
		.cra_driver_name	= "sha512-neon",
		.cra_priority		= 300,
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY |
					  CRYPTO_AHASH_ALG_FINUP_MAX,
		.cra_blocksize		= SHA512_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
} };
