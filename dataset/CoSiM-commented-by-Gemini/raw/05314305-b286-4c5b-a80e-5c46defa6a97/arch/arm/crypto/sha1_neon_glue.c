/**
 * @file sha1_neon_glue.c
 * @brief Glue code for ARM NEON-accelerated SHA1 Secure Hash Algorithm.
 * @details This file provides the necessary interface to integrate a highly
 * optimized SHA1 implementation using ARM NEON SIMD instructions with the
 * Linux kernel's generic cryptographic API (shash). It is responsible for
 * registering the NEON-accelerated algorithm and, critically, for managing
 * the NEON hardware (FPU) context during hash operations to ensure system
 * stability.
 */
// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Glue code for the SHA1 Secure Hash Algorithm assembler implementation using
 * ARM NEON instructions.
 *
 * Copyright © 2014 Jussi Kivilinna <jussi.kivilinna@iki.fi>
 *
 * This file is based on sha1_generic.c and sha1_ssse3_glue.c:
 *  Copyright (c) Alan Smithee.
 *  Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 *  Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 *  Copyright (c) Mathias Krause <minipli@googlemail.com>
 *  Copyright (c) Chandramouli Narayanan <mouli@linux.intel.com>
 */

#include <asm/neon.h>
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

/**
 * @brief Processes multiple SHA1 blocks using ARM NEON instructions.
 * @param state_h Pointer to the SHA1 state structure.
 * @param data Pointer to the input data blocks.
 * @param rounds The number of SHA1 blocks to process.
 *
 * This is the low-level entry point to the assembly code that performs the
 * core SHA1 transformation using NEON acceleration.
 */
asmlinkage void sha1_transform_neon(struct sha1_state *state_h,
				    const u8 *data, int rounds);

/**
 * @brief Implements the 'update' operation with NEON acceleration.
 * @param desc The shash descriptor containing the hash state.
 * @param data The data to be hashed.
 * @param len The length of the data.
 * @return The number of bytes not processed (if any).
 *
 * This function wraps the NEON-accelerated block processing. It crucially
 * manages the NEON FPU context by calling kernel_neon_begin() before any NEON
 * operations and kernel_neon_end() after, ensuring that the FPU state is
 * correctly saved and restored when the kernel uses SIMD instructions.
 */
static int sha1_neon_update(struct shash_desc *desc, const u8 *data,
			  unsigned int len)
{
	int remain;

	// Pre-condition: Save the current FPU state and enable NEON use in kernel context.
	kernel_neon_begin();
	// Functional Utility: Delegates the main update logic to a generic helper,
	// passing the NEON-specific transform function.
	remain = sha1_base_do_update_blocks(desc, data, len,
					    sha1_transform_neon);
	// Post-condition: Restore the FPU state.
	kernel_neon_end();

	return remain;
}

/**
 * @brief Implements the 'finup' (finalize and update) operation with NEON.
 * @param desc The shash descriptor.
 * @param data The final data chunk to be hashed.
 * @param len The length of the final data.
 * @param out The buffer to store the resulting 20-byte hash.
 * @return 0 on success.
 *
 * This function handles the final data segment, including padding, and computes
 * the final hash digest, all while safely managing the NEON FPU context.
 */
static int sha1_neon_finup(struct shash_desc *desc, const u8 *data,
			   unsigned int len, u8 *out)
{
	// Pre-condition: Save FPU state before using NEON.
	kernel_neon_begin();
	// Functional Utility: Perform the final update and padding using the NEON transform.
	sha1_base_do_finup(desc, data, len, sha1_transform_neon);
	// Post-condition: Restore FPU state.
	kernel_neon_end();

	// Functional Utility: Finalize the hash calculation and write the digest.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the NEON-accelerated SHA1 shash algorithm for the crypto API.
 *
 * This structure registers the NEON implementation of SHA1. It provides
 * pointers to the NEON-specific update/finup functions and sets a high
 * priority to ensure it is chosen over non-accelerated versions on
 * NEON-capable CPUs.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	sha1_neon_update,
	.finup		=	sha1_neon_finup,
	.descsize		= SHA1_STATE_SIZE,
	.base		=	{
		.cra_name		= "sha1",
		.cra_driver_name	= "sha1-neon",
		.cra_priority		= 250,
		.cra_flags		= CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize		= SHA1_BLOCK_SIZE,
		.cra_module		= THIS_MODULE,
	}
};

/**
 * @brief Module initialization function.
 * @return 0 on success, or -ENODEV if the CPU does not support NEON.
 *
 * Checks for NEON CPU feature support before registering the cryptographic
 * algorithm. This ensures the accelerated implementation is only loaded on
 * compatible hardware.
 */
static int __init sha1_neon_mod_init(void)
{
	// Pre-condition: Ensure the CPU has NEON capabilities before registration.
	if (!cpu_has_neon())
		return -ENODEV;

	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function.
 *
 * Unregisters the SHA1 algorithm implementation from the kernel crypto API
 * when the module is unloaded.
 */
static void __exit sha1_neon_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

module_init(sha1_neon_mod_init);
module_exit(sha1_neon_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm, NEON accelerated");
MODULE_ALIAS_CRYPTO("sha1");
