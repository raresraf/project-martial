/**
 * @file sha1_neon_glue.c
 * @brief Glue code for ARM NEON-accelerated SHA1 Secure Hash Algorithm.
 * @details This file provides the necessary interface to integrate a highly
 * optimized SHA1 implementation using ARM NEON SIMD instructions with the
 * Linux kernel's generic cryptographic API (shash). It is responsible for
 * registering the NEON-accelerated algorithm and, critically, for managing
 * the NEON hardware (FPU) context during hash operations to ensure system
 * stability and correct interaction with other kernel components that might
 * also utilize the FPU. This ensures that cryptographic operations benefit
 * from hardware acceleration while maintaining system integrity.
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
 * Functional Utility: Directly invokes the highly optimized ARM NEON assembly
 * routine for parallel processing of SHA1 data blocks. This significantly
 * accelerates the SHA1 hashing computation by leveraging SIMD capabilities.
 */
asmlinkage void sha1_transform_neon(struct sha1_state *state_h,
				    const u8 *data, int rounds);

/**
 * @brief Implements the 'update' operation with NEON acceleration and FPU context management.
 * @param desc The shash descriptor containing the hash state.
 * @param data The data to be hashed.
 * @param len The length of the data.
 * @return The number of bytes not processed (if any).
 *
 * This function wraps the NEON-accelerated block processing. It crucially
 * manages the NEON FPU context by calling kernel_neon_begin() before any NEON
 * operations and kernel_neon_end() after, ensuring that the FPU state is
 * correctly saved and restored when the kernel uses SIMD instructions.
 * Functional Utility: Orchestrates the update of the SHA1 hash state using
 * ARM NEON acceleration. It explicitly manages the NEON Floating-Point Unit (FPU)
 * context to prevent corruption of FPU registers by ensuring they are saved
 * before NEON operations and restored afterwards, providing a safe environment
 * for SIMD cryptographic processing within the kernel.
 */
static int sha1_neon_update(struct shash_desc *desc, const u8 *data,
			  unsigned int len)
{
	int remain;

	// Pre-condition: Saves the current FPU state and enables NEON operations within the kernel context.
	// Invariant: FPU state is preserved across this block, allowing safe use of NEON instructions.
	kernel_neon_begin();
	// Functional Utility: Delegates the main update logic to a generic helper,
	// passing the NEON-specific transform function. This allows the core SHA1
	// logic to remain generic while the block processing is accelerated.
	remain = sha1_base_do_update_blocks(desc, data, len,
					    sha1_transform_neon);
	// Post-condition: Restores the FPU state, making it safe for other kernel components.
	kernel_neon_end();

	return remain;
}

/**
 * @brief Implements the 'finup' (finalize and update) operation with NEON acceleration and FPU context management.
 * @param desc The shash descriptor.
 * @param data The final data chunk to be hashed.
 * @param len The length of the final data.
 * @param out The buffer to store the resulting 20-byte hash.
 * @return 0 on success.
 *
 * This function handles the final data segment, including padding, and computes
 * the final hash digest, all while safely managing the NEON FPU context.
 * Functional Utility: Finalizes the SHA1 hashing process, including padding
 * the last block and computing the final digest. It utilizes the NEON-accelerated
 * transform function while strictly managing the NEON FPU context to ensure
 * data integrity and system stability during the final cryptographic steps.
 */
static int sha1_neon_finup(struct shash_desc *desc, const u8 *data,
			   unsigned int len, u8 *out)
{
	// Pre-condition: Saves the FPU state before initiating NEON-accelerated final processing.
	// Invariant: FPU state is preserved across this block.
	kernel_neon_begin();
	// Functional Utility: Performs the final update and padding using the NEON transform,
	// ensuring all remaining data is incorporated into the hash.
	sha1_base_do_finup(desc, data, len, sha1_transform_neon);
	// Post-condition: Restores the FPU state.
	kernel_neon_end();

	// Functional Utility: Finalizes the hash calculation and writes the computed digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines and registers the NEON-accelerated SHA1 shash algorithm with the kernel's cryptographic API.
 * @details This structure registers the NEON implementation of SHA1. It provides
 * pointers to the NEON-specific update/finup functions and sets a high
 * priority (`cra_priority = 250`) to ensure this highly optimized version is
 * chosen over generic or less optimized SHA1 implementations on ARM CPUs
 * with NEON capabilities. This strategic prioritization ensures maximum
 * cryptographic performance when hardware acceleration is available.
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
 * @brief Module initialization function for the NEON-accelerated SHA1 algorithm.
 * @details This function is invoked when the kernel module is loaded.
 * Functional Utility: Prior to registering the NEON-accelerated SHA1 algorithm
 * with the kernel's crypto API, this function performs a critical check for
 * CPU NEON feature support. This ensures that the highly optimized NEON
 * implementation is only loaded and utilized on compatible hardware, preventing
 * potential system issues on non-NEON capable ARM processors.
 * @return 0 on successful registration, or -ENODEV if the CPU does not support NEON.
 */
static int __init sha1_neon_mod_init(void)
{
	// Pre-condition: Ensures the CPU possesses NEON capabilities before attempting to load the NEON-accelerated module.
	// Invariant: If NEON is not supported, the module will not register, preventing crashes.
	if (!cpu_has_neon())
		return -ENODEV;

	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for the NEON-accelerated SHA1 algorithm.
 * @details This function is invoked when the kernel module is unloaded.
 * Functional Utility: Unregisters the NEON-accelerated SHA1 algorithm from
 * the Linux kernel's cryptographic API, gracefully releasing all associated
 * resources and making the implementation unavailable. This ensures a clean
 * shutdown of the module and prevents resource leaks.
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
