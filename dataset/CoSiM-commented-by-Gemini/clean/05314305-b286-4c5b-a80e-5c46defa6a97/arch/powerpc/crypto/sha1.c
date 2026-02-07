// SPDX-License-Identifier: GPL-2.0-or-later
/**
 * @file sha1.c
 * @brief Generic PowerPC implementation of the SHA-1 Secure Hash Algorithm.
 * @details This file provides a PowerPC-optimized implementation of the SHA-1
 * (Secure Hash Algorithm 1) compression function. It integrates with the
 * Linux kernel's cryptographic API (`shash`) and serves as a baseline or
 * fallback accelerated implementation for PowerPC systems. This version
 * utilizes PowerPC-specific assembly routines for block transformation but
 * does not specifically leverage specialized hardware extensions like SPE (Signal
 * Processing Engine) or other SIMD capabilities, unlike more highly optimized
 * implementations (e.g., `sha1-spe-glue.c`).
 */
/*
 * Cryptographic API.
 *
 * powerpc implementation of the SHA1 Secure Hash Algorithm.
 *
 * Derived from cryptoapi implementation, adapted for in-place
 * scatterlist interface.
 *
 * Derived from "crypto/sha1.c"
 * Copyright (c) Alan Smithee.
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 */
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

/**
 * @brief Entry point for the PowerPC assembly implementation of SHA-1 block transformation.
 * @details This function is an `asmlinkage` declaration, indicating that its
 * implementation is provided in a separate PowerPC assembly file (`sha1-powerpc-asm.S`).
 * It performs the core SHA-1 block compression logic using optimized PowerPC
 * instructions, specifically for a single 64-byte block.
 *
 * @param state Pointer to the SHA-1 state array (5x u32 words).
 * @param src Pointer to the input data block (64-bytes).
 * Functional Utility: Dispatches to the PowerPC assembly routine for SHA-1 block processing.
 */
asmlinkage void powerpc_sha_transform(u32 *state, const u8 *src);

/**
 * @brief Processes SHA-1 blocks using the PowerPC-specific assembly transformation.
 * @details This function processes input data in 64-byte blocks, repeatedly
 * calling the `powerpc_sha_transform` assembly routine for each block. It
 * handles the iteration over multiple blocks, advancing the data pointer
 * for each transformation.
 *
 * @param sctx Pointer to the SHA-1 state structure.
 * @param data Pointer to the input data.
 * @param blocks The number of 64-byte blocks to process.
 * Functional Utility: Orchestrates the processing of multiple SHA-1 blocks using the PowerPC assembly transform.
 */
static void powerpc_sha_block(struct sha1_state *sctx, const u8 *data,
			      int blocks)
{
	// Block Logic: Iterates through the given number of 64-byte blocks.
	do {
		// Functional Utility: Invokes the PowerPC assembly routine to transform a single 64-byte SHA-1 block.
		powerpc_sha_transform(sctx->state, data);
		// Functional Utility: Advances the data pointer to the next 64-byte block.
		data += 64;
	} while (--blocks); // Functional Utility: Continues looping until all blocks are processed.
}

/**
 * @brief Implements the `shash` 'update' operation for PowerPC SHA-1.
 * @details This function integrates the PowerPC-specific SHA-1 block processing
 * into the Linux kernel's generic `shash` API for incremental updates. It acts
 * as a straightforward wrapper that delegates data buffering and partial block
 * processing to the `sha1_base_do_update_blocks` helper function, providing
 * `powerpc_sha_block` as the callback for core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param data The data to be hashed.
 * @param len  The length of the data.
 * @return 0 on success.
 * Functional Utility: Updates the SHA-1 hash state incrementally, leveraging PowerPC assembly for core block processing.
 */
static int powerpc_sha1_update(struct shash_desc *desc, const u8 *data,
			       unsigned int len)
{
	// Functional Utility: Delegates data buffering and block processing to the base SHA-1 update function,
	// using the PowerPC-specific `powerpc_sha_block` as the core block transformation.
	return sha1_base_do_update_blocks(desc, data, len, powerpc_sha_block);
}

/**
 * @brief Finalizes the SHA-1 hash computation for PowerPC.
 * @details This function handles the final steps of the SHA-1 algorithm. It applies
 * the necessary padding to the last data block, appends the total message length,
 * and then processes this final block using the PowerPC-specific transformation.
 * It utilizes the `sha1_base_do_finup` and `sha1_base_finish` helper functions,
 * providing `powerpc_sha_block` for the core block processing.
 *
 * @param desc The `shash_desc` descriptor.
 * @param src Pointer to any remaining partial input data.
 * @param len The length of the remaining partial input data.
 * @param out The buffer to store the final 20-byte SHA-1 hash digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-1 hashing process, including padding and length encoding, using PowerPC assembly.
 */
/* Add padding and return the message digest. */
static int powerpc_sha1_finup(struct shash_desc *desc, const u8 *src,
			      unsigned int len, u8 *out)
{
	// Functional Utility: Processes any remaining input data and applies SHA-1 padding using the PowerPC assembly transformation.
	sha1_base_do_finup(desc, src, len, powerpc_sha_block);
	// Functional Utility: Writes the final computed SHA-1 hash digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the PowerPC SHA-1 algorithm for the kernel's cryptographic API.
 * @details This structure registers the generic PowerPC SHA-1 algorithm
 * implementation. It specifies the algorithm's properties (digest size,
 * block size), associates the core operations (`init`, `update`, `finup`)
 * with their respective handler functions. The `cra_priority` is left at
 * its default (or a relatively low value if not explicitly set), indicating
 * that this version might be superseded by more highly optimized
 * implementations (e.g., SPE-accelerated versions) if available.
 * Functional Utility: Registers the PowerPC SHA-1 algorithm with the kernel crypto API.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	powerpc_sha1_update,
	.finup		=	powerpc_sha1_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"sha1-powerpc",
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Module initialization function for PowerPC SHA-1.
 * @details This function is the entry point when the kernel module is loaded.
 * It registers the PowerPC SHA-1 algorithm (`alg` structure) with the Linux
 * kernel's cryptographic API. This makes the `sha1-powerpc` driver available
 * to the system, providing a PowerPC-optimized SHA-1 hashing solution.
 * Functional Utility: Registers the PowerPC SHA-1 algorithm with the kernel crypto API.
 * @return 0 on successful registration, or an error code on failure.
 */
static int __init sha1_powerpc_mod_init(void)
{
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for PowerPC SHA-1.
 * @details This function is the exit point when the kernel module is unloaded.
 * Functional Utility: Unregisters the PowerPC SHA-1 algorithm (`alg` structure)
 * from the Linux kernel's cryptographic API. This cleanly removes the
 * `sha1-powerpc` driver from the system, releasing associated resources and
 * preventing any lingering references after the module is no longer in use.
 * This ensures proper resource management upon module unload.
 * Functional Utility: Unregisters the PowerPC SHA-1 algorithm from the kernel crypto API.
 */
static void __exit sha1_powerpc_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

module_init(sha1_powerpc_mod_init);
module_exit(sha1_powerpc_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm");

MODULE_ALIAS_CRYPTO("sha1");
MODULE_ALIAS_CRYPTO("sha1-powerpc");
