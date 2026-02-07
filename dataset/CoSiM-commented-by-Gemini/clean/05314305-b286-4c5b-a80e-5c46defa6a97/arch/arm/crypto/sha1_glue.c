/**
 * @file sha1_glue.c
 * @brief Glue code for ARM-optimized SHA1 Secure Hash Algorithm.
 * @details This file serves as an interface layer between the Linux kernel's
 * generic cryptographic API (shash) and a high-performance, ARM-specific
 * assembly implementation of the SHA1 algorithm. Its primary purpose is to
 * register the assembly version and map the standard shash operations (init,
 * update, final) to the corresponding optimized functions, allowing the rest
 * of the kernel to leverage this implementation transparently. This glue code
 * ensures that the ARM-optimized SHA1 block processing is seamlessly integrated
 * into the kernel's crypto framework, providing improved performance for ARM-based
 * systems.
 */
// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Cryptographic API.
 * Glue code for the SHA1 Secure Hash Algorithm assembler implementation
 *
 * This file is based on sha1_generic.c and sha1_ssse3_glue.c
 *
 * Copyright (c) Alan Smithee.
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 * Copyright (c) Mathias Krause <minipli@googlemail.com>
 */

#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/kernel.h>
#include <linux/module.h>

/**
 * @brief Processes SHA1 blocks using an ARM assembly implementation.
 * @param digest Pointer to the SHA1 state structure.
 * @param data Pointer to the input data blocks.
 * @param rounds The number of SHA1 blocks to process.
 *
 * This is the low-level entry point into the assembly-language portion of the
 * SHA1 implementation. It processes a given number of 64-byte blocks.
 * Functional Utility: This function serves as the direct invocation point
 * for the architecture-specific (ARM) SHA1 hash computation, offloading
 * cryptographic processing to highly optimized assembly routines for
 * performance gains.
 */
asmlinkage void sha1_block_data_order(struct sha1_state *digest,
		const u8 *data, int rounds);

/**
 * @brief Implements the 'update' operation using the ARM-optimized function.
 * @param desc The shash descriptor containing the hash state.
 * @param data The data to be hashed.
 * @param len The length of the data.
 * @return 0 on success.
 *
 * This function is the shash API's entry point for updating the hash state
 * with new data. It wraps the core block processing function.
 * Functional Utility: Integrates the ARM-optimized SHA1 block processing
 * function (`sha1_block_data_order`) into the generic `shash_desc` update flow,
 * efficiently consuming input data and maintaining the hash state.
 */
static int sha1_update_arm(struct shash_desc *desc, const u8 *data,
			   unsigned int len)
{
	/* Functional Utility: Compile-time assertion to ensure structural compatibility
	 * between `sha1_state` and `shash_desc`'s internal state representation.
	 * This guarantees correct memory layout for direct casting and state manipulation.
	 */
	BUILD_BUG_ON(offsetof(struct sha1_state, state) != 0);

	// Functional Utility: Delegates the block-wise update to a common helper function,
	// providing the ARM-specific block processing function as a callback.
	return sha1_base_do_update_blocks(desc, data, len,
					  sha1_block_data_order);
}

/**
 * @brief Implements the 'finup' (finalize and update) operation.
 * @param desc The shash descriptor.
 * @param data The final data chunk to be hashed.
 * @param len The length of the final data.
 * @param out The buffer to store the resulting 20-byte hash.
 * @return 0 on success.
 *
 * This function processes the last segment of data and then computes the
 * final hash digest.
 * Functional Utility: Completes the hashing process by handling any remaining
 * data, applying padding as per SHA1 specification, and computing the final
 * 20-byte message digest using the ARM-optimized block function.
 */
static int sha1_finup_arm(struct shash_desc *desc, const u8 *data,
			  unsigned int len, u8 *out)
{
	// Functional Utility: Uses a common helper for the final update and padding,
	// passing the ARM-specific block processing function.
	sha1_base_do_finup(desc, data, len, sha1_block_data_order);
	// Functional Utility: Finalizes the hash calculation and writes the computed digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @brief Defines the SHA1 shash algorithm implementation for the crypto API.
 * @details This structure registers the ARM assembly implementation of SHA1 with the
 * kernel's cryptographic subsystem. It specifies the algorithm's properties
 * (digest size, block size), associates the core operations (init, update, finup)
 * with their respective handler functions, and assigns a high priority to ensure
 * this optimized version is preferred over generic implementations.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	sha1_update_arm,
	.finup		=	sha1_finup_arm,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"sha1-asm",
		.cra_priority	=	150,
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};


/**
 * @brief Module initialization function.
 * @details This function is invoked when the kernel module is loaded.
 * Functional Utility: Registers the ARM-optimized SHA1 algorithm
 * (`alg` structure) with the Linux kernel's cryptographic API, making
 * it available for use by other kernel components. Returns 0 on success,
 * or a negative error code if registration fails.
 * @return 0 on success, or an error code on failure.
 */
static int __init sha1_mod_init(void)
{
	return crypto_register_shash(&alg);
}


/**
 * @brief Module cleanup function.
 * @details This function is invoked when the kernel module is unloaded.
 * Functional Utility: Unregisters the ARM-optimized SHA1 algorithm from
 * the Linux kernel's cryptographic API, releasing its resources and
 * making it unavailable for further cryptographic operations.
 */
static void __exit sha1_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}


module_init(sha1_mod_init);
module_exit(sha1_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm (ARM)");
MODULE_ALIAS_CRYPTO("sha1");
MODULE_AUTHOR("David McCullough <ucdevel@gmail.com>");
