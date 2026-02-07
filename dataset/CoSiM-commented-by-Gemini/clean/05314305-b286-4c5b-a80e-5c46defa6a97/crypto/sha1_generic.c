/**
 * @file sha1_generic.c
 * @brief Generic, portable implementation of the SHA-1 Secure Hash Algorithm.
 * @details This file provides a hardware-agnostic and fully portable implementation
 * of the SHA-1 (Secure Hash Algorithm 1) hash algorithm, conforming to the
 * Linux kernel's cryptographic API (`shash`). It serves as a reliable fallback
 * or default implementation for platforms and architectures that do not have
 * a more optimized or hardware-accelerated version of SHA-1 available. The
 * implementation is based on the `shash` (synchronous hash) interface,
 * providing standard `init`, `update`, and `finup` operations.
 */
// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Cryptographic API.
 *
 * SHA1 Secure Hash Algorithm.
 *
 * Derived from cryptoapi implementation, adapted for in-place
 * scatterlist interface.
 *
 * Copyright (c) Alan Smithee.
 * Copyright (c) Andrew McDonald <andrew@mcdonald.org.uk>
 * Copyright (c) Jean-Francois Dive <jef@linuxbe.org>
 */
#include <crypto/internal/hash.h>
#include <crypto/sha1.h>
#include <crypto/sha1_base.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/string.h>

/**
 * @var sha1_zero_message_hash
 * @brief The pre-computed SHA-1 hash of a zero-length message.
 *
 * @details This constant array contains the well-known 20-byte SHA-1 digest
 * for an empty input string or message. This value is defined by the SHA-1
 * standard and is useful for:
 * - **Self-Tests**: Verifying the correctness of SHA-1 implementations.
 * - **Optimizations**: Allowing algorithms to quickly return a known hash
 *   for empty inputs without performing full computation.
 * - **Consistency**: Ensuring all SHA-1 implementations produce the same
 *   output for this fundamental case.
 * Functional Utility: This constant provides the well-known digest for an
 * empty input, as defined by the SHA-1 standard. It can be used for
 * optimizations or self-tests where this specific value is expected.
 */
const u8 sha1_zero_message_hash[SHA1_DIGEST_SIZE] = {
	0xda, 0x39, 0xa3, 0xee, 0x5e, 0x6b, 0x4b, 0x0d,
	0x32, 0x55, 0xbf, 0xef, 0x95, 0x60, 0x18, 0x90,
	0xaf, 0xd8, 0x07, 0x09
};
EXPORT_SYMBOL_GPL(sha1_zero_message_hash);

/**
 * @brief Processes one or more full blocks of SHA-1 data.
 * @details This function is the core processing loop for the generic SHA-1
 * implementation. It iteratively calls the `sha1_transform` function for each
 * 64-byte block of input data (`src`), updating the algorithm's internal state
 * (`sst->state`). A temporary workspace (`temp`) is used during the transformation
 * and then explicitly zeroed for security.
 *
 * @param sst Pointer to the SHA-1 state structure.
 * @param src Pointer to the source data blocks.
 * @param blocks The number of 64-byte SHA-1 blocks to process.
 * Functional Utility: Iteratively applies the SHA-1 compression function to multiple input blocks.
 */
static void sha1_generic_block_fn(struct sha1_state *sst, u8 const *src,
				  int blocks)
{
	u32 temp[SHA1_WORKSPACE_WORDS];

	// Block Logic: Loops through each 64-byte block.
	while (blocks--) {
		// Functional Utility: Applies the core SHA-1 compression function to a single block.
		sha1_transform(sst->state, src, temp);
		// Functional Utility: Advances the source pointer to the next 64-byte block.
		src += SHA1_BLOCK_SIZE;
	}
	// Functional Utility: Securely zeroes out the temporary workspace to prevent information leakage.
	memzero_explicit(temp, sizeof(temp));
}

/**
 * @brief Updates the SHA-1 hash state with new data.
 * @details This function serves as the 'update' entry point for the `shash`
 * API for the generic SHA-1 implementation. It feeds a chunk of data into
 * the SHA-1 algorithm, efficiently managing any partial blocks buffered from
 * previous updates and processing all new full blocks. It delegates the complex
 * buffering logic to the base helper function `sha1_base_do_update_blocks`
 * and provides `sha1_generic_block_fn` for the actual 64-byte block processing.
 *
 * @param desc The `shash_desc` descriptor containing the hash state.
 * @param data Pointer to the input data.
 * @param len Length of the input data in bytes.
 * @return 0 on success.
 * Functional Utility: Increments the SHA-1 hash state with new data, handling partial blocks and using a generic block processor.
 */
static int crypto_sha1_update(struct shash_desc *desc, const u8 *data,
			      unsigned int len)
{
	// Functional Utility: Delegates data buffering, partial block handling, and full block processing to the base SHA-1 update function.
	return sha1_base_do_update_blocks(desc, data, len,
					  sha1_generic_block_fn);
}

/**
 * @brief Processes the final data chunk and computes the digest.
 * @details This function combines the 'update' and 'final' steps for the
 * generic SHA-1 algorithm. It takes the last piece of input data, applies
 * the necessary SHA-1 padding scheme (including message length append),
 * and computes the final 20-byte hash value. It leverages the base helper
 * `sha1_base_do_finup` to manage final data processing and padding,
 * and `sha1_base_finish` to write out the digest.
 *
 * @param desc The `shash_desc` descriptor containing the hash state.
 * @param data Pointer to the final input data chunk.
 * @param len Length of the final data chunk in bytes.
 * @param out Buffer to store the resulting 20-byte SHA-1 digest.
 * @return 0 on success.
 * Functional Utility: Completes the SHA-1 hashing process, including padding and final digest generation, using generic block processing.
 */
static int crypto_sha1_finup(struct shash_desc *desc, const u8 *data,
			     unsigned int len, u8 *out)
{
	// Functional Utility: Handles final data processing, padding, and delegates block processing to `sha1_generic_block_fn`.
	sha1_base_do_finup(desc, data, len, sha1_generic_block_fn);
	// Functional Utility: Writes the final computed SHA-1 hash digest to the output buffer.
	return sha1_base_finish(desc, out);
}

/**
 * @struct shash_alg alg
 * @brief Defines and registers the "sha1-generic" algorithm with the Crypto API.
 *
 * @details This structure populates the fields required by the Linux
 * cryptographic API to define a synchronous hash algorithm. It specifies the
 * digest size (`SHA1_DIGEST_SIZE`), block size (`SHA1_BLOCK_SIZE`), and
 * function pointers for initialization (`sha1_base_init`), update
 * (`crypto_sha1_update`), and finalization (`crypto_sha1_finup`).
 * The `cra_driver_name` ("sha1-generic") is the unique identifier for this
 * specific implementation, and the `cra_priority` is set to a low value (100)
 * to ensure that architecture-specific, more optimized versions are preferred
 * if available. The `CRYPTO_AHASH_ALG_BLOCK_ONLY` flag indicates that this
 * implementation can only process full blocks, relying on the base framework for
 * buffering and padding.
 * Functional Utility: This structure populates the fields required by the
 * cryptographic API to define a synchronous hash algorithm. It specifies the
 * digest size, block size, and function pointers for initialization, update,
 * and finalization. The `cra_driver_name` ("sha1-generic") is the unique
 * identifier for this specific implementation, and the priority is set to a
 * low value (100) so that architecture-specific, more optimized versions are
 * preferred if available. The `CRYPTO_AHASH_ALG_BLOCK_ONLY` flag indicates that this
 * implementation can only process full blocks, relying on the base framework for
 * buffering and padding.
 */
static struct shash_alg alg = {
	.digestsize	=	SHA1_DIGEST_SIZE,
	.init		=	sha1_base_init,
	.update		=	crypto_sha1_update,
	.finup		=	crypto_sha1_finup,
	.descsize	=	SHA1_STATE_SIZE,
	.base		=	{
		.cra_name	=	"sha1",
		.cra_driver_name=	"sha1-generic",
		.cra_priority	=	100, // Functional Utility: Low priority ensures optimized versions are preferred.
		.cra_flags	=	CRYPTO_AHASH_ALG_BLOCK_ONLY,
		.cra_blocksize	=	SHA1_BLOCK_SIZE,
		.cra_module	=	THIS_MODULE,
	}
};

/**
 * @brief Module initialization function for the generic SHA-1 implementation.
 * @details This function is the entry point when the `sha1_generic` kernel
 * module is loaded. It registers the generic SHA-1 `shash` algorithm (`alg`
 * structure) with the kernel's cryptographic API. This action makes the
 * `sha1-generic` algorithm available for other kernel subsystems that require
 * SHA-1 hashing, serving as a robust and portable baseline.
 *
 * @return 0 on successful registration, or an error code otherwise.
 * Functional Utility: Registers the generic SHA-1 algorithm, making it available system-wide.
 */
static int __init sha1_generic_mod_init(void)
{
	return crypto_register_shash(&alg);
}

/**
 * @brief Module cleanup function for the generic SHA-1 implementation.
 * @details This function is the exit point when the `sha1_generic` kernel
 * module is unloaded. It unregisters the generic SHA-1 algorithm (`alg`
 * structure) from the cryptographic API, ensuring a clean and proper
 * release of resources and preventing any lingering references after the
 * module is no longer in use.
 * Functional Utility: Unregisters the generic SHA-1 algorithm, ensuring a clean module exit.
 */
static void __exit sha1_generic_mod_fini(void)
{
	crypto_unregister_shash(&alg);
}

module_init(sha1_generic_mod_init);
module_exit(sha1_generic_mod_fini);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SHA1 Secure Hash Algorithm");

MODULE_ALIAS_CRYPTO("sha1");
MODULE_ALIAS_CRYPTO("sha1-generic");
